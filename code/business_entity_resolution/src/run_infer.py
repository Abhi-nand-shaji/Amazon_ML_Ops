#!/usr/bin/env python3
"""Decision step: model scores -> the two submission files.

Reads   artifacts/cache/candidates_test/country=<C>.parquet   (blocking)
        artifacts/cache/scores_test<tag>/country=<C>.parquet  (run_score.py)
        artifacts/decision_config<tag>.json                   (run_train.py)
Writes  output/matching_results.tsv   final matches (the leaderboard file)
        output/candidate_pairs.tsv    the exact candidate set the model scored

Every S1 entity of the test set gets exactly one row in both files (empty list when it has
no matches / candidates), and matches are by construction a subset of the candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from pipeline import candidate_countries, chunk_slices  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from run_score import scores_dir  # noqa: E402
from stage2 import BLOCKER_COLS, entity_context  # noqa: E402

# pyarrow always quotes header names ("source1_entity_id"), which the challenge validator rejects: the header is written by hand
TSV = pacsv.WriteOptions(delimiter="\t", quoting_style="none", include_header=False)


def replace_with_retry(src: Path, dst: Path, attempts: int = 60, delay: float = 1.0) -> None:
    """Atomic rename that also survives a virus scanner / search indexer briefly holding the freshly written file
    (Windows raises PermissionError / WinError 32 while another process has the file open)."""
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def joined_lists(s1_row: np.ndarray, s1_id: pa.Array, cand_id: pa.Array, keep: np.ndarray):
    """(entity ids, 'id1,id2,...' strings) for entities that have >=1 kept pair.
    Pairs must be grouped by entity (sorted by s1_row) -- true for blocker output."""
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return pa.array([], pa.string()), pa.array([], pa.string())
    rows = s1_row[idx]
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    offsets = np.r_[starts, len(idx)].astype(np.int32)
    lst = pa.ListArray.from_arrays(pa.array(offsets), cand_id.take(pa.array(idx)).cast(pa.string()))
    return s1_id.take(pa.array(idx[starts])).cast(pa.string()), pc.binary_join(lst, ",")


def group_max(sorted_group: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-row maximum of `values` inside each run of equal `sorted_group` (input grouped by entity)."""
    if len(values) == 0:
        return values.copy()
    first = np.flatnonzero(np.r_[True, sorted_group[1:] != sorted_group[:-1]])
    return np.repeat(np.maximum.reduceat(values, first), np.diff(np.r_[first, len(values)]))


def decide(p: np.ndarray, is_s2: np.ndarray, cfg: dict, s1_row: np.ndarray | None = None) -> np.ndarray:
    """Which candidate pairs become final matches under the saved decision config.

    modes:  global      p >= t
            per_source  p >= t_S2 for S2 candidates, p >= t_S3 for S3 candidates
            gated       per-source thresholds AND the entity's best candidate score >= gate (the first match needs a
                        higher bar than additional ones); needs `s1_row` (pairs grouped by entity)."""
    if cfg["mode"] == "gated":
        if s1_row is None:
            raise ValueError("gated decisions need s1_row")
        per_source = (is_s2 & (p >= cfg["s2_threshold"])) | (~is_s2 & (p >= cfg["s3_threshold"]))
        return per_source & (group_max(s1_row, p) >= cfg["gate"])
    if cfg["mode"] == "per_source":
        return (is_s2 & (p >= cfg["s2_threshold"])) | (~is_s2 & (p >= cfg["s3_threshold"]))
    return p >= cfg["global_threshold"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="`train` is only for functional tests of the writer (use --out-dir); the submission is the test split")
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="pipeline-variant suffix (candidates, scores and decision config)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    t = paths.suffix
    cfg = json.loads(paths.threshold_path().read_text())
    stage2 = None                                   # (booster, feature columns, decision config) when run_stage2 adopted it
    s2_cfg_path = paths.artifacts_dir / f"decision_config_stage2{t}.json"
    if s2_cfg_path.exists() and json.loads(s2_cfg_path.read_text()).get("adopted"):
        stage2 = (lgb.Booster(model_file=str(paths.artifacts_dir / f"stage2_model{t}.txt")),
                  json.loads((paths.artifacts_dir / f"stage2_features{t}.json").read_text()), json.loads(s2_cfg_path.read_text()))
        cfg = stage2[2]
    out_dir = Path(args.out_dir) if args.out_dir else paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_m = pa.schema([("source1_entity_id", pa.string()), ("matched_entity_ids", pa.string())])
    schema_c = pa.schema([("source1_entity_id", pa.string()), ("candidate_entity_ids", pa.string())])
    print(f"decision config: {cfg}", flush=True)

    t0 = time.time()
    seen_m, seen_c = [], []
    stats = {"pairs": 0, "matched_pairs": 0, "entities_with_match": 0, "entities_with_candidates": 0, "per_country": {}}
    n_s1_country = pq.read_table(paths.normalized_cache(args.split, "source1"), columns=["country"])["country"].value_counts().to_pylist()
    n_s1_country = {d["values"]: d["counts"] for d in n_s1_country}
    tmp_m, tmp_c = out_dir / "matching_results.tsv.tmp", out_dir / "candidate_pairs.tsv.tmp"
    # explicit file objects: pyarrow keeps a path-opened sink alive until the process exits, and on Windows the
    # final rename then fails with WinError 32 (POSIX would not care).
    with open(tmp_m, "wb") as fm, open(tmp_c, "wb") as fc, \
            pacsv.CSVWriter(fm, schema_m, write_options=TSV) as wm, pacsv.CSVWriter(fc, schema_c, write_options=TSV) as wc:
        fm.write(("\t".join(schema_m.names) + "\n").encode("utf-8"))
        fc.write(("\t".join(schema_c.names) + "\n").encode("utf-8"))
        for country in candidate_countries(paths, args.split):
            cand = pq.read_table(candidates_dir(paths, args.split) / f"country={country}.parquet",
                                 columns=["s1_row", "s1_entity_id", "cand_entity_id"] + (BLOCKER_COLS if stage2 else []))
            sc = pq.read_table(scores_dir(paths, args.split) / f"country={country}.parquet")
            s1_row = cand["s1_row"].to_numpy()
            if len(sc) != len(cand) or not np.array_equal(sc["s1_row"].to_numpy(), s1_row):
                raise RuntimeError(f"[{country}] scores are not row-aligned with the candidate file")
            s1_id = cand["s1_entity_id"].combine_chunks()
            cand_id = cand["cand_entity_id"].combine_chunks()
            is_s2 = pc.starts_with(cand_id, "S2-").to_numpy(zero_copy_only=False)
            p = sc["p"].to_numpy()
            if stage2:                              # entity-context re-scoring, in entity-aligned chunks (bounded memory)
                booster2, cols2, _ = stage2
                q = np.empty(len(p), np.float32)
                for a, b in chunk_slices(s1_row, 2_000_000):
                    ctx = entity_context(p[a:b], is_s2[a:b], s1_row[a:b], {c: cand[c].slice(a, b - a).to_numpy() for c in BLOCKER_COLS})
                    q[a:b] = booster2.predict(ctx[cols2], num_threads=os.cpu_count())
                p = q
            keep = decide(p, is_s2, cfg, s1_row)

            ents_m, list_m = joined_lists(s1_row, s1_id, cand_id, keep)
            ents_c, list_c = joined_lists(s1_row, s1_id, cand_id, np.ones(len(s1_row), bool))
            wm.write_table(pa.table({"source1_entity_id": ents_m, "matched_entity_ids": list_m}, schema=schema_m))
            wc.write_table(pa.table({"source1_entity_id": ents_c, "candidate_entity_ids": list_c}, schema=schema_c))
            seen_m.append(ents_m)
            seen_c.append(ents_c)
            stats["pairs"] += len(s1_row)
            stats["matched_pairs"] += int(keep.sum())
            stats["entities_with_match"] += len(ents_m)
            stats["entities_with_candidates"] += len(ents_c)
            n_ent = n_s1_country.get(country, 0)
            best = group_max(s1_row, p)[np.r_[True, s1_row[1:] != s1_row[:-1]]] if len(p) else np.zeros(0, np.float32)
            stats["per_country"][country] = {
                "s1_entities": n_ent, "candidate_pairs": len(s1_row), "candidates_per_entity": len(s1_row) / max(1, n_ent),
                "share_without_candidates": 1 - len(best) / max(1, n_ent),
                "entities_with_match": len(ents_m), "share_with_match": len(ents_m) / max(1, n_ent),
                "predicted_matches_per_entity": int(keep.sum()) / max(1, n_ent),
                "predicted_matches_per_entity_that_has_one": int(keep.sum()) / max(1, len(ents_m)),
                "share_s2_of_matches": float(is_s2[keep].mean()) if keep.any() else None,
                "mean_score_of_kept_pairs": float(p[keep].mean()) if keep.any() else None,
                # label-free shape of the decision problem: each entity's best candidate score (no candidate = no plausible match)
                "share_best_score_ge_0.97": float((best >= 0.97).sum() / max(1, n_ent)),
                "share_best_score_0.4_to_0.9": float(((best >= 0.4) & (best < 0.9)).sum() / max(1, n_ent)),
                "share_best_score_below_0.2_or_no_candidate": float(((best < 0.2).sum() + n_ent - len(best)) / max(1, n_ent)),
            }
            print(f"  [{country}] {len(s1_row):,} pairs -> {int(keep.sum()):,} matches for {len(ents_m):,} entities "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # entities that never made it into a partition file (no matches / no candidates) get an empty row
        all_ids = pq.read_table(paths.normalized_cache(args.split, "source1"), columns=["entity_id"])["entity_id"].combine_chunks().cast(pa.string())
        for seen, writer, schema, col in ((seen_m, wm, schema_m, "matched_entity_ids"), (seen_c, wc, schema_c, "candidate_entity_ids")):
            have = pa.chunked_array(seen, type=pa.string()) if seen else pa.chunked_array([], type=pa.string())
            missing = all_ids.filter(pc.invert(pc.is_in(all_ids, value_set=have.combine_chunks() if have.num_chunks else pa.array([], pa.string()))))
            writer.write_table(pa.table({"source1_entity_id": missing, col: pa.array([""] * len(missing), pa.string())}, schema=schema))
            print(f"  {col}: {len(missing):,} entities written with an empty list", flush=True)

    replace_with_retry(tmp_m, out_dir / "matching_results.tsv")
    replace_with_retry(tmp_c, out_dir / "candidate_pairs.tsv")
    stats.update(n_s1_entities=len(all_ids), runtime_seconds=round(time.time() - t0, 1), decision_config=cfg,
                 stage=2 if stage2 else 1)
    (paths.reports_dir / f"inference_summary{t}.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {out_dir / 'matching_results.tsv'} and {out_dir / 'candidate_pairs.tsv'}: {stats}", flush=True)


if __name__ == "__main__":
    main()
