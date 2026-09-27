#!/usr/bin/env python3
"""Score the TEST candidate pairs with the stage-2 stacker (run_stack.py), streaming per country partition.

Reads   artifacts/cache/candidates_test/country=<C>.parquet        the candidate set (unchanged: candidate_pairs.tsv)
        artifacts/cache/scores_test/country=<C>.parquet            first-stage scores p1 (run_score.py)
        artifacts/stack<exp>/stack_model.txt, stack_features.json   the stacker
        --ce-scores + --pair-index                                  cross-encoder scores from the cloud run (optional)
Writes  artifacts/cache/scores_test_stack<exp>/country=<C>.parquet  s1_row, pool_row, p (stacker probability),
        row-aligned with the candidate file, so run_infer.py --scores-dir ... --decision-config ... writes the submission.

Graph features use p1 of EVERY test pair of the partition (the complete candidate graph), exactly as in training.
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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collective import pool_side_features  # noqa: E402
from config import Paths  # noqa: E402
from pipeline import candidate_countries, iter_feature_chunks, load_meta, load_partition  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from run_score import scores_dir  # noqa: E402
from build_v4_candidates import EXTRA as V4_EXTRA  # noqa: E402
from stack_features import CE_COL, graph_features  # noqa: E402
from stack_v4 import pool_side_v4  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--exp", default="")
    ap.add_argument("--tag", default="", help="candidate / first-stage-score variant (v4: candidates_test_v4, scores_test_v4)")
    ap.add_argument("--ce-scores", default=None, help="ce_scores.parquet from kaggle_ce.py (pair_id, p_ce)")
    ap.add_argument("--pair-index", default=None, help="local_pair_index.parquet from export_ce_package.py")
    ap.add_argument("--chunk-pairs", type=int, default=1_000_000)
    ap.add_argument("--countries", default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    sdir = Paths(args.root).artifacts_dir / f"stack{('_' + args.exp) if args.exp else ''}"
    spec = json.loads((sdir / "stack_features.json").read_text())
    n_ens = int(spec.get("ensemble", 0))          # exp_stack_ensemble.py: seed-bagged members, probabilities averaged
    boosters = ([lgb.Booster(model_file=str(sdir / f"stack_model_{k}.txt")) for k in range(n_ens)] if n_ens
                else [lgb.Booster(model_file=str(sdir / "stack_model.txt"))])
    fcols = spec["feature_columns"]
    out_dir = Paths(args.root).cache_dir / f"scores_test_stack{('_' + args.exp) if args.exp else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ce_map = None
    if spec.get("uses_ce"):
        if not (args.ce_scores and args.pair_index):
            sys.exit("this stacker uses the cross-encoder score: pass --ce-scores and --pair-index")
        maps = []
        for sc_f, ix_f in zip(args.ce_scores.split(","), args.pair_index.split(",")):
            idx = pq.read_table(ix_f, columns=["pair_id", "set", "country", "s1_row", "pool_row"]).to_pandas()
            idx = idx[idx["set"] == "test"]
            maps.append(idx.merge(pd.read_parquet(sc_f), on="pair_id", how="inner"))
        ce_map = pd.concat(maps, ignore_index=True).drop_duplicates(["country", "s1_row", "pool_row"])
        print(f"cross-encoder scores for test pairs: {ce_map[CE_COL].notna().mean():.4%} of {len(ce_map):,}", flush=True)
    countries = args.countries.split(",") if args.countries else candidate_countries(paths, "test")
    meta = load_meta(paths, "test")
    t00 = time.time()
    for country in countries:
        final = out_dir / f"country={country}.parquet"
        if final.exists():
            print(f"[{country}] already scored, skipping", flush=True)
            continue
        t0 = time.time()
        cand = pq.read_table(candidates_dir(paths, "test") / f"country={country}.parquet", columns=["s1_row", "pool_row"])
        sc = pq.read_table(scores_dir(paths, "test") / f"country={country}.parquet")
        s1_row_all, pool_row_all = cand["s1_row"].to_numpy(), cand["pool_row"].to_numpy()
        if not (np.array_equal(sc["s1_row"].to_numpy(), s1_row_all) and np.array_equal(sc["pool_row"].to_numpy(), pool_row_all)):
            sys.exit(f"[{country}] first-stage scores are not row-aligned with the candidates")
        p1_all = sc["p"].to_numpy().astype(np.float64)
        extra_all = None
        if spec.get("v4"):
            extra_all = pq.read_table(candidates_dir(paths, "test") / f"country={country}.parquet", columns=V4_EXTRA).to_pandas()
            is_new = (extra_all["src_dense"].fillna(0).to_numpy() > 0) | (extra_all["src_hop"].fillna(0).to_numpy() > 0)
            ps_all = pool_side_v4(pool_row_all[~is_new], p1_all[~is_new], pool_row_all, p1_all, is_new)
        else:
            ps_all = pd.DataFrame(pool_side_features(pool_row_all, p1_all))
        p_ce_all = None
        if ce_map is not None:
            m = ce_map[ce_map["country"] == country]
            key = pd.Series(m[CE_COL].to_numpy(), index=m["s1_row"].to_numpy().astype(np.int64) << 32 | m["pool_row"].to_numpy().astype(np.int64))
            p_ce_all = key.reindex(s1_row_all.astype(np.int64) << 32 | pool_row_all.astype(np.int64)).to_numpy()
            print(f"  [{country}] cross-encoder coverage {np.isfinite(p_ce_all).mean():.4%}", flush=True)
        s1, pool, _ = load_partition(paths, "test", country, meta, s1_cols=["entity_id", "address_original"],
                                     pool_cols=["entity_id", "name_normalized", "address_normalized", "address_original"],
                                     with_full=False)
        pool_name, pool_addr = pool["name_normalized"].combine_chunks(), pool["address_normalized"].combine_chunks()
        s1_raw, pool_raw = s1["address_original"].combine_chunks(), pool["address_original"].combine_chunks()
        del pool, s1
        tmp = final.with_name(final.name + ".tmp")
        writer, a = None, 0
        for ch in iter_feature_chunks(paths, "test", args.chunk_pairs, [country]):
            b = a + len(ch.s1_row)
            if not (np.array_equal(ch.s1_row, s1_row_all[a:b]) and np.array_equal(ch.pool_row, pool_row_all[a:b])):
                sys.exit(f"[{country}] feature chunks are not aligned with the candidate file")
            X = ch.feats
            X["p1"] = p1_all[a:b].astype(np.float32)
            if p_ce_all is not None:
                X[CE_COL] = p_ce_all[a:b].astype(np.float32)
            g = graph_features(ch.s1_row, ch.pool_row, p1_all[a:b], X["is_s2"].to_numpy(), pool_name, pool_addr,
                               ps_all.iloc[a:b].reset_index(drop=True), s1_addr_raw=s1_raw, pool_addr_raw=pool_raw)
            for c in g.columns:
                X[c] = g[c].to_numpy()
            if extra_all is not None:
                for c in V4_EXTRA:
                    X[c] = extra_all[c].to_numpy()[a:b]
            q = np.mean([bo.predict(X[fcols], num_threads=args.threads) for bo in boosters], axis=0).astype(np.float32)
            t = pa.table({"s1_row": ch.s1_row.astype(np.int32), "pool_row": ch.pool_row.astype(np.int32), "p": q})
            writer = writer or pq.ParquetWriter(tmp, t.schema)
            writer.write_table(t)
            a = b
            print(f"    [{country}] {b:,}/{len(s1_row_all):,} pairs ({time.time() - t0:.0f}s)", flush=True)
        writer.close()
        if a != len(s1_row_all):
            sys.exit(f"[{country}] scored {a:,} of {len(s1_row_all):,} pairs")
        tmp.replace(final)
        print(f"[{country}] done ({time.time() - t0:.0f}s)", flush=True)
    print(f"stacker scores written to {out_dir} ({(time.time() - t00) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
