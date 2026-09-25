#!/usr/bin/env python3
"""Candidate generation, final step: shortlist -> candidate set.

Reads   artifacts/cache/shortlist_<split>[_tag]/   run_blocking.py (retrieval + ranker: <=30 pairs per S1 entity)
        artifacts/candidate_filter.{txt,json}      train_candidate_filter.py (filter model and probability floor)
Writes  artifacts/cache/candidates_<split>[_tag]/  same files and columns + `filter_score`, only the kept pairs

The candidate set is exactly what the matcher scores and what output/candidate_pairs.tsv lists. Per S1 entity it keeps
the shortlisted pairs whose filter probability clears the floor (at most `--max-candidates`, best first, ties -> lower
pool row). Training split: out-of-fold probabilities (each pair scored by a filter that never saw its entity), so the
matcher is trained and validated on candidate sets cut exactly like the test split's; test split: the filter fitted on
all sampled training entities, scored here chunk by chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import select_top  # noqa: E402
from candidate_filter import FILTER_FEATURES, country_filter_features, shortlist_countries  # noqa: E402
from config import Paths  # noqa: E402
from run_blocking import candidates_dir, file_digest, shortlist_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="pipeline-variant suffix (must match the blocking run)")
    ap.add_argument("--min-score", type=float, default=None, help="probability floor (default: the one chosen at training)")
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-pairs", type=int, default=1_000_000)
    ap.add_argument("--countries", default=None, help="comma list (lets country partitions run as parallel processes)")
    ap.add_argument("--force", action="store_true", help="discard an existing candidate set made with another rule")
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    src, dst = shortlist_dir(paths, args.split), candidates_dir(paths, args.split)
    if not (src / "_meta.json").exists():
        sys.exit(f"no shortlist in {src}: run run_blocking.py --split {args.split} first")
    fcfg = json.loads(paths.filter_config_path().read_text())
    floor = fcfg["min_score"] if args.min_score is None else args.min_score
    if fcfg["features"] != FILTER_FEATURES:
        sys.exit("candidate_filter.json was trained on a different feature list: re-run train_candidate_filter.py")
    meta = {**json.loads((src / "_meta.json").read_text()),
            "candidate_filter": {"model": file_digest(paths.filter_model_path()), "min_score": floor,
                                 "max_candidates": args.max_candidates,
                                 "scores": "out-of-fold" if args.split == "train" else "model fitted on all sampled training entities"}}
    if dst.exists():
        old = json.loads((dst / "_meta.json").read_text()) if (dst / "_meta.json").exists() else None
        if args.force:
            shutil.rmtree(dst)
        elif old != meta:
            sys.exit(f"{dst} was produced with a different configuration:\n  existing: {old}\n  requested: {meta}\n"
                     f"Re-run with --force to discard it.")
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "_meta.json").exists():
        (dst / "_meta.json").write_text(json.dumps(meta, indent=2))
    if not (dst / "s1_entities.parquet").exists():
        shutil.copyfile(src / "s1_entities.parquet", dst / "s1_entities.parquet")
    ents = pq.read_table(dst / "s1_entities.parquet", columns=["country"])["country"]
    n_by_country = {d["values"]: d["counts"] for d in pc.value_counts(ents).to_pylist()}

    oof = None
    if args.split == "train":
        oof = pq.read_table(paths.filter_oof_path())
    else:
        booster = lgb.Booster(model_file=str(paths.filter_model_path()))

    t0 = time.time()
    print(f"[{args.split}] candidate cut: filter probability >= {floor}, at most {args.max_candidates} per S1 entity "
          f"({src.name} -> {dst.name})", flush=True)
    countries = args.countries.split(",") if args.countries else shortlist_countries(paths, args.split)
    for country in countries:
        final = dst / f"country={country}.parquet"
        if final.exists():
            print(f"  [{country}] already cut, skipping", flush=True)
            continue
        short = pq.read_table(src / f"country={country}.parquet")
        s1_row, pool_row = short["s1_row"].to_numpy(), short["pool_row"].to_numpy()
        if oof is not None:
            o = oof.filter(pc.equal(oof["country"], country))
            if not (np.array_equal(o["s1_row"].to_numpy(), s1_row) and np.array_equal(o["pool_row"].to_numpy(), pool_row)):
                sys.exit(f"[{country}] out-of-fold scores are not row-aligned with the shortlist: re-run train_candidate_filter.py")
            score = o["filter_score"].to_numpy()
        else:
            score = country_filter_features(paths, args.split, country, short, chunk_pairs=args.chunk_pairs,
                                            predict=lambda X: booster.predict(X, num_threads=args.threads))
        sel = select_top(s1_row, pool_row, score, args.max_candidates, floor)
        out = short.take(pa.array(sel)).append_column("filter_score", pa.array(score[sel].astype(np.float32)))
        tmp = final.with_name(final.name + ".tmp")
        pq.write_table(out, tmp)
        tmp.replace(final)
        n_s1 = n_by_country.get(country, 0)
        n_with = len(np.unique(s1_row[sel]))
        print(f"  [{country}] {short.num_rows:,} shortlisted -> {len(sel):,} candidates: {len(sel) / max(1, n_s1):.2f} per S1 "
              f"entity (shortlist {short.num_rows / max(1, n_s1):.2f}); {n_s1 - n_with:,} of {n_s1:,} entities without "
              f"candidates ({time.time() - t0:.0f}s)", flush=True)
        del short, out, score
    print(f"[{args.split}] candidate cut complete in {time.time() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
