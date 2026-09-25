#!/usr/bin/env python3
"""Featurize and score a split's candidate pairs with the trained model, streaming.

Reads   artifacts/cache/candidates_<split>[_tag]/country=<C>.parquet
Writes  artifacts/cache/scores_<split>[_tag]/country=<C>.parquet   (row-aligned with the candidate file)
        columns: s1_row, pool_row, p (float32 match probability)

Features are computed chunk by chunk and discarded after scoring, so memory stays bounded
no matter how many pairs the split has. The decision step (thresholds, singleton gating,
writing the submission files) lives in run_infer.py and only needs these scores, so it
can be re-tuned without recomputing any features.
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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from pipeline import candidate_countries, iter_feature_chunks  # noqa: E402


def scores_dir(paths: Paths, split: str) -> Path:
    return paths.scores_dir(split)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="pipeline-variant suffix (candidates, model and scores all use it)")
    ap.add_argument("--chunk-pairs", type=int, default=1_000_000)
    ap.add_argument("--countries", default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--max-chunks", type=int, default=None, help="debug: stop after this many chunks per country")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    booster = lgb.Booster(model_file=str(paths.model_path()))
    fcols = json.loads(paths.feature_list_path().read_text())["feature_columns"]
    out_dir = scores_dir(paths, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    countries = args.countries.split(",") if args.countries else candidate_countries(paths, args.split)

    for country in countries:
        final = out_dir / f"country={country}.parquet"
        if final.exists() and not args.force:
            print(f"[{country}] scores already present, skipping")
            continue
        tmp = final.with_name(final.name + ".tmp")
        writer, n, t0 = None, 0, time.time()
        for i, ch in enumerate(iter_feature_chunks(paths, args.split, args.chunk_pairs, [country])):
            t1 = time.time()
            p = booster.predict(ch.feats[fcols], num_threads=args.threads).astype(np.float32)
            tbl = pa.table({"s1_row": ch.s1_row.astype(np.int32), "pool_row": ch.pool_row.astype(np.int32), "p": p})
            writer = writer or pq.ParquetWriter(tmp, tbl.schema)
            writer.write_table(tbl)
            n += len(p)
            print(f"    [{country}] {n:,} pairs scored ({time.time() - t0:.0f}s elapsed; predict {time.time() - t1:.1f}s)", flush=True)
            if args.max_chunks and i + 1 >= args.max_chunks:
                break
        if writer is not None:
            writer.close()
            if args.max_chunks:
                print(f"[{country}] debug run: partial scores left in {tmp}")
            else:
                tmp.replace(final)
                print(f"[{country}] scored {n:,} pairs in {time.time() - t0:.0f}s -> {final}", flush=True)


if __name__ == "__main__":
    main()
