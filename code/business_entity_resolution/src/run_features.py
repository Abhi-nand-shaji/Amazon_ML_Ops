#!/usr/bin/env python3
"""Featurize a split's blocked candidate pairs and persist them (used for TRAIN).

Reads   artifacts/cache/candidates_<split>/country=<C>.parquet   (run_blocking.py)
Writes  artifacts/cache/features_<split>/country=<C>.parquet
Columns: model features + s1_row, pool_row, s1_entity_id, cand_entity_id (+ label for train).

The TEST split is not persisted this way (tens of millions of pairs x ~80 features);
run_infer.py featurizes and scores it in a streaming fashion instead.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from pipeline import candidate_countries, iter_feature_chunks, load_meta  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402


def features_dir(paths: Paths, split: str) -> Path:
    return paths.features_dir(split)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="pipeline-variant suffix (must match the blocking run)")
    ap.add_argument("--chunk-pairs", type=int, default=1_000_000)
    ap.add_argument("--countries", default=None, help="comma list; default = every country with candidates")
    ap.add_argument("--force", action="store_true", help="recompute countries that already have a features file")
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    out_dir = features_dir(paths, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(paths, args.split)
    countries = args.countries.split(",") if args.countries else candidate_countries(paths, args.split)

    truth = None
    if args.split == "train":
        sampled = set(pq.read_table(candidates_dir(paths, "train") / "s1_entities.parquet", columns=["entity_id"])["entity_id"].to_pylist())
        truth = load_truth(paths.ground_truth(), sampled)
        print(f"ground truth loaded for {len(truth):,} sampled S1 entities "
              f"({sum(len(v) for v in truth.values()):,} true pairs)", flush=True)

    for country in countries:
        final = out_dir / f"country={country}.parquet"
        if final.exists() and not args.force:
            print(f"[{country}] features already present, skipping ({final.stat().st_size / 1e6:.0f} MB)")
            continue
        tmp = final.with_name(final.name + ".tmp")
        writer, n_rows, n_pos, t0 = None, 0, 0, time.time()
        for ch in iter_feature_chunks(paths, args.split, args.chunk_pairs, [country]):
            df = ch.feats
            df["s1_row"], df["pool_row"] = ch.s1_row, ch.pool_row
            df["s1_entity_id"], df["cand_entity_id"] = ch.s1_entity_id.to_pylist(), ch.cand_entity_id.to_pylist()
            if truth is not None:
                df["label"] = label_pairs(ch.s1_entity_id, ch.cand_entity_id, truth)
                n_pos += int(df["label"].sum())
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            n_rows += len(df)
            print(f"    [{country}] {n_rows:,} pairs featurized ({time.time() - t0:.0f}s)", flush=True)
        if writer is not None:
            writer.close()
            tmp.replace(final)
        print(f"[{country}] done: {n_rows:,} pairs" + (f", {n_pos:,} positive" if truth is not None else "")
              + f" in {time.time() - t0:.0f}s -> {final}", flush=True)


if __name__ == "__main__":
    main()
