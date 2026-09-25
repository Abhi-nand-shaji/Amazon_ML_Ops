#!/usr/bin/env python3
"""Normalize every raw source file once and cache the result as parquet.

Downstream stages (blocking, feature engineering) read the cached parquet
instead of re-running normalization, which is the expensive step.

Robustness properties (this step takes ~10-15 minutes and the box it was
developed on restarted mid-run twice):

* Each file is processed in bounded row-chunks. normalize_name/
  normalize_address each build several intermediate Series copies, so doing
  a 5M+ row file in one shot peaked at ~11-12GB RSS on a 13GB machine;
  chunking bounds the peak to O(chunk_size).
* Every chunk is written to its own part file via tmp-file + rename, so a
  killed run never leaves a half-written part that looks valid, and a
  re-run resumes from the first missing chunk instead of starting over.
* The final single-file cache is assembled from the parts under a tmp name,
  its row count is checked against the raw file, and only then renamed into
  place. The final path therefore exists if and only if it is complete.
* An explicit schema is used so chunks can never disagree on column types
  (e.g. a chunk where every postal_code is null being inferred as null-type).
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from text_normalize import normalize_address, normalize_name  # noqa: E402

SPLITS_SOURCES = [
    ("train", "source1"), ("train", "source2"), ("train", "source3"),
    ("test", "source1"), ("test", "source2"), ("test", "source3"),
]
SOURCE_TAG = {"source1": "S1", "source2": "S2", "source3": "S3"}
CHUNK_SIZE = 1_000_000

# Matches the schema of the first two cache files produced before chunking
# was introduced, so all six files share identical column types.
SCHEMA = pa.schema([
    ("entity_id", pa.large_string()),
    ("country", pa.large_string()),
    ("name_original", pa.large_string()),
    ("name_script", pa.string()),
    ("name_normalized", pa.large_string()),
    ("name_compact", pa.large_string()),
    ("name_tokens", pa.list_(pa.string())),
    ("name_sorted_tokens", pa.large_string()),
    ("name_core", pa.large_string()),
    ("address_original", pa.large_string()),
    ("address_normalized", pa.large_string()),
    ("address_compact", pa.large_string()),
    ("address_tokens", pa.list_(pa.string())),
    ("address_sorted_tokens", pa.large_string()),
    ("postal_code", pa.large_string()),
    ("house_number", pa.large_string()),
    ("source", pa.large_string()),
])


def _is_complete(final: Path) -> bool:
    if not final.exists() or final.stat().st_size == 0:
        return False
    try:
        return pq.ParquetFile(final).metadata.num_rows > 0
    except Exception:
        return False


def _normalize_chunk(chunk: pd.DataFrame, tag: str) -> pa.Table:
    name_reprs = normalize_name(chunk["business_name"])
    addr_reprs = normalize_address(chunk["business_address"])
    out = pd.DataFrame({"entity_id": chunk["entity_id"].astype(str)})
    out["country"] = chunk["country"].astype(str)
    for k, v in name_reprs.items():
        out[k] = v
    for k, v in addr_reprs.items():
        out[k] = v
    out["source"] = tag
    return pa.Table.from_pandas(out[SCHEMA.names], schema=SCHEMA, preserve_index=False)


def build_one(paths: Paths, split: str, source: str, chunk_size: int = CHUNK_SIZE, force: bool = False) -> None:
    final = paths.normalized_cache(split, source)
    if _is_complete(final) and not force:
        print(f"  {split}/{source}: already cached ({final.stat().st_size/1e6:.1f} MB), skipping")
        return

    t0 = time.time()
    df = pd.read_csv(paths.raw(split, source), sep="\t", engine="pyarrow", dtype_backend="pyarrow")
    n = len(df)
    tag = SOURCE_TAG[source]
    # chunk size is part of the dir name so parts from a different chunking never mix
    parts_dir = final.parent / f"{final.stem}.parts_cs{chunk_size}"
    parts_dir.mkdir(exist_ok=True)

    n_chunks = (n + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        part = parts_dir / f"part-{i:05d}.parquet"
        if part.exists():
            print(f"  {split}/{source}: chunk {i+1}/{n_chunks} already done (resume), skipping")
            continue
        start = i * chunk_size
        chunk = df.iloc[start : start + chunk_size].reset_index(drop=True)
        table = _normalize_chunk(chunk, tag)
        tmp = part.with_name(part.name + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(part)  # replace(), not rename(): rename() raises on Windows if the target exists
        print(f"  {split}/{source}: chunk {i+1}/{n_chunks} ({len(chunk):,} rows) written, {time.time()-t0:.1f}s elapsed", flush=True)
        del chunk, table
        gc.collect()
    del df
    gc.collect()

    parts = sorted(parts_dir.glob("part-*.parquet"))
    if len(parts) != n_chunks:
        raise RuntimeError(f"{split}/{source}: expected {n_chunks} parts, found {len(parts)}")
    tmp_final = final.with_name(final.name + ".tmp")
    writer = pq.ParquetWriter(tmp_final, SCHEMA)
    total = 0
    for part in parts:
        t = pq.read_table(part, schema=SCHEMA)
        total += t.num_rows
        writer.write_table(t)
        del t
    writer.close()
    if total != n:
        tmp_final.unlink()
        raise RuntimeError(f"{split}/{source}: merged {total:,} rows but raw file has {n:,}")
    tmp_final.replace(final)
    shutil.rmtree(parts_dir)
    print(f"  {split}/{source}: DONE {n:,} rows in {time.time()-t0:.1f}s -> {final} ({final.stat().st_size/1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--only", default=None, help="comma list like train:source1,test:source2")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--force", action="store_true", help="re-normalize even if the final cache file already exists")
    args = ap.parse_args()

    paths = Paths(args.root)
    targets = SPLITS_SOURCES
    if args.only:
        wanted = set(args.only.split(","))
        targets = [(s, src) for s, src in SPLITS_SOURCES if f"{s}:{src}" in wanted]

    for split, source in targets:
        build_one(paths, split, source, args.chunk_size, args.force)


if __name__ == "__main__":
    main()
