"""Streaming access to featurized candidate pairs, one country partition at a time.

Used by ``run_features.py`` (persist train features) and ``run_infer.py`` (score test
pairs on the fly -- the full test feature matrix is far too large to store).

Row-order contract: the blocker addresses records by *position inside the country
partition*. ``load_partition`` rebuilds the partitions with exactly the same reads and
the same sampling mask as ``run_blocking.py`` (the mask is regenerated from the seed
recorded in ``candidates_<split>/_meta.json``); ``iter_feature_chunks`` verifies the
contract against the entity ids stored next to the row positions, so a mismatch fails
loudly instead of silently corrupting every feature.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import Paths
from features import BLOCK_FEATURE_COLS, POOL_COLS, S1_COLS, PartitionContext, _group_bounds, compute_features
from run_blocking import candidates_dir, read_partition, sample_mask

PAIR_COLS = ["s1_row", "pool_row", "s1_entity_id", "cand_entity_id"] + BLOCK_FEATURE_COLS


def load_meta(paths: Paths, split: str) -> dict:
    return json.loads((candidates_dir(paths, split) / "_meta.json").read_text())


def candidate_countries(paths: Paths, split: str) -> list[str]:
    return sorted(p.stem.split("=", 1)[1] for p in candidates_dir(paths, split).glob("country=*.parquet"))


def load_partition(paths: Paths, split: str, country: str, meta: dict, s1_cols=S1_COLS, pool_cols=POOL_COLS,
                   with_full: bool = True) -> tuple[pa.Table, pa.Table, pa.Table | None]:
    """(S1 partition, S2+S3 pool partition, full S1 partition or None) in the exact order the blocker used.
    The full S1 partition is returned only when the S1 side is a sample (and `with_full`): S1-side ambiguity counts
    must be computed over every S1 record of the country, not over the sample."""
    s1_path = paths.normalized_cache(split, "source1")
    n_sample, seed = meta.get("n_s1_sample"), meta["seed"]
    s1_full = None
    if n_sample is None:
        s1_part = read_partition(s1_path, country, s1_cols)
    else:
        s1_all = pq.read_table(s1_path, columns=list(s1_cols) + ["country"])
        if with_full:
            s1_full = s1_all.filter(pc.equal(s1_all["country"], country)).select(["name_core", "address_normalized"])
        mask = sample_mask(s1_all.num_rows, n_sample, seed)
        if not mask.all():
            s1_all = s1_all.filter(pa.array(mask))
        s1_part = s1_all.filter(pc.equal(s1_all["country"], country)).drop_columns(["country"])
    pool = pa.concat_tables([
        read_partition(paths.normalized_cache(split, "source2"), country, pool_cols),
        read_partition(paths.normalized_cache(split, "source3"), country, pool_cols),
    ])
    return s1_part, pool, s1_full


def chunk_slices(s1_row: np.ndarray, target_pairs: int) -> list[tuple[int, int]]:
    """Split sorted pair rows into slices of ~target_pairs that never cut an S1 entity's run."""
    n = len(s1_row)
    if n == 0:
        return []
    bounds = _group_bounds(s1_row)
    out, start = [], 0
    while start < n:
        j = int(np.searchsorted(bounds, start + target_pairs, side="left"))
        end = int(bounds[min(j, len(bounds) - 1)])
        if end <= start:
            end = n
        out.append((start, end))
        start = end
    return out


@dataclass
class FeatureChunk:
    country: str
    ctx: PartitionContext
    feats: pd.DataFrame          # one row per pair
    s1_row: np.ndarray
    pool_row: np.ndarray
    s1_entity_id: pa.Array       # per pair
    cand_entity_id: pa.Array     # per pair


def iter_feature_chunks(paths: Paths, split: str, chunk_pairs: int = 1_000_000, countries=None,
                        verbose: bool = True) -> Iterator[FeatureChunk]:
    meta = load_meta(paths, split)
    for country in (countries or candidate_countries(paths, split)):
        t0 = time.time()
        cand = pq.read_table(candidates_dir(paths, split) / f"country={country}.parquet", columns=PAIR_COLS)
        if cand.num_rows == 0:
            continue
        s1_part, pool, s1_full = load_partition(paths, split, country, meta)
        ctx = PartitionContext(s1_part, pool, s1_full)
        del s1_part, pool, s1_full
        s1_row = cand["s1_row"].to_numpy()
        pool_row = cand["pool_row"].to_numpy()
        if verbose:
            print(f"  [{country}] partition ready: s1={ctx.n_s1:,} pool={ctx.n_pool:,} pairs={cand.num_rows:,} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        # row-order contract check on a sample of pairs
        probe = np.linspace(0, len(s1_row) - 1, num=min(len(s1_row), 5000)).astype(np.int64)
        ok = (ctx.s1_id.take(pa.array(s1_row[probe])).equals(cand["s1_entity_id"].take(pa.array(probe)).combine_chunks())
              and ctx.pool_id.take(pa.array(pool_row[probe])).equals(cand["cand_entity_id"].take(pa.array(probe)).combine_chunks()))
        if not ok:
            raise RuntimeError(f"[{country}] partition row order does not match the candidate file "
                               f"(was the cache or the sampling changed since blocking?)")
        for a, b in chunk_slices(s1_row, chunk_pairs):
            sl = slice(a, b)
            block = {c: cand[c].slice(a, b - a).to_numpy() for c in BLOCK_FEATURE_COLS}
            feats = compute_features(ctx, s1_row[sl], pool_row[sl], block)
            yield FeatureChunk(country, ctx, feats, s1_row[sl], pool_row[sl],
                               cand["s1_entity_id"].slice(a, b - a).combine_chunks(),
                               cand["cand_entity_id"].slice(a, b - a).combine_chunks())
        del ctx, cand
