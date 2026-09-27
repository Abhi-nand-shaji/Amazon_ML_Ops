#!/usr/bin/env python3
"""Second-hop candidates (second_hop.py) for the TEST split or for the training entities the stacker is fitted /
validated on (stack, val, hold of the full world).

Anchors = an entity's candidates with a test-like stacker probability >= --anchor-threshold, at most --max-anchors per
entity (best first): test -> the stacker's test scores (scores_test_stack_<exp>); training -> out-of-fold probabilities
for stack entities, the fitted stacker for val / hold (run_stack_oof.py).
Writes artifacts/cache/hop_<split>[_tag]/country=<C>.parquet: s1_row, pool_row, hop_rank, hop_conf, hop_n (the best
--top-per-entity per entity, none of them already a candidate).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import PoolIndex  # noqa: E402
from config import DEFAULT_BLOCKING, Paths  # noqa: E402
from pipeline import candidate_countries  # noqa: E402
from run_blocking import candidates_dir, load_ranker, read_partition  # noqa: E402
from second_hop import _key, second_hop_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--tag", default="full", help="training world tag (train split only)")
    ap.add_argument("--exp", default="ce", help="stacker whose test scores give the anchors")
    ap.add_argument("--anchor-threshold", type=float, default=0.9)
    ap.add_argument("--max-anchors", type=int, default=3)
    ap.add_argument("--top-per-entity", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--countries", default=None)
    args = ap.parse_args()

    base = Paths()
    paths = Paths(tag=args.tag) if args.split == "train" else base
    out = base.cache_dir / f"hop_{args.split}{paths.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(DEFAULT_BLOCKING, expansion_budget=6_000_000)
    ranker = load_ranker(base.ranker_path(), args.workers)
    t00 = time.time()
    oof = pq.read_table(base.cache_dir / "stack_oof_q.parquet").to_pandas() if args.split == "train" else None
    countries = args.countries.split(",") if args.countries else candidate_countries(paths, args.split)

    def log(m):
        print(f"[{(time.time() - t00) / 60:6.1f} min] {m}", flush=True)

    for country in countries:
        final = out / f"country={country}.parquet"
        if final.exists():
            log(f"[{country}] already done, skipping")
            continue
        cand = pq.read_table(candidates_dir(paths, args.split) / f"country={country}.parquet",
                             columns=["s1_row", "pool_row", "s1_entity_id", "cand_entity_id"]).to_pandas()
        if args.split == "test":
            cand["q"] = pq.read_table(base.cache_dir / f"scores_test_stack_{args.exp}" / f"country={country}.parquet")["p"].to_numpy()
            ents = cand
        else:
            ents = cand.merge(oof[["s1_entity_id", "cand_entity_id", "q"]], on=["s1_entity_id", "cand_entity_id"], how="inner")
        existing = np.unique(_key(ents["s1_row"], ents["pool_row"]))
        anc = ents[ents["q"] >= args.anchor_threshold].sort_values(["s1_row", "q"], ascending=[True, False])
        anc = anc[anc.groupby("s1_row").cumcount() < args.max_anchors]
        log(f"[{country}] {ents.s1_row.nunique():,} entities, {len(anc):,} anchors ({anc.s1_row.nunique():,} entities with >=1)")
        t0 = time.time()
        pool = pa.concat_tables([read_partition(paths.normalized_cache(args.split, "source2"), country),
                                 read_partition(paths.normalized_cache(args.split, "source3"), country)])
        index = PoolIndex(pool, cfg, verbose=False)
        log(f"[{country}] pool {pool.num_rows:,}, index built ({time.time() - t0:.0f}s)")
        hop = second_hop_pairs(index, pool, anc["s1_row"].to_numpy(), anc["pool_row"].to_numpy(), anc["q"].to_numpy(),
                               existing_keys=existing, ranker=ranker, chunk=args.chunk, workers=args.workers,
                               top_per_entity=args.top_per_entity, log=log)
        hop = hop.rename(columns={"entity": "s1_row"})
        tbl = pa.table({"s1_row": hop["s1_row"].to_numpy().astype(np.int32), "pool_row": hop["pool_row"].to_numpy().astype(np.int32),
                        "hop_rank": hop["hop_rank"].to_numpy().astype(np.float32), "hop_conf": hop["hop_conf"].to_numpy().astype(np.float32),
                        "hop_n": hop["hop_n"].to_numpy().astype(np.int16)})
        tmp = final.with_name(final.name + ".tmp")
        pq.write_table(tbl, tmp)
        tmp.replace(final)
        log(f"[{country}] {len(hop):,} second-hop pairs ({len(hop) / max(1, ents.s1_row.nunique()):.2f} per entity)")
        del index, pool, cand, ents
    log("done")


if __name__ == "__main__":
    main()
