#!/usr/bin/env python3
"""v4 candidate sets = first-hop candidates (blocking + filter) + kept new pairs (dense + second-hop, run_new_pairs.py).

Writes candidates_<split>_<tag> in the SAME layout as run_candidate_cut.py, so every downstream stage (features,
first-stage scores, world scoring, stacker, submission writer) runs unchanged with --tag. New pairs carry no blocker
evidence (they were not retrieved by the blocker): those columns are 0, and extra columns tell the stacker where a pair
comes from: src_dense, src_hop, dense_score, dense_rank, hop_rank, hop_conf, hop_n, new_score (NaN / 0 for first-hop
pairs). Rows stay sorted by (s1_row, pool_row) -- one entity's pairs contiguous, as every stage requires.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402

EXTRA = ["src_dense", "src_hop", "dense_score", "dense_rank", "hop_rank", "hop_conf", "hop_n", "new_score"]


def build(src_paths: Paths, dst_paths: Paths, split: str, new_dir: Path):
    src, dst = candidates_dir(src_paths, split), candidates_dir(dst_paths, split)
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "_meta.json").read_text())
    meta["v4_new_pairs"] = str(new_dir)
    (dst / "_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copyfile(src / "s1_entities.parquet", dst / "s1_entities.parquet")
    for f in sorted(src.glob("country=*.parquet")):
        country = f.stem.split("=", 1)[1]
        t0 = time.time()
        src_schema = pq.read_schema(f)
        old = pq.read_table(f).to_pandas()
        for c in EXTRA:
            old[c] = np.nan
        old["src_dense"] = 0
        old["src_hop"] = 0
        nf = new_dir / f"country={country}.parquet"
        if nf.exists():
            new = pq.read_table(nf).to_pandas()
            # blocker evidence of a new pair is 0 (numeric columns); the string columns are all set below
            add = pd.DataFrame({c: np.zeros(len(new), dtype=old[c].dtype) for c in old.columns
                                if c not in EXTRA and pd.api.types.is_numeric_dtype(old[c])})
            add["country"] = country
            add["s1_row"], add["pool_row"] = new.s1_row.to_numpy().astype(old.s1_row.dtype), new.pool_row.to_numpy().astype(old.pool_row.dtype)
            add["s1_entity_id"], add["cand_entity_id"] = new.s1_entity_id.to_numpy(), new.cand_entity_id.to_numpy()
            add["cand_source"] = np.where(new.cand_entity_id.str.startswith("S2-"), "S2", "S3")
            if "filter_score" in add:
                add["filter_score"] = new.new_score.to_numpy().astype(np.float32)
            for c in EXTRA:
                add[c] = new[c].to_numpy() if c in new else np.nan
            both = pd.concat([old, add], ignore_index=True)
        else:
            both = old
        both = both.sort_values(["s1_row", "pool_row"], kind="stable").reset_index(drop=True)
        dup = both.duplicated(["s1_row", "pool_row"]).sum()
        if dup:
            raise RuntimeError(f"[{split}/{country}] {dup} duplicated pairs: new pairs must exclude existing candidates")
        tbl = pa.Table.from_pandas(both[[c for c in old.columns]], preserve_index=False)
        # the original columns keep their exact arrow types (row-order checks compare id arrays type-strictly)
        tbl = tbl.cast(pa.schema([src_schema.field(n) if n in src_schema.names else tbl.schema.field(n)
                                  for n in tbl.schema.names]))
        tmp = dst / f"country={country}.parquet.tmp"
        pq.write_table(tbl, tmp)
        tmp.replace(dst / f"country={country}.parquet")
        n_new = int((both.src_dense.fillna(0) + both.src_hop.fillna(0) > 0).sum())
        print(f"  [{split}/{country}] {len(old):,} first-hop + {n_new:,} new = {len(both):,} pairs "
              f"({len(both) / both.s1_row.nunique():.2f} per entity with candidates) ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-tag", default="full")
    ap.add_argument("--v4", default="v4")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    base = Paths()
    if "train" in args.splits.split(","):
        build(Paths(tag=args.world_tag), Paths(tag=f"{args.world_tag}_{args.v4}"), "train",
              base.cache_dir / f"newpairs_train_{args.world_tag}")
    if "test" in args.splits.split(","):
        build(base, Paths(tag=args.v4), "test", base.cache_dir / "newpairs_test")


if __name__ == "__main__":
    main()
