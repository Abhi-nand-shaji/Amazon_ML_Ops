"""Disjoint groups of TRAINING Source-1 entities for the full-world stages (leakage control).

main_train / val / hold  the 50k main sample of run_blocking.py (--n-s1 50000, seed 42) split like run_train.py; the
                         first-stage LightGBM matcher was fitted on main_train, val drives every decision, hold is scored
                         once at the end
ranker                   the 12k entities train_ranker.py fitted the blocker's ranker on (reproduced from its seeds)
ce_extra                 extra entities the cross-encoder may be fine-tuned on (never used by the stacker)
stack                    entities the stage-2 stacker is fitted on: no model before it has seen them, so the first-stage
                         score, the cross-encoder score and the graph features behave on them exactly as on test data
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from config import RANDOM_SEED, Paths
from run_blocking import sample_mask
from train_model import entity_split3

MAIN_SAMPLE, RANKER_PER_COUNTRY, RANKER_SEED = 50_000, 6000, 7


def build_groups(paths: Paths, n_ce_extra: int, n_stack: int, seed: int = 13) -> dict:
    s1 = pq.read_table(paths.normalized_cache("train", "source1"), columns=["entity_id", "country"])
    ids = np.asarray(s1["entity_id"].to_pylist(), dtype=object)
    country = s1["country"].to_numpy(zero_copy_only=False)
    in_main = sample_mask(len(ids), MAIN_SAMPLE, RANDOM_SEED)
    main_train, val, hold = entity_split3(ids[in_main].tolist(), 0.15, 0.15, RANDOM_SEED)
    rng = np.random.default_rng(RANKER_SEED)                 # identical draw order to train_ranker.py
    ranker = []
    for c in sorted(set(country.tolist())):
        rows = np.flatnonzero((country == c) & ~in_main)
        ranker.extend(ids[np.sort(rng.choice(rows, size=min(RANKER_PER_COUNTRY, len(rows)), replace=False))].tolist())
    ranker = set(ranker)
    rest = np.array(sorted(set(ids[~in_main].tolist()) - ranker), dtype=object)
    r2 = np.random.default_rng(seed)
    perm = r2.permutation(len(rest))
    ce_extra = set(rest[perm[:n_ce_extra]].tolist())
    stack = set(rest[perm[n_ce_extra:n_ce_extra + n_stack]].tolist())
    return {"main_train": sorted(main_train), "val": sorted(val), "hold": sorted(hold), "ranker": sorted(ranker),
            "ce_extra": sorted(ce_extra), "stack": sorted(stack)}


def load_groups(paths: Paths) -> dict:
    return {k: set(v) for k, v in json.loads((paths.artifacts_dir / "world_groups.json").read_text()).items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--n-ce-extra", type=int, default=150_000)
    ap.add_argument("--n-stack", type=int, default=200_000)
    a = ap.parse_args()
    p = Paths(a.root)
    g = build_groups(p, a.n_ce_extra, a.n_stack)
    (p.artifacts_dir / "world_groups.json").write_text(json.dumps(g))
    print({k: len(v) for k, v in g.items()})
