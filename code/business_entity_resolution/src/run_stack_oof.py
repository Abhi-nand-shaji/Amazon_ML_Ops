#!/usr/bin/env python3
"""Out-of-fold stacker probabilities for the `stack` entities (entity-level K folds), plus the fitted stacker's
probabilities for val / hold. Used where a later stage needs a TEST-LIKE confidence for training entities -- e.g. the
anchors of second-hop retrieval (second_hop.py): the stacker was fitted on the stack entities, so its in-sample
probabilities there would be over-confident.

Writes artifacts/cache/stack_oof_q.parquet: s1_entity_id, cand_entity_id, q (float32), grp (stack / val / hold)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths, RANDOM_SEED  # noqa: E402
from train_model import CATEGORICAL_COLS, lgb_params  # noqa: E402
from world_groups import load_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default=r"E:\Amazon\.rescue\v2\stack_frame.parquet")
    ap.add_argument("--exp", default="ce")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    p = Paths()
    t0 = time.time()
    groups = load_groups(p)
    sdir = p.artifacts_dir / f"stack_{args.exp}"
    spec = json.loads((sdir / "stack_features.json").read_text())
    fcols = spec["feature_columns"]
    trees = lgb.Booster(model_file=str(sdir / "stack_model.txt")).num_trees()
    df = pd.read_parquet(args.frame)
    ce = pd.read_parquet(p.artifacts_dir / "ce" / "ce_train_sets.parquet", columns=["s1_entity_id", "cand_entity_id", "p_ce"])
    df = df.merge(ce, on=["s1_entity_id", "cand_entity_id"], how="left")
    grp = np.where(df.s1_entity_id.isin(groups["stack"]), "stack", np.where(df.s1_entity_id.isin(groups["val"]), "val", "hold"))
    st = np.flatnonzero(grp == "stack")
    ents = pd.Index(sorted(set(df.s1_entity_id.to_numpy()[st])))
    fold = np.random.default_rng(RANDOM_SEED).permutation(len(ents)) % args.folds
    fold_row = fold[ents.get_indexer(df.s1_entity_id.to_numpy()[st])]
    q = np.full(len(df), np.nan, np.float32)
    cat = [c for c in CATEGORICAL_COLS if c in fcols]
    params = lgb_params(args.threads, 127, 0.05)
    for k in range(args.folds):
        tr, te = st[fold_row != k], st[fold_row == k]
        b = lgb.train(params, lgb.Dataset(df.iloc[tr][fcols], label=df.label.to_numpy()[tr], categorical_feature=cat),
                      num_boost_round=trees)
        q[te] = b.predict(df.iloc[te][fcols], num_threads=args.threads)
        print(f"  fold {k + 1}/{args.folds}: {len(te):,} rows scored ({time.time() - t0:.0f}s)", flush=True)
    full = lgb.Booster(model_file=str(sdir / "stack_model.txt"))
    vh = np.flatnonzero(grp != "stack")
    q[vh] = full.predict(df.iloc[vh][fcols], num_threads=args.threads)
    out = df[["s1_entity_id", "cand_entity_id"]].assign(q=q, grp=grp)
    out.to_parquet(p.cache_dir / "stack_oof_q.parquet", index=False)
    from sklearn.metrics import roc_auc_score
    print(f"OOF AUC on stack {roc_auc_score(df.label.to_numpy()[st], q[st]):.5f}; val+hold AUC "
          f"{roc_auc_score(df.label.to_numpy()[vh], q[vh]):.5f}; wrote {p.cache_dir / 'stack_oof_q.parquet'} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
