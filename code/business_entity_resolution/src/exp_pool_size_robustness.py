#!/usr/bin/env python3
"""Robustness to the size of the candidate pool (the test set adds France, whose pool is ~1.4M records vs 4-6M for
the training countries).

A small-pool world is emulated from one training country by a *closed-family* subsample: a random fraction of the
S1 entities is kept together with all of their true matches, plus the same fraction of the unmatched pool records,
so match-count and noise distributions are preserved while every token's document frequency shrinks. The trained
matcher (with its saved decision rule) is then scored on the SAME evaluation entities in the full-size and in the
small-pool world; a large gap would mean pool-size-dependent features hurt on France.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import PoolIndex  # noqa: E402
from config import Paths  # noqa: E402
from evaluation import FastScorer, summarize  # noqa: E402
from features import BLOCK_FEATURE_COLS, PartitionContext, compute_features  # noqa: E402
from ids import id_to_int, label_pairs, load_truth  # noqa: E402
from pipeline import chunk_slices  # noqa: E402
from run_blocking import BLOCK_COLS, config_from_args, load_ranker, add_blocking_args, read_partition  # noqa: E402
from run_infer import decide  # noqa: E402
from features import POOL_COLS, S1_COLS  # noqa: E402


def evaluate_world(name, s1, pool, cfg, ranker, booster, fcols, decision, truth, eval_ids, s1_world):
    t0 = time.time()
    res = PoolIndex(pool, cfg, verbose=False).block(s1, ranker, verbose=False)
    ctx = PartitionContext(s1, pool, s1_world)        # S1-side counts over the whole (sub)world, not the evaluation sample
    probs, s1_ids, cand_ids, is_s2 = [], [], [], []
    for a, b in chunk_slices(res["s1_row"], 500_000):
        sl = slice(a, b)
        block = {c: res[c][sl] for c in BLOCK_FEATURE_COLS}
        feats = compute_features(ctx, res["s1_row"][sl], res["pool_row"][sl], block)
        probs.append(booster.predict(feats[fcols]))
        is_s2.append(feats["is_s2"].to_numpy().astype(bool))
    p, is_s2 = np.concatenate(probs), np.concatenate(is_s2)
    s1_id = s1["entity_id"].combine_chunks().take(pa.array(res["s1_row"]))
    cand_id = pool["entity_id"].combine_chunks().take(pa.array(res["pool_row"]))
    labels = label_pairs(s1_id, cand_id, truth)
    keep = decide(p, is_s2, decision, res["s1_row"])
    scorer = FastScorer(s1_id.to_numpy(zero_copy_only=False), labels, truth, eval_ids)
    s = summarize(scorer.stats(keep))
    n_true = sum(len(truth[e]) for e in eval_ids)
    print(f"[{name}] pool={pool.num_rows:,}  candidates/entity={len(p)/len(eval_ids):.1f}  candidate recall={labels.sum()/n_true:.4f}  "
          f"macro F0.5={s['macro_f05']:.4f}  singleton acc={s['singleton_accuracy']:.4f}  P={s['mean_precision_non_singleton']:.4f} "
          f"R={s['mean_recall_non_singleton']:.4f}  predicted/entity={s['mean_pred_size']:.2f} (truth {s['mean_truth_size']:.2f})  "
          f"mean p of kept={p[keep].mean():.3f}  [{time.time()-t0:.0f}s]", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model-suffix", default="", help="extra artifact suffix of the model to test (e.g. _exp-robust)")
    ap.add_argument("--country", default="India")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--n-eval", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=11)
    add_blocking_args(ap)
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    cfg = config_from_args(args)
    ranker = load_ranker(paths.ranker_path())
    sfx = paths.suffix + args.model_suffix
    booster = lgb.Booster(model_file=str(paths.artifacts_dir / f"lgbm_model{sfx}.txt"))
    fcols = json.loads((paths.artifacts_dir / f"feature_columns{sfx}.json").read_text())["feature_columns"]
    decision = json.loads((paths.artifacts_dir / f"decision_config{sfx}.json").read_text())
    val_ids = set(json.loads((paths.artifacts_dir / f"val_entities{sfx}.json").read_text()))    # not used for fitting

    all_cols = sorted(set(BLOCK_COLS) | set(POOL_COLS) | {"country"})       # blocking AND feature columns
    s1_all = pq.read_table(paths.normalized_cache("train", "source1"), columns=all_cols)
    s1_c = s1_all.filter(pc.equal(s1_all["country"], args.country))
    pool = pa.concat_tables([read_partition(paths.normalized_cache("train", "source2"), args.country, all_cols),
                             read_partition(paths.normalized_cache("train", "source3"), args.country, all_cols)])
    print(f"{args.country}: S1 {s1_c.num_rows:,}, pool {pool.num_rows:,}", flush=True)

    rng = np.random.default_rng(args.seed)
    ids_c = np.array(s1_c["entity_id"].to_pylist())
    eval_pool = np.array(sorted(val_ids & set(ids_c.tolist())))
    eval_ids = sorted(rng.choice(eval_pool, size=min(args.n_eval, len(eval_pool)), replace=False).tolist())
    gt = pd.read_csv(paths.ground_truth(), sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    gt = gt[gt["source1_entity_id"].isin(set(ids_c.tolist()))]
    truth_all = {s1: set(m.split(",")) if m.strip() else set() for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    truth = {e: truth_all[e] for e in eval_ids}
    s1_eval = s1_c.filter(pc.is_in(s1_c["entity_id"], value_set=pa.array(eval_ids)))

    # ---- closed-family subsample of the pool
    fam = rng.random(len(ids_c)) < args.fraction
    keep_entities = set(ids_c[fam].tolist()) | set(eval_ids)
    matched_in_world = {i for e in keep_entities for i in truth_all.get(e, ())}
    all_matched_set = {i for v in truth_all.values() for i in v}
    pool_ids = pool["entity_id"].to_pylist()
    take = np.array([(i in matched_in_world) or ((i not in all_matched_set) and rng.random() < args.fraction) for i in pool_ids])
    pool_small = pool.filter(pa.array(take))
    print(f"small-pool world: {pool_small.num_rows:,} pool records ({pool_small.num_rows/pool.num_rows*100:.0f}% of full); "
          f"{len(eval_ids):,} evaluation entities (validation entities, unseen in training)", flush=True)

    s1_small_world = s1_c.filter(pa.array(np.isin(ids_c, list(keep_entities))))          # the S1 entities of the sub-world
    evaluate_world("FULL pool ", s1_eval, pool, cfg, ranker, booster, fcols, decision, truth, eval_ids, s1_c)
    evaluate_world("SMALL pool", s1_eval, pool_small, cfg, ranker, booster, fcols, decision, truth, eval_ids, s1_small_world)


if __name__ == "__main__":
    main()
