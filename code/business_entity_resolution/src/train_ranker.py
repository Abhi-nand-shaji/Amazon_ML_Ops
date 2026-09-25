#!/usr/bin/env python3
"""Train the cheap candidate ranker: which ~30 of the ~1000 retrieved candidates per S1 entity survive.

Retrieval (blocking.py) is high-recall but unranked: a pair shares some rare token / bigram / prefix
with the entity. The per-entity cap needs a good ordering, and hand-written scores measured badly
(they tie, or over-reward exact-name hits). A small LightGBM over the blocker's own cheap evidence
(IDF-weighted overlap of ALL tokens, bigram / prefix / exact hits, token counts) keeps ~96% of ALL
true pairs at 20 candidates per entity -- the retrieval ceiling.

Leakage control: the ranker is trained on S1 entities OUTSIDE the main training sample
(--main-sample/--main-seed must equal the values used for run_blocking.py --n-s1/--seed), on pairs
retrieved with NO truncation, so the candidate cut never saw the labels of the entities that later
serve as validation for the matcher.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
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
from blocking import RANKER_FEATURES, PoolIndex, heuristic_rank  # noqa: E402
from config import Paths, RANDOM_SEED  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from run_blocking import BLOCK_COLS, add_blocking_args, config_from_args, read_partition, sample_mask  # noqa: E402

KS = (10, 20, 30, 50, 100)


def rank_within_entity(entity: np.ndarray, pool_row: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = np.lexsort((pool_row, -score, entity))
    e = entity[order]
    first = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    rank = np.empty(len(e), np.int64)
    rank[order] = np.arange(len(e)) - np.repeat(first, np.diff(np.r_[first, len(e)]))
    return rank


def recall_table(df: pd.DataFrame, score: np.ndarray, n_true: int) -> dict:
    rk = rank_within_entity(df["entity"].to_numpy(), df["pool_row"].to_numpy(), score)
    lab = df["label"].to_numpy() == 1
    return {k: float((lab & (rk < k)).sum() / n_true * 100) for k in KS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-per-country", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--main-sample", type=int, default=50_000)
    ap.add_argument("--main-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--neg-rate", type=float, default=0.08, help="fraction of negatives kept for training")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    add_blocking_args(ap)
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    cfg = dataclasses.replace(config_from_args(args), max_candidates=10**9)     # keep EVERY retrieved pair
    t0 = time.time()
    s1_all = pq.read_table(paths.normalized_cache("train", "source1"), columns=BLOCK_COLS)
    in_main = sample_mask(s1_all.num_rows, args.main_sample, args.main_seed)
    country_arr = s1_all["country"].to_numpy(zero_copy_only=False)
    rng = np.random.default_rng(args.seed)

    frames, n_true = [], 0
    for country in sorted(set(country_arr.tolist())):
        pool_rows = np.flatnonzero((country_arr == country) & ~in_main)
        pick = np.sort(rng.choice(pool_rows, size=min(args.n_per_country, len(pool_rows)), replace=False))
        s1 = s1_all.take(pa.array(pick))
        pool = pa.concat_tables([read_partition(paths.normalized_cache("train", "source2"), country),
                                 read_partition(paths.normalized_cache("train", "source3"), country)])
        print(f"[{country}] {s1.num_rows:,} ranker-training entities, pool {pool.num_rows:,}", flush=True)
        res = PoolIndex(pool, cfg).block(s1, ranker=None, verbose=False)
        truth = load_truth(paths.ground_truth(), set(s1["entity_id"].to_pylist()))
        n_true_c = sum(len(v) for v in truth.values())
        n_true += n_true_c
        label = label_pairs(s1["entity_id"].combine_chunks().take(pa.array(res["s1_row"])),
                            pool["entity_id"].combine_chunks().take(pa.array(res["pool_row"])), truth)
        df = pd.DataFrame({c: res[c] for c in RANKER_FEATURES + ["pool_row"]})
        df["entity"] = (res["s1_row"].astype(np.int64) + len(frames) * 10**8)       # unique across countries
        df["label"] = label
        df["heuristic"] = heuristic_rank(res)
        print(f"[{country}] {len(df):,} retrieved pairs ({len(df)/s1.num_rows:.0f}/entity); retrieval recall "
              f"{label.sum()/n_true_c*100:.2f}% of {n_true_c:,} true pairs ({time.time()-t0:.0f}s)", flush=True)
        frames.append(df)
        del pool, res
    df = pd.concat(frames, ignore_index=True)
    del frames

    held_out = (df["entity"].to_numpy() % 4 == 0)                                    # 25% of entities for evaluation
    n_true_held = None
    tr = df[~held_out]
    keep = (tr["label"].to_numpy() == 1) | (np.random.default_rng(args.seed).random(len(tr)) < args.neg_rate)
    booster = lgb.train(
        {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.9,
         "bagging_fraction": 0.9, "bagging_freq": 1, "num_threads": args.threads, "seed": RANDOM_SEED, "verbosity": -1},
        lgb.Dataset(tr.loc[keep, RANKER_FEATURES], label=tr.loc[keep, "label"]), num_boost_round=args.rounds)
    print(f"ranker trained on {int(keep.sum()):,} rows in {time.time()-t0:.0f}s", flush=True)

    te = df[held_out].reset_index(drop=True)
    n_true_held = int(round(n_true * held_out.sum() / max(1, len(held_out))))         # approx (entity-uniform)
    # exact denominator: true pairs of held-out entities = retrieved positives + never-retrieved ones; use the
    # retrieval recall measured overall to scale, so numbers are absolute % of ALL true pairs
    retrieved_frac = df["label"].sum() / n_true
    ceiling = retrieved_frac * 100
    denom = te["label"].sum() / retrieved_frac
    report = {
        "held_out_entities": int(te["entity"].nunique()), "retrieval_ceiling_pct": ceiling,
        "recall_at_k_pct": {
            "learned_ranker": recall_table(te, booster.predict(te[RANKER_FEATURES], num_threads=args.threads), denom),
            "heuristic_ov_name+1.5*ov_addr": recall_table(te, te["heuristic"].to_numpy(), denom),
            "block_score_legacy": recall_table(te, te["block_score"].to_numpy(), denom),
        },
        "feature_gain_share": (pd.Series(booster.feature_importance("gain"), index=RANKER_FEATURES)
                               .pipe(lambda s: (s / s.sum()).sort_values(ascending=False)).round(4).to_dict()),
    }
    print(json.dumps(report, indent=2), flush=True)
    booster.save_model(str(paths.ranker_path()))
    paths.report_path("ranker_report", "json").write_text(json.dumps(report, indent=2))
    print(f"saved {paths.ranker_path()} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
