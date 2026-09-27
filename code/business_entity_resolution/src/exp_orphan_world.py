#!/usr/bin/env python3
"""Does an orphan-rich world (like the test split) explain the leaderboard gap, and does training for it help?

The test pool has ~5.8 Source-2/3 records per Source-1 entity against 4.7 in training: an estimated ~40% of test pool
records belong to no Source-1 entity (26% in training). Pool-side (exclusivity) features catch a false claim when the
record's real owner competes for it; an orphan record has no owner, so nothing competes. This experiment removes a random
share `d` of the NON-selected training entities from the full world (their records become orphans), recomputes the
pool-side features of the stack / val / hold pairs from the remaining claimants, and
  1. scores the current stacker (artifacts/stack_ce) on val / hold in each simulated world, and
  2. trains stackers on stack entities in simulated worlds (single rate, and a mix of rates) and scores them the same way.
Entity-side features (context, siblings, address structure, pairwise) are unaffected by removing other entities.
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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from evaluation import FastScorer, summarize  # noqa: E402
from ids import load_truth  # noqa: E402
from run_world_score import world_p_dir  # noqa: E402
from stack_features import pool_side_for_rows  # noqa: E402
from train_model import CATEGORICAL_COLS, lgb_params, search_threshold  # noqa: E402
from world_groups import load_groups  # noqa: E402

PS_COLS = ["ps_n_claim", "ps_rank", "ps_best_other", "ps_gap_best_other", "ps_sum_other", "ps_n_other_ge50", "ps_q"]


def pool_side_for_drop(frame: pd.DataFrame, world: Paths, d: float, seed: int) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index, columns=PS_COLS, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for country, fr in frame.groupby("country"):
        wp = pq.read_table(world_p_dir(world) / f"country={country}.parquet").to_pandas()
        sel = np.unique(fr["s1_row"].to_numpy())
        others = np.setdiff1d(np.unique(wp["s1_row"].to_numpy()), sel)
        drop = others[rng.random(len(others)) < d] if d > 0 else np.zeros(0, others.dtype)
        keep = ~np.isin(wp["s1_row"].to_numpy(), drop)
        ps = pool_side_for_rows(wp["s1_row"].to_numpy()[keep], wp["pool_row"].to_numpy()[keep], wp["p1"].to_numpy()[keep],
                                fr["s1_row"].to_numpy(), fr["pool_row"].to_numpy())
        out.loc[fr.index, PS_COLS] = ps[PS_COLS].to_numpy()
        print(f"    [{country}] d={d}: removed {len(drop):,} of {len(others) + len(sel):,} S1 entities", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default=r"E:\Amazon\.rescue\v2\stack_frame.parquet")
    ap.add_argument("--ce", default=None)
    ap.add_argument("--rates", default="0,0.2,0.35")
    ap.add_argument("--train-rate", type=float, default=0.2)
    ap.add_argument("--mix", default="0,0.2,0.35", help="rates mixed (one copy each) for the mixed-world stacker")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    base, world = Paths(), Paths(tag="full")
    t0 = time.time()
    groups = load_groups(base)
    frame = pd.read_parquet(args.frame)
    ce = pd.read_parquet(args.ce or (base.artifacts_dir / "ce" / "ce_train_sets.parquet"), columns=["s1_entity_id", "cand_entity_id", "p_ce"])
    frame = frame.merge(ce, on=["s1_entity_id", "cand_entity_id"], how="left")
    spec = json.loads((base.artifacts_dir / "stack_ce" / "stack_features.json").read_text())
    fcols = spec["feature_columns"]
    grp = np.where(frame["s1_entity_id"].isin(groups["stack"]), "stack",
                   np.where(frame["s1_entity_id"].isin(groups["val"]), "val", np.where(frame["s1_entity_id"].isin(groups["hold"]), "hold", "other")))
    frame["grp"] = grp
    truth = load_truth(base.ground_truth(), groups["val"] | groups["hold"])
    idx = {g: np.flatnonzero(grp == g) for g in ("stack", "val", "hold")}
    scorers = {g: FastScorer(frame["s1_entity_id"].to_numpy()[idx[g]], frame["label"].to_numpy()[idx[g]], truth, sorted(groups[g]))
               for g in ("val", "hold")}
    print(f"frame {len(frame):,} rows ({time.time() - t0:.0f}s)", flush=True)

    rates = [float(x) for x in args.rates.split(",")]
    worlds = {}
    for d in sorted(set(rates) | {args.train_rate} | {float(x) for x in args.mix.split(",")}):
        print(f"  simulated world d={d}", flush=True)
        worlds[d] = pool_side_for_drop(frame, world, d, seed=1234 + int(d * 1000))

    def X(d, rows):
        x = frame.iloc[rows][fcols].copy()
        x[PS_COLS] = worlds[d].iloc[rows][PS_COLS].to_numpy()
        return x

    def evaluate(name, booster, t_fixed=None):
        res = {}
        for d in rates:
            p_va = booster.predict(X(d, idx["val"]), num_threads=args.threads)
            p_ho = booster.predict(X(d, idx["hold"]), num_threads=args.threads)
            t_best, f_va, _ = search_threshold(scorers["val"], p_va)
            t = t_fixed if t_fixed is not None else t_best
            s_ho = summarize(scorers["hold"].stats(p_ho >= t))
            res[d] = (t, scorers["val"].score(p_va >= t), s_ho["macro_f05"], s_ho["mean_precision_non_singleton"],
                      s_ho["mean_recall_non_singleton"], s_ho["singleton_accuracy"])
            print(f"  [{name}] world d={d}: threshold {t:.2f} -> val {res[d][1]:.4f} | hold-out {res[d][2]:.4f} "
                  f"(P {res[d][3]:.4f} R {res[d][4]:.4f} singleton {res[d][5]:.4f})", flush=True)
        return res

    print("\n=== current v3 stacker (trained in the full world), its validated threshold 0.73", flush=True)
    v3 = lgb.Booster(model_file=str(base.artifacts_dir / "stack_ce" / "stack_model.txt"))
    evaluate("v3 @0.73", v3, 0.73)
    evaluate("v3 re-tuned", v3)

    cat = [c for c in CATEGORICAL_COLS if c in fcols]

    def fit(train_rates, name):
        xs, ys = [], []
        for d in train_rates:
            xs.append(X(d, idx["stack"]))
            ys.append(frame["label"].to_numpy()[idx["stack"]])
        dtr = lgb.Dataset(pd.concat(xs, ignore_index=True), label=np.concatenate(ys), categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X(train_rates[len(train_rates) // 2], idx["val"]), label=frame["label"].to_numpy()[idx["val"]],
                          categorical_feature=cat, reference=dtr, free_raw_data=False)
        t1 = time.time()
        b = lgb.train(lgb_params(args.threads, 127, 0.05), dtr, num_boost_round=4000, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        print(f"\n=== stacker trained in world(s) {train_rates}: {b.best_iteration} trees ({time.time() - t1:.0f}s)", flush=True)
        return b

    b1 = fit([args.train_rate], f"d={args.train_rate}")
    evaluate(f"trained d={args.train_rate}, re-tuned", b1)
    b2 = fit([float(x) for x in args.mix.split(",")], "mixed")
    r2 = evaluate("trained mixed, re-tuned", b2)
    out = base.artifacts_dir / "stack_ce_orphan"
    out.mkdir(parents=True, exist_ok=True)
    b2.save_model(str(out / "stack_model.txt"), num_iteration=b2.best_iteration)
    b1.save_model(str(out / f"stack_model_d{args.train_rate}.txt"), num_iteration=b1.best_iteration)
    (out / "stack_features.json").write_text(json.dumps({**spec, "orphan_training": {"mix": args.mix, "single": args.train_rate}}, indent=2))
    print(f"\nsaved {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
