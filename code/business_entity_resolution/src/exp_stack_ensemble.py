#!/usr/bin/env python3
"""Seed-bagged stacker ensemble: K LightGBM stackers that differ only in their random seed (row / feature bagging),
probabilities averaged. Trained on the cached stacker frame (run_stack.py --cache-frame) + cross-encoder scores, with
early stopping on validation entities; the threshold is chosen on validation, hold-out is reported once.

Writes artifacts/stack_<exp>/ like run_stack.py (stack_model_<k>.txt for every member, stack_features.json with
"ensemble": K, stack_decision.json, stack_report.json), so run_stack_infer.py can score the test split with it.
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
from build_v4_candidates import EXTRA as V4_EXTRA  # noqa: E402
from config import Paths  # noqa: E402
from evaluation import FastScorer, summarize  # noqa: E402
from ids import load_truth  # noqa: E402
from run_stack import GRAPH_PREFIXES  # noqa: E402
from stack_features import CE_COL  # noqa: E402
from train_model import CATEGORICAL_COLS, lgb_params, search_threshold  # noqa: E402
from world_groups import load_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--ce", required=True, help="comma list of CE score files (s1_entity_id, cand_entity_id, p_ce)")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", default="42,7,123,2024,31337")
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    base = Paths()
    out_dir = base.artifacts_dir / f"stack_{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups = load_groups(base)
    df = pd.read_parquet(args.frame)
    ce = pd.concat([pd.read_parquet(f, columns=["s1_entity_id", "cand_entity_id", CE_COL]) for f in args.ce.split(",")],
                   ignore_index=True).drop_duplicates(["s1_entity_id", "cand_entity_id"])
    df = df.merge(ce, on=["s1_entity_id", "cand_entity_id"], how="left")
    base_cols = json.loads(base.feature_list_path().read_text())["feature_columns"]
    graph_cols = [c for c in df.columns if c.startswith(GRAPH_PREFIXES)]
    extra_cols = [c for c in V4_EXTRA if args.v4 and c in df.columns]
    fcols = base_cols + ["p1", CE_COL] + graph_cols + extra_cols
    grp = df["s1_entity_id"].map(lambda s: "stack" if s in groups["stack"] else "val" if s in groups["val"]
                                 else "hold" if s in groups["hold"] else "other")
    tr, va, ho = df[grp == "stack"], df[grp == "val"].reset_index(drop=True), df[grp == "hold"].reset_index(drop=True)
    truth = load_truth(base.ground_truth(), groups["val"] | groups["hold"])
    sc_va = FastScorer(va["s1_entity_id"].to_numpy(), va["label"].to_numpy(), truth, sorted(groups["val"]))
    sc_ho = FastScorer(ho["s1_entity_id"].to_numpy(), ho["label"].to_numpy(), truth, sorted(groups["hold"]))
    cat = [c for c in CATEGORICAL_COLS if c in fcols]
    dtr = lgb.Dataset(tr[fcols], label=tr["label"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(va[fcols], label=va["label"], categorical_feature=cat, reference=dtr, free_raw_data=False)
    print(f"frame {len(df):,} rows, {len(fcols)} features ({time.time() - t0:.0f}s)", flush=True)
    q_va, q_ho, report = np.zeros(len(va)), np.zeros(len(ho)), {"members": []}
    seeds = [int(s) for s in args.seeds.split(",")]
    for k, seed in enumerate(seeds):
        t1 = time.time()
        b = lgb.train(lgb_params(args.threads, 127, 0.05, seed=seed), dtr, num_boost_round=4000, valid_sets=[dva],
                      valid_names=["val"], callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        pv, ph = b.predict(va[fcols], num_iteration=b.best_iteration), b.predict(ho[fcols], num_iteration=b.best_iteration)
        b.save_model(str(out_dir / f"stack_model_{k}.txt"), num_iteration=b.best_iteration)
        q_va += pv
        q_ho += ph
        t, fv, _ = search_threshold(sc_va, pv)
        fh = summarize(sc_ho.stats(ph >= t))["macro_f05"]
        te, fve, _ = search_threshold(sc_va, q_va / (k + 1))
        fhe = summarize(sc_ho.stats(q_ho / (k + 1) >= te))["macro_f05"]
        report["members"].append({"seed": seed, "trees": b.best_iteration, "threshold": t, "val": fv, "hold": fh,
                                  "ensemble_so_far": {"threshold": te, "val": fve, "hold": fhe}})
        print(f"  seed {seed}: {b.best_iteration} trees, member val {fv:.4f} hold {fh:.4f} | ensemble of {k + 1}: "
              f"threshold {te:.2f} val {fve:.4f} hold {fhe:.4f} ({time.time() - t1:.0f}s)", flush=True)
    q_va /= len(seeds)
    q_ho /= len(seeds)
    thr, f_va, _ = search_threshold(sc_va, q_va)
    s_va, s_ho = summarize(sc_va.stats(q_va >= thr)), summarize(sc_ho.stats(q_ho >= thr))
    report["ENSEMBLE"] = {"threshold": thr, "val_macro_f05": s_va["macro_f05"], "hold_macro_f05": s_ho["macro_f05"],
                          "hold_precision": s_ho["mean_precision_non_singleton"], "hold_recall": s_ho["mean_recall_non_singleton"],
                          "hold_singleton_acc": s_ho["singleton_accuracy"], "members": len(seeds)}
    print(f"ENSEMBLE of {len(seeds)}: threshold {thr:.2f} val {s_va['macro_f05']:.4f} hold {s_ho['macro_f05']:.4f} "
          f"(P {s_ho['mean_precision_non_singleton']:.4f} R {s_ho['mean_recall_non_singleton']:.4f})", flush=True)
    (out_dir / "stack_features.json").write_text(json.dumps({"feature_columns": fcols, "categorical_columns": cat,
                                                             "uses_ce": True, "graph": True, "v4": bool(args.v4),
                                                             "ensemble": len(seeds)}, indent=2))
    (out_dir / "stack_decision.json").write_text(json.dumps({"mode": "global", "global_threshold": thr,
                                                            "validation_macro_f05": s_va["macro_f05"],
                                                            "holdout_macro_f05": s_ho["macro_f05"]}, indent=2))
    (out_dir / "stack_report.json").write_text(json.dumps(report, indent=2, default=float))
    for name, d, q in (("val", va, q_va), ("hold", ho, q_ho)):
        d[["country", "s1_entity_id", "cand_entity_id", "label", "p1"]].assign(q=q.astype(np.float32)).to_parquet(
            out_dir / f"stack_{name}.parquet", index=False)
    print(f"saved {out_dir} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
