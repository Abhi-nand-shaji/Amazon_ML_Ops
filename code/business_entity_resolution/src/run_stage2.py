#!/usr/bin/env python3
"""Second stage: re-score every candidate given the first-stage scores of all candidates of its entity.

Stage 2 is trained on OUT-OF-FOLD first-stage probabilities (K-fold over the training ENTITIES), so it sees the
same kind of noisy scores it will see at test time instead of over-confident in-sample ones. Validation and hold-out
entities use the main first-stage model (trained on all training entities), exactly as test inference does.

Features (see stage2.py) depend only on first-stage probabilities plus a few cheap blocker columns stored with the
candidates, so test-scale inference needs no re-featurization. The stage is adopted only if it beats the first stage
on the validation entities by a margin; the decision is recorded in decision_config_stage2<tag>.json (`adopted`).
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
from config import Paths, RANDOM_SEED  # noqa: E402
from evaluation import FastScorer, summarize  # noqa: E402
from ids import load_truth  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from run_infer import decide  # noqa: E402
from run_train import gated_thresholds, load_split_frames, per_source_thresholds  # noqa: E402
from stage2 import BLOCKER_COLS, entity_context  # noqa: E402
from train_model import CATEGORICAL_COLS, entity_max, entity_split3, lgb_params, search_threshold  # noqa: E402

MARGIN = 2e-3          # stage 2 must beat stage 1 on validation by this much to be adopted


def context_frame(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    group = pd.factorize(frame["s1_entity_id"])[0]
    return entity_context(p, frame["is_s2"].to_numpy().astype(bool), group, {c: frame[c].to_numpy() for c in BLOCKER_COLS})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exp", default="", help="suffix of the first-stage model to build on (as given to run_train --exp)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    sfx = paths.suffix + (f"_exp-{args.exp}" if args.exp else "")
    t0 = time.time()
    booster = lgb.Booster(model_file=str(paths.artifacts_dir / f"lgbm_model{sfx}.txt"))
    fcols = json.loads((paths.artifacts_dir / f"feature_columns{sfx}.json").read_text())["feature_columns"]
    report1 = json.loads((paths.reports_dir / f"validation_report{sfx}.json").read_text())
    cfg1 = json.loads((paths.artifacts_dir / f"decision_config{sfx}.json").read_text())
    rounds = int(report1["best_iteration"])
    leaves = int(report1.get("leaves", 127))

    sampled = pq.read_table(candidates_dir(paths, "train") / "s1_entities.parquet", columns=["entity_id"])["entity_id"].to_pylist()
    train_ids, val_ids, hold_ids = entity_split3(sampled, 0.15, 0.15, RANDOM_SEED)
    truth = load_truth(paths.ground_truth(), set(sampled))
    train, val, hold = load_split_frames(paths, val_ids, hold_ids, 10**6, 0)

    # ---- out-of-fold first-stage probabilities for the training entities
    fold = (pd.util.hash_array(train["s1_entity_id"].to_numpy(dtype=object)) % args.folds).astype(int)
    p_oof = np.zeros(len(train), dtype=np.float64)
    cat_idx = [fcols.index(c) for c in CATEGORICAL_COLS if c in fcols]
    for k in range(args.folds):
        tr, te = fold != k, fold == k
        m = lgb.train(lgb_params(args.threads, leaves), lgb.Dataset(train.loc[tr, fcols], label=train.loc[tr, "label"],
                                                                     categorical_feature=cat_idx), num_boost_round=rounds)
        p_oof[te] = m.predict(train.loc[te, fcols])
        print(f"  fold {k+1}/{args.folds}: {int(te.sum()):,} rows scored out-of-fold ({time.time()-t0:.0f}s)", flush=True)

    p_val = booster.predict(val[fcols])
    p_hold = booster.predict(hold[fcols])
    x_tr, x_val, x_hold = context_frame(train, p_oof), context_frame(val, p_val), context_frame(hold, p_hold)
    s2cols = list(x_tr.columns)

    # ---- stage-2 model, early stopping on validation entities
    m2 = lgb.train({**lgb_params(args.threads, 31, 0.05), "min_data_in_leaf": 200}, lgb.Dataset(x_tr, label=train["label"]),
                   num_boost_round=1500, valid_sets=[lgb.Dataset(x_val, label=val["label"])], valid_names=["val"],
                   callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.log_evaluation(100)])
    print(f"stage-2 model: {m2.num_trees()} trees (best iteration {m2.best_iteration}, {time.time()-t0:.0f}s)", flush=True)
    q_val = m2.predict(x_val, num_iteration=m2.best_iteration)
    q_hold = m2.predict(x_hold, num_iteration=m2.best_iteration)

    # ---- decision rules on stage-2 scores, selected on validation only
    scorer = FastScorer(val["s1_entity_id"].to_numpy(), val["label"].to_numpy(), truth, sorted(val_ids))
    is_s2 = val["is_s2"].to_numpy().astype(bool)
    t_glob, f_glob, _ = search_threshold(scorer, q_val)
    t2, t3, f_ps = per_source_thresholds(scorer, q_val, is_s2, t_glob)
    ent_max = entity_max(q_val, scorer.ent_idx, scorer.n)
    gate, g2, g3, f_gate = gated_thresholds(scorer, q_val, is_s2, ent_max, t2, t3)
    mode, f_sel = "global", f_glob
    if f_ps > f_sel + 1e-3:
        mode, f_sel = "per_source", f_ps
    if f_gate > f_sel + 1e-3:
        mode, f_sel = "gated", f_gate
    cfg2 = {"mode": mode, "global_threshold": t_glob, "s2_threshold": {"global": t_glob, "per_source": t2, "gated": g2}[mode],
            "s3_threshold": {"global": t_glob, "per_source": t3, "gated": g3}[mode], "gate": gate if mode == "gated" else None}

    f1 = cfg1["validation_macro_f05"]
    scorer_h = FastScorer(hold["s1_entity_id"].to_numpy(), hold["label"].to_numpy(), {e: truth[e] for e in hold_ids}, sorted(hold_ids))
    code_h = pd.factorize(hold["s1_entity_id"])[0]
    s_h1 = summarize(scorer_h.stats(decide(p_hold, hold["is_s2"].to_numpy().astype(bool), cfg1, code_h)))
    s_h2 = summarize(scorer_h.stats(decide(q_hold, hold["is_s2"].to_numpy().astype(bool), cfg2, code_h)))
    adopted = bool(f_sel > f1 + MARGIN)
    report = {"stage1_validation_f05": f1, "stage2_validation_f05": f_sel, "stage2_rule": mode, "margin_required": MARGIN,
              "adopted": adopted, "stage1_holdout_f05": s_h1["macro_f05"], "stage2_holdout_f05": s_h2["macro_f05"],
              "stage2_holdout_singleton_accuracy": s_h2["singleton_accuracy"], "stage1_holdout_singleton_accuracy": s_h1["singleton_accuracy"],
              "folds": args.folds, "stage1_rounds": rounds, "stage2_trees": m2.best_iteration,
              "feature_gain_share": (pd.Series(m2.feature_importance("gain"), index=s2cols)
                                     .pipe(lambda s: (s / s.sum()).sort_values(ascending=False)).head(12).round(4).to_dict())}
    print(json.dumps(report, indent=2), flush=True)

    m2.save_model(str(paths.artifacts_dir / f"stage2_model{sfx}.txt"), num_iteration=m2.best_iteration)
    (paths.artifacts_dir / f"stage2_features{sfx}.json").write_text(json.dumps(s2cols))
    (paths.artifacts_dir / f"decision_config_stage2{sfx}.json").write_text(json.dumps({**cfg2, "adopted": adopted,
                                                                                       "validation_macro_f05": f_sel}, indent=2))
    (paths.reports_dir / f"stage2_report{sfx}.json").write_text(json.dumps(report, indent=2))
    print(f"stage 2 {'ADOPTED' if adopted else 'not adopted'} (validation {f1:.4f} -> {f_sel:.4f}; hold-out "
          f"{s_h1['macro_f05']:.4f} -> {s_h2['macro_f05']:.4f}); {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
