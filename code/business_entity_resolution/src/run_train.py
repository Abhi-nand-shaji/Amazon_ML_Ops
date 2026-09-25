#!/usr/bin/env python3
"""Train the matcher on the featurized TRAIN sample and choose decision thresholds.

Validation mimics the real task: S1 *entities* are split (not pairs); the model trains on
train-entity candidate pairs only (all positives + capped negatives), and validation
entities are scored over their FULL uncapped candidate lists with the challenge's exact
metric (macro F0.5, singleton rule included) -- exactly what test inference will face.
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
from run_features import features_dir  # noqa: E402
from run_infer import decide  # noqa: E402
from train_model import (  # noqa: E402
    CATEGORICAL_COLS,
    DEFAULT_GRID,
    POOL_SIZE_DEPENDENT,
    entity_max,
    entity_split3,
    feature_columns,
    lgb_params,
    negative_sampling_mask,
    search_threshold,
)


def load_split_frames(paths: Paths, val_ids: set, hold_ids: set, n_hard: int, n_random: int, train_frac: float = 1.0):
    """Stream the per-country feature files: validation and hold-out rows are kept in full, training rows are
    reduced to positives + hard negatives + a random negative sample (one entity's rows are contiguous).
    `train_frac` < 1 keeps only that (deterministic, entity-level) fraction of the training entities: learning curves."""
    val_parts, hold_parts, train_parts = [], [], []
    for f in sorted(features_dir(paths, "train").glob("country=*.parquet")):
        df = pq.read_table(f).to_pandas()
        is_val = df["s1_entity_id"].isin(val_ids).to_numpy()
        is_hold = df["s1_entity_id"].isin(hold_ids).to_numpy()
        val_parts.append(df[is_val].reset_index(drop=True))
        hold_parts.append(df[is_hold].reset_index(drop=True))
        tr = df[~(is_val | is_hold)]
        if train_frac < 1.0:
            tr = tr[(pd.util.hash_array(tr["s1_entity_id"].to_numpy(dtype=object)) % 1000) < int(train_frac * 1000)]
        keep = negative_sampling_mask(tr["s1_row"].to_numpy(), tr["label"].to_numpy(), tr["prelim_score"].to_numpy(),
                                      n_hard, n_random)
        train_parts.append(tr[keep])
        print(f"  {f.name}: {len(df):,} pairs -> val {int(is_val.sum()):,} | hold-out {int(is_hold.sum()):,} | "
              f"train kept {int(keep.sum()):,} of {len(tr):,}", flush=True)
        del df, tr
    val = pd.concat(val_parts, ignore_index=True)
    hold = pd.concat(hold_parts, ignore_index=True)
    # rows stay grouped by entity (stage-2 context features need it); LightGBM does not require shuffled rows
    train = pd.concat(train_parts, ignore_index=True)
    return train, val, hold


def per_source_thresholds(scorer, probs, is_s2, t_start, rounds=3):
    """Coordinate ascent over (S2 threshold, S3 threshold) on the true metric."""
    t2 = t3 = t_start
    best = scorer.score(probs >= t_start)
    for _ in range(rounds):
        for which in ("s2", "s3"):
            for t in DEFAULT_GRID:
                a, b = (t, t3) if which == "s2" else (t2, t)
                f = scorer.score((is_s2 & (probs >= a)) | (~is_s2 & (probs >= b)))
                if f > best + 1e-12:
                    best, t2, t3 = f, float(a), float(b)
    return t2, t3, best


def gated_thresholds(scorer, probs, is_s2, ent_max, t2, t3, rounds=3):
    """Coordinate ascent over (gate, S2 threshold, S3 threshold); gate 0 reproduces the per-source rule."""
    gate, a, b = 0.0, t2, t3
    best = scorer.score((ent_max >= gate) & ((is_s2 & (probs >= a)) | (~is_s2 & (probs >= b))))
    for _ in range(rounds):
        for which in ("gate", "s2", "s3"):
            for t in DEFAULT_GRID:
                g, x, y = (t, a, b) if which == "gate" else (gate, t, b) if which == "s2" else (gate, a, t)
                f = scorer.score((ent_max >= g) & ((is_s2 & (probs >= x)) | (~is_s2 & (probs >= y))))
                if f > best + 1e-12:
                    best, gate, a, b = f, float(g), float(x), float(y)
    return gate, a, b, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--hold-frac", type=float, default=0.15,
                    help="entities never used for any decision; scored once at the end (unbiased estimate)")
    ap.add_argument("--n-hard", type=int, default=1000, help="hardest negatives per entity kept for training (default: all; "
                                                             "candidate generation already leaves ~6 per entity)")
    ap.add_argument("--n-random", type=int, default=0, help="random extra negatives per entity kept for training")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="", help="pipeline-variant suffix (must match run_blocking / run_features)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of the training entities to use (learning-curve experiments)")
    ap.add_argument("--exp", default="", help="ablation name: output artifacts get an extra `_exp-NAME` suffix so the "
                                              "main model is not overwritten (re-run without --exp to promote a winner)")
    ap.add_argument("--drop-features", default="", help="comma list of feature names to exclude from the model")
    ap.add_argument("--robust", action="store_true",
                    help="also drop features whose scale depends on the size of the country pool (IDF sums, rare-token counts, "
                         "candidate counts): the test set has a small unseen country (France, ~1.4M pool records)")
    ap.add_argument("--compare-models", dest="compare_models", action="store_true", default=True,
                    help="also fit sklearn HistGradientBoosting / LogisticRegression for the model comparison")
    ap.add_argument("--no-compare-models", dest="compare_models", action="store_false")
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    t0 = time.time()
    sampled = pq.read_table(candidates_dir(paths, "train") / "s1_entities.parquet", columns=["entity_id"])["entity_id"].to_pylist()
    train_ids, val_ids, hold_ids = entity_split3(sampled, args.val_frac, args.hold_frac, RANDOM_SEED)
    print(f"sampled S1 entities: {len(sampled):,}  -> train {len(train_ids):,} / val {len(val_ids):,} / "
          f"hold-out {len(hold_ids):,}", flush=True)
    truth = load_truth(paths.ground_truth(), set(sampled))

    print("Loading features ...", flush=True)
    train, val, hold = load_split_frames(paths, val_ids, hold_ids, args.n_hard, args.n_random, args.train_frac)
    dropped = {c for c in args.drop_features.split(",") if c} | (POOL_SIZE_DEPENDENT if args.robust else set())
    fcols = [c for c in feature_columns(train) if c not in dropped]
    if dropped:
        print(f"  dropping features: {sorted(dropped)}", flush=True)
    print(f"  train rows {len(train):,} (pos rate {train['label'].mean():.4f}) | val rows {len(val):,} "
          f"(pos rate {val['label'].mean():.4f}) | {len(fcols)} features", flush=True)

    scorer = FastScorer(val["s1_entity_id"].to_numpy(), val["label"].to_numpy(), truth, sorted(val_ids))
    log = []
    cand_meta = json.loads((candidates_dir(paths, "train") / "_meta.json").read_text())
    n_true_val = sum(len(truth[e]) for e in val_ids)
    blocking_desc = ("multi-key retrieval (rare unigrams + exact keys + address/name bigrams + name prefix) + learned ranker, "
                     f"shortlist {cand_meta['blocking_config']['max_candidates']}/entity" if cand_meta["blocking_config"].get("rescore")
                     else "legacy rare-token blocking")
    if "candidate_filter" in cand_meta:
        blocking_desc += f" + learned candidate filter (probability >= {cand_meta['candidate_filter']['min_score']})"
    cand_recall = float(val["label"].sum() / n_true_val)          # true pairs that made it into the candidate set
    avg_cands = len(val) / len(val_ids)

    def record(name, keep, model="", features="", threshold=""):
        s = summarize(scorer.stats(keep))
        log.append({"experiment": name, "blocking": blocking_desc, "features": features, "model": model, "threshold": threshold,
                    "candidate_recall": cand_recall, "avg_candidates_per_entity": avg_cands, **s})
        print(f"  [{name}] macro_f05={s['macro_f05']:.4f} singleton_acc={s['singleton_accuracy']:.4f} "
              f"P={s['mean_precision_non_singleton']:.4f} R={s['mean_recall_non_singleton']:.4f} "
              f"mean_pred={s['mean_pred_size']:.2f} (truth {s['mean_truth_size']:.2f})", flush=True)
        return s["macro_f05"]

    print("\n=== Reference points on validation entities ===")
    ceiling = record("ORACLE_given_blocking (predict exactly the true pairs that are candidates)", val["label"].to_numpy() == 1,
                     model="oracle", threshold="n/a")
    record("predict_nothing", np.zeros(len(val), bool), model="constant", threshold="n/a")
    record("B1_exact_normalized_name", val["name_exact_normalized"].to_numpy() == 1, model="rule", features="normalized name equality", threshold="n/a")
    record("B2_fuzzy_name_and_address", ((val["name_ratio"].fillna(0) > 0.85) & (val["addr_ratio"].fillna(0) > 0.5)).to_numpy(),
           model="rule", features="name Levenshtein>0.85 & address Levenshtein>0.5", threshold="0.85 / 0.5")
    t_b3, f_b3, _ = search_threshold(scorer, val["prelim_score"].to_numpy())
    record("B3_similarity_score_threshold", val["prelim_score"].to_numpy() >= t_b3, model="rule",
           features="0.5*name_ratio + 0.5*addr_ratio", threshold=f"{t_b3:.2f} (searched)")
    t_cos, _, _ = search_threshold(scorer, np.nan_to_num(val["nt_idf_cos"].to_numpy()) * 0.5 + np.nan_to_num(val["at_idf_cos"].to_numpy()) * 0.5)
    record("B3b_tfidf_cosine_threshold",
           (np.nan_to_num(val["nt_idf_cos"].to_numpy()) * 0.5 + np.nan_to_num(val["at_idf_cos"].to_numpy()) * 0.5) >= t_cos,
           model="rule", features="0.5*name IDF-cosine + 0.5*address IDF-cosine (word tokens)", threshold=f"{t_cos:.2f} (searched)")

    print("\n=== LightGBM ===", flush=True)
    cat_idx = [fcols.index(c) for c in CATEGORICAL_COLS if c in fcols]
    dtrain = lgb.Dataset(train[fcols], label=train["label"], categorical_feature=cat_idx, free_raw_data=False)
    dval = lgb.Dataset(val[fcols], label=val["label"], categorical_feature=cat_idx, reference=dtrain, free_raw_data=False)
    params = lgb_params(args.threads, args.leaves, args.lr)
    t1 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.log_evaluation(100)])
    print(f"  {booster.num_trees()} trees in {time.time() - t1:.0f}s (best_iteration={booster.best_iteration})", flush=True)
    imp = pd.Series(booster.feature_importance(importance_type="gain"), index=fcols).sort_values(ascending=False)
    print("  top features by gain:\n" + (imp / imp.sum()).head(15).round(4).to_string(), flush=True)

    probs = booster.predict(val[fcols], num_iteration=booster.best_iteration)
    t_glob, f_glob, grid = search_threshold(scorer, probs)
    record("B4_lightgbm_global_threshold", probs >= t_glob, model=f"LightGBM ({booster.best_iteration} trees, {args.leaves} leaves)",
           features=f"{len(fcols)} pairwise features", threshold=f"{t_glob:.2f} (searched)")

    is_s2 = val["is_s2"].to_numpy().astype(bool)
    t2, t3, f_ps = per_source_thresholds(scorer, probs, is_s2, t_glob)
    f_ps_rec = record("B5_lightgbm_per_source_thresholds", (is_s2 & (probs >= t2)) | (~is_s2 & (probs >= t3)),
                      model=f"LightGBM ({booster.best_iteration} trees, {args.leaves} leaves)",
                      features=f"{len(fcols)} pairwise features", threshold=f"S2 {t2:.2f} / S3 {t3:.2f} (searched)")

    # ---- other model families on the same features / validation entities (Section 12 of the brief)
    if args.compare_models:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        sub = train.sample(n=min(len(train), 500_000), random_state=RANDOM_SEED)
        Xs, ys = sub[fcols], sub["label"]
        Xf = val[fcols]
        t1 = time.time()
        hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_leaf_nodes=63, random_state=RANDOM_SEED,
                                             early_stopping=False, categorical_features=[fcols.index(c) for c in CATEGORICAL_COLS if c in fcols]).fit(Xs, ys)
        p_h = hgb.predict_proba(Xf)[:, 1]
        t_h, _, _ = search_threshold(scorer, p_h)
        record("M1_sklearn_HistGradientBoosting", p_h >= t_h, model=f"HistGradientBoosting (250 iters, 500k rows, {time.time()-t1:.0f}s)",
               features=f"{len(fcols)} pairwise features", threshold=f"{t_h:.2f} (searched)")
        t1 = time.time()
        med = Xs.median(numeric_only=True)
        def lr_matrix(X):      # NaN -> median plus explicit missing indicators for the columns that have NaN
            nan_cols = [c for c in fcols if Xs[c].isna().any()]
            out = X.fillna(med)
            for c in nan_cols:
                out[c + "_isnan"] = X[c].isna().astype(np.float32)
            return out
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0)).fit(lr_matrix(Xs), ys)
        p_l = lr.predict_proba(lr_matrix(Xf))[:, 1]
        t_l, _, _ = search_threshold(scorer, p_l)
        record("M2_sklearn_LogisticRegression", p_l >= t_l, model=f"LogisticRegression (500k rows, {time.time()-t1:.0f}s)",
               features=f"{len(fcols)} pairwise features (+missing indicators)", threshold=f"{t_l:.2f} (searched)")
    # ---- entity-gated rule: the entity's best score must clear `gate`, then per-source thresholds decide which
    # candidates are kept (the first match needs a higher bar than additional ones)
    ent_max = entity_max(probs, scorer.ent_idx, scorer.n)
    gate, g2, g3, f_gate = gated_thresholds(scorer, probs, is_s2, ent_max, t2, t3)
    record("B6_lightgbm_entity_gated", (ent_max >= gate) & ((is_s2 & (probs >= g2)) | (~is_s2 & (probs >= g3))),
           model=f"LightGBM ({booster.best_iteration} trees, {args.leaves} leaves)", features=f"{len(fcols)} pairwise features",
           threshold=f"gate {gate:.2f}; S2 {g2:.2f} / S3 {g3:.2f} (searched)")

    # A more elaborate rule is adopted only if it beats the simpler one by a margin (limits validation over-fitting).
    MARGIN = 1e-3
    mode, f_sel = "global", f_glob
    if f_ps > f_sel + MARGIN:
        mode, f_sel = "per_source", f_ps
    if f_gate > f_sel + MARGIN:
        mode, f_sel = "gated", f_gate
    cfg = {"mode": mode, "global_threshold": t_glob,
           "s2_threshold": {"global": t_glob, "per_source": t2, "gated": g2}[mode],
           "s3_threshold": {"global": t_glob, "per_source": t3, "gated": g3}[mode],
           "gate": gate if mode == "gated" else None,
           "validation_macro_f05": f_sel, "oracle_given_blocking": ceiling}
    print(f"\nSelected: {mode} decision rule -> validation macro F0.5 = {f_sel:.4f} "
          f"(global {f_glob:.4f}, per-source {f_ps:.4f}, gated {f_gate:.4f}; ceiling from blocking {ceiling:.4f})", flush=True)

    # ---- hold-out entities: scored ONCE with the selected model / thresholds / rule (nothing was tuned on them)
    truth_h = {e: truth[e] for e in hold_ids}
    scorer_h = FastScorer(hold["s1_entity_id"].to_numpy(), hold["label"].to_numpy(), truth_h, sorted(hold_ids))
    p_hold = booster.predict(hold[fcols], num_iteration=booster.best_iteration)
    keep_h = decide(p_hold, hold["is_s2"].to_numpy().astype(bool), cfg, pd.factorize(hold["s1_entity_id"])[0])
    s_h = summarize(scorer_h.stats(keep_h))
    cand_recall_h = float(hold["label"].sum() / max(1, sum(len(v) for v in truth_h.values())))
    log.append({"experiment": "FINAL_holdout_entities (never used for any decision)", "blocking": blocking_desc,
                "features": f"{len(fcols)} pairwise features", "model": "LightGBM", "threshold": f"{mode} rule, thresholds from validation",
                "candidate_recall": cand_recall_h, "avg_candidates_per_entity": len(hold) / len(hold_ids), **s_h})
    print(f"HOLD-OUT ({len(hold_ids):,} entities never used for selection): macro F0.5 = {s_h['macro_f05']:.4f}  "
          f"singleton_acc={s_h['singleton_accuracy']:.4f} P={s_h['mean_precision_non_singleton']:.4f} "
          f"R={s_h['mean_recall_non_singleton']:.4f}  candidate recall {cand_recall_h:.4f}", flush=True)
    cfg["holdout_macro_f05"] = s_h["macro_f05"]

    # ---- artifacts
    tag = paths.suffix + (f"_exp-{args.exp}" if args.exp else "")
    booster.save_model(str(paths.artifacts_dir / f"lgbm_model{tag}.txt"), num_iteration=booster.best_iteration)
    (paths.artifacts_dir / f"feature_columns{tag}.json").write_text(
        json.dumps({"feature_columns": fcols, "categorical_columns": CATEGORICAL_COLS}, indent=2))
    (paths.artifacts_dir / f"decision_config{tag}.json").write_text(json.dumps(cfg, indent=2))
    pd.DataFrame(log).to_csv(paths.reports_dir / f"experiment_log{tag}.csv", index=False)
    val_out = val.copy()                     # all features + ids + label: the edge-case diagnostics need them
    val_out["p"] = probs.astype(np.float32)
    val_out.to_parquet(paths.artifacts_dir / f"val_predictions{tag}.parquet", index=False)
    (paths.artifacts_dir / f"val_entities{tag}.json").write_text(json.dumps(sorted(val_ids)))
    grid.to_csv(paths.reports_dir / f"threshold_curve{tag}.csv", index=False)
    (paths.reports_dir / f"validation_report{tag}.json").write_text(json.dumps({
        "n_train_entities": len(train_ids), "n_val_entities": len(val_ids), "n_train_rows": len(train),
        "n_val_rows": len(val), "best_iteration": booster.best_iteration, "config": cfg,
        "top_features_gain_share": (imp / imp.sum()).head(25).round(5).to_dict(),
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=float))
    print(f"artifacts saved (tag='{args.tag}'); total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
