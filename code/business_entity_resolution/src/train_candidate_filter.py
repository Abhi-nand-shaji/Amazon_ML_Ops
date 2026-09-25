#!/usr/bin/env python3
"""Train the candidate filter (candidate_filter.py) on the training shortlist and choose its probability floor.

Reads   artifacts/cache/shortlist_train/        run_blocking.py --split train (50,000 sampled S1 entities, <=30 pairs each)
Writes  artifacts/candidate_filter.txt          LightGBM filter fitted on all sampled entities (used for the test split)
        artifacts/candidate_filter.json         features, parameters, chosen floor, size / recall trade-off, importances
        artifacts/cache/filter_oof_train.parquet  out-of-fold probability of every training-shortlist pair

Cross-fitting: the sampled S1 entities are split into folds; every training pair is scored by a model that never saw
its entity, so the candidate sets the matcher is trained and validated on are cut exactly as the test split will be
(the test split uses the model fitted on all folds).
Floor: the largest value on config.FILTER_FLOOR_GRID that still keeps at least (1 - FILTER_RECALL_BUDGET) of the
shortlisted true pairs of the matcher's VALIDATION entities; the hold-out entities play no part.
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
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_filter import FILTER_FEATURES, LGB_PARAMS, N_ROUNDS, country_filter_features, shortlist_countries  # noqa: E402
from config import FILTER_FLOOR_GRID, FILTER_FOLDS, FILTER_RECALL_BUDGET, Paths, RANDOM_SEED  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from run_blocking import file_digest, shortlist_dir  # noqa: E402
from train_model import entity_split3  # noqa: E402


def tradeoff(score, label, ent_code, n_entities, n_true_all, singles_mask=None):
    """Rows of (floor, candidates per entity, share of ALL true pairs kept, entities left without any candidate)."""
    rows = []
    for t in (0.0,) + tuple(FILTER_FLOOR_GRID):
        keep = score >= t
        per_ent = np.bincount(ent_code[keep], minlength=n_entities)
        row = {"floor": t, "candidates_per_entity": float(keep.sum() / n_entities),
               "recall_all_true_pairs": float((keep & label).sum() / n_true_all),
               "entities_without_candidates": float((per_ent == 0).mean())}
        if singles_mask is not None:
            row["true_singletons_without_candidates"] = float((per_ent[singles_mask] == 0).mean())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--folds", type=int, default=FILTER_FOLDS)
    ap.add_argument("--budget", type=float, default=FILTER_RECALL_BUDGET,
                    help="max share of the shortlisted true pairs of validation entities the floor may drop")
    ap.add_argument("--val-frac", type=float, default=0.15, help="must match run_train.py")
    ap.add_argument("--hold-frac", type=float, default=0.15, help="must match run_train.py")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    t0 = time.time()
    sdir = shortlist_dir(paths, "train")
    ents = pq.read_table(sdir / "s1_entities.parquet", columns=["entity_id"])["entity_id"].to_pylist()
    truth = load_truth(paths.ground_truth(), set(ents))
    n_true_all = sum(len(v) for v in truth.values())

    frames = []
    for country in shortlist_countries(paths, "train"):
        short = pq.read_table(sdir / f"country={country}.parquet")
        feats = country_filter_features(paths, "train", country, short)
        feats["country"] = country
        feats["s1_row"], feats["pool_row"] = short["s1_row"].to_numpy(), short["pool_row"].to_numpy()
        feats["s1_entity_id"] = short["s1_entity_id"].to_pylist()
        feats["label"] = label_pairs(short["s1_entity_id"], short["cand_entity_id"], truth)
        frames.append(feats)
        print(f"[{country}] {len(feats):,} shortlisted pairs featurized, {int(feats['label'].sum()):,} true ({time.time() - t0:.0f}s)",
              flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames
    X, y = df[FILTER_FEATURES], df["label"].to_numpy()
    ent_code_all = pd.Index(sorted(ents))
    ent_code = ent_code_all.get_indexer(df["s1_entity_id"])
    fold_of_entity = np.random.default_rng(RANDOM_SEED).permutation(len(ent_code_all)) % args.folds
    fold = fold_of_entity[ent_code]
    params = {**LGB_PARAMS, "seed": RANDOM_SEED, "num_threads": args.threads}

    oof = np.zeros(len(df), np.float32)
    for k in range(args.folds):
        m = lgb.train(params, lgb.Dataset(X[fold != k], label=y[fold != k]), num_boost_round=N_ROUNDS)
        oof[fold == k] = m.predict(X[fold == k], num_threads=args.threads)
        print(f"  fold {k + 1}/{args.folds}: fitted on {(fold != k).sum():,} pairs ({time.time() - t0:.0f}s)", flush=True)
    final = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=N_ROUNDS)
    final.save_model(str(paths.filter_model_path()))

    # ---- floor: validation entities of the matcher only
    _tr, val_ids, _hold = entity_split3(ents, args.val_frac, args.hold_frac, RANDOM_SEED)
    is_val = df["s1_entity_id"].isin(val_ids).to_numpy()
    lab = y == 1
    val_true_short = (lab & is_val).sum()
    floor, curve = 0.0, []
    for t in FILTER_FLOOR_GRID:
        lost = 1 - (lab & is_val & (oof >= t)).sum() / val_true_short
        curve.append({"floor": t, "share_of_shortlisted_true_pairs_lost": float(lost),
                      "candidates_per_entity": float(((oof >= t) & is_val).sum() / len(val_ids))})
        if lost <= args.budget:
            floor = t
    print(f"chosen floor {floor} (budget {args.budget:.2%} of the {val_true_short:,} shortlisted true pairs of the "
          f"{len(val_ids):,} validation entities)", flush=True)

    singles = np.array([not truth[e] for e in ent_code_all])
    val_code = pd.Index(sorted(val_ids)).get_indexer(df["s1_entity_id"])
    table_all = tradeoff(oof, lab, ent_code, len(ent_code_all), n_true_all, singles)
    table_val = tradeoff(oof[is_val], lab[is_val], val_code[is_val], len(val_ids), sum(len(truth[e]) for e in val_ids))
    imp = pd.Series(final.feature_importance("gain"), index=FILTER_FEATURES)
    cfg = {
        "features": FILTER_FEATURES, "lgb_params": LGB_PARAMS, "rounds": N_ROUNDS, "folds": args.folds,
        "min_score": floor, "recall_budget": args.budget, "floor_selection_on_validation_entities": curve,
        "tradeoff_all_sampled_entities_oof": table_all, "tradeoff_validation_entities_oof": table_val,
        "shortlist": {"pairs": int(len(df)), "entities": len(ent_code_all), "true_pairs_all": n_true_all,
                      "true_pairs_in_shortlist": int(lab.sum()), "ranker": json.loads((sdir / "_meta.json").read_text())["ranker"]},
        "model_digest": file_digest(paths.filter_model_path()),
        "feature_gain_share": (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict(),
    }
    paths.filter_config_path().write_text(json.dumps(cfg, indent=2))
    tbl = pa.table({"country": pa.array(df["country"].to_numpy()), "s1_row": df["s1_row"].to_numpy().astype(np.int32),
                    "pool_row": df["pool_row"].to_numpy().astype(np.int32), "filter_score": oof})
    tmp = paths.filter_oof_path().with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    tmp.replace(paths.filter_oof_path())
    sel = next(r for r in table_all if r["floor"] == floor)
    print(f"all {len(ent_code_all):,} sampled entities (out-of-fold): {sel['candidates_per_entity']:.2f} candidates per entity "
          f"(shortlist {len(df) / len(ent_code_all):.2f}), recall {sel['recall_all_true_pairs']:.4%} of all true pairs "
          f"(shortlist {lab.sum() / n_true_all:.4%})", flush=True)
    print("feature gain share:", json.dumps(cfg["feature_gain_share"]), flush=True)
    print(f"saved {paths.filter_model_path().name}, {paths.filter_config_path().name}, {paths.filter_oof_path().name} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
