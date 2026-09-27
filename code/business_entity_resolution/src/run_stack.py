#!/usr/bin/env python3
"""Stage 2 ("stacker"): LightGBM over pairwise features + first-stage score + cross-encoder score + graph features.

Fitted on the full-world `stack` entities (world_groups.py): no earlier model has seen them, so p1 (first-stage
LightGBM), p_ce (cross-encoder) and the graph features (which summarize p1 over the complete candidate graph -- every
training S1 entity competes for the pool records, as every test entity does) behave on them exactly as on the test split.
Early stopping and the decision threshold use the `val` entities; `hold` entities are scored once at the end.

Also reports, on the same entities, the first-stage model alone and the parameter-free exclusivity rule (threshold on
the exclusivity-normalized probability q), so every component's contribution is measured with the challenge metric.

Writes artifacts/stack<exp>/: stack_model.txt, stack_features.json, stack_decision.json, stack_report.json,
                              stack_val.parquet / stack_hold.parquet (predictions)
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
from pipeline import load_meta, load_partition  # noqa: E402
from run_world_score import world_feats_dir, world_p_dir  # noqa: E402
from build_v4_candidates import EXTRA as V4_EXTRA  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from stack_features import CE_COL, graph_features, pool_side_for_rows  # noqa: E402
from stack_v4 import pool_side_v4  # noqa: E402
from train_model import CATEGORICAL_COLS, lgb_params, search_threshold  # noqa: E402
from world_groups import load_groups  # noqa: E402

GRAPH_PREFIXES = ("e_", "s_", "ps_", "sib_", "as_")


def build_frame(world: Paths, country: str, v4: bool = False) -> pd.DataFrame:
    t0 = time.time()
    feats = pq.read_table(world_feats_dir(world) / f"country={country}.parquet").to_pandas()
    wp = pq.read_table(world_p_dir(world) / f"country={country}.parquet")
    if v4:
        # world_p is row-aligned with the v4 candidate file: source flags tell first-hop pairs from new ones
        cand = pq.read_table(candidates_dir(world, "train") / f"country={country}.parquet",
                             columns=["s1_row", "pool_row"] + V4_EXTRA).to_pandas()
        if not (np.array_equal(cand["s1_row"].to_numpy(), wp["s1_row"].to_numpy())
                and np.array_equal(cand["pool_row"].to_numpy(), wp["pool_row"].to_numpy())):
            raise RuntimeError(f"[{country}] world scores are not row-aligned with the v4 candidate file")
        is_new_all = (cand["src_dense"].fillna(0).to_numpy() > 0) | (cand["src_hop"].fillna(0).to_numpy() > 0)
        feats = feats.merge(cand, on=["s1_row", "pool_row"], how="left")
        is_new = (feats["src_dense"].fillna(0).to_numpy() > 0) | (feats["src_hop"].fillna(0).to_numpy() > 0)
        fh = ~is_new_all
        ps = pool_side_v4(wp["pool_row"].to_numpy()[fh], wp["p1"].to_numpy()[fh], feats["pool_row"].to_numpy(),
                          feats["p1"].to_numpy(), is_new)
        del cand
    else:
        ps = pool_side_for_rows(wp["s1_row"].to_numpy(), wp["pool_row"].to_numpy(), wp["p1"].to_numpy(),
                                feats["s1_row"].to_numpy(), feats["pool_row"].to_numpy())
    del wp
    s1, pool, _ = load_partition(world, "train", country, load_meta(world, "train"), s1_cols=["entity_id", "address_original"],
                                 pool_cols=["entity_id", "name_normalized", "address_normalized", "address_original"],
                                 with_full=False)
    g = graph_features(feats["s1_row"].to_numpy(), feats["pool_row"].to_numpy(), feats["p1"].to_numpy(),
                       feats["is_s2"].to_numpy(), pool["name_normalized"].combine_chunks(),
                       pool["address_normalized"].combine_chunks(), ps,
                       s1_addr_raw=s1["address_original"].combine_chunks(), pool_addr_raw=pool["address_original"].combine_chunks())
    del pool, s1
    out = pd.concat([feats.reset_index(drop=True), g], axis=1)
    out["country"] = country
    print(f"  [{country}] {len(out):,} rows, {g.shape[1]} graph features ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--ce", default=None, help="parquet(s) with s1_entity_id, cand_entity_id, p_ce (comma list); omit = no CE")
    ap.add_argument("--exp", default="")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--no-graph", action="store_true", help="ablation: no graph features")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--cache-frame", default=None, help="parquet path to cache / reuse the assembled frame")
    ap.add_argument("--v4", action="store_true", help="v4 candidate sets (first-hop + dense / second-hop new pairs)")
    args = ap.parse_args()

    base, world = Paths(args.root), Paths(args.root, args.tag)
    out_dir = base.artifacts_dir / f"stack{('_' + args.exp) if args.exp else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups = load_groups(base)
    if args.cache_frame and Path(args.cache_frame).exists():
        df = pd.read_parquet(args.cache_frame)
        print(f"frame loaded from cache: {len(df):,} rows", flush=True)
    else:
        countries = sorted(p.stem.split("=", 1)[1] for p in world_feats_dir(world).glob("country=*.parquet"))
        df = pd.concat([build_frame(world, c, args.v4) for c in countries], ignore_index=True)
        if args.cache_frame:
            df.to_parquet(args.cache_frame, index=False)
    if args.ce:
        ce = pd.concat([pd.read_parquet(f, columns=["s1_entity_id", "cand_entity_id", CE_COL]) for f in args.ce.split(",")],
                       ignore_index=True).drop_duplicates(["s1_entity_id", "cand_entity_id"])
        df = df.merge(ce, on=["s1_entity_id", "cand_entity_id"], how="left")
        print(f"cross-encoder scores attached: {df[CE_COL].notna().mean():.4%} of rows", flush=True)

    base_cols = json.loads(base.feature_list_path().read_text())["feature_columns"]
    graph_cols = [] if args.no_graph else [c for c in df.columns if c.startswith(GRAPH_PREFIXES)]
    extra_cols = [c for c in V4_EXTRA if args.v4 and c in df.columns]
    fcols = base_cols + ["p1"] + ([CE_COL] if args.ce else []) + graph_cols + extra_cols
    grp = df["s1_entity_id"].map(lambda s: "stack" if s in groups["stack"] else "val" if s in groups["val"]
                                 else "hold" if s in groups["hold"] else "other")
    tr, va, ho = df[grp == "stack"], df[grp == "val"].reset_index(drop=True), df[grp == "hold"].reset_index(drop=True)
    truth = load_truth(base.ground_truth(), groups["val"] | groups["hold"])
    sc_va = FastScorer(va["s1_entity_id"].to_numpy(), va["label"].to_numpy(), truth, sorted(groups["val"]))
    sc_ho = FastScorer(ho["s1_entity_id"].to_numpy(), ho["label"].to_numpy(), truth, sorted(groups["hold"]))
    print(f"rows: stack {len(tr):,} ({tr['s1_entity_id'].nunique():,} entities) | val {len(va):,} | hold {len(ho):,}; "
          f"{len(fcols)} features ({len(graph_cols)} graph) ({time.time() - t0:.0f}s)", flush=True)

    report = {}

    def record(name, p_va, p_ho, extra=None):
        t, f_va, _ = search_threshold(sc_va, p_va)
        s_va, s_ho = summarize(sc_va.stats(p_va >= t)), summarize(sc_ho.stats(p_ho >= t))
        report[name] = {"threshold": t, "val_macro_f05": s_va["macro_f05"], "hold_macro_f05": s_ho["macro_f05"],
                        "hold_precision": s_ho["mean_precision_non_singleton"], "hold_recall": s_ho["mean_recall_non_singleton"],
                        "hold_singleton_acc": s_ho["singleton_accuracy"], **(extra or {})}
        print(f"  [{name}] threshold {t:.2f}: val {s_va['macro_f05']:.4f} | hold-out {s_ho['macro_f05']:.4f} "
              f"(P {s_ho['mean_precision_non_singleton']:.4f} R {s_ho['mean_recall_non_singleton']:.4f} "
              f"singleton {s_ho['singleton_accuracy']:.4f})", flush=True)
        return t

    for name, d, g in (("val", va, "val"), ("hold", ho, "hold")):
        n_true = sum(len(truth[e]) for e in groups[g])
        report[f"candidates_{name}"] = {"candidate_recall": float(d["label"].sum() / max(1, n_true)),
                                        "candidates_per_entity": float(len(d) / len(groups[g]))}
        print(f"  [{name}] candidate recall {d['label'].sum() / max(1, n_true):.4%}, "
              f"{len(d) / len(groups[g]):.2f} candidates per entity", flush=True)
    print("\n=== reference rules on the same entities", flush=True)
    record("oracle: exactly the true candidates", va["label"].to_numpy().astype(float), ho["label"].to_numpy().astype(float))
    record("first-stage LightGBM p1", va["p1"].to_numpy(), ho["p1"].to_numpy())
    record("exclusivity-normalized q(p1)", va["ps_q"].to_numpy(), ho["ps_q"].to_numpy())
    if args.ce:
        record("cross-encoder alone", va[CE_COL].fillna(0).to_numpy(), ho[CE_COL].fillna(0).to_numpy())
        record("mean(p1, p_ce)", 0.5 * (va["p1"] + va[CE_COL].fillna(va["p1"])).to_numpy(),
               0.5 * (ho["p1"] + ho[CE_COL].fillna(ho["p1"])).to_numpy())

    print("\n=== stacker", flush=True)
    cat = [c for c in CATEGORICAL_COLS if c in fcols]
    dtr = lgb.Dataset(tr[fcols], label=tr["label"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(va[fcols], label=va["label"], categorical_feature=cat, reference=dtr, free_raw_data=False)
    t1 = time.time()
    booster = lgb.train(lgb_params(args.threads, args.leaves, args.lr), dtr, num_boost_round=args.rounds, valid_sets=[dva],
                        valid_names=["val"], callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                                                        lgb.log_evaluation(200)])
    print(f"  {booster.best_iteration} trees ({time.time() - t1:.0f}s)", flush=True)
    q_va = booster.predict(va[fcols], num_iteration=booster.best_iteration)
    q_ho = booster.predict(ho[fcols], num_iteration=booster.best_iteration)
    thr = record("STACKER", q_va, q_ho, {"trees": booster.best_iteration})
    imp = pd.Series(booster.feature_importance("gain"), index=fcols).sort_values(ascending=False)
    print("  top features by gain:\n" + (imp / imp.sum()).head(25).round(4).to_string(), flush=True)

    booster.save_model(str(out_dir / "stack_model.txt"), num_iteration=booster.best_iteration)
    (out_dir / "stack_features.json").write_text(json.dumps({"feature_columns": fcols, "categorical_columns": cat,
                                                             "uses_ce": bool(args.ce), "graph": not args.no_graph,
                                                             "v4": bool(args.v4)}, indent=2))
    (out_dir / "stack_decision.json").write_text(json.dumps({"mode": "global", "global_threshold": thr,
                                                            "validation_macro_f05": report["STACKER"]["val_macro_f05"],
                                                            "holdout_macro_f05": report["STACKER"]["hold_macro_f05"]}, indent=2))
    report["feature_gain_share_top40"] = (imp / imp.sum()).head(40).round(5).to_dict()
    (out_dir / "stack_report.json").write_text(json.dumps(report, indent=2, default=float))
    for name, d, q in (("val", va, q_va), ("hold", ho, q_ho)):
        d[["country", "s1_entity_id", "cand_entity_id", "label", "p1"]].assign(q=q.astype(np.float32)).to_parquet(
            out_dir / f"stack_{name}.parquet", index=False)
    print(f"saved {out_dir} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
