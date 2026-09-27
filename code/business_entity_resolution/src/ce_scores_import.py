#!/usr/bin/env python3
"""Map the cloud cross-encoder scores (kaggle_ce.py -> ce_scores.parquet: pair_id, p_ce) back to candidate pairs.

Reads   <kaggle_out>/ce_scores.parquet and the local_pair_index.parquet written by export_ce_package.py
Writes  artifacts/ce/ce_train_sets.parquet   s1_entity_id, cand_entity_id, set, label, p_ce for val / hold / stack pairs
                                              (input of run_stack.py --ce); test pairs are read by run_stack_infer.py
Reports coverage and AUC per set, and the macro F0.5 of the cross-encoder alone on validation / hold-out entities.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from evaluation import FastScorer, summarize  # noqa: E402
from ids import load_truth  # noqa: E402
from train_model import search_threshold  # noqa: E402
from world_groups import load_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--ce-scores", default=r"E:\Amazon\kaggle_out\ce_scores.parquet")
    ap.add_argument("--pair-index", default=r"E:\Amazon\kaggle_upload\local_pair_index.parquet")
    ap.add_argument("--out-name", default="ce_train_sets.parquet", help="file name under artifacts/ce/ (v4 new pairs: ce_train_sets_new.parquet)")
    args = ap.parse_args()
    paths = Paths(args.root)
    idx = pd.read_parquet(args.pair_index)
    ce = pd.read_parquet(args.ce_scores)
    m = idx.merge(ce, on="pair_id", how="left")
    for s, d in m.groupby("set"):
        cov = d["p_ce"].notna().mean()
        line = f"[{s}] pairs {len(d):,} scored {cov:.4%}"
        if s != "test" and d["label"].nunique() > 1 and cov > 0:
            from sklearn.metrics import roc_auc_score
            dd = d[d["p_ce"].notna()]
            line += f" AUC {roc_auc_score(dd['label'], dd['p_ce']):.5f}"
        print(line, flush=True)
    groups = load_groups(paths)
    for s in ("val", "hold"):
        d = m[m["set"] == s]
        ids = groups[s]
        truth = load_truth(paths.ground_truth(), ids)
        sc = FastScorer(d["s1_entity_id"].to_numpy(), d["label"].to_numpy(), truth, sorted(ids))
        t, f, _ = search_threshold(sc, d["p_ce"].fillna(0).to_numpy())
        su = summarize(sc.stats(d["p_ce"].fillna(0).to_numpy() >= t))
        print(f"cross-encoder alone on {s}: best threshold {t:.2f} -> macro F0.5 {su['macro_f05']:.4f} "
              f"(P {su['mean_precision_non_singleton']:.4f} R {su['mean_recall_non_singleton']:.4f})", flush=True)
    out = paths.artifacts_dir / "ce"
    out.mkdir(parents=True, exist_ok=True)
    tr = m[m["set"].isin(["val", "hold", "stack"])][["s1_entity_id", "cand_entity_id", "set", "label", "p_ce"]]
    tr.to_parquet(out / args.out_name, index=False)
    print(f"wrote {out / args.out_name} ({len(tr):,} rows)", flush=True)


if __name__ == "__main__":
    main()
