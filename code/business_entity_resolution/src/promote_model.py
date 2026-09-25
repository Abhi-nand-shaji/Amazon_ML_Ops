#!/usr/bin/env python3
"""Choose between the default and the France-robust matcher and promote the winner to the main artifact names.

The robust variant (`run_train.py --robust --exp robust`) drops the features whose scale depends on the SIZE of the
country pool (IDF sums, rare-token counts, candidate counts, exact-name / address frequencies). The test set adds France,
whose pool is 3-4x smaller than the training pools and which has no labels, so a model that does not depend on pool size
is preferred whenever it costs (almost) nothing on the labelled data:

    promote robust  <=>  its validation AND hold-out macro F0.5 are each >= the default's minus `--tolerance`.

The decision and both sets of numbers are written to reports/model_selection.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402

FILES = [("artifacts", "lgbm_model{}.txt"), ("artifacts", "feature_columns{}.json"), ("artifacts", "decision_config{}.json"),
         ("artifacts", "val_predictions{}.parquet"), ("artifacts", "val_entities{}.json"),
         ("reports", "validation_report{}.json"), ("reports", "threshold_curve{}.csv")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exp", default="robust")
    ap.add_argument("--tolerance", type=float, default=0.002)
    args = ap.parse_args()
    paths = Paths(args.root, args.tag)
    base = {"artifacts": paths.artifacts_dir, "reports": paths.reports_dir}
    sfx, exp = paths.suffix, f"{paths.suffix}_exp-{args.exp}"

    d = json.loads((base["artifacts"] / f"decision_config{sfx}.json").read_text())
    r = json.loads((base["artifacts"] / f"decision_config{exp}.json").read_text())
    ok = (r["validation_macro_f05"] >= d["validation_macro_f05"] - args.tolerance
          and r["holdout_macro_f05"] >= d["holdout_macro_f05"] - args.tolerance)
    sel = {"default": {"validation": d["validation_macro_f05"], "holdout": d["holdout_macro_f05"]},
           "robust": {"validation": r["validation_macro_f05"], "holdout": r["holdout_macro_f05"]},
           "tolerance": args.tolerance, "promoted": "robust" if ok else "default"}
    if ok:
        log_d = pd.read_csv(paths.reports_dir / f"experiment_log{sfx}.csv")
        log_r = pd.read_csv(paths.reports_dir / f"experiment_log{exp}.csv")
        for kind, name in FILES:                                          # keep the default as *_exp-default, promote robust
            src, dst_backup, dst = base[kind] / name.format(exp), base[kind] / name.format(sfx + "_exp-default"), base[kind] / name.format(sfx)
            if (base[kind] / name.format(sfx)).exists():
                shutil.copyfile(base[kind] / name.format(sfx), dst_backup)
            shutil.copyfile(src, dst)
        # merged experiment log: the promoted model's rows carry the headline names; the default's rows are kept, renamed
        log_d["experiment"] = log_d["experiment"].where(~log_d["experiment"].str.startswith(("B4", "B5", "B6", "FINAL_holdout")),
                                                         log_d["experiment"] + " [default features, not promoted]")
        keep = log_r[log_r["experiment"].str.startswith(("B4", "B5", "B6", "FINAL_holdout"))].copy()
        keep["experiment"] = keep["experiment"] + " [robust features, promoted]"
        keep.loc[keep["experiment"].str.startswith("FINAL_holdout"), "experiment"] = "FINAL_holdout_entities (never used for any decision) [robust features, promoted]"
        pd.concat([log_d, keep], ignore_index=True).to_csv(paths.reports_dir / f"experiment_log{sfx}.csv", index=False)
    (paths.reports_dir / "model_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))
    print(f"-> {'ROBUST features promoted to the main artifacts' if ok else 'default model kept'}")


if __name__ == "__main__":
    main()
