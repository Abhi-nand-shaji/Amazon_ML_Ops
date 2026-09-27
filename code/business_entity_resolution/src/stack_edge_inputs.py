#!/usr/bin/env python3
"""Edge-case / failure-taxonomy inputs for a stacker (run_stack.py), so run_edge_cases.py can describe the SUBMITTED model.

Joins the stacker's validation predictions (artifacts/stack_<exp>/stack_val.parquet: q) with the pairwise features of the
same pairs (the frame cached by run_stack.py --cache-frame) and writes, under the tag stack_<exp>:
  artifacts/val_predictions_stack_<exp>.parquet   p = stacker probability, is_s2, label, ids, the features the report uses
  artifacts/val_entities_stack_<exp>.json         the validation entities
  artifacts/decision_config_stack_<exp>.json      the stacker's global threshold
Then:  python run_edge_cases.py --tag stack_<exp>   ->  reports/edge_case_report_stack_<exp>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from world_groups import load_groups  # noqa: E402

REPORT_COLS = ["s1_entity_id", "cand_entity_id", "is_s2", "label", "name_ratio", "addr_ratio", "name_missing_1",
               "name_missing_2", "addr_missing_1", "addr_missing_2", "name_exact_normalized", "name_token_sort",
               "name_jw", "name_script_2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--exp", required=True, help="stacker experiment (artifacts/stack_<exp>)")
    ap.add_argument("--frame", required=True, help="frame parquet cached by run_stack.py --cache-frame")
    args = ap.parse_args()
    base = Paths(args.root)
    sdir = base.artifacts_dir / f"stack_{args.exp}"
    groups = load_groups(base)
    val_ids = sorted(groups["val"])
    fr = pd.read_parquet(args.frame, columns=REPORT_COLS)
    fr = fr[fr["s1_entity_id"].isin(groups["val"])]
    pred = pd.read_parquet(sdir / "stack_val.parquet", columns=["s1_entity_id", "cand_entity_id", "q"])
    d = fr.merge(pred, on=["s1_entity_id", "cand_entity_id"], how="inner").rename(columns={"q": "p"})
    if len(d) != len(pred):
        sys.exit(f"{len(pred) - len(d)} stacker predictions have no feature row: wrong --frame?")
    tag = Paths(args.root, f"stack_{args.exp}")
    d.to_parquet(tag.artifacts_dir / f"val_predictions{tag.suffix}.parquet", index=False)
    (tag.artifacts_dir / f"val_entities{tag.suffix}.json").write_text(json.dumps(val_ids))
    dec = json.loads((sdir / "stack_decision.json").read_text())
    tag.threshold_path().write_text(json.dumps({"mode": "global", "global_threshold": dec["global_threshold"]}, indent=2))
    print(f"wrote {len(d):,} validation pairs of {len(val_ids):,} entities (threshold {dec['global_threshold']}) "
          f"under tag stack_{args.exp}")


if __name__ == "__main__":
    main()
