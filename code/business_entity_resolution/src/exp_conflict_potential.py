#!/usr/bin/env python3
"""How many false positives are pool records that truly belong to ANOTHER S1 entity (a competitor that is absent from a
50k-entity sample but present in the full test world)? Upper bound of what conflict resolution can gain.
Reads the validation predictions and decision config written by run_train.py."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from evaluation import FastScorer, summarize  # noqa: E402
from ids import id_to_int, load_truth  # noqa: E402
from run_infer import decide  # noqa: E402

paths = Paths()
val = pd.read_parquet(paths.artifacts_dir / "val_predictions.parquet", columns=["s1_entity_id", "cand_entity_id", "label", "p", "is_s2"])
val_ids = json.loads((paths.artifacts_dir / "val_entities.json").read_text())
cfg = json.loads(paths.threshold_path().read_text())
keep = decide(val["p"].to_numpy(), val["is_s2"].to_numpy().astype(bool), cfg, pd.factorize(val["s1_entity_id"])[0])

gt = pd.read_csv(paths.ground_truth(), sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
gt = gt[gt["matched_entity_ids"].str.strip() != ""]
owner = gt.assign(cand=gt["matched_entity_ids"].str.split(",")).explode("cand")[["source1_entity_id", "cand"]]
owner = owner.rename(columns={"source1_entity_id": "owner"}).drop_duplicates("cand").set_index("cand")["owner"]

fp = val[keep & (val["label"] == 0).to_numpy()].copy()
fp["owner"] = fp["cand_entity_id"].map(owner)
print(f"kept pairs {int(keep.sum()):,}; false positives {len(fp):,}")
print(f"  FP whose record is OWNED by another S1 entity: {fp['owner'].notna().sum():,} ({fp['owner'].notna().mean()*100:.1f}%)")
print(f"  FP whose record belongs to nobody (distractor): {fp['owner'].isna().sum():,} ({fp['owner'].isna().mean()*100:.1f}%)")
print(f"  FP owned by an entity that is itself in the validation set: {fp['owner'].isin(set(val_ids)).sum():,}")

truth = load_truth(paths.ground_truth(), set(val_ids))
scorer = FastScorer(val["s1_entity_id"].to_numpy(), val["label"].to_numpy(), truth, sorted(val_ids))
base = summarize(scorer.stats(keep))
oracle_keep = keep.copy()
owned_fp = (keep & (val["label"] == 0).to_numpy())
owned_mask = owned_fp & val["cand_entity_id"].map(owner).notna().to_numpy()
oracle_keep[owned_mask] = False
after = summarize(scorer.stats(oracle_keep))
print(f"macro F0.5 now {base['macro_f05']:.4f}; if every 'owned by another S1' false positive were removed: {after['macro_f05']:.4f} "
      f"(singleton acc {base['singleton_accuracy']:.4f} -> {after['singleton_accuracy']:.4f}, P {base['mean_precision_non_singleton']:.4f} -> {after['mean_precision_non_singleton']:.4f})")
# how many of the false positives on true singletons are owned pool records?
sing = set(e for e in val_ids if not truth[e])
fps = fp[fp["s1_entity_id"].isin(sing)]
print(f"FP on true singletons: {len(fps):,} pairs over {fps['s1_entity_id'].nunique():,} entities; owned by another S1: {fps['owner'].notna().mean()*100:.1f}%")
