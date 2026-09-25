#!/usr/bin/env python3
"""Experiment behind the candidate filter: how small can the candidate set per S1 entity get?

Replays cuts of the 30-pair shortlist against a matcher that was trained and validated on the UNCUT shortlist, so it
needs the artifacts of a pipeline variant without the filter, e.g.
    run_candidate_cut.py --split train --tag nocut --min-score 0     (after copying shortlist_train to shortlist_train_nocut)
    run_features.py --split train --tag nocut ; run_train.py --tag nocut
(the submitted numbers were produced from the 30-candidate run that preceded the filter). Two analyses:

  ranker   cut(floor, k): keep the shortlisted pairs whose RANKER probability >= floor, at most k per entity. Reports, on the
           training sample, candidates / entity and the share of ALL true pairs kept; on the validation entities, the macro F0.5
           of the matcher restricted to the kept pairs (threshold re-searched).
  filter   fit small LightGBM filters (blocker evidence + cheap string similarities, with and without the pool-size-dependent
           statistics and the ranker probability) on the matcher's TRAIN entities and evaluate them the same way on its
           VALIDATION entities.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths, RANDOM_SEED  # noqa: E402
from evaluation import FastScorer  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from train_model import entity_split3, search_threshold  # noqa: E402

EVID = ["ov_name", "ov_addr", "idf_name", "idf_addr", "n_name_shared", "n_addr_shared", "n_addr_bigram", "n_name_bigram",
        "n_prefix", "exact_name", "exact_compact", "postal_eq", "n_strategies", "block_score", "nt_n1", "nt_n2", "at_n1", "at_n2",
        "rank_score", "block_rank", "block_gap_to_best", "is_s2"]
STR = ["name_token_set", "name_partial", "core_ratio", "core_jw", "addr_token_set", "addr_partial", "addr_ratio",
       "name_missing_2", "addr_missing_2", "housenum_match", "housenum_both_present"]
POOLDEP = {"idf_name", "idf_addr", "block_score", "n_name_shared", "n_addr_shared"}


def within_rank(ent_code, pool_row, score):
    order = np.lexsort((pool_row, -score, ent_code))
    e = ent_code[order]
    first = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    r = np.empty(len(e), np.int32)
    r[order] = np.arange(len(e)) - np.repeat(first, np.diff(np.r_[first, len(e)]))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="nocut", help="tag of the pipeline variant trained on the uncut shortlist")
    args = ap.parse_args()
    t0 = time.time()
    paths = Paths(args.root, args.tag)
    val = pd.read_parquet(paths.artifacts_dir / f"val_predictions{paths.suffix}.parquet")
    val_ids = json.loads((paths.artifacts_dir / f"val_entities{paths.suffix}.json").read_text())
    truth_v = load_truth(paths.ground_truth(), set(val_ids))
    scorer = FastScorer(val["s1_entity_id"].to_numpy(), val["label"].to_numpy(), truth_v, val_ids)
    p = val["p"].to_numpy()
    print(f"validation: {len(val_ids):,} entities, {len(val) / len(val_ids):.2f} candidates each, F0.5 {scorer.score(p >= 0.7):.5f}")

    # ---- ranker cuts: size / recall on the whole training sample, F0.5 on the validation entities
    sdir = paths.shortlist_dir("train")
    sl = pd.concat([pq.read_table(f, columns=["s1_entity_id", "cand_entity_id", "pool_row", "rank_score"]).to_pandas()
                    for f in sorted(sdir.glob("country=*.parquet"))], ignore_index=True)
    ents = pq.read_table(sdir / "s1_entities.parquet", columns=["entity_id"])["entity_id"].to_pylist()
    truth = load_truth(paths.ground_truth(), set(ents))
    n_true = sum(len(v) for v in truth.values())
    y = label_pairs(sl["s1_entity_id"], sl["cand_entity_id"], truth) == 1
    code = pd.factorize(sl["s1_entity_id"])[0]
    r, s = within_rank(code, sl["pool_row"].to_numpy(), sl["rank_score"].to_numpy()), sl["rank_score"].to_numpy()
    vcode = pd.factorize(val["s1_entity_id"])[0]
    rv, sv = within_rank(vcode, val["pool_row"].to_numpy(), val["rank_score"].to_numpy()), val["rank_score"].to_numpy()
    rows = []
    for floor in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        for k in (5, 10, 15, 20, 30):
            c, cv = (r < k) & (s >= floor), (rv < k) & (sv >= floor)
            rows.append({"cut": "ranker", "floor": floor, "k": k, "cand_per_entity": c.sum() / len(ents),
                         "recall_all_true": (c & y).sum() / n_true, "val_f05": search_threshold(scorer, p, extra_mask=cv)[1]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    # ---- learned filters: fitted on the matcher's train entities, evaluated on its validation entities
    tr_ids, _v, _h = entity_split3(ents, 0.15, 0.15, RANDOM_SEED)
    cols = sorted(set(EVID + STR)) + ["s1_entity_id", "label"]
    feats = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(paths.features_dir("train").glob("country=*.parquet"))],
                      ignore_index=True)
    tr = feats[feats["s1_entity_id"].isin(tr_ids)]
    sets = {"evidence+strings": EVID + STR, "without pool-size stats": [c for c in EVID + STR if c not in POOLDEP],
            "without pool-size stats and ranker probability": [c for c in EVID + STR if c not in POOLDEP | {"rank_score", "block_gap_to_best"}]}
    n_true_v = sum(len(v) for v in truth_v.values())
    rows = []
    for name, fs in sets.items():
        m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1,
                       "feature_fraction": 0.9, "seed": RANDOM_SEED}, lgb.Dataset(tr[fs], label=tr["label"]), num_boost_round=300)
        q = m.predict(val[fs])
        for floor in (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
            keep = q >= floor
            rows.append({"filter": name, "floor": floor, "val_cand_per_entity": keep.sum() / len(val_ids),
                         "val_recall_all_true": (keep & (val["label"].to_numpy() == 1)).sum() / n_true_v,
                         "val_f05": search_threshold(scorer, p, extra_mask=keep)[1]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
