#!/usr/bin/env python3
"""Cross-encoder scoring package for the NEW (v4) candidate pairs only: kaggle_ce.py in resume mode.

The first cloud run's output (ce_model/ with the fine-tuned weights) is attached to the new notebook as an input, so
kaggle_ce.py skips fine-tuning and only scores. Pair ids start at --id-offset so they can never collide with the first
run's pair ids (that run's ce_scores.parquet is also visible in the inputs and its ids are treated as already scored).

Writes <out>/records.parquet, pairs_score.parquet, pairs_eval.parquet, pairs_train.parquet (1 dummy row), kaggle_ce.py,
       manifest.json, and <out>/../local_pair_index_new.parquet (pair_id -> split, country, rows, ids, label).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ce_data import PartitionText  # noqa: E402
from config import Paths  # noqa: E402
from export_ce_package import PRIO, RecordTable  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from pipeline import candidate_countries  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from world_groups import load_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-tag", default="full_v4")
    ap.add_argument("--test-tag", default="v4")
    ap.add_argument("--out", default=r"E:\Amazon\kaggle_upload\er-ce-new")
    ap.add_argument("--id-offset", type=int, default=100_000_000)
    args = ap.parse_args()
    base = Paths()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups = load_groups(base)
    truth = load_truth(base.ground_truth(), groups["stack"] | groups["val"] | groups["hold"])
    rt = RecordTable()
    score, evals, index = [], [], []
    pid = args.id_offset
    for split, paths in (("train", Paths(tag=args.world_tag)), ("test", Paths(tag=args.test_tag))):
        for country in candidate_countries(paths, split):
            c = pq.read_table(candidates_dir(paths, split) / f"country={country}.parquet",
                              columns=["s1_row", "pool_row", "s1_entity_id", "cand_entity_id", "src_dense", "src_hop"]).to_pandas()
            c = c[(c.src_dense.fillna(0) > 0) | (c.src_hop.fillna(0) > 0)].reset_index(drop=True)
            if not len(c):
                continue
            txt = PartitionText(paths, split, country)
            txt.check(c.s1_row.to_numpy(), c.pool_row.to_numpy(), pa.array(c.s1_entity_id), pa.array(c.cand_entity_id))
            u1, u2 = np.unique(c.s1_row), np.unique(c.pool_row)
            rid1 = np.full(len(txt.s1_txt), -1, np.int32)
            rid2 = np.full(len(txt.pool_txt), -1, np.int32)
            rid1[u1] = rt.add(txt.s1_txt.take(pa.array(u1)))
            rid2[u2] = rt.add(txt.pool_txt.take(pa.array(u2)))
            if split == "train":
                g = np.where(c.s1_entity_id.isin(groups["val"]), "val", np.where(c.s1_entity_id.isin(groups["hold"]), "hold", "stack"))
                lab = label_pairs(pa.array(c.s1_entity_id), pa.array(c.cand_entity_id), truth)
            else:
                g = np.full(len(c), "test")
                lab = np.zeros(len(c), np.int8)
            ids = np.arange(pid, pid + len(c), dtype=np.int64)
            pid += len(c)
            left, right = rid1[c.s1_row.to_numpy()], rid2[c.pool_row.to_numpy()]
            score.append(pd.DataFrame({"pair_id": ids, "left": left, "right": right, "prio": np.array([PRIO[x] for x in g], np.int8)}))
            ev = np.isin(g, ["val", "hold"])
            if ev.any():
                evals.append(pd.DataFrame({"left": left[ev], "right": right[ev], "label": lab[ev], "set": g[ev]}))
            index.append(pd.DataFrame({"pair_id": ids, "set": g, "split": split, "country": country,
                                       "s1_row": c.s1_row.to_numpy().astype(np.int32), "pool_row": c.pool_row.to_numpy().astype(np.int32),
                                       "s1_entity_id": c.s1_entity_id.to_numpy(), "cand_entity_id": c.cand_entity_id.to_numpy(),
                                       "label": lab}))
            print(f"  [{split}/{country}] {len(c):,} new pairs ({time.time() - t0:.0f}s)", flush=True)
            del txt
    rt.write(out / "records.parquet")
    sc = pd.concat(score, ignore_index=True)
    sc.to_parquet(out / "pairs_score.parquet", index=False, compression="zstd")
    ev = pd.concat(evals, ignore_index=True)
    ev.to_parquet(out / "pairs_eval.parquet", index=False, compression="zstd")
    pd.DataFrame({"left": [0], "right": [0], "label": [0], "domain": [0], "aug": [0]}).astype(np.int32).to_parquet(out / "pairs_train.parquet", index=False)
    idx = pd.concat(index, ignore_index=True)
    idx.to_parquet(out.parent / "local_pair_index_new.parquet", index=False)
    shutil.copyfile(Path(__file__).resolve().parent / "kaggle_ce.py", out / "kaggle_ce.py")
    man = {"records": rt.n, "score_pairs_by_set": idx["set"].value_counts().to_dict(), "id_offset": args.id_offset,
           "files_mb": {f.name: round(f.stat().st_size / 1e6, 1) for f in out.glob("*.parquet")}, "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2), flush=True)


if __name__ == "__main__":
    main()
