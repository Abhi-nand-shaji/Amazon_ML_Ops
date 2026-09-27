#!/usr/bin/env python3
"""First-stage scores over the FULL training world, plus persisted features of the stacker / evaluation entities.

Reads   artifacts/cache/candidates_train_<tag>/   run_world_cut.py (every training S1 entity, cut like the test split)
        artifacts/lgbm_model.txt, feature_columns.json   the first-stage matcher (fitted on main_train entities only)
        artifacts/world_groups.json               world_groups.py
Writes  artifacts/cache/world_p_train_<tag>/country=<C>.parquet      s1_row, pool_row, p1 for EVERY world pair: the
                                                                     competition for each pool record (graph features)
        artifacts/cache/world_feats_train_<tag>/country=<C>.parquet  all pairwise features + ids + label + p1 for the
                                                                     pairs of the selected groups (default stack, val, hold)
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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from pipeline import candidate_countries, iter_feature_chunks  # noqa: E402
from world_groups import load_groups  # noqa: E402


def world_p_dir(paths: Paths) -> Path:
    return paths.cache_dir / f"world_p_train{paths.suffix}"


def world_feats_dir(paths: Paths) -> Path:
    return paths.cache_dir / f"world_feats_train{paths.suffix}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--groups", default="stack,val,hold")
    ap.add_argument("--chunk-pairs", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--countries", default=None)
    args = ap.parse_args()

    base, world = Paths(args.root), Paths(args.root, args.tag)
    booster = lgb.Booster(model_file=str(base.model_path()))
    fcols = json.loads(base.feature_list_path().read_text())["feature_columns"]
    groups = load_groups(base)
    keep = set().union(*(groups[g] for g in args.groups.split(",")))
    keep_arr = pa.array(sorted(keep), pa.string())
    truth = load_truth(base.ground_truth(), keep)
    print(f"selected entities ({args.groups}): {len(keep):,}", flush=True)
    dp, df_ = world_p_dir(world), world_feats_dir(world)
    dp.mkdir(parents=True, exist_ok=True)
    df_.mkdir(parents=True, exist_ok=True)
    countries = args.countries.split(",") if args.countries else candidate_countries(world, "train")
    for country in countries:
        fp, ff = dp / f"country={country}.parquet", df_ / f"country={country}.parquet"
        if fp.exists() and ff.exists():
            print(f"[{country}] already scored, skipping", flush=True)
            continue
        tp, tf = fp.with_name(fp.name + ".tmp"), ff.with_name(ff.name + ".tmp")
        wp = wf = None
        n = n_sel = 0
        t0 = time.time()
        for ch in iter_feature_chunks(world, "train", args.chunk_pairs, [country]):
            p = booster.predict(ch.feats[fcols], num_threads=args.threads).astype(np.float32)
            tbl = pa.table({"s1_row": ch.s1_row.astype(np.int32), "pool_row": ch.pool_row.astype(np.int32), "p1": p})
            wp = wp or pq.ParquetWriter(tp, tbl.schema)
            wp.write_table(tbl)
            sel = pc.is_in(ch.s1_entity_id, value_set=keep_arr).to_numpy(zero_copy_only=False)
            if sel.any():
                idx = np.flatnonzero(sel)
                d = ch.feats.iloc[idx].reset_index(drop=True)
                s1_ids, c_ids = ch.s1_entity_id.take(pa.array(idx)), ch.cand_entity_id.take(pa.array(idx))
                d["s1_row"], d["pool_row"] = ch.s1_row[idx].astype(np.int32), ch.pool_row[idx].astype(np.int32)
                d["s1_entity_id"], d["cand_entity_id"] = s1_ids.to_pylist(), c_ids.to_pylist()
                d["label"] = label_pairs(s1_ids, c_ids, truth)
                d["p1"] = p[idx]
                t = pa.Table.from_pandas(d, preserve_index=False)
                wf = wf or pq.ParquetWriter(tf, t.schema)
                wf.write_table(t)
                n_sel += len(idx)
            n += len(p)
            print(f"    [{country}] {n:,} pairs scored, {n_sel:,} selected ({time.time() - t0:.0f}s)", flush=True)
        for w in (wp, wf):
            if w is not None:
                w.close()
        tp.replace(fp)
        tf.replace(ff)
        print(f"[{country}] done: {n:,} pairs, {n_sel:,} selected ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
