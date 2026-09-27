#!/usr/bin/env python3
"""Candidate cut for the FULL-WORLD training run (every training Source-1 entity blocked, like the test split).

Reads   artifacts/cache/shortlist_train_<tag>/        run_blocking.py --split train --tag <tag> (no --n-s1: all entities)
        artifacts/cache/shortlist_train/ + filter_oof_train.parquet   the main 50k sample and its out-of-fold filter scores
        artifacts/candidate_filter.{txt,json}         the filter fitted on the main sample
Writes  artifacts/cache/candidates_train_<tag>/       same layout as run_candidate_cut.py output

Why a separate cut: the filter was fitted on the 50k main-sample entities, so its scores are in-sample for them. Those
entities (in particular the validation / hold-out ones every decision is measured on) keep their OUT-OF-FOLD scores --
their candidate sets are then identical to candidates_train/, which is verified -- and every other entity is scored with
the fitted filter exactly like the test split. The full world is what graph-level (collective) features need: every
Source-1 entity that could claim a Source-2/3 record is present, as at test time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import select_top  # noqa: E402
from candidate_filter import FILTER_FEATURES, country_filter_features  # noqa: E402
from config import Paths  # noqa: E402
from ids import id_to_int  # noqa: E402
from run_blocking import candidates_dir, file_digest, shortlist_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-pairs", type=int, default=1_000_000)
    args = ap.parse_args()

    base, world = Paths(args.root), Paths(args.root, args.tag)
    src, dst = shortlist_dir(world, "train"), candidates_dir(world, "train")
    fcfg = json.loads(base.filter_config_path().read_text())
    if fcfg["features"] != FILTER_FEATURES:
        sys.exit("candidate_filter.json was trained on a different feature list")
    floor = fcfg["min_score"]
    booster = lgb.Booster(model_file=str(base.filter_model_path()))
    meta = {**json.loads((src / "_meta.json").read_text()),
            "candidate_filter": {"model": file_digest(base.filter_model_path()), "min_score": floor,
                                 "max_candidates": args.max_candidates,
                                 "scores": "out-of-fold for the 50k main-sample entities, fitted filter for all others"}}
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copyfile(src / "s1_entities.parquet", dst / "s1_entities.parquet")

    t0 = time.time()
    for country in sorted(p.stem.split("=", 1)[1] for p in src.glob("country=*.parquet")):
        final = dst / f"country={country}.parquet"
        if final.exists():
            print(f"[{country}] already cut, skipping", flush=True)
            continue
        short = pq.read_table(src / f"country={country}.parquet")
        s1_row, pool_row = short["s1_row"].to_numpy(), short["pool_row"].to_numpy()
        score = country_filter_features(world, "train", country, short, chunk_pairs=args.chunk_pairs,
                                        predict=lambda X: booster.predict(X, num_threads=args.threads))
        # main-sample entities: out-of-fold scores
        sm = pq.read_table(shortlist_dir(base, "train") / f"country={country}.parquet", columns=["s1_entity_id", "cand_entity_id"])
        oof = pq.read_table(base.filter_oof_path())
        oof = oof.filter(pc.equal(oof["country"], country))
        if oof.num_rows != sm.num_rows:
            sys.exit(f"[{country}] out-of-fold scores are not aligned with the main-sample shortlist")
        main_scores = pd.DataFrame({"s1": id_to_int(sm["s1_entity_id"]), "cand": id_to_int(sm["cand_entity_id"]),
                                    "oof": oof["filter_score"].to_numpy()})
        in_main = pc.is_in(short["s1_entity_id"], value_set=pa.array(sorted(set(sm["s1_entity_id"].to_pylist())))).to_numpy(zero_copy_only=False)
        idx_main = np.flatnonzero(in_main)
        world_main = pd.DataFrame({"s1": id_to_int(short["s1_entity_id"].take(pa.array(idx_main))),
                                   "cand": id_to_int(short["cand_entity_id"].take(pa.array(idx_main)))})
        got = world_main.merge(main_scores, on=["s1", "cand"], how="left")["oof"].to_numpy()
        missing = int(np.isnan(got).sum())
        print(f"  [{country}] main-sample pairs in the world shortlist: {int(in_main.sum()):,} (main shortlist {sm.num_rows:,}); "
              f"without an out-of-fold score: {missing:,}", flush=True)
        if in_main.sum() != sm.num_rows or missing:
            sys.exit(f"[{country}] the world shortlist differs from the main-sample shortlist for main-sample entities")
        score = score.copy()
        score[in_main] = got.astype(np.float32)
        sel = select_top(s1_row, pool_row, score, args.max_candidates, floor)
        out = short.take(pa.array(sel)).append_column("filter_score", pa.array(score[sel].astype(np.float32)))
        tmp = final.with_name(final.name + ".tmp")
        pq.write_table(out, tmp)
        tmp.replace(final)
        n_s1 = len(np.unique(s1_row))
        print(f"  [{country}] {short.num_rows:,} shortlisted -> {len(sel):,} candidates ({len(sel) / max(1, n_s1):.2f} per entity) "
              f"({time.time() - t0:.0f}s)", flush=True)
        # consistency: main-sample candidate sets must equal candidates_train/
        cm = pq.read_table(candidates_dir(base, "train") / f"country={country}.parquet", columns=["s1_entity_id", "cand_entity_id"])
        a = set(zip(cm["s1_entity_id"].to_pylist(), cm["cand_entity_id"].to_pylist()))
        o_main = out.filter(pc.is_in(out["s1_entity_id"], value_set=pa.array(sorted(set(sm["s1_entity_id"].to_pylist())))))
        b = set(zip(o_main["s1_entity_id"].to_pylist(), o_main["cand_entity_id"].to_pylist()))
        print(f"  [{country}] main-sample candidate sets identical to candidates_train: {a == b} ({len(a):,} vs {len(b):,})", flush=True)
        if a != b:
            sys.exit(f"[{country}] candidate sets of the main sample differ")
        del short, out, score
    print(f"world candidate cut complete in {time.time() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
