#!/usr/bin/env python3
"""New candidate pairs (v4): dense retrieval (kaggle_dense.py) + second-hop retrieval (run_second_hop.py), filtered.

Both sources exclude an entity's existing (first-hop) candidates. A light LightGBM filter decides which new pairs join
the candidate set; it sees each source's own evidence (bi-encoder cosine and rank, anchor->candidate ranker score, anchor
confidence, number of anchors) plus cheap string similarities of the S1 and candidate records. It is fitted on the
`stack` entities' new pairs (labels from the ground truth) and its probability floor is chosen on the `val` entities:
the largest floor that still keeps --keep-share of the new pairs that are true matches (hold-out plays no part).

Writes artifacts/cache/newpairs_<split>[_tag]/country=<C>.parquet  s1_row, pool_row, src_dense, src_hop, dense_score,
       dense_rank, hop_rank, hop_conf, hop_n, nf_* (filter features), new_score (+ label for training), kept pairs only
       artifacts/new_pair_filter.txt / .json
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
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths, RANDOM_SEED  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from pipeline import load_meta, load_partition  # noqa: E402
from world_groups import load_groups  # noqa: E402

SRC_COLS = ["src_dense", "src_hop", "dense_score", "dense_rank", "hop_rank", "hop_conf", "hop_n"]
NF_COLS = ["nf_name_set", "nf_name_ratio", "nf_addr_set", "nf_addr_ratio", "nf_addr_empty", "nf_name_nonlatin"]


def load_dense(split: str, index: pd.DataFrame, dense: pd.DataFrame) -> dict:
    q = index.set_index("rid").loc[dense.q.to_numpy(), ["split", "country", "row"]].reset_index(drop=True)
    p = index.set_index("rid").loc[dense.p.to_numpy(), ["row"]].reset_index(drop=True)
    d = pd.DataFrame({"split": q.split.to_numpy(), "country": q.country.to_numpy(), "s1_row": q.row.to_numpy(),
                      "pool_row": p.row.to_numpy(), "dense_score": dense.score.to_numpy(), "dense_rank": dense.new_rank.to_numpy()})
    d = d[d.split == split]
    return {c: g.drop(columns=["split", "country"]) for c, g in d.groupby("country")}


def load_text(paths: Paths, split: str, country: str) -> tuple[dict, dict]:
    s1, pool, _ = load_partition(paths, split, country, load_meta(paths, split),
                                 s1_cols=["entity_id", "name_normalized", "address_normalized"],
                                 pool_cols=["entity_id", "name_normalized", "address_normalized", "name_script"], with_full=False)
    return ({c: s1[c].combine_chunks() for c in s1.column_names},
            {c: pool[c].combine_chunks() for c in pool.column_names})


def filter_features(text: tuple[dict, dict], d: pd.DataFrame) -> pd.DataFrame:
    s1, pool = text
    import pyarrow as pa
    r1, r2 = pa.array(d.s1_row.to_numpy()), pa.array(d.pool_row.to_numpy())
    n1 = s1["name_normalized"].take(r1).to_pylist()
    n2 = pool["name_normalized"].take(r2).to_pylist()
    a1 = s1["address_normalized"].take(r1).to_pylist()
    a2 = pool["address_normalized"].take(r2).to_pylist()
    d = d.copy()
    d["nf_name_set"] = process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    d["nf_name_ratio"] = process.cpdist(n1, n2, scorer=fuzz.ratio, workers=-1) / 100.0
    d["nf_addr_set"] = process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    d["nf_addr_ratio"] = process.cpdist(a1, a2, scorer=fuzz.ratio, workers=-1) / 100.0
    d["nf_addr_empty"] = np.array([not x for x in a2], dtype=np.int8)
    d["nf_name_nonlatin"] = ~pool["name_script"].take(r2).to_pandas().isin(["latin", "latin_accent"]).to_numpy()
    d["nf_name_nonlatin"] = d["nf_name_nonlatin"].astype(np.int8)
    d["s1_entity_id"] = s1["entity_id"].take(r1).to_pylist()
    d["cand_entity_id"] = pool["entity_id"].take(r2).to_pylist()
    return d


def union_sources(dense: pd.DataFrame | None, hop: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    if dense is not None and len(dense):
        parts.append(dense.assign(src_dense=1))
    if hop is not None and len(hop):
        parts.append(hop.assign(src_hop=1))
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["s1_row", "pool_row"])
    agg = {"src_dense": "max", "src_hop": "max", "dense_score": "max", "dense_rank": "min", "hop_rank": "max",
           "hop_conf": "max", "hop_n": "max"}
    for c in agg:
        if c not in d:
            d[c] = np.nan
    d = d.groupby(["s1_row", "pool_row"], as_index=False).agg(agg)
    d[["src_dense", "src_hop"]] = d[["src_dense", "src_hop"]].fillna(0).astype(np.int8)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", default=r"E:\Amazon\kaggle_out\dense\dense_new.parquet")
    ap.add_argument("--dense-index", default=r"E:\Amazon\kaggle_upload\dense_index.parquet")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--keep-share", type=float, default=0.9, help="share of the new true pairs (val) the floor must keep")
    ap.add_argument("--max-new-per-entity", type=float, default=3.0, help="budget: new pairs per entity (val) at the chosen floor")
    ap.add_argument("--no-dense", action="store_true", help="second-hop only (dense results not available)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-entities", type=int, default=150_000, help="test S1 entities per processing chunk")
    ap.add_argument("--skip-test", action="store_true", help="fit the filter and write the training new pairs only")
    args = ap.parse_args()
    base, world = Paths(), Paths(tag=args.tag)
    t0 = time.time()
    groups = load_groups(base)
    dense_tr = dense_te = {}
    if not args.no_dense:
        index = pd.read_parquet(args.dense_index, columns=["rid", "split", "country", "row"])
        dense = pd.read_parquet(args.dense)
        dense_tr, dense_te = load_dense("train", index, dense), load_dense("test", index, dense)
        del index, dense
    # ---- training entities: stack (fit) / val (floor) / hold (report)
    truth = load_truth(base.ground_truth(), groups["stack"] | groups["val"] | groups["hold"])
    frames = []
    for country in ("India", "US"):
        hop_f = base.cache_dir / f"hop_train{world.suffix}" / f"country={country}.parquet"
        hop = pq.read_table(hop_f).to_pandas() if hop_f.exists() else None
        d = union_sources(dense_tr.get(country), hop)
        d = filter_features(load_text(world, "train", country), d)
        keep = d.s1_entity_id.isin(groups["stack"] | groups["val"] | groups["hold"])
        d = d[keep].reset_index(drop=True)
        d["label"] = label_pairs(d.s1_entity_id, d.cand_entity_id, truth)
        d["country"] = country
        frames.append(d)
        print(f"  [train/{country}] {len(d):,} new pairs, {int(d.label.sum()):,} true ({time.time() - t0:.0f}s)", flush=True)
    tr = pd.concat(frames, ignore_index=True)
    grp = np.where(tr.s1_entity_id.isin(groups["stack"]), "stack", np.where(tr.s1_entity_id.isin(groups["val"]), "val", "hold"))
    cols = SRC_COLS + NF_COLS
    fit = grp == "stack"
    booster = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
                         "feature_fraction": 0.9, "num_threads": args.threads, "seed": RANDOM_SEED, "verbose": -1},
                        lgb.Dataset(tr.loc[fit, cols], label=tr.loc[fit, "label"]), num_boost_round=400)
    tr["new_score"] = booster.predict(tr[cols], num_threads=args.threads)
    va = tr[grp == "val"]
    n_val_ent = len(groups["val"])
    table = []
    for t in (0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3):
        k = va.new_score >= t
        table.append({"floor": t, "kept_true_share": float(va.label[k].sum() / max(1, va.label.sum())),
                      "new_per_entity": float(k.sum() / n_val_ent), "precision": float(va.label[k].mean()) if k.any() else 0.0})
    # largest floor that keeps --keep-share of the new true pairs, within the budget of new pairs per entity
    within = [r for r in table if r["new_per_entity"] <= args.max_new_per_entity] or [max(table, key=lambda r: r["floor"])]
    ok = [r for r in within if r["kept_true_share"] >= args.keep_share]
    floor = max(r["floor"] for r in ok) if ok else min(r["floor"] for r in within)
    print("  floor trade-off on val (new pairs):\n" + pd.DataFrame(table).round(4).to_string(index=False), flush=True)
    print(f"  chosen floor {floor}", flush=True)
    booster.save_model(str(base.artifacts_dir / "new_pair_filter.txt"))
    n_true_val = sum(len(truth[e]) for e in groups["val"])
    (base.artifacts_dir / "new_pair_filter.json").write_text(json.dumps(
        {"features": cols, "floor": floor, "keep_share": args.keep_share, "max_new_per_entity": args.max_new_per_entity,
         "tradeoff_val": table,
         "val_new_true_pairs": int(va.label.sum()), "val_true_pairs_total": n_true_val,
         "val_recall_gain_at_floor": float(va.label[va.new_score >= floor].sum() / n_true_val)}, indent=2))
    out_tr = base.cache_dir / f"newpairs_train{world.suffix}"
    out_tr.mkdir(parents=True, exist_ok=True)
    for country, g in tr[tr.new_score >= floor].groupby("country"):
        g.drop(columns=["country"]).to_parquet(out_tr / f"country={country}.parquet", index=False)
    print(f"  training new pairs kept: {int((tr.new_score >= floor).sum()):,} (val recall gain "
          f"{va.label[va.new_score >= floor].sum() / n_true_val:.2%} of all val true pairs)", flush=True)
    if args.skip_test:
        print(f"done, test skipped ({(time.time() - t0) / 60:.1f} min)", flush=True)
        return
    # ---- test
    out_te = base.cache_dir / "newpairs_test"
    out_te.mkdir(parents=True, exist_ok=True)
    for country in ("France", "India", "US"):
        hop_f = base.cache_dir / "hop_test" / f"country={country}.parquet"
        hop = pq.read_table(hop_f).to_pandas() if hop_f.exists() else None
        dn = dense_te.get(country)
        text = load_text(base, "test", country)
        n_s1 = len(text[0]["entity_id"])
        kept, n_all = [], 0
        for lo in range(0, n_s1, args.chunk_entities):
            hi = lo + args.chunk_entities
            dc = dn[(dn.s1_row >= lo) & (dn.s1_row < hi)] if dn is not None else None
            hc = hop[(hop.s1_row >= lo) & (hop.s1_row < hi)] if hop is not None else None
            d = union_sources(dc, hc)
            if not len(d):
                continue
            n_all += len(d)
            d = filter_features(text, d)            # ids stay: the v4 candidate files carry them
            d["new_score"] = booster.predict(d[cols], num_threads=args.threads)
            kept.append(d[d.new_score >= floor])
        if not kept:
            print(f"  [test/{country}] no new pairs", flush=True)
            continue
        d = pd.concat(kept, ignore_index=True).sort_values(["s1_row", "pool_row"]).reset_index(drop=True)
        d.to_parquet(out_te / f"country={country}.parquet", index=False)
        print(f"  [test/{country}] kept {len(d):,} of {n_all:,} new pairs ({len(d) / max(1, n_s1):.2f} per entity, "
              f"{n_s1:,} entities) ({time.time() - t0:.0f}s)", flush=True)
        del text, kept, dn, hop
    print(f"done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
