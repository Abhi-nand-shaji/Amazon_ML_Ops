#!/usr/bin/env python3
"""Build the data package for the cloud cross-encoder run (kaggle_ce.py): texts once, pairs as integer record ids.

Reads   candidates of the full training world (candidates_train_<tag>/, run_world_cut.py) and of the test split,
        artifacts/world_groups.json (world_groups.py)
Writes  <out>/records.parquet      rid, text            every record any pair refers to (serialized, lower-cased)
        <out>/pairs_train.parquet  left, right, label, domain, aug   fine-tuning pairs: main_train + ce_extra entities,
                                   plus street-swap hard negatives (aug=1, see below)
        <out>/pairs_eval.parquet   left, right, label, set          val / hold-out pairs (monitoring on the cloud side)
        <out>/pairs_score.parquet  pair_id, left, right, prio       everything to score: val, hold, stack, test
        <out>/kaggle_ce.py, manifest.json
        <out>/../local_pair_index.parquet  pair_id -> (set, country, ids, rows)   stays local: maps scores back

Street-swap hard negatives (targeted augmentation): on test, the first-stage matcher accepts French pairs with the same
name and house number but a DIFFERENT street (2% of its French matches; "55 R. Chanoine Larose" vs "55 Boulevard de la
Chauviniere"). In US/India training data that pattern is rare, so no model learns it. For a sample of US true pairs whose
candidate address has a "<number> <street>" component, the street is replaced by the street of another random US record
(no shared word) -- name and number kept -- and the pair is labelled a non-match (a different street is a different
place; true pairs in the training data never differ in the street name, only in its spelling).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ce_data import PartitionText  # noqa: E402
from config import Paths  # noqa: E402
from ids import label_pairs, load_truth  # noqa: E402
from pipeline import candidate_countries  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from world_groups import load_groups  # noqa: E402

PRIO = {"val": 0, "hold": 1, "stack": 2, "test": 3}
DOMAIN = {"India": 0, "US": 1}
ADDR_SEP = " address: "
STREET_RE = re.compile(r"^(?P<pre>(?:#\s*|no\.?\s*)?)(?P<num>\d+[a-z]?)(?P<sep>\s*-?\s+)(?P<street>[a-z][a-z .'\-]*[a-z.])$")


class RecordTable:
    def __init__(self):
        self.parts, self.n = [], 0

    def add(self, texts) -> np.ndarray:
        texts = texts if isinstance(texts, pa.Array) else pa.array(texts, pa.string())
        rids = np.arange(self.n, self.n + len(texts), dtype=np.int32)
        self.parts.append(pa.table({"rid": pa.array(rids), "text": texts.cast(pa.string())}))
        self.n += len(texts)
        return rids

    def write(self, path: Path):
        pq.write_table(pa.concat_tables(self.parts), path, compression="zstd")


def street_parts(text: str):
    name, sep, addr = text.partition(ADDR_SEP)
    if not sep:
        return None
    comps = addr.split(", ")
    for i, c in enumerate(comps):
        m = STREET_RE.match(c.strip())
        if m and len(m.group("street")) >= 4:
            return name, comps, i, m
    return None


def street_swaps(pool_texts: list[str], n_aug: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """(indices into pool_texts, new texts) for up to n_aug street-swapped variants."""
    parsed = [street_parts(t) for t in pool_texts]
    ok = np.array([p is not None for p in parsed])
    streets = [p[3].group("street") for p in parsed if p is not None]
    if not ok.any():
        return np.zeros(0, np.int64), []
    rng = np.random.default_rng(seed)
    cand = rng.permutation(np.flatnonzero(ok))[: n_aug * 2]
    idx, out = [], []
    for k in cand:
        name, comps, i, m = parsed[k]
        orig = set(m.group("street").split())
        for _ in range(10):
            s = streets[rng.integers(len(streets))]
            if not (set(s.split()) & orig):
                break
        else:
            continue
        comps = list(comps)
        comps[i] = f"{m.group('pre')}{m.group('num')}{m.group('sep')}{s}"
        idx.append(k)
        out.append(name + ADDR_SEP + ", ".join(comps))
        if len(idx) >= n_aug:
            break
    return np.array(idx, dtype=np.int64), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="full", help="full-world candidate run (train split)")
    ap.add_argument("--out", default=r"E:\Amazon\kaggle_upload\er-ce-data")
    ap.add_argument("--n-aug", type=int, default=40_000)
    ap.add_argument("--test-limit", type=int, default=None, help="smoke tests: only this many test pairs per country")
    ap.add_argument("--no-stack", action="store_true", help="smoke tests: no stack / ce_extra entities")
    args = ap.parse_args()

    base, world = Paths(args.root), Paths(args.root, args.tag)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups = load_groups(base)
    train_ids = groups["main_train"] | (set() if args.no_stack else groups["ce_extra"])
    wanted_train = {"train": train_ids, "val": groups["val"], "hold": groups["hold"]}
    if not args.no_stack:
        wanted_train["stack"] = groups["stack"]
    truth = load_truth(base.ground_truth(), train_ids | groups["val"] | groups["hold"])
    rt = RecordTable()
    frames = {k: [] for k in ("train", "eval", "score")}
    index_parts = []
    next_pair_id = 0

    def add_partition(paths: Paths, split: str, country: str, wanted: dict, limit: int | None = None):
        nonlocal next_pair_id
        cand = pq.read_table(candidates_dir(paths, split) / f"country={country}.parquet",
                             columns=["s1_row", "pool_row", "s1_entity_id", "cand_entity_id"])
        if limit is not None:
            cand = cand.slice(0, limit)
        ids = cand["s1_entity_id"].combine_chunks()
        masks = {g: (np.ones(cand.num_rows, bool) if s is None else
                     pc.is_in(ids, value_set=pa.array(sorted(s), pa.string())).to_numpy(zero_copy_only=False))
                 for g, s in wanted.items()}
        anym = np.zeros(cand.num_rows, bool)
        for m in masks.values():
            anym |= m
        if not anym.any():
            return
        tp = time.time()
        txt = PartitionText(paths, split, country)
        s1_row, pool_row = cand["s1_row"].to_numpy(), cand["pool_row"].to_numpy()
        txt.check(s1_row, pool_row, cand["s1_entity_id"], cand["cand_entity_id"])
        u1, u2 = np.unique(s1_row[anym]), np.unique(pool_row[anym])
        rid1 = np.full(len(txt.s1_txt), -1, np.int32)
        rid2 = np.full(len(txt.pool_txt), -1, np.int32)
        rid1[u1] = rt.add(txt.s1_txt.take(pa.array(u1)))
        rid2[u2] = rt.add(txt.pool_txt.take(pa.array(u2)))
        for g, m in masks.items():
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            left, right = rid1[s1_row[idx]], rid2[pool_row[idx]]
            s1i, ci = cand["s1_entity_id"].take(pa.array(idx)), cand["cand_entity_id"].take(pa.array(idx))
            lab = label_pairs(s1i, ci, truth) if split == "train" else np.zeros(len(idx), np.int8)
            if g == "train":
                frames["train"].append(pd.DataFrame({"left": left, "right": right, "label": lab,
                                                     "domain": np.int8(DOMAIN.get(country, 2)), "aug": np.int8(0)}))
                if country == "US" and args.n_aug > 0:
                    pos = idx[lab == 1]
                    pos_texts = txt.pool_txt.take(pa.array(pool_row[pos])).to_pylist()
                    k, new = street_swaps(pos_texts, args.n_aug, seed=17)
                    if len(k):
                        new_rid = rt.add(new)
                        frames["train"].append(pd.DataFrame({"left": rid1[s1_row[pos[k]]], "right": new_rid,
                                                             "label": np.int8(0), "domain": np.int8(1), "aug": np.int8(1)}))
                        print(f"    [{country}] street-swap negatives: {len(k):,}; e.g.\n      {pos_texts[k[0]]}\n   -> {new[0]}", flush=True)
                continue
            if g in ("val", "hold"):
                frames["eval"].append(pd.DataFrame({"left": left, "right": right, "label": lab, "set": g}))
            pid = np.arange(next_pair_id, next_pair_id + len(idx), dtype=np.int64)
            next_pair_id += len(idx)
            frames["score"].append(pd.DataFrame({"pair_id": pid, "left": left, "right": right, "prio": np.int8(PRIO[g])}))
            index_parts.append(pd.DataFrame({"pair_id": pid, "set": g, "split": split, "country": country,
                                             "s1_entity_id": s1i.to_pylist(), "cand_entity_id": ci.to_pylist(),
                                             "s1_row": s1_row[idx].astype(np.int32), "pool_row": pool_row[idx].astype(np.int32),
                                             "label": lab}))
        print(f"  [{split}/{country}] " + ", ".join(f"{g} {int(m.sum()):,}" for g, m in masks.items()) +
              f"; records so far {rt.n:,} ({time.time() - tp:.0f}s)", flush=True)
        del txt, cand

    src_train = world if not args.no_stack else base
    for country in candidate_countries(src_train, "train"):
        add_partition(src_train, "train", country, wanted_train)
    for country in candidate_countries(base, "test"):
        add_partition(base, "test", country, {"test": None}, args.test_limit)

    rt.write(out / "records.parquet")
    tr = pd.concat(frames["train"], ignore_index=True)
    tr.to_parquet(out / "pairs_train.parquet", index=False, compression="zstd")
    pd.concat(frames["eval"], ignore_index=True).to_parquet(out / "pairs_eval.parquet", index=False, compression="zstd")
    sc = pd.concat(frames["score"], ignore_index=True)
    sc.to_parquet(out / "pairs_score.parquet", index=False, compression="zstd")
    idx = pd.concat(index_parts, ignore_index=True)
    idx.to_parquet(out.parent / "local_pair_index.parquet", index=False)
    shutil.copyfile(Path(__file__).resolve().parent / "kaggle_ce.py", out / "kaggle_ce.py")
    manifest = {"records": rt.n, "train_pairs": len(tr), "train_positive_rate": float(tr["label"].mean()),
                "street_swap_negatives": int(tr["aug"].sum()),
                "score_pairs_by_set": idx["set"].value_counts().to_dict(),
                "files_mb": {f.name: round(f.stat().st_size / 1e6, 1) for f in out.glob("*.parquet")},
                "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"package written to {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
