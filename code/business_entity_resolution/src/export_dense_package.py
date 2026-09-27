#!/usr/bin/env python3
"""Data package for the cloud dense-retrieval run (kaggle_dense.py): a fine-tuned bi-encoder finds candidates that the
token-based retrieval misses.

Why: 3.4% of the true pairs never become candidates (hold-out ceiling 0.988); they are pairs with typos in every
distinctive word, names in another script, truncated or empty addresses -- none shares a rare token with its S1 record.
A multilingual embedding model fine-tuned on known true pairs learns those equivalences. Only pool records of the same
country are searched (every true pair is within one country).

Writes <out>/
  records.parquet        rid, text          every record: all pool records of every (split, country) partition + the S1
                                            records that are queried or used for fine-tuning
  partitions.parquet     part, split, country, pool_lo, pool_hi, query_lo, query_hi   rid ranges per partition
  train_pairs.parquet    q, pos, neg        fine-tuning triples: S1 rid, a true pool record, a hard negative (a candidate
                                            of the same entity that is NOT a match; -1 when none)
  existing.parquet       q, p               current candidates of the queried entities (excluded from the output)
  eval_pairs.parquet     q, p               true pairs of validation / hold-out entities (recall measured on the cloud)
  kaggle_dense.py, manifest.json
and <out>/../dense_index.parquet (stays local): rid -> split, country, side, row (partition position), entity_id
Queried entities: every test S1 entity; training: the 50k main sample (candidate filter) + the stacker's `stack` entities.
Fine-tuning entities: a sample of the remaining training entities (never main sample, stack, ranker).
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ce_data import PartitionText  # noqa: E402
from config import Paths  # noqa: E402
from ids import load_truth  # noqa: E402
from pipeline import candidate_countries  # noqa: E402
from run_blocking import candidates_dir  # noqa: E402
from world_groups import load_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out", default=r"E:\Amazon\kaggle_upload\er-dense-data")
    ap.add_argument("--n-train-entities", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--smoke", action="store_true", help="tiny package for a local functional test")
    args = ap.parse_args()

    base, world = Paths(args.root), Paths(args.root, args.tag)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups = load_groups(base)
    query_train = groups["main_train"] | groups["val"] | groups["hold"] | groups["stack"]
    s1_all = pq.read_table(base.normalized_cache("train", "source1"), columns=["entity_id"])["entity_id"].to_pylist()
    banned = query_train | groups["ranker"]
    rest = np.array(sorted(set(s1_all) - banned), dtype=object)
    ft = set(np.random.default_rng(args.seed).choice(rest, size=min(args.n_train_entities, len(rest)), replace=False).tolist())
    if args.smoke:
        ft = set(list(ft)[:3000])
        query_train = set(list(groups["val"])[:1500]) | set(list(groups["stack"])[:1500])
    truth = load_truth(base.ground_truth(), ft | groups["val"] | groups["hold"])
    eval_ids = groups["val"] | groups["hold"]
    print(f"fine-tuning entities {len(ft):,}; queried training entities {len(query_train):,} ({time.time() - t0:.0f}s)", flush=True)

    writer = None
    rid = 0
    parts, index_parts, train_parts, exist_parts, eval_parts = [], [], [], [], []

    def add_records(texts: pa.Array, split, country, side, rows, ids):
        nonlocal writer, rid
        n = len(texts)
        r = np.arange(rid, rid + n, dtype=np.int64)
        t = pa.table({"rid": pa.array(r), "text": texts.cast(pa.string())})
        writer = writer or pq.ParquetWriter(out / "records.parquet", t.schema, compression="zstd")
        writer.write_table(t)
        index_parts.append(pd.DataFrame({"rid": r, "split": split, "country": country, "side": side,
                                         "row": np.asarray(rows, dtype=np.int32),
                                         "entity_id": ids if ids is not None else None}))
        rid += n
        return r

    for split, paths in (("train", world), ("test", base)):
        for country in candidate_countries(paths, split):
            tp = time.time()
            txt = PartitionText(paths, split, country)
            n_pool = len(txt.pool_txt)
            pool_rows = np.arange(n_pool) if not args.smoke else np.arange(min(n_pool, 200_000))
            pool_rid = add_records(txt.pool_txt.take(pa.array(pool_rows)), split, country, "pool", pool_rows, None)
            rid_of_pool = np.full(n_pool, -1, np.int64)
            rid_of_pool[pool_rows] = pool_rid
            s1_ids = np.asarray(txt.s1_id.to_pylist(), dtype=object)
            if split == "test":
                q_rows = np.arange(len(s1_ids)) if not args.smoke else np.arange(min(len(s1_ids), 20_000))
            else:
                q_rows = np.flatnonzero(pd.Series(s1_ids).isin(query_train).to_numpy())
            q_rid = add_records(txt.s1_txt.take(pa.array(q_rows)), split, country, "query", q_rows, s1_ids[q_rows].tolist())
            rid_of_s1 = pd.Series(q_rid, index=s1_ids[q_rows])
            parts.append({"part": f"{split}/{country}", "split": split, "country": country,
                          "pool_lo": int(pool_rid[0]), "pool_hi": int(pool_rid[-1]) + 1,
                          "query_lo": int(q_rid[0]) if len(q_rid) else -1, "query_hi": int(q_rid[-1]) + 1 if len(q_rid) else -1})
            # existing candidates of the queried entities (excluded from the dense output)
            cand = pq.read_table(candidates_dir(paths, split) / f"country={country}.parquet",
                                 columns=["s1_row", "pool_row", "s1_entity_id", "cand_entity_id"]).to_pandas()
            cq = cand[cand.s1_entity_id.isin(rid_of_s1.index)]
            e = pd.DataFrame({"q": rid_of_s1.reindex(cq.s1_entity_id).to_numpy(), "p": rid_of_pool[cq.pool_row.to_numpy()]})
            exist_parts.append(e[e.p >= 0])
            pid = pd.Index(txt.pool_id.to_pylist())
            if split == "train":
                # validation / hold-out true pairs (recall measured on the cloud)
                ev = [(s, c) for s in rid_of_s1.index if s in eval_ids for c in truth.get(s, ())]
                if ev:
                    ev = pd.DataFrame(ev, columns=["s", "c"])
                    prow = pid.get_indexer(ev.c)
                    ok = prow >= 0
                    evd = pd.DataFrame({"q": rid_of_s1.reindex(ev.s[ok]).to_numpy(), "p": rid_of_pool[prow[ok]]})
                    eval_parts.append(evd[evd.p >= 0])
                # fine-tuning triples of this partition
                f_rows = np.flatnonzero(pd.Series(s1_ids).isin(ft).to_numpy())
                f_rid = add_records(txt.s1_txt.take(pa.array(f_rows)), split, country, "finetune", f_rows, s1_ids[f_rows].tolist())
                f_map = pd.Series(f_rid, index=s1_ids[f_rows])
                pos = [(s, c) for s in f_map.index for c in truth.get(s, ())]
                pos = pd.DataFrame(pos, columns=["s", "c"])
                prow = pid.get_indexer(pos.c)
                pos = pos[prow >= 0].assign(prow=prow[prow >= 0])
                negc = cand[cand.s1_entity_id.isin(f_map.index)]
                is_true = [c in truth.get(s, ()) for s, c in zip(negc.s1_entity_id, negc.cand_entity_id)]
                negc = negc[~np.asarray(is_true, bool)].sample(frac=1.0, random_state=args.seed).drop_duplicates("s1_entity_id")
                neg_of = pd.Series(negc.pool_row.to_numpy(), index=negc.s1_entity_id)
                nrow = neg_of.reindex(pos.s).fillna(-1).astype(np.int64).to_numpy()
                train_parts.append(pd.DataFrame({"q": f_map.reindex(pos.s).to_numpy(), "pos": rid_of_pool[pos.prow.to_numpy()],
                                                 "neg": np.where(nrow >= 0, rid_of_pool[np.maximum(nrow, 0)], -1)}))
            print(f"  [{split}/{country}] pool {len(pool_rows):,}, queries {len(q_rows):,}, records so far {rid:,} "
                  f"({time.time() - tp:.0f}s)", flush=True)
            del txt, cand
    writer.close()
    pd.DataFrame(parts).to_parquet(out / "partitions.parquet", index=False)
    tr = pd.concat(train_parts, ignore_index=True)
    tr = tr[(tr.q >= 0) & (tr.pos >= 0)]
    tr.to_parquet(out / "train_pairs.parquet", index=False)
    pd.concat(exist_parts, ignore_index=True).to_parquet(out / "existing.parquet", index=False)
    evp = pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame(columns=["q", "p"])
    evp.to_parquet(out / "eval_pairs.parquet", index=False)
    pd.concat(index_parts, ignore_index=True).to_parquet(out.parent / ("dense_index_smoke.parquet" if args.smoke else "dense_index.parquet"), index=False)
    shutil.copyfile(Path(__file__).resolve().parent / "kaggle_dense.py", out / "kaggle_dense.py")
    man = {"records": rid, "train_triples": len(tr), "with_hard_negative": float((tr.neg >= 0).mean()),
           "existing_pairs": int(sum(len(x) for x in exist_parts)), "eval_pairs": len(evp),
           "partitions": pd.DataFrame(parts).to_dict("records"),
           "files_mb": {f.name: round(f.stat().st_size / 1e6, 1) for f in out.glob("*.parquet")},
           "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2), flush=True)
    print(f"package written to {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
