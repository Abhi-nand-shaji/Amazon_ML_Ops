#!/usr/bin/env python3
"""Candidate generation, step 1 of 2: retrieval + learned ranker -> shortlist, one parquet file per country.

Train: blocks a random sample of Source-1 entities against the COMPLETE Source-2/3 pool. The pool is
never subsampled, so candidate density and the "competitor" structure match test-time conditions;
only the number of Source-1 entities (hence the feature-computation cost) is reduced.
Test: blocks every Source-1 entity.

Per country the pool index is built once and S1 is queried in chunks (memory stays bounded, and a
killed run resumes from the first missing chunk).

Output directory: artifacts/cache/shortlist_<split>[_<tag>]/
    _meta.json             config + sample + ranker description (reproducibility / staleness check)
    s1_entities.parquet    every S1 entity that was blocked (incl. ones that got 0 candidates)
    country=<C>.parquet    the best `max_candidates` (30) pairs per S1 entity with the ranker probability `rank_score`

Step 2, run_candidate_cut.py, keeps the shortlisted pairs above a probability floor: that final, much smaller
candidate set (artifacts/cache/candidates_<split>/) is what the matcher scores and candidate_pairs.tsv lists.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import PoolIndex  # noqa: E402
from config import DEFAULT_BLOCKING, Paths, RANDOM_SEED  # noqa: E402

BLOCK_COLS = ["entity_id", "country", "source", "name_tokens", "address_tokens",
              "name_normalized", "name_compact", "name_core", "postal_code"]
EVIDENCE_COLS = ["n_name_shared", "n_addr_shared", "idf_name", "idf_addr", "exact_name", "exact_compact",
                 "postal_eq", "n_strategies", "block_score", "n_addr_bigram", "n_name_bigram", "n_prefix",
                 "ov_name", "ov_addr", "rank_score"]


def shortlist_dir(paths: Paths, split: str) -> Path:
    return paths.shortlist_dir(split)


def candidates_dir(paths: Paths, split: str) -> Path:
    """The final candidate set (run_candidate_cut.py output) that every downstream stage reads."""
    return paths.candidates_dir(split)


def read_partition(path: Path, country: str, columns=BLOCK_COLS) -> pa.Table:
    return pq.read_table(path, columns=columns, filters=[("country", "=", country)])


def sample_mask(n: int, n_sample: int | None, seed: int) -> np.ndarray:
    if n_sample is None or n_sample >= n:
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=n_sample, replace=False)] = True
    return mask


def add_blocking_args(ap: argparse.ArgumentParser) -> None:
    d = DEFAULT_BLOCKING
    ap.add_argument("--max-candidates", type=int, default=d.max_candidates, help="shortlist size per S1 entity")
    ap.add_argument("--df-cap", type=int, default=d.df_cap)
    ap.add_argument("--k-name", type=int, default=d.k_rare_name)
    ap.add_argument("--k-addr", type=int, default=d.k_rare_addr)
    ap.add_argument("--cap-addr-bigram", type=int, default=d.cap_addr_bigram)
    ap.add_argument("--cap-name-bigram", type=int, default=d.cap_name_bigram)
    ap.add_argument("--prefix-lens", default=",".join(map(str, d.prefix_lens)), help="comma list; empty disables")
    ap.add_argument("--cap-prefix", type=int, default=d.cap_prefix)
    ap.add_argument("--no-rescore", action="store_true", help="legacy: rank by the rare-token score only")
    ap.add_argument("--expansion-budget", type=int, default=d.expansion_budget,
                    help="max (S1, pool) expansions materialized per batch: bounds memory (lower it when using --workers)")


def config_from_args(args):
    return dataclasses.replace(
        DEFAULT_BLOCKING, max_candidates=args.max_candidates, df_cap=args.df_cap, k_rare_name=args.k_name,
        k_rare_addr=args.k_addr, cap_addr_bigram=args.cap_addr_bigram, cap_name_bigram=args.cap_name_bigram,
        prefix_lens=tuple(int(x) for x in args.prefix_lens.split(",") if x), cap_prefix=args.cap_prefix,
        rescore=not args.no_rescore, expansion_budget=args.expansion_budget)


class _Ranker:
    """LightGBM booster used as the per-entity cut ranker (`.predict` takes the RANKER_FEATURES matrix). Its prediction
    threads are bounded so that several S1 chunks can be ranked concurrently without oversubscribing the CPU."""

    def __init__(self, path: Path, threads: int):
        self.booster, self.threads = lgb.Booster(model_file=str(path)), threads

    def predict(self, matrix):
        return self.booster.predict(matrix, num_threads=self.threads)


def load_ranker(path: Path, workers: int = 1) -> _Ranker:
    import os
    return _Ranker(path, max(1, (os.cpu_count() or 4) // max(1, workers)))


def file_digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _write_part(path: Path, s1_ids: pa.Array, pool: pa.Table, res: dict, offset: int, country: str) -> tuple[int, int]:
    n = len(res["s1_row"])
    s1_row = res["s1_row"].astype(np.int64) + offset
    table = pa.table({
        "country": pa.array([country] * n, type=pa.large_string()),
        "s1_row": s1_row.astype(np.int32), "pool_row": res["pool_row"],
        "s1_entity_id": s1_ids.take(pa.array(s1_row)),
        "cand_entity_id": pool["entity_id"].combine_chunks().take(pa.array(res["pool_row"])),
        "cand_source": pool["source"].combine_chunks().take(pa.array(res["pool_row"])),
        **{k: res[k] for k in EVIDENCE_COLS},
    })
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return n, len(np.unique(res["s1_row"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="pipeline-variant suffix for every derived artifact")
    ap.add_argument("--n-s1", type=int, default=None, help="random sample of S1 entities to block (default: all)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--chunk-s1", type=int, default=200_000, help="S1 records per query chunk (bounds memory; resume unit)")
    ap.add_argument("--workers", type=int, default=1,
                    help="threads querying different S1 chunks of the same pool index concurrently (numpy / pyarrow / "
                         "LightGBM release the GIL); use with a smaller --chunk-s1 and --expansion-budget")
    ap.add_argument("--ranker", default=None, help="path of the ranker model (default: artifacts/ranker<tag>.txt)")
    ap.add_argument("--no-ranker", action="store_true", help="rank with the built-in heuristic instead of a learned ranker")
    ap.add_argument("--only-countries", default=None, help="comma list (debug)")
    ap.add_argument("--force", action="store_true", help="discard existing output for this split and recompute")
    add_blocking_args(ap)
    args = ap.parse_args()

    paths = Paths(args.root, args.tag)
    cfg = config_from_args(args)
    ranker_path = None if args.no_ranker else Path(args.ranker) if args.ranker else paths.ranker_path()
    if ranker_path is not None and not ranker_path.exists():
        sys.exit(f"ranker model not found: {ranker_path}\nTrain it first (train_ranker.py) or pass --no-ranker.")
    ranker = load_ranker(ranker_path, args.workers) if ranker_path else None
    out_dir = shortlist_dir(paths, args.split)
    meta = {"split": args.split, "n_s1_sample": args.n_s1, "seed": args.seed, "blocking_config": dataclasses.asdict(cfg),
            "ranker": file_digest(ranker_path) if ranker_path else None}

    if out_dir.exists():
        old = json.loads((out_dir / "_meta.json").read_text()) if (out_dir / "_meta.json").exists() else None
        if args.force:
            shutil.rmtree(out_dir)
        elif old != meta:
            sys.exit(f"{out_dir} was produced with a different configuration:\n  existing: {old}\n  requested: {meta}\n"
                     f"Re-run with --force to discard it (or use a different --tag).")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    s1_all = pq.read_table(paths.normalized_cache(args.split, "source1"), columns=BLOCK_COLS)
    mask = sample_mask(s1_all.num_rows, args.n_s1, args.seed)
    s1_all = s1_all.filter(pa.array(mask)) if not mask.all() else s1_all
    print(f"[{args.split}] S1 entities to block: {s1_all.num_rows:,} (sampled={not mask.all()})  ranker={meta['ranker']}  "
          f"config={meta['blocking_config']}", flush=True)

    ent_path = out_dir / "s1_entities.parquet"
    if not ent_path.exists():
        tmp = ent_path.with_name(ent_path.name + ".tmp")
        pq.write_table(s1_all.select(["entity_id", "country"]), tmp)
        tmp.replace(ent_path)

    countries = sorted(pc.unique(s1_all["country"]).to_pylist())
    if args.only_countries:
        countries = [c for c in countries if c in set(args.only_countries.split(","))]
    print(f"[{args.split}] country partitions: {countries}", flush=True)

    for country in countries:
        final = out_dir / f"country={country}.parquet"
        if final.exists():
            print(f"  country={country!r}: already done, skipping", flush=True)
            continue
        tc = time.time()
        s1_part = s1_all.filter(pc.equal(s1_all["country"], country))
        s1_ids = s1_part["entity_id"].combine_chunks()
        n1, chunk = s1_part.num_rows, args.chunk_s1
        n_chunks = (n1 + chunk - 1) // chunk
        parts_dir = out_dir / f"country={country}.parts"
        parts_dir.mkdir(exist_ok=True)
        todo = [i for i in range(n_chunks) if not (parts_dir / f"part-{i:05d}.parquet").exists()]
        print(f"  country={country!r}: s1={n1:,} in {n_chunks} chunk(s), {len(todo)} to do", flush=True)
        if todo:
            pool = pa.concat_tables([
                read_partition(paths.normalized_cache(args.split, "source2"), country),
                read_partition(paths.normalized_cache(args.split, "source3"), country),
            ])
            print(f"  country={country!r}: pool={pool.num_rows:,} (loaded in {time.time()-tc:.1f}s)", flush=True)
            index = PoolIndex(pool, cfg)
            pool = pool.select(["entity_id", "source"])      # only the ids are needed from here on: frees ~2 GB per country
            gc.collect()

            def work(i):
                tk = time.time()
                a = i * chunk
                res = index.block(s1_part.slice(a, chunk), ranker, verbose=False)
                n, n_with = _write_part(parts_dir / f"part-{i:05d}.parquet", s1_ids, pool, res, a, country)
                return i, a, n, n_with, time.time() - tk

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                for i, a, n, n_with, dt in ex.map(work, todo):          # results arrive in chunk order
                    print(f"    chunk {i+1}/{n_chunks}: {n:,} pairs, {n / max(1, min(chunk, n1 - a)):.1f} per S1 entity, "
                          f"{min(chunk, n1 - a) - n_with:,} with 0 candidates, {dt:.0f}s "
                          f"(total {time.time()-tc:.0f}s)", flush=True)
            del pool, index
        # merge the parts into the single per-country file
        parts = sorted(parts_dir.glob("part-*.parquet"))
        tmp = final.with_name(final.name + ".tmp")
        writer, total = None, 0
        for p in parts:
            t = pq.read_table(p)
            writer = writer or pq.ParquetWriter(tmp, t.schema)
            writer.write_table(t)
            total += t.num_rows
        if writer is not None:
            writer.close()
        else:                                    # no pairs at all: write an empty, correctly typed file
            pq.write_table(pa.table({"country": pa.array([], pa.large_string())}), tmp)
        tmp.replace(final)
        shutil.rmtree(parts_dir)
        print(f"  country={country!r}: {total:,} pairs | {total/max(1,n1):.1f} per S1 entity | {time.time()-tc:.0f}s", flush=True)
        del s1_part

    print(f"[{args.split}] blocking complete in {time.time()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
