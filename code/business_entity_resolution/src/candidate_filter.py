"""Candidate generation, step 2: a light learned filter over the blocker's shortlist.

The blocker (blocking.py / run_blocking.py) keeps the 30 retrieved pairs per S1 entity that its ranker likes best --
97% of all true pairs -- but that ranker only sees token-overlap evidence, so most of the 30 are obviously wrong.
This filter adds a handful of cheap character-level similarities (rapidfuzz, computed with exactly the functions the
matcher uses) and missing-field / house-number flags to the blocker's evidence, and predicts for every shortlisted
pair whether it can be a match. Pairs below a probability floor are dropped (run_candidate_cut.py); what remains is
the candidate set -- about 6 per S1 entity instead of 30 -- that the matcher scores and candidate_pairs.tsv lists.

Left to the matcher on purpose: pool / S1 ambiguity statistics, sparse token-matrix IDF statistics, scripts, and the
comparison of a candidate's string similarities with the entity's other candidates. Also left out: the blocker's raw
IDF sums / rare-token counts and the ranker probability, whose scale depends on the size of the country pool (the
unseen test country France has a 3-4x smaller pool and a visibly inflated ranker probability); dropping them cost
nothing on the validation entities.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from blocking import _list_lengths, strip_leading_zeros
from config import Paths
from features import _ADDR_SCORERS, _CORE_SCORERS, _NAME_SCORERS, _combine, _fuzzy, _np, _rank_desc_within, _take
from pipeline import chunk_slices, load_partition
from run_blocking import shortlist_dir

# Blocker evidence stored in the shortlist that the filter uses (overlap ratios, rule hits, flags: no pool-size scale).
EVIDENCE = ["ov_name", "ov_addr", "n_addr_bigram", "n_name_bigram", "n_prefix", "exact_name", "exact_compact",
            "postal_eq", "n_strategies"]
FILTER_FEATURES = EVIDENCE + [
    "s1_ntok_name", "s1_ntok_addr", "c_ntok_name", "c_ntok_addr",       # token counts of both records
    "block_rank", "is_s2",                                               # rank inside the entity's shortlist, source
    "name_token_set", "name_partial", "core_ratio", "core_jw", "addr_token_set", "addr_partial", "addr_ratio",
    "name_missing_2", "addr_missing_2", "housenum_both_present", "housenum_match",
]
SHORTLIST_COLS = ["s1_row", "pool_row", "s1_entity_id", "cand_entity_id", "rank_score"] + EVIDENCE
S1_COLS = ["entity_id", "name_normalized", "name_core", "address_normalized", "house_number", "name_tokens", "address_tokens"]
POOL_COLS = S1_COLS + ["source"]

_NAME = {k: _NAME_SCORERS[k] for k in ("token_set", "partial")}
_CORE = {k: _CORE_SCORERS[k] for k in ("ratio", "jw")}
_ADDR = {k: _ADDR_SCORERS[k] for k in ("token_set", "partial", "ratio")}

LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
              "feature_fraction": 0.9, "verbose": -1}
N_ROUNDS = 300


class FilterContext:
    """The per-record fields of one country partition the filter needs; rows are partition positions, exactly as
    the blocker addresses them."""

    def __init__(self, s1_tbl: pa.Table, pool_tbl: pa.Table):
        cols = ("name_normalized", "name_core", "address_normalized", "house_number")
        self.s1 = {c: _combine(s1_tbl[c]) for c in cols}
        self.pool = {c: _combine(pool_tbl[c]) for c in cols}
        for side in (self.s1, self.pool):              # house numbers: '019' == '19' (as in features.PartitionContext)
            side["house_number"] = strip_leading_zeros(side["house_number"])
        self.s1_id, self.pool_id = _combine(s1_tbl["entity_id"]), _combine(pool_tbl["entity_id"])
        self.pool_is_s2 = _np(pc.equal(_combine(pool_tbl["source"]), "S2"), np.int8)
        self.s1_ntok_name, self.s1_ntok_addr = _list_lengths(s1_tbl["name_tokens"]), _list_lengths(s1_tbl["address_tokens"])
        self.c_ntok_name, self.c_ntok_addr = _list_lengths(pool_tbl["name_tokens"]), _list_lengths(pool_tbl["address_tokens"])
        self.s1_name_len = _np(pc.utf8_length(self.s1["name_normalized"]), np.float32)
        self.pool_name_len = _np(pc.utf8_length(self.pool["name_normalized"]), np.float32)
        self.s1_addr_len = _np(pc.utf8_length(self.s1["address_normalized"]), np.float32)
        self.pool_addr_len = _np(pc.utf8_length(self.pool["address_normalized"]), np.float32)


def filter_features(ctx: FilterContext, s1_row: np.ndarray, pool_row: np.ndarray, short: dict) -> pd.DataFrame:
    """FILTER_FEATURES for shortlisted pairs given as partition row positions. `s1_row` must be sorted (an entity's
    pairs contiguous, as the blocker writes them); `short` maps EVIDENCE + ["rank_score"] to per-pair arrays."""
    n1, n2 = _take(ctx.s1["name_normalized"], s1_row), _take(ctx.pool["name_normalized"], pool_row)
    c1, c2 = _take(ctx.s1["name_core"], s1_row), _take(ctx.pool["name_core"], pool_row)
    a1, a2 = _take(ctx.s1["address_normalized"], s1_row), _take(ctx.pool["address_normalized"], pool_row)
    len_n1, len_n2 = ctx.s1_name_len[s1_row], ctx.pool_name_len[pool_row]
    len_a1, len_a2 = ctx.s1_addr_len[s1_row], ctx.pool_addr_len[pool_row]
    name_missing = (len_n1 == 0) | (len_n2 == 0)
    addr_missing = (len_a1 == 0) | (len_a2 == 0)
    core_missing = _np(pc.or_(pc.equal(pc.utf8_length(c1), 0), pc.equal(pc.utf8_length(c2), 0)), bool)

    f = {c: np.asarray(short[c]) for c in EVIDENCE}
    f["s1_ntok_name"], f["s1_ntok_addr"] = ctx.s1_ntok_name[s1_row], ctx.s1_ntok_addr[s1_row]
    f["c_ntok_name"], f["c_ntok_addr"] = ctx.c_ntok_name[pool_row], ctx.c_ntok_addr[pool_row]
    f["block_rank"] = _rank_desc_within(s1_row, np.asarray(short["rank_score"], dtype=np.float32))
    f["is_s2"] = ctx.pool_is_s2[pool_row]
    f.update(_fuzzy(n1.to_pylist(), n2.to_pylist(), "name", _NAME, name_missing))
    f.update(_fuzzy(c1.to_pylist(), c2.to_pylist(), "core", _CORE, core_missing))
    f.update(_fuzzy(a1.to_pylist(), a2.to_pylist(), "addr", _ADDR, addr_missing))
    f["name_missing_2"], f["addr_missing_2"] = (len_n2 == 0).astype(np.int8), (len_a2 == 0).astype(np.int8)
    h1, h2 = _take(ctx.s1["house_number"], s1_row), _take(ctx.pool["house_number"], pool_row)
    both_h = _np(pc.and_(pc.is_valid(h1), pc.is_valid(h2)), bool)
    f["housenum_both_present"] = both_h.astype(np.int8)
    f["housenum_match"] = (_np(pc.fill_null(pc.equal(h1, h2), False), bool) & both_h).astype(np.int8)
    out = pd.DataFrame(f)[FILTER_FEATURES]
    assert len(out) == len(s1_row)
    return out


def shortlist_meta(paths: Paths, split: str) -> dict:
    return json.loads((shortlist_dir(paths, split) / "_meta.json").read_text())


def shortlist_countries(paths: Paths, split: str) -> list[str]:
    return sorted(p.stem.split("=", 1)[1] for p in shortlist_dir(paths, split).glob("country=*.parquet"))


def country_filter_features(paths: Paths, split: str, country: str, short: pa.Table, predict=None,
                            chunk_pairs: int = 1_000_000, verbose: bool = True):
    """Filter features for every pair of one country's shortlist table (same row order). With `predict`
    (DataFrame -> probabilities) the features are scored chunk by chunk and only the probabilities are returned,
    so memory stays bounded on the test split."""
    t0 = time.time()
    s1_part, pool, _ = load_partition(paths, split, country, shortlist_meta(paths, split), S1_COLS, POOL_COLS, with_full=False)
    ctx = FilterContext(s1_part, pool)
    del s1_part, pool
    check_row_order(ctx, short, country)
    s1_row, pool_row = short["s1_row"].to_numpy(), short["pool_row"].to_numpy()
    if verbose:
        print(f"  [{country}] filter context ready: {short.num_rows:,} shortlisted pairs ({time.time() - t0:.0f}s)", flush=True)
    out = []
    for a, b in chunk_slices(s1_row, chunk_pairs):
        block = {c: short[c].slice(a, b - a).to_numpy() for c in EVIDENCE + ["rank_score"]}
        feats = filter_features(ctx, s1_row[a:b], pool_row[a:b], block)
        out.append(np.asarray(predict(feats), dtype=np.float32) if predict is not None else feats)
        if verbose and len(s1_row) > chunk_pairs:
            print(f"    [{country}] {b:,} / {len(s1_row):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    if predict is not None:
        return np.concatenate(out) if out else np.zeros(0, np.float32)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=FILTER_FEATURES)


def check_row_order(ctx: FilterContext, short: pa.Table, country: str) -> None:
    """The shortlist addresses records by partition position: verify on a sample of pairs that the rebuilt
    partition holds the same entity ids at those positions (fails loudly instead of corrupting every feature)."""
    n = short.num_rows
    if n == 0:
        return
    probe = np.linspace(0, n - 1, num=min(n, 5000)).astype(np.int64)
    s1_row, pool_row = pa.array(short["s1_row"].to_numpy()[probe]), pa.array(short["pool_row"].to_numpy()[probe])
    ok = (ctx.s1_id.take(s1_row).equals(short["s1_entity_id"].combine_chunks().take(pa.array(probe)))
          and ctx.pool_id.take(pool_row).equals(short["cand_entity_id"].combine_chunks().take(pa.array(probe))))
    if not ok:
        raise RuntimeError(f"[{country}] partition row order does not match the shortlist (cache or sampling changed?)")
