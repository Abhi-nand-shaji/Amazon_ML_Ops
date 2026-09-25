"""Pairwise feature engineering for candidate (S1, pool) pairs -- vectorized.

Scale drives the design (up to ~10^8 candidate pairs for the test split):

* Nothing is computed with per-pair Python loops. String similarities run
  through ``rapidfuzz.process.cpdist`` (multithreaded C++ over aligned string
  arrays); token-level overlap features come from int-coded sparse token
  matrices (one row per record, one column per vocabulary token), so the
  intersection / IDF statistics of a whole chunk of pairs are a couple of
  sparse element-wise products and mat-vec products.
* Pairs are addressed by row position inside one *country partition*
  (``s1_row`` into the S1 table, ``pool_row`` into the S2+S3 pool table); this
  is exactly what the blocker emits. Only the handful of Arrow columns the
  features need are held in memory (``PartitionContext``).
* ``country`` is deliberately NOT a model feature: the test set contains a
  country that never appears in training (France), and a category the model
  has never seen would fall into an arbitrary learned branch. Partitioning by
  country happens upstream (blocking); the features themselves are
  country-agnostic.
* When either side of a field is missing, similarity features are NaN
  (LightGBM's native missing-value handling) rather than 0 or 1 -- "no
  information" must not look like "different" (or, for two blanks, "identical").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from scipy import sparse

from blocking import canon_flat, strip_leading_zeros

# Columns a partition needs (from the normalized parquet cache) to compute features.
S1_COLS = ["entity_id", "name_normalized", "name_core", "name_script", "name_tokens",
           "address_normalized", "address_tokens", "postal_code", "house_number"]
POOL_COLS = S1_COLS + ["source"]

# Fixed vocabulary so category codes mean the same thing in every run and split.
SCRIPT_VOCAB = ["latin", "latin_accent", "devanagari", "tamil", "telugu", "kannada", "gujarati",
                "bengali", "malayalam", "gurmukhi", "oriya", "other"]
_SCRIPT_CODE = {s: float(i) for i, s in enumerate(SCRIPT_VOCAB)}


def _combine(col):
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


def _np(arr, dtype=None):
    out = arr.to_numpy(zero_copy_only=False)
    return out if dtype is None else out.astype(dtype, copy=False)


def _dedup_rc(row: np.ndarray, code: np.ndarray, n_codes: int):
    """Unique (row, code) pairs (a token repeated inside one record counts once)."""
    key = np.unique(row.astype(np.int64) * n_codes + code)
    return (key // n_codes).astype(np.int32), (key % n_codes).astype(np.int32)


def _group_bounds(sorted_ids: np.ndarray) -> np.ndarray:
    """Start index of each run of equal values (input sorted by group), plus the end sentinel."""
    if len(sorted_ids) == 0:
        return np.zeros(1, dtype=np.int64)
    first = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    return np.r_[first, len(sorted_ids)]


def _rank_desc_within(sorted_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """1-based descending rank of `values` inside each run of equal `sorted_ids`
    (ties broken by position). NaN sorts last."""
    v = np.where(np.isnan(values), -np.inf, values)
    order = np.lexsort((np.arange(len(v)), -v, sorted_ids))
    b = _group_bounds(sorted_ids[order])
    counts = np.diff(b)
    rank_sorted = np.arange(len(v)) - np.repeat(b[:-1], counts) + 1
    out = np.empty(len(v), dtype=np.int32)
    out[order] = rank_sorted
    return out


def _group_max(sorted_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-row maximum of `values` within its run of equal `sorted_ids` (input sorted by group)."""
    if len(values) == 0:
        return values.copy()
    b = _group_bounds(sorted_ids)
    v = np.where(np.isnan(values), -np.inf, values)
    mx = np.maximum.reduceat(v, b[:-1])
    out = np.repeat(mx, np.diff(b))
    return np.where(np.isinf(out), np.nan, out).astype(values.dtype)


# ---------------------------------------------------------------------------
# Token statistics
# ---------------------------------------------------------------------------
class TokenStats:
    """Sparse token matrices for one list<string> column over the S1 rows and the
    pool rows of a partition, sharing a single vocabulary.

    IDF is document frequency over the *pool* (the side candidates are drawn
    from); a token that never occurs in the pool gets the maximum IDF.
    """

    def __init__(self, s1_list, pool_list):
        a1, a2 = _combine(s1_list), _combine(pool_list)
        n1, n2 = len(a1), len(a2)
        f1, f2 = canon_flat(pc.list_flatten(a1)), canon_flat(pc.list_flatten(a2))   # '019' == '19'
        p1, p2 = _np(pc.list_parent_indices(a1)), _np(pc.list_parent_indices(a2))
        enc = pc.dictionary_encode(pa.concat_arrays([f1, f2]))
        codes = _np(enc.indices)
        vocab = enc.dictionary
        n_v = max(len(vocab), 1)
        c1, c2 = codes[: len(f1)], codes[len(f1):]
        r1, k1 = _dedup_rc(p1, c1, n_v)
        r2, k2 = _dedup_rc(p2, c2, n_v)
        self.A1 = sparse.csr_matrix((np.ones(len(r1), np.float32), (r1, k1)), shape=(n1, n_v))
        self.A2 = sparse.csr_matrix((np.ones(len(r2), np.float32), (r2, k2)), shape=(n2, n_v))
        df = np.asarray(self.A2.sum(axis=0)).ravel()
        self.idf = np.log((n2 + 1.0) / (df + 1.0)).astype(np.float32)
        self.idf2 = self.idf * self.idf
        self.is_num = _np(pc.match_substring_regex(vocab, r"\d"), np.float32) if len(vocab) else np.zeros(n_v, np.float32)
        # per-record totals (computed once, gathered per pair)
        self.tot1 = self._totals(self.A1)
        self.tot2 = self._totals(self.A2)

    def _totals(self, A) -> dict:
        return {
            "n": np.asarray(A.sum(axis=1)).ravel().astype(np.float32),
            "idf": (A @ self.idf).astype(np.float32),
            "idf2": (A @ self.idf2).astype(np.float32),
            "num": (A @ self.is_num).astype(np.float32),
        }

    def pair_features(self, s1_rows: np.ndarray, pool_rows: np.ndarray, prefix: str) -> dict:
        M = self.A1[s1_rows].multiply(self.A2[pool_rows]).tocsr()     # shared-token indicator matrix
        inter = np.asarray(M.sum(axis=1)).ravel().astype(np.float32)
        inter_idf = (M @ self.idf).astype(np.float32)
        inter_idf2 = (M @ self.idf2).astype(np.float32)
        inter_num = (M @ self.is_num).astype(np.float32)
        n1, n2 = self.tot1["n"][s1_rows], self.tot2["n"][pool_rows]
        idf1, idf2 = self.tot1["idf"][s1_rows], self.tot2["idf"][pool_rows]
        sq1, sq2 = self.tot1["idf2"][s1_rows], self.tot2["idf2"][pool_rows]
        num1, num2 = self.tot1["num"][s1_rows], self.tot2["num"][pool_rows]

        def div(a, b, fill=np.nan):
            out = np.full(len(a), fill, dtype=np.float32)
            np.divide(a, b, out=out, where=b > 0)
            return out

        union = n1 + n2 - inter
        num_union = num1 + num2 - inter_num
        num_jac = div(inter_num, num_union)                       # NaN when neither side has a numeric token
        num_jac = np.where((num_union > 0) & (inter_num == 0), 0.0, num_jac).astype(np.float32)
        return {
            f"{prefix}_n1": n1, f"{prefix}_n2": n2,
            f"{prefix}_inter": inter,
            f"{prefix}_jaccard": div(inter, union),
            f"{prefix}_dice": div(2 * inter, n1 + n2),
            f"{prefix}_cont1": div(inter, n1),                    # share of S1 tokens found in the candidate
            f"{prefix}_cont2": div(inter, n2),                    # share of candidate tokens found in S1
            f"{prefix}_idf_cos": div(inter_idf2, np.sqrt(sq1) * np.sqrt(sq2)),
            f"{prefix}_idf_ov1": div(inter_idf, idf1),
            f"{prefix}_idf_ov2": div(inter_idf, idf2),
            f"{prefix}_num_jaccard": num_jac,
        }


# ---------------------------------------------------------------------------
# Partition context
# ---------------------------------------------------------------------------
class PartitionContext:
    """Everything needed to featurize pairs of one country partition."""

    def __init__(self, s1_tbl: pa.Table, pool_tbl: pa.Table, s1_full: pa.Table | None = None):
        """`s1_full` = every S1 record of the same country partition and split (needed when `s1_tbl` is only a sample of it:
        S1-side ambiguity counts must not depend on the sampling)."""
        self.n_s1, self.n_pool = s1_tbl.num_rows, pool_tbl.num_rows
        cols = ("name_normalized", "name_core", "address_normalized", "postal_code", "house_number")
        self.s1 = {c: _combine(s1_tbl[c]) for c in cols}
        self.pool = {c: _combine(pool_tbl[c]) for c in cols}
        for side in (self.s1, self.pool):      # house numbers: '019' == '19' (nulls stay null)
            side["house_number"] = strip_leading_zeros(side["house_number"])
        self.s1_id, self.pool_id = _combine(s1_tbl["entity_id"]), _combine(pool_tbl["entity_id"])
        self.pool_is_s2 = _np(pc.equal(_combine(pool_tbl["source"]), "S2"), np.int8)
        self.name_tokens = TokenStats(s1_tbl["name_tokens"], pool_tbl["name_tokens"])
        self.addr_tokens = TokenStats(s1_tbl["address_tokens"], pool_tbl["address_tokens"])
        self.s1_script = self._script_codes(s1_tbl["name_script"])
        self.pool_script = self._script_codes(pool_tbl["name_script"])
        # ambiguity of each pool record inside its country pool: how many pool records carry the same exact normalized
        # name / core name / address (chain names, malls). Pool-side only: unbiased even when S1 is a sample.
        self.pool_name_freq = self._freq_per_row(self.pool["name_normalized"])
        self.pool_core_freq = self._freq_per_row(self.pool["name_core"])
        self.pool_addr_freq = self._freq_per_row(self.pool["address_normalized"])
        # ... and the same on the S1 side: how many S1 ENTITIES share the name / address (a chain shares its name across
        # entities, a mall its address); computed over the full S1 split, never over a sample
        universe = s1_full if s1_full is not None else s1_tbl
        self.s1_core_freq = self._freq_in(self.s1["name_core"], universe["name_core"])
        self.s1_addr_freq = self._freq_in(self.s1["address_normalized"], universe["address_normalized"])
        self.s1_name_len = _np(pc.utf8_length(self.s1["name_normalized"]), np.float32)
        self.pool_name_len = _np(pc.utf8_length(self.pool["name_normalized"]), np.float32)
        self.s1_addr_len = _np(pc.utf8_length(self.s1["address_normalized"]), np.float32)
        self.pool_addr_len = _np(pc.utf8_length(self.pool["address_normalized"]), np.float32)

    @staticmethod
    def _freq_per_row(arr) -> np.ndarray:
        """For every row: the number of rows of `arr` holding the identical string."""
        enc = pc.dictionary_encode(arr)
        codes = _np(enc.indices)
        return np.bincount(codes, minlength=len(enc.dictionary)).astype(np.float32)[codes]

    @staticmethod
    def _freq_in(query, universe) -> np.ndarray:
        """For every query string: the number of rows of `universe` holding the identical string (query rows are part of it)."""
        enc = pc.dictionary_encode(_combine(universe))
        counts = np.bincount(_np(enc.indices), minlength=len(enc.dictionary)).astype(np.float32)
        codes = _np(pc.index_in(_combine(query), value_set=enc.dictionary).fill_null(-1))
        return np.where(codes >= 0, counts[np.maximum(codes, 0)], 1.0).astype(np.float32)

    @staticmethod
    def _script_codes(col) -> np.ndarray:
        arr = _combine(col)
        codes = np.full(len(arr), np.nan, dtype=np.float32)
        for name, code in _SCRIPT_CODE.items():
            codes[_np(pc.equal(arr, name), bool)] = code
        return np.where(np.isnan(codes), _SCRIPT_CODE["other"], codes).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def _take(arr, rows: np.ndarray):
    return arr.take(pa.array(rows))


def _fuzzy(a: list, b: list, prefix: str, scorers: dict, invalid: np.ndarray) -> dict:
    out = {}
    for key, (fn, scale) in scorers.items():
        res = process.cpdist(a, b, scorer=fn, workers=-1)
        res = (res / scale).astype(np.float32)
        res[invalid] = np.nan
        out[f"{prefix}_{key}"] = res
    return out


_NAME_SCORERS = {
    "ratio": (fuzz.ratio, 100.0), "jw": (JaroWinkler.normalized_similarity, 1.0),
    "token_sort": (fuzz.token_sort_ratio, 100.0), "token_set": (fuzz.token_set_ratio, 100.0),
    "partial": (fuzz.partial_ratio, 100.0),
}
_CORE_SCORERS = {k: v for k, v in _NAME_SCORERS.items() if k != "partial"}
_ADDR_SCORERS = _NAME_SCORERS

# Columns produced by the blocker that are used as model features (retrieval evidence + the ranking score).
BLOCK_FEATURE_COLS = ["n_name_shared", "n_addr_shared", "idf_name", "idf_addr", "exact_name", "exact_compact",
                      "postal_eq", "n_strategies", "block_score", "n_addr_bigram", "n_name_bigram", "n_prefix",
                      "ov_name", "ov_addr", "rank_score"]


def compute_features(ctx: PartitionContext, s1_row: np.ndarray, pool_row: np.ndarray, block: dict) -> pd.DataFrame:
    """Feature matrix (one row per pair, same order) for pairs given as partition row positions.

    ``s1_row`` must be sorted (all pairs of an S1 entity contiguous) -- true for blocker
    output -- because the within-entity relative features use run boundaries. ``block``
    maps the BLOCK_FEATURE_COLS names to per-pair arrays.
    """
    n = len(s1_row)
    f: dict[str, np.ndarray] = {}

    # ---- string fields, gathered once per pair
    n1, n2 = _take(ctx.s1["name_normalized"], s1_row), _take(ctx.pool["name_normalized"], pool_row)
    c1, c2 = _take(ctx.s1["name_core"], s1_row), _take(ctx.pool["name_core"], pool_row)
    a1, a2 = _take(ctx.s1["address_normalized"], s1_row), _take(ctx.pool["address_normalized"], pool_row)

    len_n1, len_n2 = ctx.s1_name_len[s1_row], ctx.pool_name_len[pool_row]
    len_a1, len_a2 = ctx.s1_addr_len[s1_row], ctx.pool_addr_len[pool_row]
    name_missing = (len_n1 == 0) | (len_n2 == 0)
    addr_missing = (len_a1 == 0) | (len_a2 == 0)
    core_missing = _np(pc.or_(pc.equal(pc.utf8_length(c1), 0), pc.equal(pc.utf8_length(c2), 0)), bool)

    f["name_missing_1"], f["name_missing_2"] = (len_n1 == 0).astype(np.int8), (len_n2 == 0).astype(np.int8)
    f["addr_missing_1"], f["addr_missing_2"] = (len_a1 == 0).astype(np.int8), (len_a2 == 0).astype(np.int8)
    f["name_exact_normalized"] = (_np(pc.fill_null(pc.equal(n1, n2), False), bool) & ~name_missing).astype(np.int8)
    f["name_exact_core"] = (_np(pc.fill_null(pc.equal(c1, c2), False), bool) & ~core_missing).astype(np.int8)
    f["addr_exact_normalized"] = (_np(pc.fill_null(pc.equal(a1, a2), False), bool) & ~addr_missing).astype(np.int8)

    # ---- fuzzy string similarity (multithreaded C++)
    ln1, ln2 = n1.to_pylist(), n2.to_pylist()
    f.update(_fuzzy(ln1, ln2, "name", _NAME_SCORERS, name_missing))
    del ln1, ln2
    lc1, lc2 = c1.to_pylist(), c2.to_pylist()
    f.update(_fuzzy(lc1, lc2, "core", _CORE_SCORERS, core_missing))
    del lc1, lc2
    la1, la2 = a1.to_pylist(), a2.to_pylist()
    f.update(_fuzzy(la1, la2, "addr", _ADDR_SCORERS, addr_missing))
    del la1, la2

    # ---- lengths
    f["name_len_1"], f["name_len_2"] = len_n1, len_n2
    f["name_len_diff"] = np.abs(len_n1 - len_n2)
    f["addr_len_1"], f["addr_len_2"] = len_a1, len_a2
    f["addr_len_diff"] = np.abs(len_a1 - len_a2)

    # ---- token overlap (sparse)
    nt = ctx.name_tokens.pair_features(s1_row, pool_row, "nt")
    at = ctx.addr_tokens.pair_features(s1_row, pool_row, "at")
    for k, v in nt.items():
        if k not in ("nt_n1", "nt_n2"):
            v[name_missing] = np.nan
        f[k] = v
    for k, v in at.items():
        if k not in ("at_n1", "at_n2"):
            v[addr_missing] = np.nan
        f[k] = v

    # ---- postal / house number
    p1, p2 = _take(ctx.s1["postal_code"], s1_row), _take(ctx.pool["postal_code"], pool_row)
    both_p = _np(pc.and_(pc.is_valid(p1), pc.is_valid(p2)), bool)
    f["postal_both_present"] = both_p.astype(np.int8)
    f["postal_match"] = (_np(pc.fill_null(pc.equal(p1, p2), False), bool) & both_p).astype(np.int8)
    h1, h2 = _take(ctx.s1["house_number"], s1_row), _take(ctx.pool["house_number"], pool_row)
    both_h = _np(pc.and_(pc.is_valid(h1), pc.is_valid(h2)), bool)
    f["housenum_both_present"] = both_h.astype(np.int8)
    f["housenum_match"] = (_np(pc.fill_null(pc.equal(h1, h2), False), bool) & both_h).astype(np.int8)

    # ---- script of the names (fixed vocabulary; country itself is intentionally not a feature)
    f["name_script_1"], f["name_script_2"] = ctx.s1_script[s1_row], ctx.pool_script[pool_row]
    f["name_script_match"] = (f["name_script_1"] == f["name_script_2"]).astype(np.int8)

    # ---- ambiguity of the candidate inside its pool (log1p of the number of identical name / core name / address records)
    f["pool_name_freq"] = np.log1p(ctx.pool_name_freq[pool_row])
    f["pool_core_freq"] = np.log1p(ctx.pool_core_freq[pool_row])
    pa_freq = np.log1p(ctx.pool_addr_freq[pool_row])
    pa_freq[len_a2 == 0] = np.nan                    # empty address: the "identical address" count is meaningless
    f["pool_addr_freq"] = pa_freq
    f["s1_core_freq"] = np.log1p(ctx.s1_core_freq[s1_row])
    f["s1_addr_freq"] = np.where(len_a1 == 0, np.nan, np.log1p(ctx.s1_addr_freq[s1_row])).astype(np.float32)

    # ---- cross-field
    nr = np.nan_to_num(f["name_ratio"], nan=0.0)
    ar = np.nan_to_num(f["addr_ratio"], nan=0.0)
    f["sim_min"], f["sim_max"], f["sim_prod"] = np.minimum(nr, ar), np.maximum(nr, ar), nr * ar

    # ---- blocker evidence
    f["is_s2"] = ctx.pool_is_s2[pool_row]
    for c in BLOCK_FEATURE_COLS:
        f[c] = np.asarray(block[c])

    # ---- relative to the other candidates of the same S1 entity
    prelim = (0.5 * nr + 0.5 * ar).astype(np.float32)
    f["prelim_score"] = prelim
    f["prelim_rank"] = _rank_desc_within(s1_row, prelim)
    f["prelim_gap_to_best"] = _group_max(s1_row, prelim) - prelim
    f["name_ratio_gap"] = _group_max(s1_row, f["name_ratio"]) - f["name_ratio"]
    f["addr_ratio_gap"] = _group_max(s1_row, f["addr_ratio"]) - f["addr_ratio"]
    rank_score = np.asarray(block["rank_score"], dtype=np.float32)
    f["block_rank"] = _rank_desc_within(s1_row, rank_score)
    f["block_gap_to_best"] = _group_max(s1_row, rank_score) - rank_score
    b = _group_bounds(s1_row)
    f["n_candidates_for_entity"] = np.repeat(np.diff(b), np.diff(b)).astype(np.int32)

    # ---- ambiguity context: how many candidates of the same entity already agree strongly? A name-only candidate
    # (no address) is far more plausible when nobody else agrees on the name, and a chain name shared by many candidates
    # is ambiguous unless the address decides.
    def group_count(mask):
        return np.repeat(np.add.reduceat(mask.astype(np.int32), b[:-1]), np.diff(b)) if n else np.zeros(0, np.int32)

    f["n_name_ge90"] = group_count(nr >= 0.9)
    f["n_addr_ge80"] = group_count(ar >= 0.8)
    f["n_both_strong"] = group_count((nr >= 0.85) & (ar >= 0.7))
    f["is_name_top1"] = (nr >= np.nan_to_num(_group_max(s1_row, nr.astype(np.float32)), nan=1.0)).astype(np.int8)

    out = pd.DataFrame(f)
    assert len(out) == n
    return out
