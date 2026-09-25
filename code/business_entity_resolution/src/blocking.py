"""Candidate generation (blocking): union of several retrieval rules, then a learned cheap ranking.

Scale drives every choice here (~10M pool records, up to 1.7M Source-1 entities per split):

* Tokens are dictionary-encoded to int32 codes with pyarrow and everything after that is
  vectorized numpy on integer arrays -- no Python-level loops, no per-token strings.
* Retrieval rules (a pair is a candidate if ANY rule fires):
    1. rare unigram tokens: each S1 entity looks up its K rarest name / address tokens (pool
       document frequency <= df_cap);
    2. exact keys: normalized name, compact name, postal code;
    3. address bigrams and name bigrams (adjacent token pairs, order-insensitive): far more
       specific than single tokens, so they retrieve true matches whose individual tokens are all
       common ("ekta vihar", "805 union") -- the typical case for non-Latin names with a clean
       address, or clean names with a missing address;
    4. compact-core-name prefix: catches concatenated / domain-style names
       ("gurgaontechnologiesclinic.com" vs "Gurgaon Technologies Clinic").
  Every rule has a document-frequency cap, which keeps the expansion volume bounded.
* Ranking. A hand-written score over the retrieved pairs ties massively (a name made of common
  words leaves one rare address token; hundreds of candidates share it) and loses the true match at
  the per-entity cap. Every retrieved pair therefore gets cheap evidence features -- IDF-weighted
  token overlap of name and address over ALL tokens (padded token matrices, ~1 microsecond per pair),
  bigram / prefix / exact-key hits, token counts -- and a small LightGBM `ranker` (train_ranker.py)
  turns them into a match probability. The best ``max_candidates`` (30) per entity form the *shortlist*,
  which keeps ~97% of ALL true pairs (the retrieval ceiling), where the hand-written score kept ~64%.
* Final cut (run_candidate_cut.py, ``select_top`` with a probability floor): most shortlisted pairs have a
  ranker probability near 0, so the candidate set the matcher scores keeps only pairs above a small
  probability floor -- a few candidates for an easy entity, more for an ambiguous one, none when nothing
  in the pool is plausible.
* Country is a hard partition: 0 of 7,638,365 training true matches cross a country boundary
  (reports/data_analysis_report.md). Partitions are processed independently, so any country value
  -- including France, which never appears in training -- flows through with no special-casing.

The pool side (token indexes, key postings, dense matrices) is built once per country partition
(``PoolIndex``) and then queried with any number of S1 chunks (``PoolIndex.block``). Rows are
addressed by their position inside the partition tables; entity-id strings are attached only to the
final (already truncated) pair table.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from config import DEFAULT_BLOCKING, BlockingConfig


def _combine(col):
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


def _dedup_pairs(row: np.ndarray, code: np.ndarray, n_codes: int):
    key = np.unique(row.astype(np.int64) * n_codes + code)
    return (key // n_codes).astype(np.int32), (key % n_codes).astype(np.int32)


def _rank_within_sorted_groups(group: np.ndarray) -> np.ndarray:
    """0-based rank of each element within its run of equal `group` values
    (input must already be sorted by group)."""
    if len(group) == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    counts = np.diff(np.r_[first, len(group)])
    return np.arange(len(group)) - np.repeat(first, counts)


def select_top(s1_row: np.ndarray, pool_row: np.ndarray, score: np.ndarray, max_candidates: int,
               min_score: float | None = None, min_keep: int = 0) -> np.ndarray:
    """Ascending indices of the pairs kept for each S1 entity: best `score` first (ties -> lower pool row), at most
    `max_candidates`, and with `min_score` only pairs scoring >= min_score (the best `min_keep` of an entity are kept
    regardless). The blocker's shortlist truncation and the final candidate cut (run_candidate_cut.py) both use this
    one ordering, so cutting a stored shortlist gives exactly the pairs a blocker run with the tighter rule would keep."""
    order = np.lexsort((pool_row, -score, s1_row))
    rank = _rank_within_sorted_groups(s1_row[order])
    ok = rank < max_candidates
    if min_score is not None:
        ok &= (score[order] >= min_score) | (rank < min_keep)
    sel = order[ok]
    sel.sort()
    return sel


def _expand(post_off: np.ndarray, post_rows: np.ndarray, row: np.ndarray, code: np.ndarray):
    """For each (query_row, code) pair, emit one output per posting of `code`.
    Returns (query_row_repeated, pool_row, run_index) where run_index maps each
    output back to its input pair (so per-pair weights can be gathered)."""
    starts = post_off[code]
    cnt = post_off[code + 1] - starts
    total = int(cnt.sum())
    if total == 0:
        e = np.zeros(0, dtype=np.int32)
        return e, e, np.zeros(0, dtype=np.int64)
    run_index = np.repeat(np.arange(len(code)), cnt)
    run_start = np.cumsum(cnt) - cnt
    within = np.arange(total) - np.repeat(run_start, cnt)
    pool = post_rows[np.repeat(starts, cnt) + within]
    return row[run_index], pool, run_index


def _list_lengths(col) -> np.ndarray:
    return pc.list_value_length(_combine(col)).to_numpy(zero_copy_only=False).astype(np.int16)


# RE2 (Arrow's regex engine) has no lookahead: drop the leading zeros but always keep one digit ('0' stays '0').
_LEADING_ZEROS = r"^0+([0-9])"


def strip_leading_zeros(strings):
    return pc.replace_substring_regex(strings, _LEADING_ZEROS, r"\1")


def canon_flat(flat):
    """Numeric tokens differing only by leading zeros are the same token ('019' == '19', '010001' == '10001');
    seen in real noisy addresses. Applied to every flattened token array (pool and S1) before encoding."""
    return strip_leading_zeros(flat)


class TokenIndex:
    """Inverted index (token code -> pool rows) for one list<string> column."""

    def __init__(self, list_col, n_rows: int, keep_flat: bool = False):
        arr = _combine(list_col)
        flat = canon_flat(pc.list_flatten(arr))
        parent = pc.list_parent_indices(arr).to_numpy()
        enc = pc.dictionary_encode(flat)
        codes = enc.indices.to_numpy(zero_copy_only=False)
        self.vocab = enc.dictionary
        self.n_codes = len(self.vocab)
        self.n_rows = n_rows
        if keep_flat:   # token sequence in original order (needed for bigrams / dense matrices)
            self.flat_parent = parent.astype(np.int32)
            self.flat_codes = codes.astype(np.int32)
        row, code = _dedup_pairs(parent, codes, max(self.n_codes, 1))
        self.df = np.bincount(code, minlength=self.n_codes).astype(np.int64)
        order = np.argsort(code, kind="stable")
        self.post_rows = row[order]
        self.post_off = np.concatenate([[0], np.cumsum(self.df)]).astype(np.int64)
        self.idf = np.log((n_rows + 1.0) / (self.df + 1.0)).astype(np.float32)

    def drop_flat(self):
        self.__dict__.pop("flat_parent", None)
        self.__dict__.pop("flat_codes", None)

    def encode_flat(self, list_col):
        """Token sequence of the query side in original order: (parent row, code) with code -1 for
        tokens that never occur in the pool (they can not match anything)."""
        arr = _combine(list_col)
        flat = canon_flat(pc.list_flatten(arr))
        parent = pc.list_parent_indices(arr).to_numpy()
        codes = pc.index_in(flat, value_set=self.vocab).fill_null(-1).to_numpy(zero_copy_only=False)
        return parent.astype(np.int32), codes.astype(np.int32)

    def encode_queries(self, list_col):
        """S1 token lists -> deduped (row, code) over tokens present in the pool vocab."""
        parent, codes = self.encode_flat(list_col)
        keep = codes >= 0
        return _dedup_pairs(parent[keep], codes[keep], max(self.n_codes, 1))

    def select_rarest(self, row: np.ndarray, code: np.ndarray, k: int, df_cap: int):
        ok = self.df[code] <= df_cap
        row, code = row[ok], code[ok]
        order = np.lexsort((code, self.df[code], row))
        row, code = row[order], code[order]
        keep = _rank_within_sorted_groups(row) < k
        return row[keep], code[keep]


class KeyIndex:
    """Exact-value index over a string column (empty/null values never match)."""

    def __init__(self, pool_col):
        arr = _combine(pool_col)
        arr = pc.if_else(pc.equal(arr, ""), pa.scalar(None, arr.type), arr)
        enc = pc.dictionary_encode(arr, null_encoding="mask")
        self.vocab = enc.dictionary
        self.n_codes = len(self.vocab)
        codes = enc.indices.fill_null(-1).to_numpy(zero_copy_only=False)
        valid = np.flatnonzero(codes >= 0)
        vcodes = codes[valid]
        self.df = np.bincount(vcodes, minlength=self.n_codes).astype(np.int64)
        order = np.argsort(vcodes, kind="stable")
        self.post_rows = valid[order].astype(np.int32)
        self.post_off = np.concatenate([[0], np.cumsum(self.df)]).astype(np.int64)

    def pairs(self, query_col, cap: int):
        arr = _combine(query_col)
        arr = pc.if_else(pc.equal(arr, ""), pa.scalar(None, arr.type), arr)
        codes = pc.index_in(arr, value_set=self.vocab).fill_null(-1).to_numpy(zero_copy_only=False)
        rows = np.flatnonzero(codes >= 0)
        codes = codes[rows]
        ok = self.df[codes] <= cap
        rows, codes = rows[ok].astype(np.int32), codes[ok]
        s1_row, pool_row, _ = _expand(self.post_off, self.post_rows, rows, codes)
        return s1_row, pool_row


# ---------------------------------------------------------------------------
# Additional retrieval families over arbitrary int64 keys
# ---------------------------------------------------------------------------
class KeyPostings:
    """Inverted index over int64 keys: (pool row, key) pairs -> sorted unique keys with posting lists."""

    def __init__(self, rows: np.ndarray, keys: np.ndarray):
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        self.post_rows = rows[order].astype(np.int32)
        first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(ks) else np.zeros(0, dtype=np.int64)
        self.uk = ks[first]
        self.off = np.r_[first, len(ks)].astype(np.int64)
        self.df = np.diff(self.off)

    def query(self, rows: np.ndarray, keys: np.ndarray, cap: int):
        """(row, posting-list index) for query (row, key) pairs whose key exists in the pool with df <= cap,
        sorted by row."""
        if len(self.uk) == 0 or len(keys) == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int64)
        pos = np.minimum(np.searchsorted(self.uk, keys), len(self.uk) - 1)
        ok = (self.uk[pos] == keys) & (self.df[pos] <= cap)
        rows, pos = rows[ok].astype(np.int32), pos[ok]
        order = np.argsort(rows, kind="stable")
        return rows[order], pos[order]


def adjacent_pair_keys(parent: np.ndarray, codes: np.ndarray, n_codes: int):
    """Order-insensitive adjacent-token pair keys of every record: (row, key). Pairs that straddle an
    unknown token (code -1) or a record boundary are not formed."""
    if len(codes) < 2:
        return np.zeros(0, np.int32), np.zeros(0, np.int64)
    ok = (parent[1:] == parent[:-1]) & (codes[1:] >= 0) & (codes[:-1] >= 0)
    a, b = codes[:-1][ok].astype(np.int64), codes[1:][ok].astype(np.int64)
    return parent[:-1][ok], np.minimum(a, b) * n_codes + np.maximum(a, b)


class PrefixIndex:
    """Postings over the first `length` characters of the compact core name (space-free); names shorter
    than `length` get no key."""

    def __init__(self, pool_core, length: int):
        self.length = length
        enc = pc.dictionary_encode(self._prefixes(pool_core), null_encoding="mask")
        self.vocab = enc.dictionary
        codes = enc.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
        rows = np.flatnonzero(codes >= 0).astype(np.int32)
        self.post = KeyPostings(rows, codes[rows])

    def _prefixes(self, col):
        compact = pc.replace_substring(_combine(col), " ", "")
        pre = pc.utf8_slice_codeunits(compact, 0, self.length)
        return pc.if_else(pc.greater_equal(pc.utf8_length(compact), self.length), pre, pa.scalar(None, pre.type))

    def query(self, s1_core, cap: int):
        codes = pc.index_in(self._prefixes(s1_core), value_set=self.vocab).fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
        rows = np.flatnonzero(codes >= 0).astype(np.int32)
        return self.post.query(rows, codes[rows], cap)


# ---------------------------------------------------------------------------
# Dense token matrices: full-evidence overlap of a pair in ~1 microsecond
# ---------------------------------------------------------------------------
def dense_matrix(parent: np.ndarray, codes: np.ndarray, n_rows: int, width: int) -> np.ndarray:
    """(n_rows, width) int32 matrix of each record's first `width` known tokens, -1 padded."""
    keep = codes >= 0
    parent, codes = parent[keep], codes[keep]
    m = np.full((n_rows, width), -1, dtype=np.int32)
    if len(parent) == 0:
        return m
    pos = _rank_within_sorted_groups(parent)
    ok = pos < width
    m[parent[ok], pos[ok]] = codes[ok]
    return m


def idf_mass(parent: np.ndarray, codes: np.ndarray, idf: np.ndarray, n_rows: int, oov_idf: float) -> np.ndarray:
    """Per-record total IDF of ALL tokens (unknown tokens weigh `oov_idf`, the maximum IDF)."""
    known = codes >= 0
    tot = np.bincount(parent[known], weights=idf[codes[known]], minlength=n_rows)
    return (tot + np.bincount(parent[~known], minlength=n_rows) * oov_idf).astype(np.float32)


def shared_idf(m_s1: np.ndarray, m_pool: np.ndarray, idf: np.ndarray, s1_rows: np.ndarray, pool_rows: np.ndarray,
               chunk: int = 400_000) -> np.ndarray:
    """IDF mass of the S1 tokens (first `width` per record) that also occur in the candidate, per pair."""
    out = np.empty(len(s1_rows), dtype=np.float32)
    for a in range(0, len(s1_rows), chunk):
        x = m_s1[s1_rows[a:a + chunk]]
        y = m_pool[pool_rows[a:a + chunk]]
        hit = (x[:, :, None] == y[:, None, :]).any(axis=2) & (x >= 0)
        out[a:a + chunk] = (idf[np.maximum(x, 0)] * hit).sum(axis=1)
    return out


class DensePool:
    """Pool-side padded token matrix + IDF masses of one field."""

    def __init__(self, idx: TokenIndex, width: int):
        self.idf = idx.idf
        self.m_pool = dense_matrix(idx.flat_parent, idx.flat_codes, idx.n_rows, width)
        self.oov_idf = float(np.log(idx.n_rows + 1.0))
        self.tot_pool = idf_mass(idx.flat_parent, idx.flat_codes, idx.idf, idx.n_rows, self.oov_idf)
        self.width = width


class DenseS1:
    """S1-side counterpart of DensePool for one chunk of S1 records."""

    def __init__(self, idx: TokenIndex, pool: DensePool, s1_list):
        parent, codes = idx.encode_flat(s1_list)
        n1 = len(_combine(s1_list))
        self.m_s1 = dense_matrix(parent, codes, n1, pool.width)
        self.tot_s1 = idf_mass(parent, codes, idx.idf, n1, pool.oov_idf)
        self.pool = pool

    def dice(self, s1_rows: np.ndarray, pool_rows: np.ndarray) -> np.ndarray:
        """IDF-weighted Dice overlap in [0, 1]; 0 when both sides are empty."""
        sh = shared_idf(self.m_s1, self.pool.m_pool, self.pool.idf, s1_rows, pool_rows)
        den = self.tot_s1[s1_rows] + self.pool.tot_pool[pool_rows]
        return np.divide(2.0 * sh, den, out=np.zeros(len(sh), np.float32), where=den > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _agg_by_key(key: np.ndarray, weights: np.ndarray | None):
    uk, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, minlength=len(uk))
    w = np.bincount(inv, weights=weights, minlength=len(uk)) if weights is not None else None
    return uk, n, w


def _lookup(uk_all: np.ndarray, uk: np.ndarray, values: np.ndarray, dtype) -> np.ndarray:
    out = np.zeros(len(uk_all), dtype=dtype)
    if len(uk):
        out[np.searchsorted(uk_all, uk)] = values
    return out


OUT_COLS = ["s1_row", "pool_row", "n_name_shared", "n_addr_shared", "idf_name", "idf_addr", "exact_name",
            "exact_compact", "postal_eq", "n_strategies", "block_score",
            "n_addr_bigram", "n_name_bigram", "n_prefix", "ov_name", "ov_addr", "rank_score",
            "s1_ntok_name", "s1_ntok_addr", "c_ntok_name", "c_ntok_addr"]
_OUT_DTYPES = {"s1_row": np.int32, "pool_row": np.int32, "n_name_shared": np.int16, "n_addr_shared": np.int16,
               "idf_name": np.float32, "idf_addr": np.float32, "exact_name": np.int8, "exact_compact": np.int8,
               "postal_eq": np.int8, "n_strategies": np.int8, "block_score": np.float32,
               "n_addr_bigram": np.int16, "n_name_bigram": np.int16, "n_prefix": np.int8,
               "ov_name": np.float32, "ov_addr": np.float32, "rank_score": np.float32,
               "s1_ntok_name": np.int16, "s1_ntok_addr": np.int16, "c_ntok_name": np.int16, "c_ntok_addr": np.int16}

# Evidence columns the learned ranker consumes (the pair-level outputs above; order matters).
RANKER_FEATURES = ["ov_name", "ov_addr", "idf_name", "idf_addr", "n_name_shared", "n_addr_shared", "n_addr_bigram",
                   "n_name_bigram", "n_prefix", "exact_name", "exact_compact", "postal_eq", "n_strategies",
                   "block_score", "s1_ntok_name", "s1_ntok_addr", "c_ntok_name", "c_ntok_addr"]


def heuristic_rank(ev: dict) -> np.ndarray:
    """Ranking used when no learned ranker is supplied: address overlap weighted above name overlap
    (measured on the training sample: far better than an unweighted sum at small caps)."""
    return (ev["ov_name"] + 1.5 * ev["ov_addr"]).astype(np.float32)


class PoolIndex:
    """Everything derived from the pool side of one country partition, built once and queried per S1 chunk."""

    def __init__(self, pool_tbl: pa.Table, cfg: BlockingConfig = DEFAULT_BLOCKING, verbose: bool = True):
        t0 = time.time()
        self.cfg, self.n = cfg, pool_tbl.num_rows
        need_flat = bool(cfg.cap_addr_bigram or cfg.cap_name_bigram or cfg.rescore)
        self.name_idx = TokenIndex(pool_tbl["name_tokens"], self.n, keep_flat=need_flat)
        self.addr_idx = TokenIndex(pool_tbl["address_tokens"], self.n, keep_flat=need_flat)
        self.exact = {"name": KeyIndex(pool_tbl["name_normalized"]), "comp": KeyIndex(pool_tbl["name_compact"]),
                      "post": KeyIndex(pool_tbl["postal_code"])}
        self.bigram = {}
        if cfg.cap_addr_bigram:
            self.bigram["addr_bigram"] = KeyPostings(*adjacent_pair_keys(self.addr_idx.flat_parent, self.addr_idx.flat_codes, self.addr_idx.n_codes))
        if cfg.cap_name_bigram:
            self.bigram["name_bigram"] = KeyPostings(*adjacent_pair_keys(self.name_idx.flat_parent, self.name_idx.flat_codes, self.name_idx.n_codes))
        self.prefix = [PrefixIndex(pool_tbl["name_core"], L) for L in cfg.prefix_lens]
        self.dense = None
        if cfg.rescore:
            self.dense = {"name": DensePool(self.name_idx, cfg.dense_width_name),
                          "addr": DensePool(self.addr_idx, cfg.dense_width_addr)}
        self.name_idx.drop_flat()
        self.addr_idx.drop_flat()
        self.ntok_name = _list_lengths(pool_tbl["name_tokens"])
        self.ntok_addr = _list_lengths(pool_tbl["address_tokens"])
        if verbose:
            print(f"    pool index built in {time.time()-t0:.1f}s (pool={self.n:,}, name vocab={self.name_idx.n_codes:,}, "
                  f"addr vocab={self.addr_idx.n_codes:,})", flush=True)

    def block(self, s1_tbl: pa.Table, ranker=None, verbose: bool = True) -> dict:
        """Candidate pairs for a chunk of S1 records (s1_row indexes into `s1_tbl`).

        s1_tbl needs the same columns as the pool table (name_tokens, address_tokens, name_normalized,
        name_compact, postal_code, and name_core when prefix rules are on). `ranker` (optional) is an
        object with `.predict(matrix)` over RANKER_FEATURES; without one, heuristic_rank is used.
        Returns a dict of numpy arrays, one entry per pair, with the keys in OUT_COLS.
        """
        cfg, npool = self.cfg, self.n
        n1 = s1_tbl.num_rows
        empty = {k: np.zeros(0, _OUT_DTYPES[k]) for k in OUT_COLS}
        if n1 == 0 or npool == 0:
            return empty
        t0 = time.time()
        name_idx, addr_idx = self.name_idx, self.addr_idx
        nr, nc = name_idx.select_rarest(*name_idx.encode_queries(s1_tbl["name_tokens"]), cfg.k_rare_name, cfg.df_cap)
        ar, ac = addr_idx.select_rarest(*addr_idx.encode_queries(s1_tbl["address_tokens"]), cfg.k_rare_addr, cfg.df_cap)

        def pair_keys(s1_row, pool_row):
            return np.unique(s1_row.astype(np.int64) * npool + pool_row)

        exact_keys = {
            "name": pair_keys(*self.exact["name"].pairs(s1_tbl["name_normalized"], cfg.exact_key_cap)),
            "comp": pair_keys(*self.exact["comp"].pairs(s1_tbl["name_compact"], cfg.exact_key_cap)),
            "post": pair_keys(*self.exact["post"].pairs(s1_tbl["postal_code"], cfg.exact_key_cap)),
        }

        families = []   # (label, KeyPostings, query rows sorted, query posting indices)
        if "addr_bigram" in self.bigram:
            p1, c1 = addr_idx.encode_flat(s1_tbl["address_tokens"])
            r, q = self.bigram["addr_bigram"].query(*adjacent_pair_keys(p1, c1, addr_idx.n_codes), cfg.cap_addr_bigram)
            families.append(("addr_bigram", self.bigram["addr_bigram"], r, q))
        if "name_bigram" in self.bigram:
            p1, c1 = name_idx.encode_flat(s1_tbl["name_tokens"])
            r, q = self.bigram["name_bigram"].query(*adjacent_pair_keys(p1, c1, name_idx.n_codes), cfg.cap_name_bigram)
            families.append(("name_bigram", self.bigram["name_bigram"], r, q))
        for px in self.prefix:
            r, q = px.query(s1_tbl["name_core"], cfg.cap_prefix)
            families.append((f"prefix{px.length}", px.post, r, q))

        dn = da = None
        if cfg.rescore:
            dn = DenseS1(name_idx, self.dense["name"], s1_tbl["name_tokens"])
            da = DenseS1(addr_idx, self.dense["addr"], s1_tbl["address_tokens"])
        s1_ntok_name, s1_ntok_addr = _list_lengths(s1_tbl["name_tokens"]), _list_lengths(s1_tbl["address_tokens"])
        if verbose:
            fam = ", ".join(f"{lab}: {len(rows):,}" for lab, _p, rows, _q in families)
            print(f"    s1 chunk of {n1:,} prepared in {time.time()-t0:.1f}s (lookups: name {len(nr):,}, addr {len(ar):,}; {fam})", flush=True)

        # Batch S1 rows so the number of materialized (s1, pool) expansions stays under the budget; a row's
        # whole candidate set lives in one batch, so the per-entity truncation can be applied batch-by-batch.
        vol = np.bincount(nr, weights=name_idx.df[nc], minlength=n1) + np.bincount(ar, weights=addr_idx.df[ac], minlength=n1)
        for _lab, post, rows, pos in families:
            vol = vol + np.bincount(rows, weights=post.df[pos], minlength=n1)
        batch_id = (np.cumsum(vol) // cfg.expansion_budget).astype(np.int64)
        bounds = np.flatnonzero(np.r_[True, batch_id[1:] != batch_id[:-1]])
        bounds = np.r_[bounds, n1]

        out_parts = []
        for b in range(len(bounds) - 1):
            r0, r1 = int(bounds[b]), int(bounds[b + 1])

            def rows_in(row):  # rows are sorted -> contiguous slice
                lo, hi = np.searchsorted(row, [r0, r1], side="left")
                return slice(lo, hi)

            sn, sa = rows_in(nr), rows_in(ar)
            s1n, pooln, run_n = _expand(name_idx.post_off, name_idx.post_rows, nr[sn], nc[sn])
            s1a, poola, run_a = _expand(addr_idx.post_off, addr_idx.post_rows, ar[sa], ac[sa])
            key_n = s1n.astype(np.int64) * npool + pooln
            key_a = s1a.astype(np.int64) * npool + poola
            uk_n, cnt_n, sc_n = _agg_by_key(key_n, name_idx.idf[nc[sn]][run_n]) if len(key_n) else (key_n, key_n, key_n)
            uk_a, cnt_a, sc_a = _agg_by_key(key_a, addr_idx.idf[ac[sa]][run_a]) if len(key_a) else (key_a, key_a, key_a)

            lo_k, hi_k = np.int64(r0) * npool, np.int64(r1) * npool
            ek = {}
            for name, keys in exact_keys.items():
                i0, i1 = np.searchsorted(keys, [lo_k, hi_k], side="left")
                ek[name] = keys[i0:i1]

            fam_counts = {}
            for label, post, rows, pos in families:
                sl = rows_in(rows)
                fs, fp, _ = _expand(post.off, post.post_rows, rows[sl], pos[sl])
                if len(fs):
                    fam_counts[label] = _agg_by_key(fs.astype(np.int64) * npool + fp, None)[:2]

            uk = np.unique(np.concatenate([uk_n, uk_a, ek["name"], ek["comp"], ek["post"]] + [v[0] for v in fam_counts.values()]))
            if len(uk) == 0:
                continue
            ev = {
                "n_name_shared": _lookup(uk, uk_n, cnt_n, np.int16), "n_addr_shared": _lookup(uk, uk_a, cnt_a, np.int16),
                "idf_name": _lookup(uk, uk_n, sc_n, np.float32), "idf_addr": _lookup(uk, uk_a, sc_a, np.float32),
                "exact_name": _lookup(uk, ek["name"], np.ones(len(ek["name"]), np.int8), np.int8),
                "exact_compact": _lookup(uk, ek["comp"], np.ones(len(ek["comp"]), np.int8), np.int8),
                "postal_eq": _lookup(uk, ek["post"], np.ones(len(ek["post"]), np.int8), np.int8),
                "n_addr_bigram": _lookup(uk, *fam_counts["addr_bigram"], np.int16) if "addr_bigram" in fam_counts else np.zeros(len(uk), np.int16),
                "n_name_bigram": _lookup(uk, *fam_counts["name_bigram"], np.int16) if "name_bigram" in fam_counts else np.zeros(len(uk), np.int16),
            }
            n_pre = np.zeros(len(uk), np.int8)
            for label, (fk, _fc) in fam_counts.items():
                if label.startswith("prefix"):
                    n_pre = n_pre + _lookup(uk, fk, np.ones(len(fk), np.int8), np.int8)
            ev["n_prefix"] = n_pre
            ev["n_strategies"] = ((ev["n_name_shared"] > 0).astype(np.int8) + (ev["n_addr_shared"] > 0) + ev["exact_name"]
                                  + ev["exact_compact"] + ev["postal_eq"] + (ev["n_addr_bigram"] > 0)
                                  + (ev["n_name_bigram"] > 0) + (ev["n_prefix"] > 0)).astype(np.int8)
            ev["block_score"] = (ev["idf_name"] + ev["idf_addr"] + 6.0 * ev["exact_name"] + 4.0 * ev["exact_compact"]
                                 + 3.0 * ev["postal_eq"]).astype(np.float32)
            s1_row = (uk // npool).astype(np.int32)
            pool_row = (uk % npool).astype(np.int32)
            ev["s1_ntok_name"], ev["s1_ntok_addr"] = s1_ntok_name[s1_row], s1_ntok_addr[s1_row]
            ev["c_ntok_name"], ev["c_ntok_addr"] = self.ntok_name[pool_row], self.ntok_addr[pool_row]
            if cfg.rescore:
                ev["ov_name"], ev["ov_addr"] = dn.dice(s1_row, pool_row), da.dice(s1_row, pool_row)
            else:
                ev["ov_name"] = ev["ov_addr"] = np.zeros(len(uk), np.float32)
            if ranker is not None:
                rank = ranker.predict(np.column_stack([ev[c] for c in RANKER_FEATURES]).astype(np.float32)).astype(np.float32)
            elif cfg.rescore:
                rank = heuristic_rank(ev)
            else:
                rank = ev["block_score"]          # legacy ranking (rare-token score)
            # per-entity truncation to the shortlist: best ranking score first (ties: lower pool row)
            sel = select_top(s1_row, pool_row, rank, cfg.max_candidates)
            part = {"s1_row": s1_row[sel] + 0, "pool_row": pool_row[sel], "rank_score": rank[sel]}
            for c in OUT_COLS:
                if c not in part:
                    part[c] = ev[c][sel]
            out_parts.append(part)
            if verbose and len(bounds) > 2:
                print(f"    batch {b+1}/{len(bounds)-1}: rows [{r0:,},{r1:,}) -> {len(sel):,} pairs "
                      f"({len(uk):,} before truncation), {time.time()-t0:.1f}s", flush=True)
            del s1n, pooln, s1a, poola, key_n, key_a, uk, ev
            gc.collect()

        if not out_parts:
            return empty
        return {c: np.concatenate([p[c] for p in out_parts]).astype(_OUT_DTYPES[c], copy=False) for c in OUT_COLS}


def block_partition(s1_tbl: pa.Table, pool_tbl: pa.Table, cfg: BlockingConfig = DEFAULT_BLOCKING, verbose: bool = True,
                    ranker=None) -> dict:
    """One-shot convenience: build the pool index and block a single S1 table against it."""
    if s1_tbl.num_rows == 0 or pool_tbl.num_rows == 0:
        return {k: np.zeros(0, _OUT_DTYPES[k]) for k in OUT_COLS}
    return PoolIndex(pool_tbl, cfg, verbose).block(s1_tbl, ranker, verbose)
