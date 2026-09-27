"""Collective (graph-level) evidence for a candidate pair, computed from first-stage scores of the WHOLE candidate graph.

The candidate graph links Source-1 entities to Source-2/3 records. Pairwise models score each edge on its own; two
structural facts of this task are ignored that way:

1. Exclusivity (pool side). In the ground truth every Source-2/3 record belongs to at most one Source-1 entity
   (7,638,365 true pairs, every pool id appears once). When several Source-1 entities claim the same record, at most one
   of them can be right. `pool_side_features` describes the competition for the pair's record: how many entities claim
   it, the best competing score, this entity's rank among the claimants, and the exclusivity-normalized probability
   q = o / (1 + sum of all claimants' odds), o = p / (1 - p) -- the posterior that THIS entity owns the record if claims
   were independent and the record has at most one owner (q == p when nobody else claims it).

2. Transitivity (entity side, the "triangle" test). All true records of one entity are noisy copies of the same business,
   so they resemble EACH OTHER. A candidate that is confident on its own but unlike the entity's other confident
   candidates (a "fractured triangle": e.g. same name and house number but another street, while the entity's two
   confident records agree on the right street) is suspicious; one that resembles them is corroborated ("cohesive
   triangle"). `sibling_features` compares every candidate with the other candidates of its entity (name / address
   similarity, rapidfuzz, the same normalized strings the matcher uses) and summarizes that agreement weighted by the
   siblings' first-stage scores.

Everything is vectorized; pairs are addressed by partition row positions like the rest of the pipeline and must be
grouped by entity (s1_row sorted), which is how blocking writes them.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz, process

EPS = 1e-6


def _runs(sorted_key: np.ndarray):
    first = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]]) if len(sorted_key) else np.zeros(0, np.int64)
    return first, np.diff(np.r_[first, len(sorted_key)])


def pool_side_features(pool_row: np.ndarray, p: np.ndarray, prefix: str = "ps_") -> dict:
    """Competition for each pair's pool record among all Source-1 entities that have it as a candidate."""
    n = len(p)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    order = np.lexsort((np.arange(n), -p, pool_row))
    key, ps = pool_row[order], p[order]
    first, counts = _runs(key)
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    rank_sorted = np.arange(n) - np.repeat(first, counts)
    top1 = np.repeat(ps[first], counts)
    second = np.where(counts > 1, ps[np.minimum(first + 1, n - 1)], 0.0)
    top2 = np.repeat(second, counts)
    odds = ps / (1 - ps)
    sum_odds = np.repeat(np.add.reduceat(odds, first), counts) if n else odds
    sum_p = np.repeat(np.add.reduceat(ps, first), counts) if n else ps
    n_ge50 = np.repeat(np.add.reduceat((ps >= 0.5).astype(np.int32), first), counts) if n else np.zeros(0, np.int32)
    best_other = np.where(rank_sorted == 0, top2, top1)
    out = {
        f"{prefix}n_claim": np.repeat(counts, counts),
        f"{prefix}rank": rank_sorted,
        f"{prefix}best_other": best_other,
        f"{prefix}gap_best_other": ps - best_other,
        f"{prefix}sum_other": sum_p - ps,
        f"{prefix}n_other_ge50": n_ge50 - (ps >= 0.5),
        f"{prefix}q": odds / (1.0 + sum_odds),
    }
    return {k: (v[inv].astype(np.float32) if v.dtype.kind == "f" else v[inv].astype(np.int32)) for k, v in out.items()}


def exclusive_q(pool_row: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Exclusivity-normalized probability only (see module docstring)."""
    return pool_side_features(pool_row, p)["ps_q"]


def _within_entity_pairs(s1_row: np.ndarray, max_k: int = 40):
    """All ordered index pairs (i, j), i != j, of rows belonging to the same entity (rows grouped by entity).
    Entities with more than `max_k` candidates only compare their first `max_k` rows (bounded cost)."""
    first, counts = _runs(s1_row)
    counts_c = np.minimum(counts, max_k)
    sizes = counts_c * (counts_c - 1)
    total = int(sizes.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ent = np.repeat(np.arange(len(first)), sizes)
    start = np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
    local = np.arange(total) - start
    k = counts_c[ent]
    a = local // (k - 1)
    b = local % (k - 1)
    b = b + (b >= a)                                     # skip the diagonal
    return first[ent] + a, first[ent] + b


def sibling_features(s1_row: np.ndarray, pool_row: np.ndarray, p: np.ndarray, pool_name: pa.Array, pool_addr: pa.Array,
                     prefix: str = "sib_") -> dict:
    """Agreement of each candidate with the other candidates of its entity (transitivity / triangle evidence).

    pool_name / pool_addr: normalized name / address of every pool record of the partition (indexed by pool_row).
    For candidate i of entity e and every other candidate j of e: s_ij = similarity of the two POOL records
    (name token-set ratio, address token-set ratio; NaN when a side is empty). Returned per candidate i:
      top_p            highest first-stage score among the OTHER candidates of the entity
      sim_top_name/addr similarity of i to that best other candidate (fractured vs cohesive triangle)
      wsim_name/addr   first-stage-score-weighted mean similarity of i to the other candidates
      n_agree          number of other candidates with p >= 0.5 that resemble i (name and address both >= 0.8, or one
                       >= 0.9 with the other side missing)
      n_disagree       number of other candidates with p >= 0.5 that clearly differ from i (name or address < 0.5)
      best_agree_p     highest first-stage score among the candidates that resemble i
    """
    n = len(p)
    p = np.asarray(p, dtype=np.float64)
    i, j = _within_entity_pairs(s1_row)
    out = {f"{prefix}{k}": np.full(n, np.nan, np.float32) for k in ("top_p", "sim_top_name", "sim_top_addr", "wsim_name",
                                                                     "wsim_addr", "best_agree_p")}
    out[f"{prefix}n_agree"] = np.zeros(n, np.int32)
    out[f"{prefix}n_disagree"] = np.zeros(n, np.int32)
    if len(i) == 0:
        return out
    ri, rj = pool_row[i], pool_row[j]
    # similarity is symmetric: compute it once per unordered pair
    lo = i < j
    na, nb = pool_name.take(pa.array(ri[lo])).to_pylist(), pool_name.take(pa.array(rj[lo])).to_pylist()
    aa, ab = pool_addr.take(pa.array(ri[lo])).to_pylist(), pool_addr.take(pa.array(rj[lo])).to_pylist()
    sn_u = process.cpdist(na, nb, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32) / 100.0
    sa_u = process.cpdist(aa, ab, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32) / 100.0
    empty_n = np.fromiter((not x or not y for x, y in zip(na, nb)), bool, count=len(na))
    empty_a = np.fromiter((not x or not y for x, y in zip(aa, ab)), bool, count=len(aa))
    sn_u[empty_n] = np.nan
    sa_u[empty_a] = np.nan
    # map every ordered pair to its unordered similarity
    key_u = np.minimum(i[lo], j[lo]).astype(np.int64) * (n + 1) + np.maximum(i[lo], j[lo])
    key = np.minimum(i, j).astype(np.int64) * (n + 1) + np.maximum(i, j)
    pos = np.searchsorted(key_u, key) if np.all(np.diff(key_u) >= 0) else None
    if pos is None:
        o = np.argsort(key_u)
        key_u, sn_u, sa_u = key_u[o], sn_u[o], sa_u[o]
        pos = np.searchsorted(key_u, key)
    sn, sa = sn_u[pos], sa_u[pos]
    pj = p[j]
    # best other candidate of each row i (ties -> first)
    order = np.lexsort((j, -pj, i))
    io = i[order]
    first, counts = _runs(io)
    best = order[first]
    rows = io[first]
    out[f"{prefix}top_p"][rows] = pj[best]
    out[f"{prefix}sim_top_name"][rows] = sn[best]
    out[f"{prefix}sim_top_addr"][rows] = sa[best]
    w = pj
    for nm, s in (("wsim_name", sn), ("wsim_addr", sa)):
        ok = ~np.isnan(s)
        num = np.bincount(i[ok], weights=(w * s)[ok], minlength=n)
        den = np.bincount(i[ok], weights=w[ok], minlength=n)
        v = np.full(n, np.nan, np.float32)
        np.divide(num, den, out=v, where=den > 0, casting="unsafe")
        out[f"{prefix}{nm}"] = v
    sn0, sa0 = np.nan_to_num(sn, nan=-1.0), np.nan_to_num(sa, nan=-1.0)
    agree = ((sn0 >= 0.8) & (sa0 >= 0.8)) | ((sn0 >= 0.9) & np.isnan(sa)) | ((sa0 >= 0.9) & np.isnan(sn))
    disagree = ((sn0 >= 0) & (sn0 < 0.5)) | ((sa0 >= 0) & (sa0 < 0.5))
    conf = pj >= 0.5
    out[f"{prefix}n_agree"] = np.bincount(i[agree & conf], minlength=n).astype(np.int32)
    out[f"{prefix}n_disagree"] = np.bincount(i[disagree & conf], minlength=n).astype(np.int32)
    bap = np.full(n, -1.0)
    np.maximum.at(bap, i[agree], pj[agree])
    out[f"{prefix}best_agree_p"] = np.where(bap < 0, np.nan, bap).astype(np.float32)
    return out
