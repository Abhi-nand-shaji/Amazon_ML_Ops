"""Pool-side (exclusivity) features for v4 candidate sets = first-hop candidates + new pairs (dense / second-hop).

Competition for a pool record is always measured against its FIRST-HOP claimants (the S1 entities whose blocker
candidates include it), plus the pair itself. Other entities' new pairs never count as competitors: in training only the
stack / val / hold entities get new pairs while at test time every entity does, so counting them would make the feature
mean different things in training and at test time. For a first-hop pair the result equals collective.pool_side_features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def pool_side_v4(fh_pool_row: np.ndarray, fh_p1: np.ndarray, pool_row: np.ndarray, p1: np.ndarray, is_new: np.ndarray,
                 prefix: str = "ps_") -> pd.DataFrame:
    """fh_*: every first-hop pair of the partition (pool row, first-stage probability).
    pool_row / p1 / is_new: the rows to describe (first-hop or new)."""
    fh_p = np.clip(np.asarray(fh_p1, np.float64), EPS, 1 - EPS)
    p = np.clip(np.asarray(p1, np.float64), EPS, 1 - EPS)
    is_new = np.asarray(is_new, bool)
    n_pool = int(max(fh_pool_row.max(initial=0), pool_row.max(initial=0))) + 1
    cnt = np.bincount(fh_pool_row, minlength=n_pool)
    s = np.bincount(fh_pool_row, weights=fh_p, minlength=n_pool)
    so = np.bincount(fh_pool_row, weights=fh_p / (1 - fh_p), minlength=n_pool)
    ge = np.bincount(fh_pool_row, weights=(fh_p >= 0.5).astype(np.float64), minlength=n_pool)
    # top-2 first-hop probabilities per pool record
    order = np.lexsort((-fh_p, fh_pool_row))
    r_sorted, p_sorted = fh_pool_row[order], fh_p[order]
    first = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]]) if len(r_sorted) else np.zeros(0, np.int64)
    top1 = np.zeros(n_pool)
    top2 = np.zeros(n_pool)
    top1[r_sorted[first]] = p_sorted[first]
    sizes = np.diff(np.r_[first, len(r_sorted)])
    ok = sizes >= 2
    top2[r_sorted[first[ok]]] = p_sorted[first[ok] + 1]
    # number of first-hop claimants with a HIGHER probability (composite sorted key: row + p / 2)
    key = r_sorted.astype(np.float64) + p_sorted * 0.5
    key_sorted = np.sort(key)
    q_hi = np.searchsorted(key_sorted, pool_row.astype(np.float64) + 1.0 - 1e-12, side="left")
    q_gt = np.searchsorted(key_sorted, pool_row.astype(np.float64) + p * 0.5, side="right")
    n_greater = (q_hi - q_gt).astype(np.int32)
    r = pool_row
    c, sm, sodd, g50, t1, t2 = cnt[r], s[r], so[r], ge[r], top1[r], top2[r]
    odds = p / (1 - p)
    # first-hop pair: it is itself one of the claimants -> exclude itself; new pair: an extra claimant
    self_is_top = (~is_new) & (p >= t1 - 1e-12)
    best_other = np.where(is_new, t1, np.where(self_is_top, t2, t1))
    n_claim = np.where(is_new, c + 1, c)
    sum_other = np.where(is_new, sm, sm - p)
    n_other_ge50 = np.where(is_new, g50, g50 - (p >= 0.5))
    q = np.where(is_new, odds / (1.0 + sodd + odds), odds / (1.0 + sodd))
    return pd.DataFrame({
        f"{prefix}n_claim": n_claim.astype(np.int32), f"{prefix}rank": n_greater,
        f"{prefix}best_other": best_other.astype(np.float32), f"{prefix}gap_best_other": (p - best_other).astype(np.float32),
        f"{prefix}sum_other": np.maximum(sum_other, 0).astype(np.float32), f"{prefix}n_other_ge50": np.maximum(n_other_ge50, 0).astype(np.int32),
        f"{prefix}q": q.astype(np.float32),
    })
