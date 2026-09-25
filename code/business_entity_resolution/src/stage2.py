"""Second stage: re-decide each candidate given the first-stage scores of ALL candidates of its S1 entity.

A pair's first-stage probability says how much the pair itself looks like a match; whether the entity has a
match at all (singleton control), which candidate is the best, and how many matches to expect depend on the
*other* candidates: "0.55 with everything else below 0.05" is a different situation from "0.55 next to a 0.97".
These entity-context features are derived from the first-stage probabilities only, so at test scale they can be
computed from the saved scores (no re-featurization); cheap blocker columns stored with the candidates
(ranking score, overlap of name / address, number of retrieval rules that fired) are appended.

Rows must be grouped by entity (contiguous runs), which is how blocking writes them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUTS = (0.1, 0.3, 0.5, 0.7, 0.9)
BLOCKER_COLS = ["rank_score", "ov_name", "ov_addr", "n_strategies"]


def _runs(group: np.ndarray):
    first = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    return first, np.diff(np.r_[first, len(group)])


def _group_stats(group: np.ndarray, p: np.ndarray, prefix: str) -> dict:
    """Rank / best / second best / sums / counts of p inside each run of equal `group` values."""
    n = len(p)
    order = np.lexsort((np.arange(n), -p, group))              # by group, best first (stable on ties)
    g_sorted, p_sorted = group[order], p[order]
    first, counts = _runs(g_sorted)
    rank_sorted = np.arange(n) - np.repeat(first, counts)
    rank = np.empty(n, np.int32)
    rank[order] = rank_sorted
    top = []
    for k in range(3):                                           # k-th best p of the group (0 when the group is shorter)
        idx = np.minimum(first + k, n - 1)
        val = np.where(counts > k, p_sorted[idx], 0.0)
        top.append(np.repeat(val, counts)[np.argsort(order, kind="stable")])
    out = {
        f"{prefix}rank": rank, f"{prefix}top1": top[0], f"{prefix}top2": top[1], f"{prefix}top3": top[2],
        f"{prefix}sum": np.repeat(np.add.reduceat(p_sorted, first), counts)[np.argsort(order, kind="stable")],
        f"{prefix}count": np.repeat(counts, counts)[np.argsort(order, kind="stable")].astype(np.int32),
    }
    for c in CUTS:
        n_above = np.add.reduceat((p_sorted >= c).astype(np.int32), first)
        out[f"{prefix}n_ge_{int(c * 100)}"] = np.repeat(n_above, counts)[np.argsort(order, kind="stable")]
    # best OTHER candidate: the top1 for everyone except the best itself, which sees top2
    out[f"{prefix}best_other"] = np.where(rank == 0, out[f"{prefix}top2"], out[f"{prefix}top1"])
    out[f"{prefix}gap_to_best_other"] = p - out[f"{prefix}best_other"]
    return out


def entity_context(p: np.ndarray, is_s2: np.ndarray, group: np.ndarray, blocker: dict | None = None) -> pd.DataFrame:
    """Stage-2 feature frame (one row per pair, same order). `group` identifies the entity of each row."""
    p = np.asarray(p, dtype=np.float64)
    f = {"p1": p.astype(np.float32), "is_s2": np.asarray(is_s2, dtype=np.int8)}
    f.update({k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in _group_stats(group, p, "e_").items()})
    # the same statistics inside the pair's own source (S2 vs S3 candidates of the entity)
    src_group = group.astype(np.int64) * 2 + np.asarray(is_s2, dtype=np.int64)
    order = np.argsort(src_group, kind="stable")                 # runs of (entity, source)
    inv = np.argsort(order, kind="stable")
    s = _group_stats(src_group[order], p[order], "s_")
    f.update({k: (v[inv].astype(np.float32) if v.dtype.kind == "f" else v[inv]) for k, v in s.items()})
    if blocker:
        for c in BLOCKER_COLS:
            f[c] = np.asarray(blocker[c], dtype=np.float32)
    return pd.DataFrame(f)
