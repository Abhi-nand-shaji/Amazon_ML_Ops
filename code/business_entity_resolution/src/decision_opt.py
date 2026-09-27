"""Expected-F0.5-optimal decisions per Source-1 entity (plug-in rule on calibrated match probabilities).

The challenge metric is F0.5 per S1 entity, averaged. For one entity with candidates sorted by probability
q_1 >= q_2 >= ... >= q_n, the prediction "the top k" scores, for true labels y and M true pairs that are not candidates,
    F(k) = 1.25 TP / (1.25 TP + 0.25 FN + FP),  TP = y_1+..+y_k,  FP = k - TP,  FN = y_{k+1}+..+y_n + M,
with the singleton rule F = 1 when nothing is true and nothing is predicted. Treating the labels as independent
Bernoulli(q_i) and M ~ Poisson(lam) (candidate generation misses ~0.12 true pairs per entity), E[F(k)] is computed EXACTLY:
the prefix / suffix positive counts are Poisson-binomial, so their distributions follow from a convolution recursion,
and E[F(k)] = sum_{a,b,m} P(TP_k = a) P(FN_in_k = b) P(M = m) F(a, k - a, b + m).
The chosen k maximizes E[F(k)]. Compared with one global threshold this adapts to each entity: it keeps a 0.6 candidate
when the entity has nothing better (a likely true pair would be lost otherwise), drops it next to three 0.99 candidates
(it would mostly add a false merge), and predicts nothing when an empty list is the better bet (singleton protection).
Vectorized per candidate-count group; exact, no sampling.
"""
from __future__ import annotations

import math

import numpy as np

M_MAX = 4


def _f_table(n: int) -> np.ndarray:
    """F[k, a, c] for k predicted, a true among them, c false negatives in total (0 <= a <= k <= n, c <= n + M_MAX)."""
    k = np.arange(n + 1)[:, None, None]
    a = np.arange(n + 1)[None, :, None]
    c = np.arange(n + 1 + M_MAX)[None, None, :]
    fp = k - a
    num = 1.25 * a
    den = 1.25 * a + 0.25 * c + fp
    f = np.divide(num, den, out=np.zeros(np.broadcast_shapes(k.shape, a.shape, c.shape)), where=den > 0)
    f = np.where((a == 0) & (fp == 0) & (c == 0), 1.0, f)          # nothing true, nothing predicted
    f = np.where(a > k, 0.0, f)                                     # impossible cells
    return f


def _poibin_prefix(q: np.ndarray) -> np.ndarray:
    """P[e, k, a] = P(sum of the first k labels = a), q: (E, n)."""
    E, n = q.shape
    P = np.zeros((E, n + 1, n + 1))
    P[:, 0, 0] = 1.0
    for k in range(1, n + 1):
        qk = q[:, k - 1:k]
        P[:, k, :] = P[:, k - 1, :] * (1 - qk)
        P[:, k, 1:] += P[:, k - 1, :-1] * qk
    return P


def ef05_best_k(q_sorted: np.ndarray, lam: float) -> np.ndarray:
    """Best number of top candidates to predict for each row of q_sorted (E, n), sorted descending."""
    E, n = q_sorted.shape
    P = _poibin_prefix(q_sorted)                                  # prefix counts
    S = _poibin_prefix(q_sorted[:, ::-1])[:, ::-1, :]             # S[e, k, b] = P(sum of labels k+1..n = b)
    pm = np.array([math.exp(-lam) * lam ** m / math.factorial(m) for m in range(M_MAX + 1)])
    pm[-1] += max(0.0, 1.0 - pm.sum())
    F = _f_table(n)                                               # (k, a, c)
    # G[k, a, b] = sum_m pm[m] F[k, a, b + m]
    G = np.zeros((n + 1, n + 1, n + 1))
    for m, w in enumerate(pm):
        G += w * F[:, :, m:m + n + 1]
    EF = np.einsum("eka,ekb,kab->ek", P, S, G, optimize=True)
    return EF.argmax(axis=1), EF


def ef05_decide(entity: np.ndarray, q: np.ndarray, lam: float = 0.12, max_n: int = 40, batch: int = 50_000) -> np.ndarray:
    """Keep-mask under the expected-F0.5-optimal rule. `entity`: group id per row (any order); `q`: probabilities."""
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
    order = np.lexsort((-q, entity))
    ent_sorted = entity[order]
    first = np.flatnonzero(np.r_[True, ent_sorted[1:] != ent_sorted[:-1]])
    counts = np.diff(np.r_[first, len(order)])
    keep_sorted = np.zeros(len(order), dtype=bool)
    q_sorted_all = q[order]
    for n in np.unique(counts):
        grp = first[counts == n]
        n_eff = int(min(n, max_n))
        for s in range(0, len(grp), batch):
            g = grp[s:s + batch]
            qs = q_sorted_all[g[:, None] + np.arange(n_eff)[None, :]]
            k, _ = ef05_best_k(qs, lam)
            pos = g[:, None] + np.arange(n_eff)[None, :]
            keep_sorted[pos] = np.arange(n_eff)[None, :] < k[:, None]
    keep = np.zeros(len(order), dtype=bool)
    keep[order] = keep_sorted
    return keep
