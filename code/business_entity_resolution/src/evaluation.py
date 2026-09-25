"""The challenge's exact scoring metric: per-entity F0.5, macro-averaged.

Implemented as a vectorized outer-join rather than a Python loop per entity,
since validation can involve hundreds of thousands of Source-1 entities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pairs_from_dict(d: dict) -> pd.DataFrame:
    rows = [(s1, cid) for s1, ids in d.items() for cid in ids]
    if not rows:
        return pd.DataFrame(columns=["s1", "cid"])
    return pd.DataFrame(rows, columns=["s1", "cid"])


def macro_f05(pred: dict, truth: dict, required_ids) -> tuple[float, pd.DataFrame]:
    """pred, truth: {s1_entity_id: set/list of matched ids}. required_ids: all
    S1 ids that must be scored (missing from the dict = empty prediction).
    Returns (macro_f05, per_entity_dataframe[s1, tp, fp, fn, precision, recall, f05, truth_size, pred_size]).

    Convenience wrapper for one-off calls. Threshold-search loops that score
    dozens of decision rules should use ``FastScorer`` instead: rebuilding a
    Python dict (and re-exploding it into pairs) on every grid point is
    needless overhead at millions-of-rows scale.
    """
    pred_pairs = _pairs_from_dict(pred)
    truth_pairs = _pairs_from_dict(truth)
    return score_pairs(pred_pairs, truth_pairs, required_ids)


def score_pairs(pred_pairs: pd.DataFrame, truth_pairs: pd.DataFrame, required_ids) -> tuple[float, pd.DataFrame]:
    """Same computation as macro_f05, but takes pre-built (s1, cid) pair
    frames directly -- the shared core so callers can build ``truth_pairs``
    once and reuse it across many threshold values."""
    required_ids = pd.Index(required_ids).unique()
    if len(pred_pairs) or len(truth_pairs):
        merged = pred_pairs.merge(truth_pairs, on=["s1", "cid"], how="outer", indicator=True)
        tp = merged[merged["_merge"] == "both"].groupby("s1").size()
        fp = merged[merged["_merge"] == "left_only"].groupby("s1").size()
        fn = merged[merged["_merge"] == "right_only"].groupby("s1").size()
    else:
        tp = fp = fn = pd.Series(dtype="int64")

    stats = pd.DataFrame(index=required_ids)
    stats["tp"] = tp.reindex(required_ids).fillna(0).astype(int)
    stats["fp"] = fp.reindex(required_ids).fillna(0).astype(int)
    stats["fn"] = fn.reindex(required_ids).fillna(0).astype(int)
    stats["truth_size"] = stats["tp"] + stats["fn"]
    stats["pred_size"] = stats["tp"] + stats["fp"]

    denom_p = stats["tp"] + stats["fp"]
    stats["precision"] = np.where(denom_p > 0, stats["tp"] / denom_p.replace(0, np.nan), 0.0)
    denom_r = stats["tp"] + stats["fn"]
    stats["recall"] = np.where(denom_r > 0, stats["tp"] / denom_r.replace(0, np.nan), 1.0)
    # true singleton (truth empty): recall is vacuous; score is 1.0 iff pred also empty
    is_singleton = stats["truth_size"] == 0
    beta2 = 0.25
    num = (1 + beta2) * stats["precision"] * stats["recall"]
    den = beta2 * stats["precision"] + stats["recall"]
    f05 = np.where(den > 0, num / den.replace(0, np.nan), 0.0)
    f05 = np.where(is_singleton, np.where(stats["pred_size"] == 0, 1.0, 0.0), f05)
    stats["f05"] = f05
    stats.loc[is_singleton, "precision"] = np.where(stats.loc[is_singleton, "pred_size"] == 0, np.nan, 0.0)
    stats.loc[is_singleton, "recall"] = np.nan

    return float(stats["f05"].mean()), stats.reset_index(names="s1_entity_id")


class FastScorer:
    """Threshold-independent state precomputed once so that scoring any
    decision rule (a boolean "keep this candidate row" mask) costs a couple of
    ``np.bincount`` calls instead of rebuilding dicts / merging pair frames.

    Requires each candidate row to carry its ground-truth label
    (label == 1 iff (s1, candidate) is a true match). True matches that were
    never candidates (blocking failures) need no special handling: they are
    counted as false negatives through ``truth_size - tp``.

    Uses the closed form  F0.5 = 1.25*tp / (1.25*tp + 0.25*fn + fp)  (identical
    to the challenge's 1.25*P*R / (0.25*P + R) whenever tp+fp>0), with the
    challenge's singleton rule: no true matches -> 1.0 iff nothing predicted.
    """

    def __init__(self, s1_ids, labels, truth: dict, required_ids):
        self.required = pd.Index(required_ids).unique()
        self.n = len(self.required)
        idx = self.required.get_indexer(pd.Index(s1_ids))
        if (idx < 0).any():
            raise ValueError("candidate rows reference S1 ids that are not in required_ids")
        self.ent_idx = idx
        self.label = np.asarray(labels, dtype=np.float64)
        self.truth_size = np.fromiter((len(truth.get(s, ())) for s in self.required), dtype=np.int64, count=self.n)

    def _counts(self, keep):
        keep = np.asarray(keep, dtype=bool)
        idx = self.ent_idx[keep]
        pred_size = np.bincount(idx, minlength=self.n).astype(np.int64)
        tp = np.bincount(idx, weights=self.label[keep], minlength=self.n).astype(np.int64)
        return tp, pred_size

    def _f05(self, tp, pred_size):
        fp = pred_size - tp
        fn = self.truth_size - tp
        den = 1.25 * tp + 0.25 * fn + fp
        f05 = np.divide(1.25 * tp, den, out=np.zeros(self.n), where=den > 0)
        return np.where(self.truth_size == 0, (pred_size == 0).astype(float), f05)

    def score(self, keep) -> float:
        tp, pred_size = self._counts(keep)
        return float(self._f05(tp, pred_size).mean())

    def stats(self, keep) -> pd.DataFrame:
        """Per-entity table with the same columns as ``score_pairs`` returns."""
        tp, pred_size = self._counts(keep)
        fp = pred_size - tp
        fn = self.truth_size - tp
        singleton = self.truth_size == 0
        precision = np.where(pred_size > 0, tp / np.maximum(pred_size, 1), 0.0)
        recall = np.where(self.truth_size > 0, tp / np.maximum(self.truth_size, 1), np.nan)
        precision = np.where(singleton & (pred_size == 0), np.nan, precision)
        return pd.DataFrame({
            "s1_entity_id": self.required, "tp": tp, "fp": fp, "fn": fn,
            "truth_size": self.truth_size, "pred_size": pred_size,
            "precision": precision, "recall": recall, "f05": self._f05(tp, pred_size),
        })


def summarize(stats: pd.DataFrame) -> dict:
    is_singleton = stats["truth_size"] == 0
    non_singleton = ~is_singleton
    out = {
        "n_entities": len(stats),
        "macro_f05": float(stats["f05"].mean()),
        "n_singletons_true": int(is_singleton.sum()),
        "singleton_accuracy": float((stats.loc[is_singleton, "pred_size"] == 0).mean()) if is_singleton.any() else float("nan"),
        "false_merge_rate_on_singletons": float((stats.loc[is_singleton, "pred_size"] > 0).mean()) if is_singleton.any() else float("nan"),
        "mean_precision_non_singleton": float(stats.loc[non_singleton, "precision"].mean()) if non_singleton.any() else float("nan"),
        "mean_recall_non_singleton": float(stats.loc[non_singleton, "recall"].mean()) if non_singleton.any() else float("nan"),
        "mean_f05_non_singleton": float(stats.loc[non_singleton, "f05"].mean()) if non_singleton.any() else float("nan"),
        "mean_pred_size": float(stats["pred_size"].mean()),
        "mean_truth_size": float(stats["truth_size"].mean()),
        "total_tp": int(stats["tp"].sum()),
        "total_fp": int(stats["fp"].sum()),
        "total_fn": int(stats["fn"].sum()),
    }
    return out
