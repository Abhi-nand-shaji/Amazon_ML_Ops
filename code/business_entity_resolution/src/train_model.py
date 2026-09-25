"""Helpers for training the LightGBM matcher and choosing decision thresholds
against the challenge's true entity-level F0.5 metric."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED

ID_COLS = ["s1_entity_id", "cand_entity_id"]
# Columns that must never reach the feature matrix: ids and row positions identify a pair, and
# "label" is the answer (if it leaked in the model would train perfectly and then crash at
# inference, where real feature frames have no label).
NON_FEATURE_COLS = set(ID_COLS) | {"label", "s1_row", "pool_row"}
# Script codes come from a FIXED vocabulary (features.SCRIPT_VOCAB), so they are stable
# integer categories across train and test. `country` is intentionally not a feature.
CATEGORICAL_COLS = ["name_script_2"]
# Features whose scale or distribution depends on the SIZE of the country pool: IDF sums grow with log(pool size),
# rare-token counts and candidate counts depend on df caps relative to the pool. The test set adds France with a much
# smaller pool (~1.4M vs 4-6M), so these could shift; `--robust` in run_train.py drops them (validated by ablation).
POOL_SIZE_DEPENDENT = {"idf_name", "idf_addr", "block_score", "n_name_shared", "n_addr_shared", "n_candidates_for_entity",
                       "pool_name_freq", "pool_core_freq", "pool_addr_freq", "s1_core_freq", "s1_addr_freq"}


def lgb_params(threads: int, leaves: int = 127, lr: float = 0.05, seed: int = RANDOM_SEED) -> dict:
    """Hyper-parameters of the pairwise matcher (shared by the main run and the stage-2 out-of-fold models)."""
    return {"objective": "binary", "metric": ["binary_logloss", "auc"], "learning_rate": lr, "num_leaves": leaves,
            "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "num_threads": threads, "seed": seed, "verbosity": -1}


# The S1 name is 100% ASCII in training but 2.4% non-ASCII in test (accented French names), so any feature that depends on
# the S1 script is undefined territory at test time: the model can neither have learned it nor be trusted on it.
S1_SCRIPT_DEPENDENT = {"name_script_1", "name_script_match"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and c not in S1_SCRIPT_DEPENDENT]


def entity_split(s1_ids, val_frac: float = 0.15, seed: int = RANDOM_SEED):
    """Split S1 ENTITIES (not pairs) into train / validation id sets."""
    s1_ids = np.array(sorted(set(s1_ids)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(s1_ids))
    n_val = int(len(s1_ids) * val_frac)
    return set(s1_ids[shuffled[n_val:]]), set(s1_ids[shuffled[:n_val]])


def entity_split3(s1_ids, val_frac: float = 0.15, hold_frac: float = 0.15, seed: int = RANDOM_SEED):
    """Train / validation / hold-out ENTITY split. Validation entities drive every decision (early stopping,
    thresholds, rule selection); hold-out entities are only scored once at the end, so their F0.5 is an
    unbiased estimate of what the selection procedure delivers."""
    s1_ids = np.array(sorted(set(s1_ids)))
    perm = np.random.default_rng(seed).permutation(len(s1_ids))
    n_val, n_hold = int(len(s1_ids) * val_frac), int(len(s1_ids) * hold_frac)
    return (set(s1_ids[perm[n_val + n_hold:]]), set(s1_ids[perm[:n_val]]), set(s1_ids[perm[n_val:n_val + n_hold]]))


def _rank_within(sorted_group: np.ndarray) -> np.ndarray:
    first = np.flatnonzero(np.r_[True, sorted_group[1:] != sorted_group[:-1]])
    counts = np.diff(np.r_[first, len(sorted_group)])
    return np.arange(len(sorted_group)) - np.repeat(first, counts)


def negative_sampling_mask(s1_row: np.ndarray, label: np.ndarray, prelim: np.ndarray,
                           n_hard: int, n_random: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Rows to train on: every positive, the `n_hard` most similar negatives of each entity (by
    prelim_score), plus `n_random` random other negatives per entity.

    The random part matters: training only on the hardest negatives would show the model nothing
    about easy candidates, yet inference scores every candidate of every entity."""
    keep = label == 1
    neg = np.flatnonzero(label == 0)
    if len(neg) == 0:
        return keep
    order = neg[np.lexsort((-prelim[neg], s1_row[neg]))]           # by entity, hardest first
    rank = _rank_within(s1_row[order])
    keep[order[rank < n_hard]] = True
    rest = order[rank >= n_hard]
    if n_random and len(rest):
        rng = np.random.default_rng(seed)
        r = rng.random(len(rest))
        o2 = np.lexsort((r, s1_row[rest]))                          # random order inside each entity
        rest = rest[o2]
        keep[rest[_rank_within(s1_row[rest]) < n_random]] = True
    return keep


DEFAULT_GRID = np.round(np.arange(0.05, 0.9901, 0.01), 4)


def search_threshold(scorer, scores, grid=None, extra_mask=None):
    """Best single threshold on `scores` under the true macro-F0.5 metric.
    `scorer` is an evaluation.FastScorer; `extra_mask` (optional) is ANDed in.
    Returns (best_threshold, best_f05, table)."""
    grid = DEFAULT_GRID if grid is None else grid
    scores = np.asarray(scores)
    rows, best = [], (-1.0, None)
    for t in grid:
        keep = scores >= t
        if extra_mask is not None:
            keep = keep & extra_mask
        f = scorer.score(keep)
        rows.append({"threshold": float(t), "macro_f05": f})
        if f > best[0]:
            best = (f, float(t))
    return best[1], best[0], pd.DataFrame(rows)


def entity_max(scores: np.ndarray, ent_idx: np.ndarray, n_entities: int) -> np.ndarray:
    """Per-row: the best score among all candidates of the row's entity."""
    mx = np.full(n_entities, -np.inf)
    np.maximum.at(mx, ent_idx, scores)
    return mx[ent_idx]
