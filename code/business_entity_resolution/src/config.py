"""Shared paths and constants for the entity-resolution pipeline.

All paths default to the conventional challenge layout, resolved relative to
the ``student_resource/`` repository root. Every entry point accepts
``--root`` to override this if the package is relocated.
"""
from __future__ import annotations

from pathlib import Path

# code/business_entity_resolution/src/config.py -> student_resource/
DEFAULT_ROOT = Path(__file__).resolve().parents[3]

RANDOM_SEED = 42


class Paths:
    def __init__(self, root: str | Path | None = None, tag: str = ""):
        """`tag` isolates a whole pipeline variant: every derived artifact (candidates, features, scores,
        model, decision config, reports) gets the suffix ``_<tag>`` so experiments can live side by side."""
        self.root = Path(root).resolve() if root else DEFAULT_ROOT
        self.tag = tag
        self.suffix = f"_{tag}" if tag else ""
        self.dataset = self.root / "dataset"
        self.train_dir = self.dataset / "train"
        self.test_dir = self.dataset / "test"
        self.output_dir = self.root / "output"
        self.code_dir = self.root / "code" / "business_entity_resolution"
        self.artifacts_dir = self.code_dir / "artifacts"
        self.cache_dir = self.artifacts_dir / "cache"
        self.reports_dir = self.code_dir / "reports"
        for d in (self.output_dir, self.artifacts_dir, self.cache_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # raw inputs
    def raw(self, split: str, source: str) -> Path:
        d = self.train_dir if split == "train" else self.test_dir
        return d / f"{split}_{source}.tsv"

    def ground_truth(self) -> Path:
        return self.train_dir / "train_ground_truth.tsv"

    # cached normalized parquet per split/source
    def normalized_cache(self, split: str, source: str) -> Path:
        return self.cache_dir / f"{split}_{source}_normalized.parquet"

    # pipeline stages (all honour the tag suffix)
    def shortlist_dir(self, split: str) -> Path:
        """run_blocking.py output: retrieval + learned ranker, the best 30 pairs per S1 entity with their ranker probability."""
        return self.cache_dir / f"shortlist_{split}{self.suffix}"

    def candidates_dir(self, split: str) -> Path:
        """run_candidate_cut.py output: the final candidate set -- exactly the pairs the matcher scores (candidate_pairs.tsv)."""
        return self.cache_dir / f"candidates_{split}{self.suffix}"

    def features_dir(self, split: str) -> Path:
        return self.cache_dir / f"features_{split}{self.suffix}"

    def scores_dir(self, split: str) -> Path:
        return self.cache_dir / f"scores_{split}{self.suffix}"

    def ranker_path(self) -> Path:
        return self.artifacts_dir / f"ranker{self.suffix}.txt"

    def filter_model_path(self) -> Path:
        return self.artifacts_dir / f"candidate_filter{self.suffix}.txt"

    def filter_config_path(self) -> Path:
        return self.artifacts_dir / f"candidate_filter{self.suffix}.json"

    def filter_oof_path(self) -> Path:
        """Out-of-fold filter probabilities for every pair of the training shortlist (row order of the shortlist files)."""
        return self.cache_dir / f"filter_oof_train{self.suffix}.parquet"

    def model_path(self) -> Path:
        return self.artifacts_dir / f"lgbm_model{self.suffix}.txt"

    def feature_list_path(self) -> Path:
        return self.artifacts_dir / f"feature_columns{self.suffix}.json"

    def threshold_path(self) -> Path:
        return self.artifacts_dir / f"decision_config{self.suffix}.json"

    def report_path(self, stem: str, ext: str) -> Path:
        return self.reports_dir / f"{stem}{self.suffix}.{ext}"


# ---------------------------------------------------------------------------
# Blocking configuration
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockingConfig:
    """Candidate generation is a union of (a) rare-token lookups and (b) exact
    keys, executed on integer token codes (see blocking.py).

    Why "K rarest tokens" instead of "any shared token": at ~10M pool records
    a single moderately common token (df in the thousands) would pull in
    thousands of candidates per entity and the pre-truncation pair count would
    reach billions. Each entity instead looks up only its K rarest tokens per
    field (those that exist in the pool and have df <= df_cap), which keeps
    the expansion small while true matches -- which usually share several
    informative tokens across name and address -- still surface.
    """
    df_cap: int = 1500              # tokens with pool document frequency above this are never used to look up candidates
    k_rare_name: int = 4            # max rarest name tokens per S1 entity used for lookup
    k_rare_addr: int = 5            # max rarest address tokens per S1 entity used for lookup
    exact_key_cap: int = 300        # exact-name / compact-name / postal groups larger than this are skipped
    max_candidates: int = 60        # per-entity cap after ranking
    expansion_budget: int = 25_000_000  # max (s1,pool) expansions materialized per batch (bounds memory)

    # ---- additional retrieval key families (0 / () = family disabled) -----------------------------
    # Adjacent-token pairs ("bigrams", order-insensitive) are far more specific than single tokens, so
    # they retrieve true matches whose individual tokens are all common (e.g. "ekta vihar", "805 union").
    cap_addr_bigram: int = 0        # address bigrams with pool df above this are never used
    cap_name_bigram: int = 0
    # Prefix of the compact core name (name without legal suffixes, spaces removed): catches
    # concatenated / domain-style names ("gurgaontechnologiesclinic.com" vs "Gurgaon Technologies Clinic").
    prefix_lens: tuple = ()
    cap_prefix: int = 0

    # ---- ranking ------------------------------------------------------------------------------------
    # rescore=True ranks each entity's candidates by full IDF-weighted token overlap (common tokens
    # included, computed on padded token matrices) instead of by the rare-token score, which ties
    # massively when a name consists only of common words.
    rescore: bool = False
    dense_width_name: int = 8
    dense_width_addr: int = 16


# Chosen from a retrieval-cap sweep on the training sample (India, 4,000 entities): df_cap 300 with K=3/4 rare
# tokens retrieves 95.7% of all true pairs at 444 candidates per entity, versus 96.4% at 1,273 for
# df_cap 1500 / K=4/5 -- a 65% cut in volume for 0.7 points of retrieval recall (worth ~0.15 points of F0.5).
# max_candidates=30 is applied AFTER the learned ranker: at 20-30 per entity it keeps ~96% of all true pairs.
DEFAULT_BLOCKING = BlockingConfig(
    df_cap=300, k_rare_name=3, k_rare_addr=4, max_candidates=30,
    cap_addr_bigram=300, cap_name_bigram=300, prefix_lens=(8,), cap_prefix=300, rescore=True,
)

# Candidate filter (candidate_filter.py, train_candidate_filter.py, run_candidate_cut.py): the shortlist of 30 is cut to the
# pairs whose filter probability clears a floor. The floor is the largest value on the grid that still keeps at least
# (1 - FILTER_RECALL_BUDGET) of the shortlisted true pairs of the matcher's validation entities.
FILTER_FOLDS = 5
FILTER_RECALL_BUDGET = 0.005
FILTER_FLOOR_GRID = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)

# Negative sampling cap per entity when building the *training* dataset for
# the classifier (inference always scores every candidate; this only bounds
# how many negatives feed model fitting).
MAX_NEGATIVES_PER_ENTITY_TRAIN = 20

TFIDF_CHAR_NGRAM_RANGE = (2, 4)
TFIDF_MAX_FEATURES = 60_000
