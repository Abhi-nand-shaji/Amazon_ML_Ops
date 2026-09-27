"""Second-stage ("stacker") feature frame: pairwise features + first-stage score + cross-encoder score + graph features.

Graph features (collective.py, stage2.py), all derived from the FIRST-stage score p1 of the whole candidate graph:
  e_* / s_*   entity context: the pair's rank and the score distribution among all candidates of its S1 entity
              (overall / within the same source)
  ps_*        pool-side competition for the pair's S2/S3 record among all S1 entities that have it as a candidate
              (exclusivity), incl. the exclusivity-normalized probability ps_q
  sib_*       triangle / transitivity evidence: agreement of the candidate with the entity's other candidates
  as_*        address structure: house number and street name compared separately (address_structure.py)
Rows must be grouped by entity (s1_row runs), and the pool-side arrays must come from the COMPLETE candidate graph of the
country partition (every S1 entity), which is what run_world_score.py (training) and the test scores provide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from address_structure import structure_features
from collective import pool_side_features, sibling_features
from pipeline import chunk_slices
from stage2 import entity_context

CE_COL = "p_ce"


def pool_side_for_rows(all_s1_row: np.ndarray, all_pool_row: np.ndarray, all_p1: np.ndarray,
                       s1_row: np.ndarray, pool_row: np.ndarray) -> pd.DataFrame:
    """Pool-side features computed over the complete graph (all_*), returned for the requested (s1_row, pool_row) rows."""
    ps = pool_side_features(all_pool_row, all_p1)
    key_all = all_s1_row.astype(np.int64) << 32 | all_pool_row.astype(np.int64)
    key = s1_row.astype(np.int64) << 32 | pool_row.astype(np.int64)
    order = np.argsort(key_all, kind="stable")
    pos = np.searchsorted(key_all[order], key)
    pos = np.minimum(pos, len(order) - 1)
    idx = order[pos]
    if not np.array_equal(key_all[idx], key):
        raise RuntimeError("pool-side features: some requested pairs are not in the complete candidate graph")
    return pd.DataFrame({k: v[idx] for k, v in ps.items()})


def graph_features(s1_row: np.ndarray, pool_row: np.ndarray, p1: np.ndarray, is_s2: np.ndarray, pool_name: pa.Array,
                   pool_addr: pa.Array, pool_side: pd.DataFrame, chunk_rows: int = 400_000,
                   s1_addr_raw: pa.Array | None = None, pool_addr_raw: pa.Array | None = None) -> pd.DataFrame:
    """Entity-context + sibling + pool-side (+ address-structure, when the ORIGINAL addresses of the partition's S1 and
    pool records are given) features for rows grouped by entity (same order as the inputs)."""
    parts = []
    for a, b in chunk_slices(s1_row, chunk_rows):
        ctx = entity_context(p1[a:b], is_s2[a:b], s1_row[a:b]).drop(columns=["p1", "is_s2"])
        sib = sibling_features(s1_row[a:b], pool_row[a:b], p1[a:b], pool_name, pool_addr)
        for k, v in sib.items():
            ctx[k] = v
        if s1_addr_raw is not None and pool_addr_raw is not None:
            for k, v in structure_features(s1_addr_raw, pool_addr_raw, s1_row[a:b], pool_row[a:b]).items():
                ctx[k] = v
        parts.append(ctx)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for c in pool_side.columns:
        out[c] = pool_side[c].to_numpy()
    return out
