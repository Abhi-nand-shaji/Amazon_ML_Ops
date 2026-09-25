"""Entity-id helpers: 'S2-123456789' <-> compact int64 keys, and ground-truth positive pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

TAG_SCALE = 10**10


def id_to_int(arr) -> np.ndarray:
    """'S2-123456789' -> source_tag * 1e10 + number: a compact int64 key that avoids string
    work in joins. Works on pyarrow arrays and anything pyarrow can wrap."""
    if not isinstance(arr, (pa.Array, pa.ChunkedArray)):
        arr = pa.array(arr, type=pa.string())
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    tag = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64()).to_numpy(zero_copy_only=False)
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy(zero_copy_only=False)
    return tag.astype(np.int64) * TAG_SCALE + num


def load_truth(gt_path, s1_ids=None) -> dict:
    """{s1_entity_id: set(matched ids)} from the ground-truth TSV, optionally restricted to `s1_ids`."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    if s1_ids is not None:
        gt = gt[gt["source1_entity_id"].isin(set(s1_ids))]
    return {s1: (set(m.split(",")) if m.strip() else set())
            for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def positive_pairs(truth: dict) -> pd.DataFrame:
    """int64 (s1, cand) frame with one row per true (S1, S2/S3) pair."""
    s1s, cands = [], []
    for s1, ids in truth.items():
        for c in ids:
            s1s.append(s1)
            cands.append(c)
    if not s1s:
        return pd.DataFrame({"s1": np.zeros(0, np.int64), "cand": np.zeros(0, np.int64), "label": np.zeros(0, np.int8)})
    return pd.DataFrame({"s1": id_to_int(s1s), "cand": id_to_int(cands), "label": np.int8(1)})


def label_pairs(s1_entity_id, cand_entity_id, truth: dict) -> np.ndarray:
    """int8 array, 1 where (s1, cand) is a true match. Vectorized (int64 hash join)."""
    pairs = pd.DataFrame({"s1": id_to_int(s1_entity_id), "cand": id_to_int(cand_entity_id)})
    pos = positive_pairs(truth)
    merged = pairs.merge(pos, how="left", on=["s1", "cand"])
    assert len(merged) == len(pairs), "duplicate ground-truth pairs would duplicate candidate rows"
    return merged["label"].fillna(0).to_numpy(dtype=np.int8)
