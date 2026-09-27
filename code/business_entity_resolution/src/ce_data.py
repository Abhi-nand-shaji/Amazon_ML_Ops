"""Text of candidate pairs for the cross-encoder, addressed exactly like the rest of the pipeline (partition row positions).

Records are serialized with Arrow compute kernels (vectorized, no per-record Python), and only the rows a pair list
actually references are materialized as Python strings for the tokenizer.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pipeline import load_meta, load_partition

TEXT_COLS = ["entity_id", "name_original", "address_original"]


def serialize_table(tbl: pa.Table) -> pa.Array:
    """``name: <name> address: <address>`` (lower-cased, trimmed; missing -> empty) for every row of a record table."""
    def clean(col):
        arr = tbl[col].combine_chunks() if isinstance(tbl[col], pa.ChunkedArray) else tbl[col]
        return pc.utf8_lower(pc.utf8_trim_whitespace(pc.fill_null(arr.cast(pa.string()), "")))
    return pc.binary_join_element_wise("name: ", clean("name_original"), " address: ", clean("address_original"), "")


class PartitionText:
    """Serialized S1 / pool records of one country partition, in the blocker's row order."""

    def __init__(self, paths, split: str, country: str):
        meta = load_meta(paths, split)
        s1, pool, _ = load_partition(paths, split, country, meta, s1_cols=TEXT_COLS, pool_cols=TEXT_COLS + ["source"],
                                     with_full=False)
        self.s1_id = s1["entity_id"].combine_chunks()
        self.pool_id = pool["entity_id"].combine_chunks()
        self.s1_txt = serialize_table(s1)
        self.pool_txt = serialize_table(pool)

    def check(self, s1_row: np.ndarray, pool_row: np.ndarray, s1_ids, cand_ids, n_probe: int = 2000) -> None:
        if len(s1_row) == 0:
            return
        probe = np.linspace(0, len(s1_row) - 1, num=min(len(s1_row), n_probe)).astype(np.int64)
        s1_ids = s1_ids.combine_chunks() if isinstance(s1_ids, pa.ChunkedArray) else pa.array(s1_ids)
        cand_ids = cand_ids.combine_chunks() if isinstance(cand_ids, pa.ChunkedArray) else pa.array(cand_ids)
        ok = (self.s1_id.take(pa.array(s1_row[probe])).equals(s1_ids.take(pa.array(probe)))
              and self.pool_id.take(pa.array(pool_row[probe])).equals(cand_ids.take(pa.array(probe))))
        if not ok:
            raise RuntimeError("partition row order does not match the pair list (cache or sampling changed?)")

    def pairs(self, s1_row: np.ndarray, pool_row: np.ndarray) -> tuple[list[str], list[str]]:
        return (self.s1_txt.take(pa.array(s1_row)).to_pylist(), self.pool_txt.take(pa.array(pool_row)).to_pylist())
