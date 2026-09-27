"""Second-hop (transitive) candidate retrieval: search from an entity's CONFIDENT matches, not only from its S1 record.

Measured on validation / hold-out entities: 3.4% of true pairs never become candidates, yet 95% of those missed records
belong to an entity with at least one FOUND true record, and 71% of them closely resemble such a found sibling (name
token-set >= 0.9 or address >= 0.85) -- noisy copies of one business share details the S1 record lacks ("block a 909
ground fl" in two copies, absent from the S1 address), and a copy whose name is written in another script still carries
the same plot / sector numbers as a Latin-script sibling.

So each entity's confident matches ("anchors") are used as queries against the pool with the SAME retrieval engine as
the S1 records (blocking.PoolIndex: rare tokens, exact keys, bigrams, prefixes, learned ranker); what they retrieve --
other than the entity's existing candidates -- becomes a second-hop candidate of the entity, with the anchor's ranker
score and confidence as evidence. Simulated on val / hold (anchors = stacker probability >= 0.9): up to 3 anchors per
entity recover roughly a third of the missed true pairs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa

from blocking import PoolIndex


def _key(a, b):
    return np.asarray(a, dtype=np.int64) << 32 | np.asarray(b, dtype=np.int64)


def second_hop_pairs(index: PoolIndex, pool_tbl: pa.Table, anchor_entity: np.ndarray, anchor_pool_row: np.ndarray,
                     anchor_conf: np.ndarray, existing_keys: np.ndarray | None = None, ranker=None, chunk: int = 20_000,
                     workers: int = 1, top_per_entity: int | None = None, log=None) -> pd.DataFrame:
    """Second-hop candidates. anchor_* are aligned arrays (entity code, pool row of the anchor, its confidence);
    existing_keys = _key(entity, pool_row) of the current candidates (excluded). Returns one row per (entity, pool_row):
    hop_rank (best anchor->candidate ranker score), hop_conf (confidence of that anchor), hop_n (anchors that retrieved it)."""
    starts = list(range(0, len(anchor_pool_row), chunk))

    def work(a):
        rows = anchor_pool_row[a:a + chunk]
        res = index.block(pool_tbl.take(pa.array(rows)), ranker, verbose=False)
        if len(res["s1_row"]) == 0:
            return None
        k = res["s1_row"].astype(np.int64)
        return pd.DataFrame({"entity": anchor_entity[a:a + chunk][k], "anchor_row": rows[k], "pool_row": res["pool_row"],
                             "hop_rank": res["rank_score"], "hop_conf": anchor_conf[a:a + chunk][k]})

    parts = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, d in enumerate(ex.map(work, starts)):
            if d is not None:
                parts.append(d[d["pool_row"] != d["anchor_row"]])
            if log and (i + 1) % 10 == 0:
                log(f"      anchor chunks {i + 1}/{len(starts)}")
    if not parts:
        return pd.DataFrame(columns=["entity", "pool_row", "hop_rank", "hop_conf", "hop_n"])
    d = pd.concat(parts, ignore_index=True)
    if existing_keys is not None and len(existing_keys):
        d = d[~np.isin(_key(d["entity"], d["pool_row"]), existing_keys)]
    d = d.sort_values(["entity", "pool_row", "hop_rank"], ascending=[True, True, False])
    n = d.groupby(["entity", "pool_row"]).size().rename("hop_n")
    best = d.drop_duplicates(["entity", "pool_row"]).merge(n, left_on=["entity", "pool_row"], right_index=True)
    best = best[["entity", "pool_row", "hop_rank", "hop_conf", "hop_n"]]
    if top_per_entity:
        best = best.sort_values(["entity", "hop_rank"], ascending=[True, False])
        best = best[best.groupby("entity").cumcount() < top_per_entity]
    return best.sort_values(["entity", "pool_row"]).reset_index(drop=True)
