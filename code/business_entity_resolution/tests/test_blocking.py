"""Checks the vectorized blocking against a brute-force pure-Python reference.

Run:  python3 -m unittest discover -s code/business_entity_resolution/tests -v
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blocking import block_partition  # noqa: E402
from config import BlockingConfig  # noqa: E402


def make_tables(seed=0, n_pool=400, n_s1=70):
    rng = np.random.default_rng(seed)
    name_vocab = [f"n{i}" for i in range(50)]
    addr_vocab = [f"a{i}" for i in range(140)]

    def zipf(vocab):
        p = 1.0 / np.arange(1, len(vocab) + 1)
        return p / p.sum()

    pn, pa_ = zipf(name_vocab), zipf(addr_vocab)

    def draw(vocab, p, k_max):
        k = int(rng.integers(0, k_max + 1))            # includes empty token lists
        return [str(t) for t in rng.choice(vocab, size=k, p=p)]  # with replacement -> in-row duplicates

    def table(n, plant_from=None):
        names = [draw(name_vocab, pn, 4) for _ in range(n)]
        addrs = [draw(addr_vocab, pa_, 6) for _ in range(n)]
        if plant_from is not None:                      # copy some rows so exact keys / overlaps occur
            for i in range(0, n, 3):
                j = int(rng.integers(0, len(plant_from[0])))
                names[i], addrs[i] = list(plant_from[0][j]), list(plant_from[1][j])
        postal = [None if rng.random() < 0.6 else f"{int(rng.integers(0, 25)):05d}" for _ in range(n)]
        return names, addrs, postal

    pool = table(n_pool)
    s1 = table(n_s1, plant_from=pool)

    def to_arrow(names, addrs, postal):
        return pa.table({
            "name_tokens": pa.array(names, type=pa.list_(pa.string())),
            "address_tokens": pa.array(addrs, type=pa.list_(pa.string())),
            "name_normalized": pa.array([" ".join(t) for t in names], type=pa.large_string()),
            "name_compact": pa.array(["".join(t) for t in names], type=pa.large_string()),
            "postal_code": pa.array(postal, type=pa.large_string()),
        })

    return to_arrow(*s1), to_arrow(*pool), s1, pool


def reference(s1, pool, cfg):
    """Pure-Python restatement of the blocking rules (no numpy tricks)."""
    n_pool = len(pool[0])
    out = {}  # (i, j) -> dict

    def rare_lookup(q_toks, p_toks, k):
        df, first = Counter(), {}
        pos = 0
        for row in p_toks:
            for t in row:
                first.setdefault(t, len(first))      # dictionary_encode: order of first appearance
            for t in set(row):
                df[t] += 1
        postings = {}
        for j, row in enumerate(p_toks):
            for t in set(row):
                postings.setdefault(t, []).append(j)
        res = {}
        for i, row in enumerate(q_toks):
            cand = [t for t in set(row) if t in df and df[t] <= cfg.df_cap]
            sel = sorted(cand, key=lambda t: (df[t], first[t]))[:k]
            for t in sel:
                idf = float(np.float32(np.log((n_pool + 1.0) / (df[t] + 1.0))))
                for j in postings[t]:
                    d = res.setdefault((i, j), [0, 0.0])
                    d[0] += 1
                    d[1] += idf
        return res

    name_res = rare_lookup(s1[0], pool[0], cfg.k_rare_name)
    addr_res = rare_lookup(s1[1], pool[1], cfg.k_rare_addr)

    def exact(q_vals, p_vals):
        cnt = Counter(v for v in p_vals if v)
        res = set()
        for i, v in enumerate(q_vals):
            if v and 0 < cnt[v] <= cfg.exact_key_cap:
                res |= {(i, j) for j, pv in enumerate(p_vals) if pv == v}
        return res

    ex_name = exact([" ".join(t) for t in s1[0]], [" ".join(t) for t in pool[0]])
    ex_comp = exact(["".join(t) for t in s1[0]], ["".join(t) for t in pool[0]])
    ex_post = exact(s1[2], pool[2])

    keys = set(name_res) | set(addr_res) | ex_name | ex_comp | ex_post
    for k in keys:
        nn, sn = name_res.get(k, [0, 0.0])
        na, sa = addr_res.get(k, [0, 0.0])
        out[k] = dict(
            n_name_shared=nn, n_addr_shared=na, idf_name=sn, idf_addr=sa,
            exact_name=int(k in ex_name), exact_compact=int(k in ex_comp), postal_eq=int(k in ex_post),
        )
    return out


class BlockingTest(unittest.TestCase):
    def check(self, seed, cfg):
        s1_tbl, pool_tbl, s1, pool = make_tables(seed)
        got = block_partition(s1_tbl, pool_tbl, cfg, verbose=False)
        ref = reference(s1, pool, cfg)
        return got, ref

    def test_pair_sets_and_evidence_match_reference(self):
        cfg = BlockingConfig(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, max_candidates=10**6)
        for seed in range(6):
            got, ref = self.check(seed, cfg)
            got_pairs = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(got["s1_row"], got["pool_row"]))}
            self.assertEqual(set(got_pairs), set(ref), f"seed {seed}: candidate pair sets differ")
            self.assertEqual(len(got_pairs), len(got["s1_row"]), "duplicate pairs emitted")
            for k, i in got_pairs.items():
                r = ref[k]
                self.assertEqual(int(got["n_name_shared"][i]), r["n_name_shared"])
                self.assertEqual(int(got["n_addr_shared"][i]), r["n_addr_shared"])
                self.assertAlmostEqual(float(got["idf_name"][i]), r["idf_name"], places=3)
                self.assertAlmostEqual(float(got["idf_addr"][i]), r["idf_addr"], places=3)
                self.assertEqual(int(got["exact_name"][i]), r["exact_name"])
                self.assertEqual(int(got["exact_compact"][i]), r["exact_compact"])
                self.assertEqual(int(got["postal_eq"][i]), r["postal_eq"])
                strategies = (r["n_name_shared"] > 0) + (r["n_addr_shared"] > 0) + r["exact_name"] + r["exact_compact"] + r["postal_eq"]
                self.assertEqual(int(got["n_strategies"][i]), strategies)

    def test_multi_batch_equals_single_batch(self):
        base = dict(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, max_candidates=10**6)
        s1_tbl, pool_tbl, _, _ = make_tables(3)
        one = block_partition(s1_tbl, pool_tbl, BlockingConfig(**base, expansion_budget=10**9), verbose=False)
        many = block_partition(s1_tbl, pool_tbl, BlockingConfig(**base, expansion_budget=40), verbose=False)
        a = set(zip(one["s1_row"].tolist(), one["pool_row"].tolist()))
        b = set(zip(many["s1_row"].tolist(), many["pool_row"].tolist()))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_truncation_keeps_highest_scoring_candidates(self):
        full_cfg = BlockingConfig(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, max_candidates=10**6)
        cut_cfg = BlockingConfig(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, max_candidates=3)
        s1_tbl, pool_tbl, _, _ = make_tables(1)
        full = block_partition(s1_tbl, pool_tbl, full_cfg, verbose=False)
        cut = block_partition(s1_tbl, pool_tbl, cut_cfg, verbose=False)
        by_row_full, by_row_cut = {}, {}
        for r, s in zip(full["s1_row"], full["block_score"]):
            by_row_full.setdefault(int(r), []).append(float(s))
        for r, p, s in zip(cut["s1_row"], cut["pool_row"], cut["block_score"]):
            by_row_cut.setdefault(int(r), []).append(float(s))
        for r, scores in by_row_full.items():
            kept = sorted(by_row_cut.get(r, []), reverse=True)
            self.assertEqual(len(kept), min(3, len(scores)))
            expected_top = sorted(scores, reverse=True)[: len(kept)]
            np.testing.assert_allclose(kept, expected_top, atol=1e-4)

    def test_empty_inputs(self):
        s1_tbl, pool_tbl, _, _ = make_tables(0)
        out = block_partition(s1_tbl.slice(0, 0), pool_tbl, verbose=False)
        self.assertEqual(len(out["s1_row"]), 0)
        out = block_partition(s1_tbl, pool_tbl.slice(0, 0), verbose=False)
        self.assertEqual(len(out["s1_row"]), 0)


if __name__ == "__main__":
    unittest.main()
