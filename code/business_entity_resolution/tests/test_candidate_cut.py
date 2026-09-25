"""Final candidate selection: the per-entity cut and the candidate filter's features.

Run:  python -m unittest discover -s code/business_entity_resolution/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import select_top  # noqa: E402
from candidate_filter import EVIDENCE, FILTER_FEATURES, FilterContext, filter_features  # noqa: E402
from features import BLOCK_FEATURE_COLS, PartitionContext, compute_features  # noqa: E402
from test_features import POOL_RAW, S1_RAW, _table  # noqa: E402


def reference_cut(s1_row, pool_row, score, k, floor=None, min_keep=0):
    keep = []
    for e in np.unique(s1_row):
        idx = np.flatnonzero(s1_row == e)
        idx = sorted(idx, key=lambda i: (-score[i], pool_row[i]))           # best first, ties -> lower pool row
        for r, i in enumerate(idx[:k]):
            if floor is None or score[i] >= floor or r < min_keep:
                keep.append(i)
    return np.array(sorted(keep), dtype=np.int64)


class SelectTopTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n_ent, per = 40, 25
        self.s1_row = np.repeat(np.arange(n_ent), per).astype(np.int32)
        self.pool_row = np.concatenate([np.sort(rng.choice(1000, per, replace=False)) for _ in range(n_ent)]).astype(np.int32)
        # coarse scores -> many ties, which must be broken by the lower pool row
        self.score = (rng.integers(0, 12, size=n_ent * per) / 11.0).astype(np.float32)

    def test_matches_reference_with_cap_floor_and_min_keep(self):
        for k, floor, m in [(30, None, 0), (5, None, 0), (30, 0.5, 0), (8, 0.3, 0), (8, 0.95, 2), (1, 0.0, 0)]:
            got = select_top(self.s1_row, self.pool_row, self.score, k, floor, m)
            np.testing.assert_array_equal(got, reference_cut(self.s1_row, self.pool_row, self.score, k, floor, m),
                                          err_msg=f"k={k} floor={floor} min_keep={m}")

    def test_cutting_a_shortlist_equals_cutting_the_full_list(self):
        """A shortlist (top 12) cut again with a tighter rule keeps exactly what the tighter rule keeps directly."""
        short = select_top(self.s1_row, self.pool_row, self.score, 12)
        again = short[select_top(self.s1_row[short], self.pool_row[short], self.score[short], 6, 0.4)]
        np.testing.assert_array_equal(again, select_top(self.s1_row, self.pool_row, self.score, 6, 0.4))

    def test_entity_with_nothing_above_the_floor_keeps_no_candidate(self):
        score = self.score.copy()
        score[self.s1_row == 7] = 0.01
        got = select_top(self.s1_row, self.pool_row, score, 30, 0.05)
        self.assertFalse(np.isin(7, self.s1_row[got]))
        self.assertTrue(np.all(score[got] >= 0.05))


class FilterFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s1, pool = _table(S1_RAW, False), _table(POOL_RAW, True)
        grid = [(i, j) for i in range(len(S1_RAW)) for j in range(len(POOL_RAW))]      # sorted by s1_row
        cls.s1_row = np.array([g[0] for g in grid])
        cls.pool_row = np.array([g[1] for g in grid])
        n = len(grid)
        block = {c: np.ones(n, dtype=np.float32) for c in BLOCK_FEATURE_COLS}
        block["rank_score"] = np.linspace(1, 5, n, dtype=np.float32)
        cls.block = block
        cls.f = filter_features(FilterContext(s1, pool), cls.s1_row, cls.pool_row, {c: block[c] for c in EVIDENCE + ["rank_score"]})
        cls.m = compute_features(PartitionContext(s1, pool), cls.s1_row, cls.pool_row, block)
        cls.s1_tok = [len(x) for x in s1["name_tokens"].to_pylist()]
        cls.pool_addr_tok = [len(x) for x in pool["address_tokens"].to_pylist()]

    def test_columns(self):
        self.assertEqual(list(self.f.columns), FILTER_FEATURES)
        self.assertEqual(len(self.f), len(self.s1_row))
        for banned in ("rank_score", "idf_name", "idf_addr", "block_score", "n_name_shared", "n_addr_shared", "country", "label"):
            self.assertNotIn(banned, self.f.columns)

    def test_shared_features_equal_the_matchers_values(self):
        """The filter must compute exactly what the matcher computes for the same pair (same functions, same NaN rules)."""
        for c in ("name_token_set", "name_partial", "core_ratio", "core_jw", "addr_token_set", "addr_partial", "addr_ratio",
                  "name_missing_2", "addr_missing_2", "housenum_both_present", "housenum_match", "is_s2", "block_rank",
                  "ov_name", "ov_addr", "exact_name", "n_strategies"):
            np.testing.assert_allclose(self.f[c].to_numpy(dtype=float), self.m[c].to_numpy(dtype=float), rtol=1e-6,
                                       equal_nan=True, err_msg=c)

    def test_token_counts(self):
        np.testing.assert_array_equal(self.f["s1_ntok_name"].to_numpy(), np.array(self.s1_tok)[self.s1_row])
        np.testing.assert_array_equal(self.f["c_ntok_addr"].to_numpy(), np.array(self.pool_addr_tok)[self.pool_row])

    def test_missing_address_gives_nan(self):
        k = int(np.flatnonzero((self.s1_row == 0) & (self.pool_row == 1))[0])      # S2-2 has an empty address
        self.assertEqual(self.f["addr_missing_2"][k], 1)
        for c in ("addr_ratio", "addr_token_set", "addr_partial"):
            self.assertTrue(np.isnan(self.f[c][k]), c)


if __name__ == "__main__":
    unittest.main()
