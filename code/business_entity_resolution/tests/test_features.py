"""Feature engineering: the vectorized implementation must agree with slow, obvious references.

Run:  python -m unittest discover -s code/business_entity_resolution/tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import BLOCK_FEATURE_COLS, PartitionContext, TokenStats, compute_features  # noqa: E402
from prepare_cache import _normalize_chunk  # noqa: E402


def _list_col(rows):
    return pa.array(rows, type=pa.list_(pa.string()))


class TokenStatsTest(unittest.TestCase):
    def test_pair_features_match_set_reference(self):
        rng = np.random.default_rng(1)
        vocab = [f"w{i}" for i in range(30)] + ["12", "5", "411001", "7b"]
        s1 = [list(rng.choice(vocab, size=int(rng.integers(0, 6)), replace=False)) for _ in range(40)]
        pool = [list(rng.choice(vocab, size=int(rng.integers(0, 7)), replace=False)) for _ in range(120)]
        ts = TokenStats(_list_col(s1), _list_col(pool))

        s1_rows = rng.integers(0, len(s1), size=300)
        pool_rows = rng.integers(0, len(pool), size=300)
        got = ts.pair_features(s1_rows, pool_rows, "t")

        df = {}
        for row in pool:
            for tok in set(row):
                df[tok] = df.get(tok, 0) + 1
        idf = lambda t: math.log((len(pool) + 1.0) / (df.get(t, 0) + 1.0))  # noqa: E731
        has_digit = lambda t: any(ch.isdigit() for ch in t)  # noqa: E731

        for i, (r1, r2) in enumerate(zip(s1_rows, pool_rows)):
            a, b = set(s1[r1]), set(pool[r2])
            inter, union = a & b, a | b
            self.assertEqual(got["t_inter"][i], len(inter))
            self.assertEqual(got["t_n1"][i], len(a))
            self.assertEqual(got["t_n2"][i], len(b))
            if union:
                self.assertAlmostEqual(got["t_jaccard"][i], len(inter) / len(union), places=5)
                self.assertAlmostEqual(got["t_dice"][i], 2 * len(inter) / (len(a) + len(b)), places=5)
            else:
                self.assertTrue(np.isnan(got["t_jaccard"][i]))
            if a:
                self.assertAlmostEqual(got["t_cont1"][i], len(inter) / len(a), places=5)
                self.assertAlmostEqual(got["t_idf_ov1"][i], sum(idf(t) for t in inter) / sum(idf(t) for t in a), places=4)
            if b:
                self.assertAlmostEqual(got["t_cont2"][i], len(inter) / len(b), places=5)
            if a and b:
                cos = sum(idf(t) ** 2 for t in inter) / (
                    math.sqrt(sum(idf(t) ** 2 for t in a)) * math.sqrt(sum(idf(t) ** 2 for t in b)))
                self.assertAlmostEqual(got["t_idf_cos"][i], cos, places=4)
            na, nb = {t for t in a if has_digit(t)}, {t for t in b if has_digit(t)}
            if na | nb:
                self.assertAlmostEqual(got["t_num_jaccard"][i], len(na & nb) / len(na | nb), places=5)
            else:
                self.assertTrue(np.isnan(got["t_num_jaccard"][i]))


S1_RAW = [
    ("S1-1", "Raj Investments Private Limited", "12 MG Road, Pune, Maharashtra 411001", "India"),
    ("S1-2", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC 27262", "US"),
    ("S1-3", "B+ Retail Inc", "1712 Montebello Avenue, Phoenix, AZ", "US"),
]
POOL_RAW = [
    ("S2-1", "Raj Investments Pvt Ltd", "12 M.G. Road, Pune 411001", "India"),
    ("S2-2", "राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", "", "India"),
    ("S3-1", "Orelee Barbershop", "1795 Westchester Dr, High Point, NC", "US"),
    ("S3-2", "Prime Money", "17560 Ellis Road, Tahlequah, OK", "US"),
    ("S2-3", "B Retail Incorporated", "1712 Montebello Ave, Phoenix, AZ", "US"),
]


def _table(raw, tag_from_id):
    df = pd.DataFrame(raw, columns=["entity_id", "business_name", "business_address", "country"])
    tbl = _normalize_chunk(df, "S1")
    if tag_from_id:  # pool: source comes from the id prefix
        src = pa.array([r[0][:2] for r in raw], type=pa.large_string())
        tbl = tbl.set_column(tbl.schema.get_field_index("source"), "source", src)
    return tbl


class ComputeFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, cls.pool = _table(S1_RAW, False), _table(POOL_RAW, True)
        cls.ctx = PartitionContext(cls.s1, cls.pool)
        grid = [(i, j) for i in range(len(S1_RAW)) for j in range(len(POOL_RAW))]   # sorted by s1_row
        cls.s1_row = np.array([g[0] for g in grid])
        cls.pool_row = np.array([g[1] for g in grid])
        n = len(grid)
        cls.block = {c: np.ones(n, dtype=np.float32) for c in BLOCK_FEATURE_COLS}
        cls.block["rank_score"] = np.linspace(1, 5, n, dtype=np.float32)
        cls.f = compute_features(cls.ctx, cls.s1_row, cls.pool_row, cls.block)
        cls.n1 = cls.s1["name_normalized"].to_pylist()
        cls.n2 = cls.pool["name_normalized"].to_pylist()
        cls.a1 = cls.s1["address_normalized"].to_pylist()
        cls.a2 = cls.pool["address_normalized"].to_pylist()

    def test_shape_and_no_label_or_country_leak(self):
        self.assertEqual(len(self.f), len(self.s1_row))
        for banned in ("label", "country", "s1_entity_id", "cand_entity_id"):
            self.assertNotIn(banned, self.f.columns)

    def test_name_similarities_match_rapidfuzz(self):
        for k, (i, j) in enumerate(zip(self.s1_row, self.pool_row)):
            x, y = self.n1[i], self.n2[j]
            self.assertAlmostEqual(self.f["name_ratio"][k], fuzz.ratio(x, y) / 100, places=5)
            self.assertAlmostEqual(self.f["name_jw"][k], JaroWinkler.normalized_similarity(x, y), places=5)
            self.assertAlmostEqual(self.f["name_token_sort"][k], fuzz.token_sort_ratio(x, y) / 100, places=5)
            self.assertAlmostEqual(self.f["name_token_set"][k], fuzz.token_set_ratio(x, y) / 100, places=5)
            self.assertAlmostEqual(self.f["name_partial"][k], fuzz.partial_ratio(x, y) / 100, places=5)

    def test_missing_address_gives_nan_similarity_not_zero(self):
        k = int(np.flatnonzero((self.s1_row == 0) & (self.pool_row == 1))[0])   # S2-2 has an empty address
        self.assertEqual(self.f["addr_missing_2"][k], 1)
        for col in ("addr_ratio", "addr_jw", "addr_token_set", "at_jaccard", "at_idf_cos"):
            self.assertTrue(np.isnan(self.f[col][k]), col)
        self.assertFalse(np.isnan(self.f["name_ratio"][k]))

    def test_true_pairs_score_higher_than_wrong_pairs(self):
        def val(i, j, col):
            return float(self.f[col][int(np.flatnonzero((self.s1_row == i) & (self.pool_row == j))[0])])
        # S1-2 (Orelee) vs S3-1 (Orelee Barbershop) must beat S1-2 vs S3-2 (Prime Money)
        self.assertGreater(val(1, 2, "name_ratio"), val(1, 3, "name_ratio"))
        self.assertGreater(val(1, 2, "addr_ratio"), val(1, 3, "addr_ratio"))
        self.assertEqual(val(1, 2, "housenum_match"), 1.0)
        self.assertEqual(val(1, 3, "housenum_match"), 0.0)
        self.assertEqual(val(0, 0, "postal_match"), 1.0)     # both Pune records carry 411001
        self.assertEqual(val(2, 4, "name_script_match"), 1.0)

    def test_exact_flags(self):
        f = self.f
        # Normalization contracts "Private Limited" -> "pvt ltd" and "Incorporated" -> "inc", so exactly two
        # toy pairs are identical after normalization: (Raj, Raj Pvt Ltd) and (B+ Retail Inc, B Retail Incorporated).
        exact = {(int(i), int(j)) for i, j, v in zip(self.s1_row, self.pool_row, f["name_exact_normalized"]) if v == 1}
        self.assertEqual(exact, {(0, 0), (2, 4)})
        k = int(np.flatnonzero((self.s1_row == 0) & (self.pool_row == 0))[0])
        self.assertEqual(f["name_exact_core"][k], 1)                 # "raj investments" once suffixes are dropped

    def test_devanagari_name_is_transliterated_and_flagged(self):
        k = int(np.flatnonzero((self.s1_row == 0) & (self.pool_row == 1))[0])
        self.assertEqual(self.f["name_script_2"][k], 2.0)            # devanagari
        self.assertEqual(self.f["name_script_1"][k], 0.0)            # latin
        self.assertGreater(self.f["name_jw"][k], 0.5)                # transliteration recovers real similarity

    def test_s1_frequency_features_use_the_full_universe(self):
        # the S1 table here is a sample; frequencies must come from the full S1 universe passed in
        full = _table(S1_RAW + [("S1-9", "Orelee's Barbershop", "somewhere else 12", "US"),
                                ("S1-10", "Orelee's Barbershop", "another place 7", "US")], False)
        ctx = PartitionContext(self.s1, self.pool, full.select(["name_core", "address_normalized"]))
        f = compute_features(ctx, self.s1_row, self.pool_row, self.block)
        core_ref = pd.Series(full["name_core"].to_pylist())
        s1_core = self.s1["name_core"].to_pylist()
        for k, i in enumerate(self.s1_row):
            self.assertAlmostEqual(float(f["s1_core_freq"][k]), np.log1p((core_ref == s1_core[i]).sum()), places=5)
        self.assertGreater(float(f["s1_core_freq"][int(np.flatnonzero(self.s1_row == 1)[0])]), float(np.log1p(1)))   # Orelee: 3 S1 entities share it

    def test_pool_frequency_features(self):
        names = self.pool["name_normalized"].to_pylist()
        addrs = self.pool["address_normalized"].to_pylist()
        cnt_n = pd.Series(names).map(pd.Series(names).value_counts())
        cnt_a = pd.Series(addrs).map(pd.Series(addrs).value_counts())
        for k, j in enumerate(self.pool_row):
            self.assertAlmostEqual(float(self.f["pool_name_freq"][k]), np.log1p(cnt_n[j]), places=5)
            if addrs[j] == "":
                self.assertTrue(np.isnan(self.f["pool_addr_freq"][k]))
            else:
                self.assertAlmostEqual(float(self.f["pool_addr_freq"][k]), np.log1p(cnt_a[j]), places=5)

    def test_within_entity_relative_features_match_pandas(self):
        df = pd.DataFrame({"s1": self.s1_row, "prelim": self.f["prelim_score"].to_numpy(),
                           "block": self.block["rank_score"], "ar": self.f["addr_ratio"].to_numpy()})
        g = df.groupby("s1")
        np.testing.assert_array_equal(self.f["prelim_rank"], g["prelim"].rank(ascending=False, method="first").astype(int))
        np.testing.assert_allclose(self.f["prelim_gap_to_best"], g["prelim"].transform("max") - df["prelim"], atol=1e-6)
        np.testing.assert_array_equal(self.f["block_rank"], g["block"].rank(ascending=False, method="first").astype(int))
        np.testing.assert_allclose(self.f["addr_ratio_gap"], g["ar"].transform("max") - df["ar"], atol=1e-6, equal_nan=True)
        self.assertTrue((self.f["n_candidates_for_entity"] == len(POOL_RAW)).all())
        nr, ar = self.f["name_ratio"].fillna(0).to_numpy(), self.f["addr_ratio"].fillna(0).to_numpy()
        ref = pd.DataFrame({"g": self.s1_row, "n90": nr >= 0.9, "a80": ar >= 0.8, "both": (nr >= 0.85) & (ar >= 0.7)})
        for col, src in (("n_name_ge90", "n90"), ("n_addr_ge80", "a80"), ("n_both_strong", "both")):
            np.testing.assert_array_equal(self.f[col], ref.groupby("g")[src].transform("sum").astype(int))
        ref["nr"] = nr
        np.testing.assert_array_equal(self.f["is_name_top1"], (ref["nr"] >= ref.groupby("g")["nr"].transform("max")).astype(int))


if __name__ == "__main__":
    unittest.main()
