"""Plumbing correctness: chunking, negative sampling, output list building, id/label helpers.

Run:  python -m unittest discover -s code/business_entity_resolution/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids import id_to_int, label_pairs  # noqa: E402
from pipeline import chunk_slices  # noqa: E402
from run_infer import decide, joined_lists  # noqa: E402
from train_model import negative_sampling_mask  # noqa: E402


class ChunkSlicesTest(unittest.TestCase):
    def test_slices_cover_everything_and_never_cut_an_entity(self):
        rng = np.random.default_rng(0)
        counts = rng.integers(1, 40, size=500)
        s1_row = np.repeat(np.arange(len(counts)), counts)
        for target in (1, 7, 100, 1000, 10**9):
            sl = chunk_slices(s1_row, target)
            self.assertEqual(sl[0][0], 0)
            self.assertEqual(sl[-1][1], len(s1_row))
            for (a, b), (c, _d) in zip(sl, sl[1:]):
                self.assertEqual(b, c)                                   # contiguous
                self.assertNotEqual(s1_row[b - 1], s1_row[b])            # boundary is an entity boundary
            for a, b in sl[:-1]:
                self.assertGreaterEqual(b - a, min(target, len(s1_row)))  # chunks reach the target size

    def test_empty_input(self):
        self.assertEqual(chunk_slices(np.zeros(0, dtype=np.int64), 10), [])


class NegativeSamplingTest(unittest.TestCase):
    def test_positives_kept_hardest_kept_and_budget_respected(self):
        rng = np.random.default_rng(1)
        n_ent, per = 50, 40
        s1_row = np.repeat(np.arange(n_ent), per)
        label = (rng.random(len(s1_row)) < 0.1).astype(np.int8)
        prelim = rng.random(len(s1_row)).astype(np.float32)
        keep = negative_sampling_mask(s1_row, label, prelim, n_hard=5, n_random=3)
        self.assertTrue(keep[label == 1].all())
        for e in range(n_ent):
            m = s1_row == e
            neg = m & (label == 0)
            self.assertLessEqual(int((keep & neg).sum()), 8)
            hardest = np.flatnonzero(neg)[np.argsort(-prelim[neg], kind="stable")[:5]]
            self.assertTrue(keep[hardest].all())
        self.assertLess(keep.sum(), len(keep))

    def test_no_negatives(self):
        keep = negative_sampling_mask(np.array([0, 0, 1]), np.array([1, 1, 1], np.int8), np.zeros(3, np.float32), 5, 5)
        self.assertTrue(keep.all())


class SplitTest(unittest.TestCase):
    def test_entity_split3_is_disjoint_deterministic_and_complete(self):
        from train_model import entity_split3
        ids = [f"S1-{i}" for i in range(1000)]
        a, b = entity_split3(ids, 0.15, 0.15, 42), entity_split3(ids, 0.15, 0.15, 42)
        self.assertEqual(a, b)
        tr, va, ho = a
        self.assertFalse(tr & va or tr & ho or va & ho)
        self.assertEqual((len(tr), len(va), len(ho)), (700, 150, 150))
        self.assertEqual(tr | va | ho, set(ids))
        self.assertNotEqual(a, entity_split3(ids, 0.15, 0.15, 7))


class OutputListsTest(unittest.TestCase):
    def test_joined_lists_group_by_entity(self):
        s1_row = np.array([0, 0, 0, 1, 1, 3])
        s1_id = pa.array(["S1-a", "S1-a", "S1-a", "S1-b", "S1-b", "S1-d"])
        cand = pa.array(["S2-1", "S3-2", "S2-3", "S2-4", "S3-5", "S3-6"])
        keep = np.array([True, False, True, False, False, True])
        ents, lists = joined_lists(s1_row, s1_id, cand, keep)
        self.assertEqual(ents.to_pylist(), ["S1-a", "S1-d"])
        self.assertEqual(lists.to_pylist(), ["S2-1,S2-3", "S3-6"])

    def test_nothing_kept(self):
        ents, lists = joined_lists(np.array([0]), pa.array(["S1-a"]), pa.array(["S2-1"]), np.array([False]))
        self.assertEqual(len(ents), 0)
        self.assertEqual(len(lists), 0)

    def test_decide_modes(self):
        p = np.array([0.4, 0.6, 0.4, 0.6])
        is_s2 = np.array([True, True, False, False])
        g = decide(p, is_s2, {"mode": "global", "global_threshold": 0.5})
        self.assertEqual(g.tolist(), [False, True, False, True])
        ps = decide(p, is_s2, {"mode": "per_source", "s2_threshold": 0.3, "s3_threshold": 0.5})
        self.assertEqual(ps.tolist(), [True, True, False, True])

    def test_gated_mode_requires_the_entity_best_score(self):
        p = np.array([0.55, 0.40, 0.35, 0.30])                  # entity A: rows 0-1 (best .55), entity B: rows 2-3 (best .35)
        is_s2 = np.array([True, False, True, False])
        s1_row = np.array([0, 0, 1, 1])
        cfg = {"mode": "gated", "gate": 0.5, "s2_threshold": 0.3, "s3_threshold": 0.3}
        self.assertEqual(decide(p, is_s2, cfg, s1_row).tolist(), [True, True, False, False])   # B never clears the gate
        with self.assertRaises(ValueError):
            decide(p, is_s2, cfg)


class IdsTest(unittest.TestCase):
    def test_id_to_int_roundtrip_order_and_tags(self):
        v = id_to_int(pa.array(["S1-5", "S2-5", "S3-5", "S2-123456789"]))
        self.assertEqual(v[0], 10**10 + 5)
        self.assertEqual(v[1], 2 * 10**10 + 5)
        self.assertEqual(v[2], 3 * 10**10 + 5)
        self.assertEqual(v[3], 2 * 10**10 + 123456789)
        self.assertEqual(len(set(v.tolist())), 4)

    def test_label_pairs(self):
        truth = {"S1-1": {"S2-10", "S3-11"}, "S1-2": set()}
        labels = label_pairs(pa.array(["S1-1", "S1-1", "S1-1", "S1-2"]), pa.array(["S2-10", "S2-99", "S3-11", "S2-10"]), truth)
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
