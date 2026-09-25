"""Metric correctness: the challenge's per-entity F0.5, macro-averaged, with its singleton rule.

Run:  python3 -m unittest discover -s code/business_entity_resolution/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import FastScorer, macro_f05  # noqa: E402


class MetricTest(unittest.TestCase):
    def test_readme_worked_example(self):
        # predicted [S2-47, S2-193, S3-812] vs truth [S2-47, S3-812]: P=2/3, R=1 -> 0.714
        f, _ = macro_f05({"S1-1": ["S2-47", "S2-193", "S3-812"]}, {"S1-1": ["S2-47", "S3-812"]}, ["S1-1"])
        self.assertAlmostEqual(f, 0.7142857, places=6)

    def test_singleton_rules(self):
        truth = {"a": [], "b": [], "c": ["S2-1"], "d": ["S2-1", "S2-2"]}
        pred = {"a": [], "b": ["S2-9"], "c": [], "d": ["S2-1", "S2-2"]}
        f, st = macro_f05(pred, truth, list(truth))
        st = st.set_index("s1_entity_id")
        self.assertEqual(st.loc["a", "f05"], 1.0)   # true singleton, correctly empty
        self.assertEqual(st.loc["b", "f05"], 0.0)   # true singleton, any prediction is a false merge
        self.assertEqual(st.loc["c", "f05"], 0.0)   # has matches, predicted none
        self.assertEqual(st.loc["d", "f05"], 1.0)
        self.assertAlmostEqual(f, 0.5)

    def test_entities_without_any_prediction_or_candidate_are_still_scored(self):
        f, st = macro_f05({}, {"a": ["S2-1"], "b": []}, ["a", "b", "c"])  # c: unknown to truth -> treated as singleton
        self.assertEqual(len(st), 3)
        self.assertAlmostEqual(f, (0 + 1 + 1) / 3)

    def test_precision_is_weighted_over_recall(self):
        # 1 TP, 1 FP (P=.5,R=1) must score lower than 1 TP, 1 FN (P=1,R=.5)
        fp_case, _ = macro_f05({"a": ["S2-1", "S2-x"]}, {"a": ["S2-1"]}, ["a"])
        fn_case, _ = macro_f05({"a": ["S2-1"]}, {"a": ["S2-1", "S2-2"]}, ["a"])
        self.assertLess(fp_case, fn_case)

    def test_fast_scorer_matches_reference_on_random_data(self):
        rng = np.random.default_rng(0)
        ents = [f"S1-{i}" for i in range(300)]
        truth, rows = {}, []
        for e in ents:
            k = 0 if rng.random() < 0.1 else int(rng.integers(1, 6))
            tr = {f"S2-{e}-{j}" for j in range(k)}
            truth[e] = tr
            for c in tr:
                if rng.random() < 0.85:  # some true matches never become candidates (blocking misses)
                    rows.append((e, c, 1, float(np.clip(rng.normal(0.8, 0.2), 0, 1))))
            for j in range(int(rng.integers(0, 6))):
                rows.append((e, f"S3-neg-{e}-{j}", 0, float(np.clip(rng.normal(0.3, 0.25), 0, 1))))
        df = pd.DataFrame(rows, columns=["s1", "cid", "label", "p"])
        scorer = FastScorer(df.s1.to_numpy(), df.label.to_numpy(), truth, ents)
        for t in np.linspace(0, 1.01, 25):
            keep = (df.p >= t).to_numpy()
            pred = {}
            for s1, cid in zip(df.s1[keep], df.cid[keep]):
                pred.setdefault(s1, set()).add(cid)
            ref, _ = macro_f05(pred, truth, ents)
            self.assertAlmostEqual(scorer.score(keep), ref, places=12)


if __name__ == "__main__":
    unittest.main()
