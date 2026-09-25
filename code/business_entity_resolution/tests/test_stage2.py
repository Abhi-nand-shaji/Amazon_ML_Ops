"""Stage-2 entity-context features agree with a pandas groupby reference.

Run:  python -m unittest discover -s code/business_entity_resolution/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2 import CUTS, entity_context  # noqa: E402


class EntityContextTest(unittest.TestCase):
    def test_matches_pandas_reference(self):
        rng = np.random.default_rng(3)
        counts = rng.integers(1, 12, size=60)
        group = np.repeat(np.arange(len(counts)), counts)
        p = np.round(rng.random(len(group)), 3)                       # rounding creates ties on purpose
        is_s2 = rng.random(len(group)) < 0.5
        f = entity_context(p, is_s2, group)
        df = pd.DataFrame({"g": group, "p": p, "s2": is_s2.astype(int)})
        g = df.groupby("g")["p"]

        np.testing.assert_allclose(f["e_sum"], g.transform("sum"), atol=1e-5)
        np.testing.assert_array_equal(f["e_count"], g.transform("size"))
        np.testing.assert_array_equal(f["e_rank"], g.rank(ascending=False, method="first").astype(int) - 1)
        np.testing.assert_allclose(f["e_top1"], g.transform("max"), atol=1e-6)
        for c in CUTS:
            np.testing.assert_array_equal(f[f"e_n_ge_{int(c*100)}"], df.assign(x=(df.p >= c).astype(int)).groupby("g")["x"].transform("sum"))
        # best OTHER candidate = max over the entity's other rows (0 when the entity has a single candidate)
        best_other = [max([v for j, v in enumerate(p[group == e]) if j != k] or [0.0]) for e, k in zip(group, _pos_in_group(group))]
        np.testing.assert_allclose(f["e_best_other"], best_other, atol=1e-6)
        np.testing.assert_allclose(f["e_gap_to_best_other"], p - np.array(best_other), atol=1e-6)
        # same statistics inside the (entity, source) subgroups
        gs = df.groupby(["g", "s2"])["p"]
        np.testing.assert_allclose(f["s_sum"], gs.transform("sum"), atol=1e-5)
        np.testing.assert_array_equal(f["s_count"], gs.transform("size"))
        np.testing.assert_array_equal(f["s_rank"], gs.rank(ascending=False, method="first").astype(int) - 1)

    def test_single_candidate_entities(self):
        f = entity_context(np.array([0.8, 0.2]), np.array([True, False]), np.array([0, 1]))
        self.assertEqual(f["e_best_other"].tolist(), [0.0, 0.0])
        self.assertEqual(f["e_rank"].tolist(), [0, 0])
        self.assertAlmostEqual(float(f["e_gap_to_best_other"][0]), 0.8, places=6)


def _pos_in_group(group):
    out, seen = [], {}
    for e in group:
        out.append(seen.get(e, 0))
        seen[e] = seen.get(e, 0) + 1
    return out


if __name__ == "__main__":
    unittest.main()
