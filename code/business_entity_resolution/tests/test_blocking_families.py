"""The extra retrieval families (bigrams, name prefix) and the full-evidence re-ranking, each checked
against a slow, obvious pure-Python reference.

Run:  python -m unittest discover -s code/business_entity_resolution/tests -v
"""
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import block_partition  # noqa: E402
from config import BlockingConfig  # noqa: E402
from test_blocking import make_tables  # noqa: E402

# Old rules off: df_cap=0 -> no unigram lookups, exact_key_cap=0 -> no exact keys. Only the new families remain.
ONLY_NEW = dict(df_cap=0, k_rare_name=2, k_rare_addr=3, exact_key_cap=0, max_candidates=10**6)


def with_core(tbl, tokens):
    return tbl.append_column("name_core", pa.array([" ".join(t) for t in tokens[0]], type=pa.large_string()))


def ref_bigrams(q_toks, p_toks, cap):
    """{(i, j): number of matching bigram occurrence pairs}. Bigrams are adjacent token pairs, unordered;
    the query side only forms pairs of tokens that exist in the pool vocabulary."""
    vocab = {t for row in p_toks for t in row}
    def bigs(row):
        return [tuple(sorted((a, b))) for a, b in zip(row, row[1:]) if a in vocab and b in vocab]
    postings = {}
    for j, row in enumerate(p_toks):
        for k in bigs(row):
            postings.setdefault(k, []).append(j)
    out = {}
    for i, row in enumerate(q_toks):
        for k in bigs(row):
            plist = postings.get(k, [])
            if 1 <= len(plist) <= cap:
                for j in plist:
                    out[(i, j)] = out.get((i, j), 0) + 1
    return out


def ref_prefix(q_names, p_names, length, cap):
    key = lambda toks: "".join(toks)[:length] if len("".join(toks)) >= length else None  # noqa: E731
    p_keys = [key(t) for t in p_names]
    cnt = Counter(k for k in p_keys if k is not None)
    out = set()
    for i, t in enumerate(q_names):
        k = key(t)
        if k is not None and 1 <= cnt.get(k, 0) <= cap:
            out |= {(i, j) for j, pk in enumerate(p_keys) if pk == k}
    return out


def pairs_of(got):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(got["s1_row"], got["pool_row"]))}


class FamilyTest(unittest.TestCase):
    def test_address_and_name_bigrams_match_reference(self):
        for seed in range(6):
            s1_tbl, pool_tbl, s1, pool = make_tables(seed)
            cfg = BlockingConfig(**ONLY_NEW, cap_addr_bigram=4, cap_name_bigram=3)
            got = block_partition(s1_tbl, pool_tbl, cfg, verbose=False)
            ref_a = ref_bigrams(s1[1], pool[1], 4)
            ref_n = ref_bigrams(s1[0], pool[0], 3)
            gp = pairs_of(got)
            self.assertEqual(set(gp), set(ref_a) | set(ref_n), f"seed {seed}: bigram candidate sets differ")
            for k, i in gp.items():
                self.assertEqual(int(got["n_addr_bigram"][i]), ref_a.get(k, 0), (seed, k))
                self.assertEqual(int(got["n_name_bigram"][i]), ref_n.get(k, 0), (seed, k))
                self.assertEqual(int(got["n_name_shared"][i]) + int(got["n_addr_shared"][i]), 0)
            self.assertGreater(len(gp), 0)

    def test_prefix_family_matches_reference(self):
        for seed in range(6):
            s1_tbl, pool_tbl, s1, pool = make_tables(seed)
            for length in (2, 4):
                cfg = BlockingConfig(**ONLY_NEW, prefix_lens=(length,), cap_prefix=6)
                got = block_partition(with_core(s1_tbl, s1), with_core(pool_tbl, pool), cfg, verbose=False)
                self.assertEqual(set(pairs_of(got)), ref_prefix(s1[0], pool[0], length, 6), (seed, length))
                self.assertTrue((got["n_prefix"] == 1).all())

    def test_new_families_only_add_pairs_never_remove(self):
        base = dict(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, max_candidates=10**6)
        s1_tbl, pool_tbl, s1, pool = make_tables(2)
        old = block_partition(s1_tbl, pool_tbl, BlockingConfig(**base), verbose=False)
        new = block_partition(with_core(s1_tbl, s1), with_core(pool_tbl, pool),
                              BlockingConfig(**base, cap_addr_bigram=5, cap_name_bigram=5, prefix_lens=(3,), cap_prefix=5),
                              verbose=False)
        self.assertLessEqual(set(pairs_of(old)), set(pairs_of(new)))
        self.assertGreater(len(pairs_of(new)), len(pairs_of(old)))


class LeadingZeroTest(unittest.TestCase):
    def test_zero_padded_numbers_are_the_same_token(self):
        def tbl(addr):
            n = len(addr)
            return pa.table({
                "name_tokens": pa.array([["x"]] * n, type=pa.list_(pa.string())),
                "address_tokens": pa.array(addr, type=pa.list_(pa.string())),
                "name_normalized": pa.array(["x"] * n, type=pa.large_string()),
                "name_compact": pa.array(["x"] * n, type=pa.large_string()),
                "postal_code": pa.array([None] * n, type=pa.large_string())})
        pool = tbl([["010001", "mabelvale", "plaza"], ["77", "other"]])
        s1 = tbl([["10001", "mabelvale", "plaza"]])
        got = block_partition(s1, pool, BlockingConfig(df_cap=10, k_rare_name=1, k_rare_addr=4, exact_key_cap=0,
                                                       max_candidates=10, cap_addr_bigram=10), verbose=False)
        by_pool = dict(zip(got["pool_row"].tolist(), got["n_addr_shared"].tolist()))
        self.assertEqual(by_pool.get(0), 3)                    # '010001'=='10001', 'mabelvale', 'plaza'
        i = got["pool_row"].tolist().index(0)
        self.assertEqual(int(got["n_addr_bigram"][i]), 2)      # ('10001','mabelvale') and ('mabelvale','plaza')


class ChunkingTest(unittest.TestCase):
    def test_chunked_queries_equal_whole_table_query(self):
        """An entity's candidates must not depend on which other S1 entities share its chunk."""
        from blocking import PoolIndex
        s1_tbl, pool_tbl, s1, pool = make_tables(3)
        cfg = BlockingConfig(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, cap_addr_bigram=5,
                             cap_name_bigram=5, prefix_lens=(3,), cap_prefix=5, rescore=True, max_candidates=4,
                             dense_width_name=10, dense_width_addr=10, expansion_budget=60)
        idx = PoolIndex(with_core(pool_tbl, pool), cfg, verbose=False)
        whole = idx.block(with_core(s1_tbl, s1), verbose=False)
        ref = {(int(a), int(b)): float(s) for a, b, s in zip(whole["s1_row"], whole["pool_row"], whole["rank_score"])}
        got, n = {}, s1_tbl.num_rows
        for a in range(0, n, 17):
            chunk = idx.block(with_core(s1_tbl, s1).slice(a, 17), verbose=False)
            for r, p, s in zip(chunk["s1_row"], chunk["pool_row"], chunk["rank_score"]):
                got[(int(r) + a, int(p))] = float(s)
        self.assertEqual(set(got), set(ref))
        for k, v in ref.items():
            self.assertAlmostEqual(got[k], v, places=5)
        self.assertGreater(len(ref), 0)


class ThreadingTest(unittest.TestCase):
    def test_concurrent_chunk_queries_equal_sequential(self):
        """run_blocking.py --workers N queries one read-only pool index from several threads."""
        from concurrent.futures import ThreadPoolExecutor
        from blocking import PoolIndex
        s1_tbl, pool_tbl, s1, pool = make_tables(4, n_pool=600, n_s1=120)
        cfg = BlockingConfig(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, cap_addr_bigram=5,
                             cap_name_bigram=5, prefix_lens=(3,), cap_prefix=5, rescore=True, max_candidates=5,
                             dense_width_name=10, dense_width_addr=10, expansion_budget=200)
        idx = PoolIndex(with_core(pool_tbl, pool), cfg, verbose=False)
        s1c = with_core(s1_tbl, s1)
        starts = list(range(0, s1c.num_rows, 20))
        seq = [idx.block(s1c.slice(a, 20), verbose=False) for a in starts]
        with ThreadPoolExecutor(4) as ex:
            par = list(ex.map(lambda a: idx.block(s1c.slice(a, 20), verbose=False), starts))
        for x, y in zip(seq, par):
            for k in x:
                np.testing.assert_array_equal(x[k], y[k], err_msg=k)


class RescoreTest(unittest.TestCase):
    CFG = dict(df_cap=12, k_rare_name=2, k_rare_addr=3, exact_key_cap=6, cap_addr_bigram=5, cap_name_bigram=5,
               dense_width_name=10, dense_width_addr=10)   # widths above the max token count -> no truncation

    def test_rescoring_changes_ranking_not_the_candidate_set(self):
        s1_tbl, pool_tbl, _, _ = make_tables(4)
        a = block_partition(s1_tbl, pool_tbl, BlockingConfig(**self.CFG, max_candidates=10**6, rescore=False), verbose=False)
        b = block_partition(s1_tbl, pool_tbl, BlockingConfig(**self.CFG, max_candidates=10**6, rescore=True), verbose=False)
        self.assertEqual(set(pairs_of(a)), set(pairs_of(b)))

    def test_overlap_scores_match_idf_dice_reference(self):
        s1_tbl, pool_tbl, s1, pool = make_tables(5)
        got = block_partition(s1_tbl, pool_tbl, BlockingConfig(**self.CFG, max_candidates=10**6, rescore=True), verbose=False)
        n_pool = len(pool[0])

        def dice_fn(q_toks, p_toks):
            df = Counter(t for row in p_toks for t in set(row))
            idf = lambda t: float(np.float32(math.log((n_pool + 1.0) / (df.get(t, 0) + 1.0))))  # noqa: E731
            def f(i, j):
                a, b = q_toks[i], p_toks[j]
                sh = sum(idf(t) for t in a if t in b)                      # S1 slots present in the candidate
                den = sum(idf(t) for t in a) + sum(idf(t) for t in b)      # unknown tokens: df=0 -> max idf
                return 0.0 if den == 0 else 2 * sh / den
            return f

        f_name, f_addr = dice_fn(s1[0], pool[0]), dice_fn(s1[1], pool[1])
        for i, (r, c) in enumerate(zip(got["s1_row"], got["pool_row"])):
            self.assertAlmostEqual(float(got["ov_name"][i]), f_name(int(r), int(c)), places=3)
            self.assertAlmostEqual(float(got["ov_addr"][i]), f_addr(int(r), int(c)), places=3)

    def test_truncation_keeps_the_best_by_rank_score(self):
        s1_tbl, pool_tbl, _, _ = make_tables(1)
        full = block_partition(s1_tbl, pool_tbl, BlockingConfig(**self.CFG, max_candidates=10**6, rescore=True), verbose=False)
        cut = block_partition(s1_tbl, pool_tbl, BlockingConfig(**self.CFG, max_candidates=3, rescore=True), verbose=False)
        by_full, by_cut = {}, {}
        for r, s in zip(full["s1_row"], full["rank_score"]):
            by_full.setdefault(int(r), []).append(float(s))
        for r, s in zip(cut["s1_row"], cut["rank_score"]):
            by_cut.setdefault(int(r), []).append(float(s))
        for r, scores in by_full.items():
            kept = sorted(by_cut.get(r, []), reverse=True)
            self.assertEqual(len(kept), min(3, len(scores)))
            np.testing.assert_allclose(kept, sorted(scores, reverse=True)[: len(kept)], atol=1e-5)

    def test_shared_common_words_no_longer_tie(self):
        """The failure this whole change is about. Every name word is too common to be looked up
        (df > df_cap); a single address token ('dalles') is rare enough, so the ~300 pool records
        sharing it all tie on the rare-token score and the per-entity cap keeps an arbitrary few. Only the
        true match also shares all three (common) name words -- which the full-overlap ranking sees."""
        import dataclasses
        words = ["pediatric", "dental", "associates"]
        n_noise, n_dalles = 700, 300
        # noise names: two of the three words (each word then appears in ~2/3 of 700 rows = ~467 > df_cap)
        names = [[w for k, w in enumerate(words) if k != i % 3] + [f"x{i}"] for i in range(n_noise)]
        addrs = [(["dalles"] if i < n_dalles else []) + [f"n{i}"] for i in range(n_noise)]
        names.append(list(words))                                   # the true match is the LAST pool row
        addrs.append(["dalles"])                                    # no token unique to it: retrieval is via 'dalles' alone
        true_row = n_noise
        def to_arrow(n, a):
            return pa.table({
                "name_tokens": pa.array(n, type=pa.list_(pa.string())), "address_tokens": pa.array(a, type=pa.list_(pa.string())),
                "name_normalized": pa.array([" ".join(x) for x in n], type=pa.large_string()),
                "name_compact": pa.array(["".join(x) for x in n], type=pa.large_string()),
                "postal_code": pa.array([None] * len(n), type=pa.large_string())})
        cfg = BlockingConfig(df_cap=400, k_rare_name=3, k_rare_addr=3, exact_key_cap=0, max_candidates=5,
                             dense_width_name=6, dense_width_addr=6)
        s1_tbl, pool_tbl = to_arrow([list(words)], [["dalles"]]), to_arrow(names, addrs)
        old = block_partition(s1_tbl, pool_tbl, dataclasses.replace(cfg, rescore=False), verbose=False)
        new = block_partition(s1_tbl, pool_tbl, dataclasses.replace(cfg, rescore=True), verbose=False)
        self.assertNotIn(true_row, old["pool_row"].tolist(), "old ranking: complete tie, cap keeps the lowest pool rows")
        self.assertIn(true_row, new["pool_row"].tolist(), "new ranking must keep the true match under the same cap")
        self.assertEqual(int(new["pool_row"][int(np.argmax(new["rank_score"]))]), true_row)


if __name__ == "__main__":
    unittest.main()
