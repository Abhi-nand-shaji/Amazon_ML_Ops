# Blocking experiment log (candidate generation)

All numbers are measured on the training sample (ground truth known). "Recall" = share of ALL true (S1, S2/S3) pairs
that are in the candidate set; the matcher can never recover a pair that is not a candidate.

| # | Blocking strategy | Data | Retrieval recall (no cap) | Recall at the cap | Candidates / S1 entity |
|---|---|---|---|---|---|
| 1 | Legacy: rare unigram tokens (K=4 name / 5 address, df<=1500) + exact keys (name, compact name, postal); ranked and capped by the rare-token score | 50k entities | - | **82.1% @200**, 78% @60 | 175 (cap 200) |
| 1b | same, uncapped retrieval | India 4,000 entities | 88.8% | 77% @50 | 1,057 |
| 2 | + address / name bigrams + compact-name-prefix keys (df caps 300); capped by a hand-written score (exact-key bonuses) | India 4,000 | **96.4%** | 84.2% @50, 86.1% @60 | 1,273 |
| 3 | same retrieval, capped by the plain sum `ov_name + ov_addr` (IDF-weighted overlap of ALL tokens) | India 4,000 | 96.4% | 85.1% @50 | 1,273 |
| 4 | same retrieval, capped by `ov_name + 1.5*ov_addr` | India 4,000 | 96.4% | 92.1% @50, 90.3% @20 | 1,273 |
| 5 | same retrieval, capped by a **learned ranker** (LightGBM on cheap blocker evidence) | India 4,000 (2-fold CV) | 96.4% | **96.1% @50, 95.6% @20** | 1,273 |
| 6 | tighter retrieval caps (df<=300, K=3/4) + learned ranker (**final**) | India 4,000 | 95.7% | ~95.5% @30 | 444 pre-cap |
| 7 | retrieval + ranker shortlist, full 50,000-entity sample (entities never used to train the ranker) | India + US | 97.0% | **96.98% @30** | 29.78 |
| 8 | shortlist cut to a fixed top-k by the ranker | 50k | - | 96.81% @20, 96.60% @15, 96.01% @10 | 20 / 15 / 10 |
| 9 | shortlist cut by a floor on the ranker probability (0.01 / 0.05 / 0.1) | 50k | - | 96.77% / 96.32% / 95.92% | 14.6 / 9.6 / 8.0 |
| 10 | shortlist cut by the **learned candidate filter** (probability >= 0.015; out-of-fold, **final**) | 50k | - | **96.55%** | **6.08** |

Effect of the cut on the matcher (experiments 8-10 replayed with the matcher trained on the 30-candidate sets, validation
entities, threshold re-searched; `src/exp_candidate_cut.py`): macro F0.5 0.9625 with all
30 candidates; fixed top-10 0.9614; ranker floor 0.01 / 0.05 / 0.1: 0.9623 / 0.9610 / 0.9602; learned filter at 6.1 candidates
per entity: **0.9625 (unchanged)**, and still 0.9624 at 5.2 per entity. The ranker probability is a poor basis for the cut because
it only sees token overlap (true matches written as a handle / domain / scrambled name, e.g. `@znfederation`, sit at shortlist rank
20+), and it is inflated in France (same-region addresses share many tokens): a ranker floor of 0.05 keeps 25 candidates per French
entity against 11 (India) and 8 (US), and a fixed top-10 removes 2.6% of the matches predicted in France against 1.0% / 0.2%.
Filter variants with and without the pool-size-dependent statistics and the ranker probability were equal on validation (6.4-6.6
candidates per entity at the same recall), so the final filter uses neither.

Retrieval-cap sweep (India, 4,000 entities, retrieval only): default df<=1500/K=4/5 -> 96.41% at 1,273 pairs/entity;
df<=500 -> 95.91% at 554; df<=300, K=3/4 -> 95.67% at 444; df<=150 -> 95.49% at 377; no unigrams at all -> 94.70% at 338;
bigram caps 150 -> 94.76% at 304; bigram caps 600 -> 96.37% at 736. The final configuration trades 0.7 points of
retrieval recall (worth ~0.15 points of F0.5) for a 65% smaller candidate volume.

Learned ranker vs hand-written scores on held-out entities (India + US, 3,000 entities; % of all true pairs kept at k
candidates per entity): learned 95.6 / 96.6 / 96.8 (k = 10 / 20 / 30); `ov_name + 1.5 ov_addr` 91.4 / 93.3 / 93.9;
legacy rare-token score 54.9 / 60.0 / 63.6 (`reports/ranker_report.json`). The retrieval ceiling is 97.0%: at 30 candidates the ranker loses only 0.17 points.
