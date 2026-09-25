# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [fill in]  
**Team Members:** [fill in]  
**Submission Date:** 2026-09-25 (pipeline built and validated on this date)

---

## 1. Executive Summary

The solution is **candidate generation in three cheap steps + a supervised matcher**. Inside each country partition, (1) inverted
indices retrieve every pool record that shares a *specific* key with the Source-1 (S1) entity (rare tokens, exact keys, **address / name
bigrams**, **compact-name prefix**); (2) a **small learned ranker** over the retrieval evidence keeps a shortlist of the best 30
(96.978% of all true pairs); (3) a **light learned candidate filter** (a few character-level similarities on top of the
retrieval evidence) drops shortlisted pairs whose match probability is below 0.015. The final candidate set holds
**96.549% of all true pairs with 6.08 candidates per entity** (the first design held 82% with 175; on the test split
7.1 per entity). A LightGBM matcher over ~90 vectorized pairwise features scores exactly these candidates, and the decision
threshold is chosen on the challenge's own metric (per-entity F0.5, singleton rule included) using S1 *entities* held out from training.
On 7,500 hold-out entities never used for any decision the macro F0.5 is **0.9647** (simple baselines: 0.48 - 0.73).

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data (`code/business_entity_resolution/reports/data_analysis_report.md`):
* **Scale**: 2.2M S1, 5.0M S2, 5.3M S3 training records (1.7M / 4.9M / 5.1M test). A Cartesian product is impossible (about 2x10^13 same-country S1 x pool pairs in the training split alone).
* **Country is a perfect blocking key**: 0 of 7,638,365 true pairs cross a country boundary. The test set adds France (15% of S1) which never
  appears in training, so `country` is used only to *partition*, never as a model feature or a hard-coded vocabulary.
* **Structure**: S1 is deduplicated; each S2/S3 record belongs to at most one S1 entity; 5.6% of S1 entities are true singletons, the mean
  entity has 3.46 matches (max 11), 80% have matches in both sources. A singleton predicted with any match scores 0, a correctly empty
  prediction scores 1, so singleton control is worth ~5.6 points of the metric.
* **Noise**: only 21.9% of true pairs have identical names after normalization; 14.4% share no name token at all. Names carry typos, accent
  corruption, legal-suffix and word-order changes, concatenated / domain-style forms (`gurgaontechnologiesclinic.com`), and 11-19% of S2/S3 names
  are non-ASCII (about half plain Latin text with corrupted accents, the rest Devanagari and 8 other Indic scripts; 13.9% of true pairs have exactly one non-ASCII name)
  while their addresses stay mostly Latin.
  Addresses lose components (3.4% are empty), get abbreviated (`Rd`, `Ave`, `KA` for `Karnataka`), reordered, zero-padded (`019` vs `19`) or truncated.
  Postal codes exist in only ~7% of addresses.

### 2.2 Solution Strategy
**Approach Type:** Blocking (multi-key retrieval -> learned ranker -> learned candidate filter) -> gradient-boosted pairwise classifier -> entity-level threshold.
**Core Innovation:** (1) *retrieval keys that stay specific when every word is common* (adjacent-token bigrams, compact-name prefix) combined with
a *learned cheap ranker* instead of a hand-written score - the earlier rare-token score ties massively (hundreds of candidates share the one rare
address token) and its per-entity cap kept an arbitrary subset; (2) an *adaptive, learned candidate cut*: instead of a fixed number per entity,
each entity keeps only the shortlisted records that are plausible - one or two for an easy entity, more for an ambiguous one, none when nothing in
the pool is plausible (which is also the right answer for a singleton) - 29.78 -> 6.08 candidates per entity for
96.978% -> 96.549% recall and no measurable change in F0.5; (3) evaluation that mimics the real task (entity-level split, exact metric,
out-of-fold candidate cuts, a hold-out never used for selection); (4) everything vectorized (int-coded tokens, sparse token matrices,
multithreaded C++ string similarity), so the 1.7M-entity test split runs end to end on one laptop (see 3.1).

---

## 3. Candidate Generation (Blocking)

Candidate generation has three steps; the output of the last one is the candidate set the matcher scores and `candidate_pairs.tsv` lists.

- **Step 1 - retrieval, blocking keys used** (a pool record is retrieved if ANY key fires; every key family has a document-frequency cap):
  1. rare unigram tokens (each S1 entity looks up its 3 rarest name / 4 rarest address tokens with pool df <= 300);
  2. exact keys: normalized name, compact name, postal code;
  3. **address bigrams** and **name bigrams** (adjacent tokens, order-insensitive, df <= 300) - retrieve non-Latin-name pairs through their clean address and
     missing-address pairs through their name (85% of the pairs the first design never retrieved share an address bigram);
  4. **compact core-name prefix** (first 8 characters of the space-free name without legal suffixes) - concatenated / domain-style names;
  5. leading-zero-insensitive numeric tokens (`019` = `19`).
  Country is a hard partition (0 of 7.6M true pairs cross it).
- **Step 2 - ranking to a shortlist**: every retrieved pair (~450 per entity) gets cheap evidence (IDF-weighted Dice overlap of name and address over
  *all* tokens on padded int token matrices at ~1 us/pair, hit counts per rule, token counts); a small LightGBM ranker (trained on 12,000 S1 entities
  outside the matcher's training sample, so it never saw validation labels) keeps the best 30 per entity: 29.78 per entity,
  96.978% of all true pairs.
- **Step 3 - learned candidate filter** (`candidate_filter.py`, `train_candidate_filter.py`, `run_candidate_cut.py`): most of the 30 are obviously wrong
  (the median shortlisted pair has a ranker probability below 0.01), so a second, still cheap LightGBM filter scores each shortlisted pair from the
  retrieval evidence (overlaps, rule hits, exact-key flags, token counts, rank in the shortlist) plus seven character-level similarities
  (token-set / partial ratio of the names, ratio / Jaro-Winkler of the legal-suffix-free core names, token-set / partial / plain ratio of the addresses)
  and missing-field / house-number flags; a pair survives if its probability is at least **0.015**. The floor is the largest value that still keeps
  at least 99.5% of the shortlisted true pairs of the matcher's validation entities. The filter is cross-fitted (5 folds of S1 entities): every
  training pair is cut by a filter that never saw its entity, so the matcher is trained and validated on candidate sets cut exactly like the test
  split's (which uses the filter fitted on all folds). The ranker probability and the pool-size-dependent statistics are deliberately *not* filter inputs:
  the unseen test country France has a 3-4x smaller pool and a visibly inflated ranker probability (median at shortlist rank 10: 0.50 in France,
  0.03 in India, 0.01 in the US), and leaving them out cost nothing on validation. Most important filter inputs: `block_rank` (62%), `ov_addr` (11%), `addr_token_set` (4%), `core_ratio` (4%), `name_partial` (3%), `ov_name` (3%).
  Size / recall trade-off of the floor (out-of-fold, 50,000 training entities):

| probability floor | candidates / entity | recall of all true pairs | entities without candidates | true singletons without candidates |
|---|---|---|---|---|
| 0 (no filter) | 29.78 | 96.978% | 0.01% | 0.1% |
| 0.001 | 10.99 | 96.939% | 0.01% | 0.1% |
| 0.002 | 9.18 | 96.896% | 0.01% | 0.1% |
| 0.003 | 8.36 | 96.867% | 0.01% | 0.1% |
| 0.005 | 7.51 | 96.809% | 0.01% | 0.1% |
| 0.0075 | 6.94 | 96.746% | 0.02% | 0.3% |
| 0.01 | 6.58 | 96.679% | 0.04% | 0.6% |
| 0.0125 | 6.31 | 96.612% | 0.06% | 1.0% |
| 0.015 **(chosen)** | 6.08 | 96.549% | 0.08% | 1.3% |
| 0.02 | 5.72 | 96.387% | 0.12% | 2.0% |
| 0.025 | 5.45 | 96.236% | 0.17% | 2.8% |
| 0.03 | 5.25 | 96.118% | 0.23% | 3.8% |
| 0.04 | 4.94 | 95.889% | 0.39% | 6.3% |
| 0.05 | 4.73 | 95.677% | 0.56% | 9.2% |

Floor chosen on the matcher's validation entities: the largest grid value that loses at most 0.5% of their shortlisted true pairs. Filter features by gain: `block_rank` 61.6%, `ov_addr` 11.3%, `addr_token_set` 4.2%, `core_ratio` 3.6%, `name_partial` 3.4%, `ov_name` 3.3%, `n_strategies` 2.3%, `housenum_match` 2.2%.

- **Candidate pairs generated**: 12,288,781 for 1,732,544 test S1 entities (**7.1 per entity**; the shortlist had 51,672,105,
  29.8 per entity); candidate reduction ratio > 0.99999 versus the same-country cross product. `output/candidate_pairs.tsv` is exactly
  this set, the matcher scores exactly these pairs, and every predicted match is one of them. Per country: B.5.
- **How true matches were not lost**: recall was measured per step, per rule and per cap on ground truth, and the design was iterated on the failure
  categories: first design 82.1% recall at 175 candidates/entity -> retrieval + ranker 96.978% at 29.78 -> final 96.549% at
  6.08 (tables in Appendix B.2). A plain floor on the ranker probability needs about 15 candidates per entity for the recall the filter keeps
  with about 6 (experiment 9 in B.2). The misses are dominated by non-Latin names with sparse addresses and by candidates with missing addresses and
  scrambled names (`reports/blocking_report.md`).

### 3.1 Scalability ("blocking has to scale")
* **No pairwise comparison outside the retrieved set.** Retrieval is inverted-index lookups: each S1 entity reads at most K rare-token postings and the
  bigram / prefix / exact-key postings of its own keys, and every key with a document frequency above its cap (300) is never read, so the work per entity is
  bounded by (number of keys x cap) no matter how large the pool grows; the ranker and the filter cost O(1) per retrieved / shortlisted pair, and the
  filter only ever sees 30 pairs per entity. Total cost is linear in the number of S1 entities (queries) plus linear in the pool (index build).
* **Partitioned and streaming.** Country partitions are independent (0 of 7.6M true pairs cross a country), the pool index is built once per partition and
  queried in S1 chunks with bounded memory (`--chunk-s1`, `--expansion-budget`), and every stage is resumable per chunk / country. At larger scale the same
  code shards by partition (country, then region / postal prefix) and by S1 chunk across machines.
* **Measured on one laptop** (8 cores, 15 GB RAM, no GPU): test split = 1.73M S1 entities against 10M pool records: retrieval + ranker 71 min, candidate
  filter 13 min, featurize + score the 12,288,781 candidates 15 min, decisions and output files under a minute.

---

## 4. Matching Model

**Features used** (~90, all country-agnostic, NaN when a field is missing on either side):
- Name features: exact equality (normalized / legal-suffix-free core), Levenshtein ratio, Jaro-Winkler, token-sort, token-set, partial ratio (on the normalized
  name and on the core name), token Jaccard / Dice / containment in both directions / IDF cosine / IDF overlap, numeric-token overlap, length differences,
  script code of both names (fixed vocabulary), and rule evidence from blocking.
- Address features: exact equality, the same five string similarities, token Jaccard / Dice / containment / IDF cosine / IDF overlap, numeric-token overlap
  (house / plot numbers), house-number and postal-code agreement, length differences.
- Other: S2 vs S3, blocker evidence (overlap scores, hits per rule, ranker score), position of the candidate inside its entity (rank / gap to the best by several
  scores), ambiguity context (how many candidates of the entity already agree strongly on name / address), and cross-field products / min / max.
  Pool-ambiguity statistics - how many pool records / S1 entities carry the identical name, core name or address, plus IDF sums and rare-token counts - are used;
  they are the largest single gain of the matcher (see B.1) but their scale depends on the size of the country pool, which is a documented risk for France.

**Model type:** LightGBM (MIT licence, 733 trees, 127 leaves) trained on S1-entity-disjoint data; sklearn HistGradientBoosting and LogisticRegression were
trained on the same features for the model comparison (Appendix B). The matcher is trained, validated and applied on the *final* candidate sets (after the
filter), so its within-entity features (rank, gaps, number of candidates, how many candidates agree strongly) describe exactly the set it decides on.
Two matcher variants were trained on identical data: the default feature set (validation 0.9628, hold-out 0.9647) and a France-robust set that drops the features whose scale depends on the size of the country pool (validation 0.9552, hold-out 0.9560). Because France has no labels and a 3-4x smaller pool, the robust set is promoted whenever it is within 0.002 of the default on both validation and hold-out: **default features were promoted**.

**Threshold selection method:** search on validation entities for the threshold maximising macro F0.5 *under the challenge's own scoring rule*
(per entity, singleton = 1 only when empty). Per-source (S2/S3) thresholds and an entity-level gate (the best candidate must clear a higher bar) were tried and
adopted only when they beat the simple rule by a margin; a single global threshold (0.74) was selected.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** hold-out entities (never used for training, early stopping, threshold or rule selection): **0.9647**
  (validation entities: 0.9628; the ceiling if every true pair that is a candidate were predicted is 0.9876). Reference points on the same
  validation entities: exact normalized name 0.481, fuzzy name + address rule 0.677, best similarity-score threshold 0.726, TF-IDF cosine threshold 0.722,
  LightGBM 0.9628. On the hold-out: singleton accuracy 0.976, mean precision 0.985, mean recall 0.918, candidate recall 0.966.
- **Common false positives (wrong merges):** False merges are rare (160 false merges on validation entities that do have matches and 14 true singletons that received a prediction; table B.3, `reports/edge_case_report.md`). Two patterns explain them. (1) About half are pool records that truly belong to *another* S1 entity - typically a branch or chain outlet with the same name on the same street or in the same town (e.g. `Internal Medicine Legacy Health Inc | 6355 Shedd Road, Eloy` versus a record of `... | 6376 Shedd Road`). That competing entity is absent from a 50k-entity training sample but present in the full test world; enforcing 'each S2/S3 record belongs to at most one S1 entity' is therefore the most promising remaining precision gain (Appendix C). (2) The rest are distractor records that belong to no S1 entity but carry an almost identical name and a similar address.
- **Common false negatives (missed matches):** Misses split into candidate-generation misses (3.44% of true pairs: never retrieved, cut from the shortlist, or removed by the candidate filter) and matching misses (4.48% of true pairs are candidates that the matcher rejected). Candidate-generation misses are dominated by non-Latin / accented names whose address is sparse or written differently (e.g. a Tamil name with `TN` instead of `Tamil Nadu`), candidates with no address and a scrambled name (`Ruiz Erndoicrighltoyg`), and concatenated names with a noisy address. The weakest edge cases are candidates with a missing address (recall 0.841 - only the name is available and equal names exist elsewhere), non-Latin names (recall 0.857) and 'ambiguous' entities in which a false candidate also scores >= 0.3 (F0.5 0.907, precision 0.926).

Per-edge-case results (26 cases of the brief, with counts, candidate recall, precision, recall, F0.5, failure taxonomy and examples) are in
`code/business_entity_resolution/reports/edge_case_report.md`; the summary is in Appendix B.

---

## 6. Conclusion

Candidate generation was the decisive lever: multi-key retrieval with a learned ranker lifted the recall of the candidate set from 82% to 97% at one sixth of the first design's volume, and a light learned candidate filter then shrank the set the matcher scores from 30 to 6.08 candidates per entity while keeping 96.549% of all true pairs and the same F0.5 - the smaller candidate set the challenge asks for. Features that measure how ambiguous a name or address is inside its country were the most valuable matcher addition. The pipeline reaches a hold-out macro F0.5 of 0.9647 against 0.48 - 0.73 for simple baselines and a ceiling of 0.9876 given the candidate set, and regenerates both submission files from the raw data on one laptop. The main remaining headroom is enforcing the one-S1-owner-per-record constraint, transliteration of non-Devanagari scripts, and labelled French data, which the challenge does not provide.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (all source in `src/`, unit tests in `tests/`, `README.md`, pinned `requirements.txt`). Pipeline entry points, in order:
`run_all.py` runs the whole chain: `prepare_cache.py` (normalize once, chunked/resumable) -> `train_ranker.py` -> `run_blocking.py --split train` (retrieval +
ranker shortlist) -> `train_candidate_filter.py` (cross-fitted filter + floor) -> `run_candidate_cut.py --split train` -> `run_features.py --split train` ->
`run_train.py` (matcher, thresholds, model comparison, baselines, experiment log) -> `promote_model.py` -> `run_blocking.py --split test` ->
`run_candidate_cut.py --split test` -> `run_score.py --split test` -> `run_infer.py` (writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`)
-> `utils/validate_submission.py`. `run_blocking_report.py` and `run_edge_cases.py` produce the diagnostics. Exact commands are in the README.

### B. Additional Results
#### B.1 Experiment log (validation entities; last row = hold-out entities never used for any decision)
Blocking strategy, features, model, threshold, candidate recall, precision, recall, F0.5, singleton accuracy and average candidates per entity of every
experiment (baselines 1-3b, LightGBM with the three decision rules, sklearn model families, hold-out):

| experiment | features | model | threshold | candidate recall | precision | recall | F0.5 | singleton acc. | avg cand./entity |
|---|---|---|---|---|---|---|---|---|---|
| ORACLE_given_blocking (predict exactly the true pairs that are candidates) |  | oracle |  | 0.9656 | 0.9962 | 0.9650 | 0.9876 | 1.0000 | 6.0755 |
| predict_nothing |  | constant |  | 0.9656 | 0.0000 | 0.0000 | 0.0549 | 1.0000 | 6.0755 |
| B1_exact_normalized_name | normalized name equality | rule |  | 0.9656 | 0.5980 | 0.2900 | 0.4813 | 0.8180 | 6.0755 |
| B2_fuzzy_name_and_address | name Levenshtein>0.85 & address Levenshtein>0.5 | rule | 0.85 / 0.5 | 0.9656 | 0.7831 | 0.5168 | 0.6770 | 0.6553 | 6.0755 |
| B3_similarity_score_threshold | 0.5*name_ratio + 0.5*addr_ratio | rule | 0.68 (searched) | 0.9656 | 0.7667 | 0.8098 | 0.7263 | 0.1553 | 6.0755 |
| B3b_tfidf_cosine_threshold | 0.5*name IDF-cosine + 0.5*address IDF-cosine (word tokens) | rule | 0.63 (searched) | 0.9656 | 0.7784 | 0.7377 | 0.7217 | 0.2209 | 6.0755 |
| B4_lightgbm_global_threshold | 90 pairwise features | LightGBM (733 trees, 127 leaves) | 0.74 (searched) | 0.9656 | 0.9828 | 0.9183 | 0.9628 | 0.9660 | 6.0755 |
| B5_lightgbm_per_source_thresholds | 90 pairwise features | LightGBM (733 trees, 127 leaves) | S2 0.78 / S3 0.64 (searched) | 0.9656 | 0.9827 | 0.9206 | 0.9633 | 0.9660 | 6.0755 |
| M1_sklearn_HistGradientBoosting | 90 pairwise features | HistGradientBoosting (250 iters, 500k rows, 46s) | 0.70 (searched) | 0.9656 | 0.9807 | 0.9144 | 0.9599 | 0.9612 | 6.0755 |
| M2_sklearn_LogisticRegression | 90 pairwise features (+missing indicators) | LogisticRegression (500k rows, 13s) | 0.67 (searched) | 0.9656 | 0.9530 | 0.8523 | 0.9140 | 0.8495 | 6.0755 |
| B6_lightgbm_entity_gated | 90 pairwise features | LightGBM (733 trees, 127 leaves) | gate 0.00; S2 0.78 / S3 0.64 (searched) | 0.9656 | 0.9827 | 0.9206 | 0.9633 | 0.9660 | 6.0755 |
| FINAL_holdout_entities (never used for any decision) | 90 pairwise features | LightGBM | global rule, thresholds from validation | 0.9658 | 0.9849 | 0.9176 | 0.9647 | 0.9758 | 6.1005 |

#### B.2 Candidate generation experiments
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


#### B.3 Failure taxonomy and edge cases (validation entities)
| type | count | share of true pairs |
|---|---|---|
| A blocking (true match not a candidate) | 894 | 3.44% |
| B matching: true candidate rejected | 1,164 | 4.48% |
| B matching: false merge on an entity with matches | 160 | 0.62% |
| C singleton with a predicted match | 14 | 0.05% |

Per edge case (the 26 cases of the brief that can be evaluated on the labelled data; entities may belong to several cases):

Entities can belong to several cases. Precision / recall are pooled over the case's entities; F0.5 is the macro mean.

| edge case | S1 entities | candidate recall | precision | recall | macro F0.5 | A blocking failures | B matching failures | C singleton failures |
|---|---|---|---|---|---|---|---|---|
| 1 true singleton | 412 |  | 0.000 |  | 0.966 | 0 | 0 | 14 |
| 2 one-to-one match | 420 | 0.971 | 0.972 | 0.898 | 0.887 | 12 | 42 | 0 |
| 3 multiple matches | 6668 | 0.966 | 0.994 | 0.921 | 0.967 | 882 | 1,282 | 0 |
| 4 cross-source match (S2 and S3) | 6043 | 0.966 | 0.994 | 0.922 | 0.968 | 812 | 1,203 | 0 |
| 5 multiple matches within one source | 5759 | 0.966 | 0.994 | 0.922 | 0.969 | 809 | 1,170 | 0 |
| 6 name-only strong (name>=.85, address weak/missing) | 1342 | 0.970 | 0.994 | 0.887 | 0.951 | 169 | 492 | 0 |
| 7 address-only strong (address>=.85, name weak/missing) | 601 | 0.965 | 0.992 | 0.898 | 0.957 | 89 | 191 | 0 |
| 8 name+address both noisy (both < .7) | 849 | 0.969 | 0.994 | 0.896 | 0.954 | 110 | 281 | 0 |
| 9 name collision (false candidate with name>=.9) | 2511 | 0.942 | 0.986 | 0.890 | 0.944 | 493 | 541 | 8 |
| 10 address collision (false candidate with address>=.9) | 2047 | 0.964 | 0.991 | 0.922 | 0.964 | 244 | 341 | 5 |
| 11 missing name (either side) | 0 |  |  |  |  |  |  |  |
| 12 missing address (either side) | 865 | 0.976 | 0.993 | 0.841 | 0.929 | 86 | 510 | 0 |
| 13 multiple missing fields | 0 |  |  |  |  |  |  |  |
| 14 normalization-only (identical after normalization) | 4656 | 0.975 | 0.994 | 0.935 | 0.976 | 471 | 843 | 0 |
| 15 token-order (token-sort>=.95, plain ratio<.9) | 1429 | 0.976 | 0.995 | 0.943 | 0.980 | 147 | 228 | 0 |
| 16 typo (name JW>=.9 but not identical) | 5454 | 0.975 | 0.994 | 0.932 | 0.974 | 525 | 1,043 | 0 |
| 17 transliteration / non-Latin name | 821 | 0.930 | 0.988 | 0.857 | 0.933 | 219 | 261 | 0 |
| 22 ambiguous (true candidate AND a false candidate p>=.3) | 621 | 0.957 | 0.926 | 0.907 | 0.907 | 96 | 269 | 0 |

Cases that are structural rather than measurable: country conflict (0 of 7.6M true pairs cross a country, candidates are generated per country partition), unseen country
(France, test only), duplicate candidates (candidates are a deduplicated union), empty ground truth (parsed as zero matches), output-subset invariant, S1 ids in
predictions and duplicate predicted ids (all verified on the real output files by `run_edge_cases.py` and `utils/validate_submission.py`).

#### B.4 Source 2 versus Source 3
| source | true pairs in candidates | mean name similarity (true pairs) | mean address similarity (true pairs) | address missing | non-Latin name | name identical after normalization | precision | recall (of candidate true pairs) |
|---|---|---|---|---|---|---|---|---|
| S2 | 12079 | 0.837 | 0.849 | 0.037 | 0.077 | 0.294 | 0.992 | 0.954 |
| S3 | 13025 | 0.836 | 0.775 | 0.036 | 0.042 | 0.299 | 0.993 | 0.953 |

#### B.5 Test-set prediction statistics per country (France has no labels: only prediction rates can be compared)
| country | s1_entities | candidate_pairs | candidates_per_entity | share_without_candidates | entities_with_match | share_with_match | predicted_matches_per_entity | predicted_matches_per_entity_that_has_one | share_s2_of_matches | mean_score_of_kept_pairs | share_best_score_ge_0.97 | share_best_score_0.4_to_0.9 | share_best_score_below_0.2_or_no_candidate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| France | 259,452 | 2,417,100 | 9.3162 | 0.0004 | 247,200 | 0.9528 | 3.5269 | 3.7017 | 0.4787 | 0.9859 | 0.9396 | 0.0174 | 0.0267 |
| India | 809,986 | 5,879,544 | 7.2588 | 0.0001 | 755,038 | 0.9322 | 3.1282 | 3.3559 | 0.4862 | 0.9877 | 0.9149 | 0.0157 | 0.0516 |
| US | 663,106 | 3,992,137 | 6.0204 | 0.0009 | 623,878 | 0.9408 | 3.3520 | 3.5628 | 0.4837 | 0.9914 | 0.9336 | 0.0077 | 0.0491 |

#### B.6 Most important model features (share of total gain)
`rank_score` (43.2%), `at_num_jaccard` (10.4%), `block_gap_to_best` (6.6%), `name_ratio` (3.1%), `nt_idf_ov2` (1.6%), `name_partial` (1.6%), `s1_core_freq` (1.5%), `name_jw` (1.4%)

### C. Limitations and honest notes
* France is unseen in training and has no labels, so its accuracy cannot be measured. Nothing in the pipeline is country-specific (`country` only partitions the
  data). Label-free checks on the test predictions (`reports/inference_summary.json`):

| | France | India | US |
|---|---|---|---|
| S1 entities | 259,452 | 809,986 | 663,106 |
| candidates per entity | 9.32 | 7.26 | 6.02 |
| entities without any candidate | 0.0% | 0.0% | 0.1% |
| entities with a predicted match | 95.3% | 93.2% | 94.1% |
| predicted matches per entity that has one | 3.70 | 3.36 | 3.56 |
| best candidate score >= 0.97 | 94.0% | 91.5% | 93.4% |
| best candidate score in [0.4, 0.9) | 1.7% | 1.6% | 0.8% |
| best score < 0.2 or no candidate | 2.7% | 5.2% | 4.9% |

  France has more candidates per entity and fewer entities without any plausible candidate than India and the US. That can mean more true matches per French
  entity or easier matching, or over-merging because its pool is 3-4x smaller and its addresses share many tokens inside a region (`Pays de la Loire`), which
  inflates token-overlap evidence; without labels the two cannot be told apart. Mitigations in place: the candidate filter does not use the ranker probability or
  pool-size-dependent statistics, and a matcher variant without pool-size-dependent features was trained for comparison (Section 4); it was not promoted because it
  is worse on the labelled countries. A small-pool simulation was not run.
* Non-Devanagari Indic scripts are not transliterated (only Devanagari has a rule-based transliterator); such records are matched through their addresses.
* The matcher was trained on a 50,000-entity sample of the 2.2M training entities (all candidates of those entities); more training entities were not tried.
* The pool-side constraint "each S2/S3 record belongs to at most one S1 entity" is not enforced at inference; on validation, false positives that are records
  owned by another S1 entity account for 51% of all false positives (89 of 176; removing exactly those would raise validation F0.5 from 0.9628 to 0.9658, an
  upper bound of +0.30 points; `src/exp_conflict_potential.py`).
* No external data, APIs, geocoders or pre-trained models are used anywhere; all statistics come from the provided files.
