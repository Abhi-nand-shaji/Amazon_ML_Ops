# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [fill in]
**Team Members:** [fill in]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

The solution has four layers. Each one was added only when it improved the challenge metric on held-out S1 *entities*.

1. **Candidate generation (CPU).** Inside each country partition, inverted indices retrieve every pool record that shares a
   *specific* key with the Source-1 (S1) entity: rare tokens, exact keys, **address / name bigrams**, and a **compact-name prefix**.
   A small **learned ranker** keeps a shortlist of 30 (96.978% of all true pairs). A light **learned candidate
   filter** then keeps 6.08 candidates per entity that hold 96.549% of all true pairs. Then **second-hop retrieval** reaches pairs that no blocking key connects. An entity's confidently matched pool records (anchors) become new queries against the same indices, and a learned new-pair filter keeps the plausible results. That adds 1.3 candidates per entity on validation and raises the recall of the candidate set to 97.90% (hold-out entities).
2. **First-stage matcher.** LightGBM over ~90 vectorized pairwise features gives a match probability `p1` for every candidate pair.
   This is computed for the *whole* world, not just a sample. All 2.2M training S1 entities are blocked and cut exactly like
   the test split, so the competition for every pool record is known.
3. **Cross-encoder.** `microsoft/mdeberta-v3-base` (MIT licence, 278M parameters, multilingual) was fine-tuned on 1.17M
   labelled training pairs, with a country-adversarial head and targeted hard negatives. It reads both records as text
   and scores every candidate pair (training was done on a free Kaggle 2x T4 GPU session).
4. **Graph stacker.** A second LightGBM combines the pairwise features, `p1`, the cross-encoder score, and **collective
   (graph) features** computed over the complete candidate graph:
   * *exclusivity*: every S2/S3 record belongs to at most one S1 entity, so a record claimed by a stronger entity is
     unlikely to match;
   * *entity context*;
   * *sibling agreement* among an entity's candidates;
   * *parsed house number vs street*.

   The stacker is trained on 200,000 S1 entities that no earlier model saw. Its threshold is chosen with the
   challenge's own metric (per-entity F0.5, singleton rule included) on validation entities.

On 7,500 hold-out entities that were never used for any decision, macro F0.5 rose with each layer:
**first-stage matcher 0.9647 -> graph stacker 0.9742 -> + cross-encoder 0.9818 -> + second-hop candidates 0.9847**. The public
leaderboard gave 0.93 / 0.95 / 0.9739 for the same versions. Simple baselines score 0.48 - 0.73.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data (`code/business_entity_resolution/reports/data_analysis_report.md`):
* **Scale**: 2.2M S1, 5.0M S2 and 5.3M S3 training records (1.7M / 4.9M / 5.1M test). A Cartesian product is impossible:
  there are about 2x10^13 same-country S1 x pool pairs in the training split alone.
* **Country is a perfect blocking key**: 0 of 7,638,365 true pairs cross a country boundary. The test set adds France
  (15% of S1), which never appears in training. So `country` is used only to *partition* the data, never as a model feature
  or a hard-coded vocabulary.
* **Structure**: S1 is deduplicated, and each S2/S3 record belongs to at most one S1 entity (every pool id appears once in
  the ground truth). 5.6% of S1 entities are true singletons. The mean entity has 3.46 matches (max 11), and 80% have
  matches in both sources. A singleton predicted with any match scores 0 and a correctly empty prediction scores 1, so
  singleton control is worth ~5.6 points of the metric.
* **Noise**:
  * Names: only 21.9% of true pairs have identical names after normalization, and 14.4% share no name token at all.
    Names carry typos, accent corruption, legal-suffix and word-order changes, and concatenated / domain-style forms
    (`gurgaontechnologiesclinic.com`).
  * Scripts: 11-19% of S2/S3 names are non-ASCII. About half are plain Latin text with corrupted accents; the rest are
    Devanagari and 8 other Indic scripts. 13.9% of true pairs have exactly one non-ASCII name, while the addresses stay
    mostly Latin.
  * Addresses lose components (3.4% are empty), get abbreviated (`Rd`, `Ave`, `KA` for `Karnataka`), reordered,
    zero-padded (`019` vs `19`) or truncated. Postal codes exist in only ~7% of addresses.

### 2.2 Solution Strategy
**Approach Type:** Hybrid.
* Blocking: multi-key retrieval -> learned ranker -> learned candidate filter -> second-hop retrieval with a learned new-pair filter.
* Pairwise matching: gradient-boosted features + fine-tuned transformer cross-encoder.
* Collective graph resolution: a stacker over the complete candidate graph.
* Decision: an entity-level threshold on the exact metric.

**Core Innovations:**
1. *Retrieval keys that stay specific when every word is common* (adjacent-token bigrams, compact-name prefix), combined
   with a *learned cheap ranker* and an *adaptive, learned candidate cut*. An easy entity keeps one or two candidates, an
   ambiguous one keeps more, and none are kept when nothing in the pool is plausible (which is also the right answer for a
   singleton): 29.78 -> 6.08 candidates per entity for 96.978% -> 96.549% recall.
2. *Full-world collective resolution.* The stacker needs to know, for each pool record, which other S1 entities claim it
   and how strongly. That is only meaningful if the training world looks like the test world. So all 2.2M training S1
   entities go through the same blocking and cut as the test split. Graph features computed on this full world behave on
   training entities exactly as on test entities. On a 50k-entity sample, by contrast, most competitors of a pool record
   would be missing.
3. *A transformer cross-encoder as one input of the stacker*, not as the final decision. It sees both records as text
   (typos, transliteration, abbreviations, reordered components). The stacker adds what a pair-level model cannot know:
   exclusivity and the entity's other candidates.
4. *Evaluation that mimics the real task*: an entity-level split, the exact metric, out-of-fold candidate cuts, strict
   entity groups per model (see 4.4), and a hold-out that is never used for selection.
5. *Everything vectorized and streaming* (int-coded tokens, sparse token matrices, multithreaded C++ string similarity).
   The 1.7M-entity test split runs through every CPU stage on one laptop (see 3.1).

---

## 3. Candidate Generation (Blocking)

Candidate generation has four steps. The output of the last one is the candidate set that the matchers score
and that `candidate_pairs.tsv` lists.

- **Step 1 - retrieval, blocking keys used.** A pool record is retrieved if ANY key fires. Every key family has a
  document-frequency cap.
  1. Rare unigram tokens: each S1 entity looks up its 3 rarest name tokens and 4 rarest address tokens with pool df <= 300.
  2. Exact keys: normalized name, compact name, postal code.
  3. **Address bigrams** and **name bigrams** (adjacent tokens, order-insensitive, df <= 300). These retrieve
     non-Latin-name pairs through their clean address, and missing-address pairs through their name. 85% of the pairs that
     the first design never retrieved share an address bigram.
  4. **Compact core-name prefix** (first 8 characters of the space-free name without legal suffixes), for concatenated /
     domain-style names.
  5. Leading-zero-insensitive numeric tokens (`019` = `19`).

  Country is a hard partition (0 of 7.6M true pairs cross it).
- **Step 2 - ranking to a shortlist.** Every retrieved pair (~450 per entity) gets cheap evidence: IDF-weighted Dice
  overlap of name and address over *all* tokens (computed on padded int token matrices at ~1 us/pair), hit counts per rule,
  and token counts. A small LightGBM ranker keeps the best 30 per entity: 29.78 per entity and
  96.978% of all true pairs. The ranker was trained on 12,000 S1 entities outside every other model's training
  and evaluation entities.
- **Step 3 - learned candidate filter** (`candidate_filter.py`, `train_candidate_filter.py`, `run_candidate_cut.py`).
  Most of the 30 are obviously wrong: the median shortlisted pair has a ranker probability below 0.01.
  * *Inputs*: a second, still cheap LightGBM scores each shortlisted pair from the retrieval evidence (overlaps, rule
    hits, exact-key flags, token counts, rank in the shortlist), plus seven character-level similarities and
    missing-field / house-number flags.
  * *Floor*: a pair survives if its probability is at least **0.015**. That floor is the largest value that still
    keeps 99.5% of the shortlisted true pairs of the validation entities.
  * *Cross-fitting*: the filter is cross-fitted over 5 folds of S1 entities, so every training pair is cut by a filter that
    never saw its entity.
  * *Left out on purpose*: the ranker probability and pool-size-dependent statistics. The unseen test country France has a
    3-4x smaller pool and a visibly inflated ranker probability.
  * *Most important inputs*: `block_rank` (62%), `ov_addr` (11%), `addr_token_set` (4%), `core_ratio` (4%), `name_partial` (3%), `ov_name` (3%).

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

- **Step 4 - second-hop retrieval** (`second_hop.py`, `run_second_hop.py`, `run_new_pairs.py`, `build_v4_candidates.py`). A true match that shares no specific key with its S1 record often shares one with another record of the same business: typically a record in the other source with a cleaner address or a Latin-script name.
  * *Queries*: each S1 entity queries the pool indices again with up to 3 of its **anchors**, its candidates with stacker probability >= 0.9. The stacker probabilities are out-of-fold for training entities (`run_stack_oof.py`). The anchor's ranker keeps the top 10 new records per entity, excluding existing candidates.
  * *Filter*: a light LightGBM **new-pair filter** scores each new pair from the anchor evidence (anchor confidence and rank, number of anchors that reached the record) plus four string similarities. It is fitted on the `stack` entities, and its floor (0.003) is chosen on validation entities: it keeps 90% of the new true pairs, capped at 3 new pairs per entity.
  * *Result*: validation recall of all true pairs rises by 1.25% at 1.3 new candidates per entity. The new pairs join the candidate set, `candidate_pairs.tsv` and every downstream stage.

  Floor trade-off (new pairs of the validation entities):

| floor | kept_true_share | new_per_entity | precision |
|---|---|---|---|
| 0.0010 | 0.9641 | 2.5452 | 0.0183 |
| 0.0020 | 0.9392 | 1.6539 | 0.0274 |
| 0.0030 | 0.9006 | 1.2652 | 0.0344 |
| 0.0050 | 0.8508 | 0.8452 | 0.0486 |
| 0.0100 | 0.8149 | 0.4236 | 0.0929 |
| 0.0200 | 0.7459 | 0.2211 | 0.1628 |
| 0.0300 | 0.7155 | 0.1549 | 0.2229 |
| 0.0500 | 0.6575 | 0.1065 | 0.2979 |
| 0.0750 | 0.6381 | 0.0800 | 0.3850 |
| 0.1000 | 0.5967 | 0.0659 | 0.4372 |
| 0.1500 | 0.5552 | 0.0511 | 0.5248 |
| 0.2000 | 0.5331 | 0.0440 | 0.5848 |
| 0.3000 | 0.4945 | 0.0357 | 0.6679 |

- **Candidate pairs generated**: 15,280,744 for 1,732,544 test S1 entities (**8.82 per entity**).
  The candidate reduction ratio is > 0.99999 versus the same-country cross product. `output/candidate_pairs.tsv` is
  exactly this set, the models score exactly these pairs, and every predicted match is one of them. Per-country
  statistics are in Appendix B.5.
- **How true matches were not lost**: recall was measured per step, per rule and per cap on ground truth, and the design
  was iterated on the failure categories. The first design reached 82.1% recall at 175 candidates per entity.
  Retrieval + ranker reached 96.978% at 29.78, and the filter kept 96.549% at
  6.08, and second-hop retrieval brought it to 97.90% at 7.37 candidates per entity (hold-out entities) (tables in Appendix B.2). A plain floor on the ranker probability needs about 15
  candidates per entity for the recall that the filter keeps with about 6. The remaining misses are dominated by
  non-Latin names with sparse addresses, and by candidates with missing addresses and scrambled names
  (`reports/blocking_report.md`).

### 3.1 Scalability ("blocking has to scale")
* **No pairwise comparison outside the retrieved set.** Retrieval is inverted-index lookups. Each S1 entity reads at most K
  rare-token postings plus the bigram / prefix / exact-key postings of its own keys. A key whose document frequency is
  above its cap (300) is never read, so the work per entity is bounded by (number of keys x cap), however large the pool
  grows. The ranker and the filter cost O(1) per retrieved / shortlisted pair. Total cost is linear in the number of S1
  entities (queries) plus linear in the pool (index build).
* **Partitioned and streaming.** Country partitions are independent. The pool index is built once per partition and
  queried in S1 chunks with bounded memory. Every stage is resumable per chunk / country. At larger scale the same code
  shards by partition (country, then region / postal prefix) and by S1 chunk across machines.
* **Measured cost.** CPU stages ran on one laptop (8 cores / 16 threads, 15 GB RAM); GPU stages ran on a free Kaggle
  session (2x Tesla T4):
  * test blocking (retrieval + ranker, 1.73M S1 entities against 10M pool records): 71 min;
  * candidate filter: 13 min;
  * first-stage featurize + score: 15 min;
  * cross-encoder: fine-tuning 96 min, then scoring 13.6M pairs at ~1,400 pairs/s in 164 min;
  * stacker inference on the test split (all graph features): 17 min.

  * **Second-hop retrieval** reuses the blocker's indices, and each entity issues at most 3 anchor queries, so its cost is linear in the number of entities. On the test split it produced 16.2M raw second-hop pairs in 2 h 25 min with 3 threads. The new-pair filter and the candidate build took 5 min, and scoring the 3.0M kept test pairs with the cross-encoder took 41 min on 2x T4.

---

## 4. Matching Model

### 4.1 First-stage matcher (pairwise features, LightGBM)
**Features used** (~90, all country-agnostic, NaN when a field is missing on either side):
- **Name features**:
  - exact equality (normalized / legal-suffix-free core);
  - Levenshtein ratio, Jaro-Winkler, token-sort, token-set and partial ratio, on the normalized name and on the core name;
  - token Jaccard / Dice / containment in both directions / IDF cosine / IDF overlap;
  - numeric-token overlap, length differences, script code of both names, and rule evidence from blocking.
- **Address features**:
  - exact equality and the same five string similarities;
  - token Jaccard / Dice / containment / IDF cosine / IDF overlap;
  - numeric-token overlap (house / plot numbers), house-number and postal-code agreement, length differences.
- **Other features**:
  - S2 vs S3, and blocker evidence;
  - the candidate's position inside its entity (rank / gap to the best by several scores) and ambiguity context;
  - pool-ambiguity statistics (how many pool records / S1 entities carry the identical name, core name or address);
  - cross-field products / min / max.

**Model type:** LightGBM (MIT licence, 733 trees, 127 leaves), trained on the 35,000 `main_train` entities.
sklearn HistGradientBoosting and LogisticRegression were trained on the same features for comparison (Appendix B.1).
Two matcher variants were trained on identical data: the default feature set (validation 0.9628, hold-out 0.9647) and a France-robust set that drops the features whose scale depends on the size of the country pool (validation 0.9552, hold-out 0.9560). Because France has no labels and a 3-4x smaller pool, the robust set is promoted whenever it is within 0.002 of the default on both validation and hold-out: **default features were promoted**.

### 4.2 Cross-encoder (transformer, fine-tuned)
* **Model**: `microsoft/mdeberta-v3-base`, multilingual DeBERTa-v3 (MIT licence, 278M parameters, well under the
  8B limit). The pre-trained weights come from Hugging Face. The model is fine-tuned only on the provided training data,
  and no external data is looked up.
* **Input**: Ditto-style serialization of both records, `name: <name> address: <address>`, as one sequence pair
  (lower-cased, max 128 tokens; the mean pair is 54 tokens). The [CLS] representation feeds a match head.
* **Training data**: 1,169,095 candidate pairs (52.8% positive) of the `main_train` + `ce_extra` entities (185,000
  entities that the stacker never sees), with labels from the ground truth.
* **Targeted hard negatives**: on test, the first-stage matcher accepted French pairs with the same name and house
  number but a *different street* (~2% of its French matches). That pattern is rare in US / India training data.
  40,000 **street-swap negatives** teach it: US true pairs whose street is replaced by another record's street, keeping
  the name and number, labelled non-match.
* **Domain robustness for the unseen country**: a gradient-reversal head predicts the country from the [CLS] vector
  (weight 0.1). The encoder is pushed towards country-independent evidence: country-head accuracy fell from 0.63 to
  0.573, which equals the class prior (57.3% of the training pairs are US).
* **Training setup**: 1 epoch, AdamW, learning rate 3e-5 (head 1e-3), 5% warm-up and linear decay, fp16 mixed precision,
  length-bucketed batches, on 2x T4 (Kaggle) in 96 min.
* **Quality**: pair AUC 0.9985 on both validation and hold-out pairs. On its own, with the best threshold, it reaches a
  hold-out macro F0.5 of 0.9725, above the whole first-stage LightGBM matcher (0.9647).
* **Scoring**: every candidate pair of the test split and of the stacker / validation / hold-out entities (13.6M pairs)
  was scored by one worker process per GPU with resumable shards.

### 4.3 Graph stacker (collective resolution)
The stacker is LightGBM (314 trees, 127 leaves, early stopping on validation entities). It uses the
first-stage features plus `p1`, the cross-encoder score `p_ce`, and five families of collective features. All of them are
computed from `p1` over the **complete candidate graph** of the country partition: every S1 entity, both in training and
at test time.
* **Exclusivity / pool side (`ps_*`)**:
  * for the pair's S2/S3 record: how many S1 entities claim it, the pair's rank among them, the best / sum of the other
    claimants' `p1`, and how many others have `p1 >= 0.5`;
  * the **exclusivity-normalized probability** `ps_q = o / (1 + sum of all claimants' odds)`, with `o = p1 / (1 - p1)`.

  Used alone as a parameter-free rule, `ps_q` already lifts hold-out F0.5 from 0.9647 to 0.9676. The ground truth
  assigns every S2/S3 record to at most one S1 entity.
* **Entity context (`e_*`, `s_*`)**: rank, top-3 scores, gap to the best other candidate, and the score sum within the
  entity (`e_`) and within the entity's same-source candidates (`s_`).
* **Sibling / triangle agreement (`sib_*`)**: how similar the candidate is to the entity's other high-scoring candidates
  (name / address token-set similarity of the two *pool* records, weighted by their `p1`). True matches of one entity
  describe the same business, so they agree with each other; a distractor usually does not.
* **Address structure (`as_*`)**: the original address is parsed into house number and street, which are compared
  separately. The features are same number, street similarity, *same number but different street*, and similarity of the
  remaining components.
* **Candidate source (v4 only)**: `src_hop` (the pair came from second-hop retrieval), the new-pair filter score, the anchor's confidence (`hop_conf`) and rank, and the number of anchors that reached the record (`hop_n`). For a new pair, pool-side competition is measured against the first-hop claimants only (`stack_v4.py`). New pairs exist for every test entity but only for the stacker / validation / hold-out entities in training, so counting other entities' new pairs would change the feature's meaning.

**Training entities.** `stack` = 200,000 S1 entities outside every earlier model's training data. On them `p1`, `p_ce`
and the graph features are all out-of-sample, exactly as on the test split. Early stopping and the threshold use the
7,500 `val` entities, and the 7,500 `hold` entities are scored once at the end. Most important stacker inputs (share of
total gain): `p_ce` (85.6%), `ps_q` (8.3%), `ps_gap_best_other` (3.6%), `ps_sum_other` (0.3%), `p1` (0.1%), `new_score` (0.1%), `sim_min` (0.1%), `ps_best_other` (0.1%), `s_gap_to_best_other` (0.1%), `s_top1` (0.1%), `ps_rank` (0.1%), `ps_n_claim` (0.0%).

### 4.4 Entity groups (no leakage between stages)
The 2.2M training S1 entities are split once (`world_groups.json`):

| group | entities | used for |
|---|---:|---|
| `main_train` | 35,000 | first-stage matcher, candidate filter (cross-fitted), cross-encoder fine-tuning |
| `ranker` | 12,000 | blocking ranker only |
| `ce_extra` | 150,000 | cross-encoder fine-tuning only |
| `stack` | 200,000 | stacker training; its anchors for second-hop retrieval use out-of-fold stacker probabilities |
| `val` | 7,500 | early stopping, thresholds, all selection decisions |
| `hold` | 7,500 | final evaluation only |

All other training entities are still part of the world. They are blocked and scored, so they compete for pool records
exactly as the other 1.7M test entities do.

**Threshold selection method:** a grid search on validation entities for the global threshold that maximizes macro
F0.5 *under the challenge's own scoring rule* (per entity, singleton = 1 only when empty). Other rules were also tried:
per-source (S2/S3) thresholds and an entity-level gate for the first-stage matcher (a single global threshold (0.74) was selected), and an exact
expected-F0.5 decision per entity for the stacker (`decision_opt.py`, within 0.0002 of the global threshold). The
simple rule was kept (stacker threshold 0.74).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro)** of the submitted model on hold-out entities (never used for training, early stopping,
  threshold or rule selection): **0.9847**. On the same entities, precision is 0.9944, recall
  0.9592, and singleton accuracy 0.9956. Validation F0.5 is 0.9831. The ceiling, if exactly the
  true candidates were predicted, is 0.9913.

  Contribution of each component, measured on the same validation / hold-out entities:

| stage | threshold | validation F0.5 | hold-out F0.5 | hold-out precision | hold-out recall | hold-out singleton acc. |
|---|---|---|---|---|---|---|
| 1. first-stage LightGBM matcher (v1) | 0.7400 | 0.9628 | 0.9647 | 0.9849 | 0.9176 | 0.9758 |
| 2. parameter-free exclusivity rule q(p1) | 0.5300 | 0.9670 | 0.9676 | 0.9855 | 0.9281 | 0.9692 |
| 3. graph stacker: p1 + collective features (v2) | 0.6500 | 0.9727 | 0.9742 | 0.9878 | 0.9410 | 0.9868 |
| 4. cross-encoder alone (mDeBERTa-v3-base) | 0.7900 | 0.9715 | 0.9725 | 0.9908 | 0.9283 | 0.9780 |
| 5. mean of p1 and the cross-encoder | 0.5500 | 0.9747 | 0.9764 | 0.9896 | 0.9452 | 0.9868 |
| 6. graph stacker + cross-encoder (v3) | 0.7300 | 0.9802 | 0.9818 | 0.9944 | 0.9490 | 0.9978 |
| ceiling: exactly the true candidates of the v4 candidate sets | 0.0500 | 0.9907 | 0.9913 | 0.9966 | 0.9772 | 1.0000 |
| 7. + second-hop candidates (cross-encoder on first-hop pairs only) | 0.8300 | 0.9826 | 0.9841 | 0.9946 | 0.9560 | 0.9978 |
| 8. + cross-encoder scores of the new pairs (v4) | 0.7400 | 0.9831 | 0.9847 | 0.9944 | 0.9592 | 0.9956 |
| 9. v4 with 5 seed-bagged stackers averaged (tried, not adopted) | 0.6700 | 0.9834 | 0.9848 | 0.9942 | 0.9602 | 0.9956 |

  Public leaderboard: v1 (first-stage matcher) 0.93, v2 (graph stacker) 0.95, v3 (+ cross-encoder)
  0.9739. The public leaderboard uses a subset of the test set, and its entities are not those of our hold-out.

  Reference points of the first design, on the validation entities:
  * exact normalized name: 0.481;
  * fuzzy name + address rule: 0.677;
  * best similarity-score threshold: 0.726;
  * TF-IDF cosine threshold: 0.722;
  * first-stage LightGBM: 0.9628.
- **Common false positives (wrong merges):** False merges are rare for the submitted model: 30 false merges on validation entities that do have matches (0.12% of their true pairs) and 2 true singletons with a prediction (table B.3, `reports/edge_case_report_stack_v4hce.md`). The pattern is a pool record that belongs to *another* S1 entity or to none, with the same or almost the same name and a compatible or empty address, e.g. `Lyra Inc | 1445 Harlem Avenue, Oak Park` versus `LYRASYN | HARLEM AVENUE, OAK PARK`. In the first design, records owned by another S1 entity were 51% of all false positives. The exclusivity features and the cross-encoder removed most of them: hold-out precision rose from 0.985 (first-stage matcher) to 0.994.
- **Common false negatives (missed matches):** Misses split into two groups. *Candidate-generation misses* (2.18% of true pairs) were never retrieved, were cut from the shortlist, or were removed by the candidate filter. *Matching misses* (1.79% of true pairs) are candidates that the model rejected. Most matching misses have one clear cause: an **address-less pool record whose name is a generic business name that several S1 entities share**. `Rev It Up Cafe` exists as three S1 entities (MO, CA, VA), and the pool holds three address-less copies (`Rev It-Up Cafe`, `REV-IT UP  CAFE`, `REV IT UP CAFE`), one owned by each entity. Nothing in the records says which copy belongs to which entity. Among address-less candidates whose normalized name equals the S1 name, 84% are not the entity's own, and almost all of those belong to another S1 entity. A wrong copy has a median of 54 S1 claimants, against 3 for a true one. Under F0.5 the best decision is to reject such a copy unless the entity is the only plausible owner, and that is what the stacker learned (the missing-address edge case has recall 0.884). Most candidate-generation misses are non-Latin names with a sparse or differently written address (the non-Latin case has recall 0.961), and records with no address and a scrambled name.

Per-edge-case results for the submitted model (26 cases of the brief, with counts, candidate recall, precision,
recall, F0.5, failure taxonomy and examples) are in
`code/business_entity_resolution/reports/edge_case_report_stack_v4hce.md` (the first design's are in `edge_case_report.md`).
The summary is in Appendix B.3.

---

## 6. Conclusion

Three levers moved the metric, each measured on the same hold-out entities. (1) **Candidate generation**: multi-key retrieval with a learned ranker lifted the recall of the candidate set from 82% to 97% at one sixth of the first design's volume, and a light learned candidate filter reduced it to 6.08 candidates per entity with no loss in F0.5. (2) **Collective resolution on the full world**: the constraint that every pool record has at most one owner, turned into features over the complete candidate graph of all 2.2M training entities, took hold-out F0.5 from 0.9647 to 0.9742. (3) **A fine-tuned multilingual cross-encoder** used as an input of the graph stacker took it to 0.9818; on its own it would have been 0.9725. The submitted pipeline reaches 0.9847 on hold-out entities never used for any decision, against 0.48 - 0.73 for simple baselines. What remains is mostly information that the data do not contain (address-less copies of names shared by several businesses) and true pairs that no retrieval key reaches.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains all source in `src/`, unit tests in `tests/`, `README.md` and a pinned
`requirements.txt` (plus `requirements-gpu.txt` for the two GPU scripts). Pipeline entry points, in order:
1. **First design** (`run_all.py` runs this whole chain):
   `prepare_cache.py` -> `train_ranker.py` -> `run_blocking.py --split train` -> `train_candidate_filter.py` ->
   `run_candidate_cut.py` -> `run_features.py` -> `run_train.py` -> `promote_model.py` -> `run_blocking.py --split test`
   -> `run_candidate_cut.py --split test` -> `run_score.py --split test`.
2. **Full world**: `world_groups.py` -> `run_blocking.py --split train --tag full` (all 2.2M training entities) ->
   `run_world_cut.py` -> `run_world_score.py`.
3. **Cross-encoder**: `export_ce_package.py` -> `kaggle_ce.py` (GPU, Kaggle) -> `ce_scores_import.py`.
4. **New candidates (v4)**: `run_stack_oof.py` (anchors for training entities) -> `run_second_hop.py --split train/test` -> `run_new_pairs.py` (new-pair filter) -> `build_v4_candidates.py` -> `run_world_score.py --tag full_v4h` / `run_score.py --tag v4h` -> `export_ce_newpairs.py` -> `kaggle_ce.py` (scoring only, the fine-tuned model ships in the package) -> `ce_scores_import.py`.
5. **Stacker and output**: `run_stack.py` -> `run_stack_infer.py` -> `run_infer.py --scores-dir ... --decision-config ...`
   (writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`) -> `utils/validate_submission.py`.

Feature code: `features.py` (pairwise), `collective.py`, `stack_features.py`, `address_structure.py`, `second_hop.py`, `stack_v4.py`.
The exact commands are in the README.

### B. Additional Results
#### B.1 Experiment log of the first-stage matcher (validation entities; last row = hold-out entities never used for any decision)
Blocking strategy, features, model, threshold, candidate recall, precision, recall, F0.5, singleton accuracy and average
candidates per entity of every first-design experiment (baselines 1-3b, LightGBM with the three decision rules, sklearn
model families, hold-out). The stage-2 experiments are the table in Section 5.

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




#### B.3 Failure taxonomy and edge cases (submitted model, validation entities)
| type | count | share of true pairs |
|---|---|---|
| A blocking (true match not a candidate) | 568 | 2.18% |
| B matching: true candidate rejected | 465 | 1.79% |
| B matching: false merge on an entity with matches | 30 | 0.12% |
| C singleton with a predicted match | 2 | 0.01% |

Per edge case (the 26 cases of the brief that can be evaluated on the labelled data; entities may belong to several cases):

Entities can belong to several cases. Precision / recall are pooled over the case's entities; F0.5 is the macro mean.

| edge case | S1 entities | candidate recall | precision | recall | macro F0.5 | A blocking failures | B matching failures | C singleton failures |
|---|---|---|---|---|---|---|---|---|
| 1 true singleton | 412 |  | 0.000 |  | 0.995 | 0 | 0 | 2 |
| 2 one-to-one match | 420 | 0.971 | 0.993 | 0.948 | 0.944 | 12 | 13 | 0 |
| 3 multiple matches | 6668 | 0.978 | 0.999 | 0.960 | 0.985 | 556 | 482 | 0 |
| 4 cross-source match (S2 and S3) | 6043 | 0.979 | 0.999 | 0.961 | 0.986 | 512 | 440 | 0 |
| 5 multiple matches within one source | 5759 | 0.979 | 0.999 | 0.962 | 0.986 | 492 | 447 | 0 |
| 6 name-only strong (name>=.85, address weak/missing) | 1354 | 0.984 | 0.999 | 0.931 | 0.972 | 90 | 302 | 0 |
| 7 address-only strong (address>=.85, name weak/missing) | 606 | 0.986 | 1.000 | 0.964 | 0.989 | 37 | 58 | 0 |
| 8 name+address both noisy (both < .7) | 952 | 0.983 | 0.999 | 0.965 | 0.991 | 70 | 72 | 0 |
| 9 name collision (false candidate with name>=.9) | 2699 | 0.964 | 0.998 | 0.942 | 0.974 | 328 | 228 | 2 |
| 10 address collision (false candidate with address>=.9) | 2173 | 0.978 | 0.999 | 0.959 | 0.983 | 157 | 146 | 1 |
| 11 missing name (either side) | 0 |  |  |  |  |  |  |  |
| 12 missing address (either side) | 882 | 0.985 | 0.998 | 0.884 | 0.951 | 56 | 379 | 0 |
| 13 multiple missing fields | 0 |  |  |  |  |  |  |  |
| 14 normalization-only (identical after normalization) | 4657 | 0.984 | 0.999 | 0.966 | 0.988 | 302 | 353 | 0 |
| 15 token-order (token-sort>=.95, plain ratio<.9) | 1434 | 0.984 | 0.999 | 0.972 | 0.992 | 95 | 84 | 0 |
| 16 typo (name JW>=.9 but not identical) | 5469 | 0.985 | 0.999 | 0.967 | 0.989 | 327 | 411 | 0 |
| 17 transliteration / non-Latin name | 835 | 0.978 | 0.999 | 0.961 | 0.990 | 70 | 56 | 0 |
| 22 ambiguous (true candidate AND a false candidate p>=.3) | 222 | 0.979 | 0.961 | 0.958 | 0.960 | 16 | 46 | 0 |

Some cases are structural rather than measurable:
* country conflict: 0 of 7.6M true pairs cross a country, and candidates are generated per country partition;
* unseen country: France appears in test only;
* duplicate candidates: candidates are a deduplicated union;
* empty ground truth: parsed as zero matches;
* output-subset invariant, S1 ids in predictions, and duplicate predicted ids: all verified on the real output files by
  `run_edge_cases.py` and `utils/validate_submission.py`.

#### B.4 Source 2 versus Source 3
| source | true pairs in candidates | mean name similarity (true pairs) | mean address similarity (true pairs) | address missing | non-Latin name | name identical after normalization | precision | recall (of candidate true pairs) |
|---|---|---|---|---|---|---|---|---|
| S2 | 12221 | 0.834 | 0.846 | 0.037 | 0.084 | 0.290 | 0.999 | 0.983 |
| S3 | 13209 | 0.832 | 0.771 | 0.036 | 0.047 | 0.295 | 0.999 | 0.981 |

#### B.5 Test-set prediction statistics per country of the submitted model (France has no labels: only prediction rates can be compared)
| country | s1_entities | candidate_pairs | candidates_per_entity | share_without_candidates | entities_with_match | share_with_match | predicted_matches_per_entity | predicted_matches_per_entity_that_has_one | share_s2_of_matches | mean_score_of_kept_pairs | share_best_score_ge_0.97 | share_best_score_0.4_to_0.9 | share_best_score_below_0.2_or_no_candidate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| France | 259,452 | 3,389,829 | 13.0653 | 0.0004 | 243,244 | 0.9375 | 3.1976 | 3.4106 | 0.4838 | 0.9928 | 0.9287 | 0.0088 | 0.0530 |
| India | 809,986 | 7,372,023 | 9.1014 | 0.0001 | 757,945 | 0.9358 | 3.3032 | 3.5300 | 0.4846 | 0.9986 | 0.9347 | 0.0011 | 0.0626 |
| US | 663,106 | 4,518,892 | 6.8147 | 0.0009 | 624,345 | 0.9415 | 3.3768 | 3.5864 | 0.4834 | 0.9984 | 0.9404 | 0.0013 | 0.0566 |

#### B.6 Most important first-stage features (share of total gain)
`rank_score` (43.2%), `at_num_jaccard` (10.4%), `block_gap_to_best` (6.6%), `name_ratio` (3.1%), `nt_idf_ov2` (1.6%), `name_partial` (1.6%), `s1_core_freq` (1.5%), `name_jw` (1.4%)

### C. Limitations and honest notes
* **Pre-trained models and compute.** Two pre-trained open models are used, both under the MIT licence and far below the
  8B-parameter limit: `microsoft/mdeberta-v3-base` (cross-encoder). A fine-tuned dense bi-encoder retriever (`kaggle_dense.py`, `intfloat/multilingual-e5-small`, MIT) was also built, but its cloud run did not finish in time and it is not part of the submission. They are fine-tuned only on the
  provided training data. No external data, API, geocoder or lookup service is used anywhere. The GPU parts ran on a
  free Kaggle session (2x T4); `src/kaggle_*.py` run unchanged on any CUDA machine (see the README).
* **France** is unseen in training and has no labels, so its accuracy cannot be measured. Nothing in the pipeline is
  country-specific, and `country` only partitions the data. Mitigations:
  * the candidate filter does not use pool-size-dependent inputs;
  * the cross-encoder has a country-adversarial head;
  * the street-swap negatives target a France-specific false-merge pattern.

  A leaderboard probe (the v3 submission with every French prediction removed) implied a French F0.5 of about 0.972,
  close to India + US together (about 0.974).
* **Leaderboard versus hold-out gap.** The public leaderboard is below our hold-out for every version: by 3.5 points
  for v1, 2.4 for v2 and 0.8 for v3. The gap shrank as the models got stronger, and for v3 it is not specific
  to France. It is not explained by the composition of the test entities: their score distribution and candidate
  statistics match the validation entities. It is also not explained by a simulation of S1 entities whose true records
  are missing from the pool (0.06 to 0.13 points). The hold-out stays the basis of every decision, because only it is
  untouched by selection.
* Non-Devanagari Indic scripts are not transliterated by rules (only Devanagari has a rule-based transliterator). The
  multilingual transformer handles them from text, and second-hop retrieval reaches many of them through the business's other, Latin-script records.
* Remaining headroom is mostly in candidate generation: the hold-out ceiling given the final candidate sets is
  0.9913, against an achieved 0.9847.
