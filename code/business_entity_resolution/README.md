# Business Entity Resolution - reproducible pipeline

Links every Source-1 (S1) record to the Source-2 / Source-3 records that describe the same business.
Data: ~2.2M S1 / 5.0M S2 / 5.3M S3 training records, ~1.7M / 4.9M / 5.1M test records. Metric: per-S1-entity
F0.5, macro-averaged; a true singleton scores 1.0 only when nothing is predicted.

The CPU stages run on one machine (developed on Windows 11, Ryzen 7 / 16 threads, 15 GB RAM). The two transformer
stages (`src/kaggle_ce.py`, `src/kaggle_dense.py`) run on a cloud GPU (Kaggle "GPU T4 x2"). Nothing outside the provided
`dataset/` is used as data. No API, geocoder or lookup service is used. The only pre-trained weights are MIT-licensed
open models (`microsoft/mdeberta-v3-base`, `intfloat/multilingual-e5-small`), fine-tuned only on the provided training
data. Libraries: `requirements.txt` (CPU) and `requirements-gpu.txt` (GPU scripts).

## Method in one page

1. **Normalization** (`src/text_normalize.py`): NFKD accent folding, rule-based Devanagari -> Latin transliteration,
   `&`->`and`, suffix / abbreviation canonicalization, token lists, compact forms, postal code and house-number
   extraction, coarse script detection. Cached once as parquet (`src/prepare_cache.py`, chunked and resumable).
   Numeric tokens are compared without leading zeros (`019` = `19`) in blocking and features.
2. **Blocking** (`src/blocking.py`, `src/run_blocking.py`) - per country partition (0 of 7.6M true pairs cross a
   country; any country incl. the unseen France just flows through):
   * retrieval = union of rare unigram tokens, exact keys (name / compact name / postal), **address and name
     bigrams**, and a **compact-core-name prefix** key, each with a document-frequency cap;
   * every retrieved pair gets cheap evidence (IDF-weighted overlap of *all* tokens on padded token matrices, hit
     counts per rule, token counts) and a small **learned ranker** (`src/train_ranker.py`, LightGBM) keeps a
     *shortlist* of the best 30 per entity: ~97.0% of *all* true pairs (a hand-written score kept ~64%);
   * a light **learned candidate filter** (`src/candidate_filter.py`, `src/train_candidate_filter.py`,
     `src/run_candidate_cut.py`) scores each shortlisted pair from the retrieval evidence plus seven cheap string
     similarities and keeps it only if its probability clears a floor chosen on validation entities: **~6 candidates
     per entity** (instead of 30) holding ~96.5% of all true pairs, with no measurable change in F0.5. It is
     cross-fitted over S1 entities, so training / validation candidate sets are cut exactly like the test split's.
     This final set is what the matcher scores and `output/candidate_pairs.tsv` lists.
3. **Pairwise features** (`src/features.py`): multithreaded rapidfuzz similarities (ratio, Jaro-Winkler,
   token sort/set, partial) on normalized name, core name (legal suffix removed) and address; sparse-matrix token overlap
   (Jaccard, Dice, containment both ways, IDF cosine, numeric-token overlap); postal / house-number agreement; script
   codes; blocker evidence; rank / gap / ambiguity counts of each candidate inside its entity. Missing fields give NaN,
   never 0 or 1. `country` is deliberately **not** a feature (France is unseen in training).
4. **Matcher** (`src/run_train.py`): LightGBM binary classifier. The S1 *entities* are split into train / validation /
   hold-out: validation entities drive early stopping, thresholds and rule selection; hold-out entities are scored once at
   the end. Baselines (exact name, fuzzy rule, similarity / TF-IDF thresholds) and other model families (sklearn
   HistGradientBoosting, LogisticRegression) are evaluated on the same entities with the challenge metric.
5. **Optional second stage** (`src/stage2.py`, `src/run_stage2.py`): entity-context features of the first-stage
   probabilities (rank, gap to the best other candidate, counts above cut-offs, per source), trained on out-of-fold
   scores; adopted only if it beats the first stage on validation by a margin (`decision_config_stage2*.json`).
6. **Inference** (`src/run_score.py`, `src/run_infer.py`): candidate pairs of the test split are featurized and scored
   in a streaming fashion, thresholded, and written as `output/matching_results.tsv` (+ `candidate_pairs.tsv`).
7. **Full training world + graph stacker** (`src/world_groups.py`, `run_world_cut.py`, `run_world_score.py`,
   `collective.py`, `stack_features.py`, `address_structure.py`, `run_stack.py`, `run_stack_infer.py`).
   * *World*: every training S1 entity (2.2M) is blocked and cut like the test split, and the first-stage model scores
     every pair. The entities are split once into fixed groups: `main_train` / `ranker` / `ce_extra` / `stack` / `val` / `hold`.
   * *Stacker*: a second LightGBM is trained on the 200k `stack` entities. Its inputs are the pairwise features, `p1`,
     the cross-encoder score, and collective features over the complete graph: pool-side exclusivity
     (`ps_q = o / (1 + sum of all claimants' odds)`), entity context, sibling agreement, and house number vs street.
8. **Cross-encoder** (`src/export_ce_package.py` -> `kaggle_ce.py` on a GPU -> `ce_scores_import.py`).
   * *Model*: `microsoft/mdeberta-v3-base` fine-tuned on 1.17M pairs of the `main_train` + `ce_extra` entities, with
     40k "same number, different street" hard negatives and a country-adversarial head.
   * *Input*: `name: … address: …` pairs without special tokens.
   * *Released weights*: `artifacts/ce/ce_model/` (Git LFS; usage in its README).
9. **Second-hop candidates** (`src/run_stack_oof.py`, `second_hop.py`, `run_second_hop.py`, `run_new_pairs.py`,
   `build_v4_candidates.py`, `stack_v4.py`).
   * *Retrieval*: each entity's confident matches (anchors, stacker probability >= 0.9, out-of-fold for training
     entities) query the pool indices again.
   * *Filter*: a learned new-pair filter keeps about 1.3 new candidates per entity. Candidate recall rises from 96.5%
     to 97.9% of all true pairs.
   * *Scoring*: the new pairs are scored by the same cross-encoder (`export_ce_newpairs.py` -> `kaggle_ce.py`,
     scoring only), then enter the stacker with their source features.

## Reproduce

One command regenerates everything from the raw data (stages are resumable and skip finished work):

```bash
python -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt   # Windows: .venv\Scripts\...
cd student_resource                     # the provided dataset/ (TSV files) must be in student_resource/dataset/
.venv/bin/python code/business_entity_resolution/src/run_all.py              # ~4-6 h on the reference machine
.venv/bin/python code/business_entity_resolution/src/run_all.py --list       # stage names; resume with --from <stage>
```

or stage by stage:

```bash
S=code/business_entity_resolution/src
python $S/prepare_cache.py                                              # normalize all six sources (~20 min)
python $S/train_ranker.py                                               # candidate ranker (trained on entities outside the sample below)
python $S/run_blocking.py --split train --n-s1 50000 --workers 3 --chunk-s1 4000 --expansion-budget 6000000   # -> shortlist_train
python $S/train_candidate_filter.py                                     # cross-fitted candidate filter + probability floor
python $S/run_candidate_cut.py --split train                            # shortlist -> candidate set (out-of-fold filter scores)
python $S/run_features.py --split train                                 # pairwise features (+ labels)
python $S/run_train.py                                                  # LightGBM matcher + thresholds + baselines + model comparison
python $S/run_train.py --robust --exp robust --no-compare-models        # variant without pool-size-dependent features (for the unseen country)
python $S/promote_model.py                                              # keeps the default unless the robust variant is within 0.002 on validation AND hold-out
python $S/run_stage2.py                                                 # optional second stage (adopted only if it helps; not used for the submission)
python $S/run_blocking.py --split test --workers 3 --chunk-s1 50000 --expansion-budget 6000000    # -> shortlist_test
python $S/run_candidate_cut.py --split test                             # -> candidates_test (= candidate_pairs.tsv)
python $S/run_score.py --split test                                     # featurize + score (streaming)
python $S/run_infer.py                                                  # -> output/matching_results.tsv, output/candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
python $S/run_blocking_report.py ; python $S/run_edge_cases.py          # diagnostics -> reports/
python -m unittest discover -s code/business_entity_resolution/tests -v # 60 unit tests
```

Stage 2 (the submitted v4 pipeline), after the commands above:

```bash
S=code/business_entity_resolution/src; A=code/business_entity_resolution/artifacts
python $S/world_groups.py                                                # fixed entity groups -> artifacts/world_groups.json
python $S/run_blocking.py --split train --tag full --ranker $A/ranker.txt --workers 3 --chunk-s1 50000 --expansion-budget 6000000   # all 2.2M
python $S/run_world_cut.py --tag full && python $S/run_world_score.py --tag full   # candidate cut + first-stage scores of the world
python $S/export_ce_package.py --tag full                                           # -> kaggle_upload/er-ce-data (upload as a Kaggle dataset)
#   GPU: python kaggle_ce.py  (fine-tunes + scores; ~4.5 h on 2x T4)   -> ce_scores.parquet, ce_model/
python $S/ce_scores_import.py --ce-scores <ce_scores.parquet>            # -> artifacts/ce/ce_train_sets.parquet
python $S/run_stack.py --tag full --exp ce --ce $A/ce/ce_train_sets.parquet             # v3 stacker
python $S/run_stack_infer.py --exp ce --ce-scores <ce_scores.parquet> --pair-index <local_pair_index.parquet>
python $S/run_stack_oof.py                                               # out-of-fold stacker probabilities (anchors)
python $S/run_second_hop.py --split train --tag full && python $S/run_second_hop.py --split test
python $S/run_new_pairs.py --no-dense                                    # learned new-pair filter (floor chosen on val)
python $S/build_v4_candidates.py --v4 v4h                                # candidates_train_full_v4h / candidates_test_v4h
python $S/run_world_score.py --tag full_v4h
cp $A/lgbm_model.txt $A/lgbm_model_v4h.txt; cp $A/feature_columns.json $A/feature_columns_v4h.json
python $S/run_score.py --split test --tag v4h
python $S/export_ce_newpairs.py --world-tag full_v4h --test-tag v4h      # -> kaggle_upload/er-ce-new (+ ce_weights_fp16.pt)
#   GPU: python kaggle_ce.py  (scoring only, ~45 min on 2x T4)          -> ce_new/ce_scores.parquet
python $S/ce_scores_import.py --ce-scores <ce_new/ce_scores.parquet> --pair-index <local_pair_index_new.parquet> --out-name ce_train_sets_new.parquet
python $S/run_stack.py --tag full_v4h --exp v4hce --v4 --ce $A/ce/ce_train_sets.parquet,$A/ce/ce_train_sets_new.parquet
python $S/run_stack_infer.py --tag v4h --exp v4hce --ce-scores <ce_scores.parquet>,<ce_new/ce_scores.parquet> \
       --pair-index <local_pair_index.parquet>,<local_pair_index_new.parquet>
python $S/run_infer.py --tag v4h --out-dir output --scores-dir $A/cache/scores_test_stack_v4hce --decision-config $A/stack_v4hce/stack_decision.json
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Every stage accepts `--tag NAME` to keep a variant's artifacts side by side (`candidates_train_NAME`, `lgbm_model_NAME.txt`,
...); `run_train.py --exp NAME` writes ablation artifacts without touching the main model. Fixed seeds throughout.
Memory: the largest country partition needs ~6 GB for the pool index; lower `--workers` / `--expansion-budget` on small machines.

## Layout

```
src/text_normalize.py   field normalization           src/features.py      pairwise features (vectorized)
src/prepare_cache.py    normalized parquet cache      src/train_model.py   split / sampling / threshold helpers
src/blocking.py         retrieval + ranking           src/evaluation.py    exact challenge metric (+ fast scorer)
src/run_blocking.py     blocking driver (shortlist)   src/run_train.py     matcher training + thresholds + baselines
src/train_ranker.py     candidate ranker              src/stage2.py, run_stage2.py   optional entity-context stage
src/candidate_filter.py, train_candidate_filter.py, run_candidate_cut.py   learned candidate filter: shortlist -> candidate set
src/pipeline.py         chunked feature iterator      src/run_score.py     streaming scoring
src/run_all.py          end-to-end driver             src/run_infer.py     submission files
src/eda.py, run_blocking_report.py, run_edge_cases.py      diagnostics (exp_candidate_cut.py: the cut experiments;
                        exp_pool_size_robustness.py: small-pool simulation, written but not run)
tests/                  unit tests (normalization, metric, blocking + families, features, plumbing, stage 2)
reports/                data analysis, blocking report + experiment log, validation report, experiment log, edge cases
```
