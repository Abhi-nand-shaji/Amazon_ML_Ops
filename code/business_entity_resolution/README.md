# Business Entity Resolution - reproducible pipeline

Links every Source-1 (S1) record to the Source-2 / Source-3 records that describe the same business.
Data: ~2.2M S1 / 5.0M S2 / 5.3M S3 training records, ~1.7M / 4.9M / 5.1M test records. Metric: per-S1-entity
F0.5, macro-averaged; a true singleton scores 1.0 only when nothing is predicted.

Everything runs on one machine (developed on Windows 11, Ryzen 7 / 16 threads, 15 GB RAM) in a few hours; nothing outside
the provided `dataset/` is used (no external data, APIs or pre-trained models). Only pip-installable libraries listed in
`requirements.txt`.

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
