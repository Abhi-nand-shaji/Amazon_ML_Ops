# Business Entity Resolution at Scale (Amazon ML Challenge 2026)

Given business records from three independent sources with noisy names and addresses (typos, abbreviations, missing
fields, transliterated scripts), find for every Source-1 entity all Source-2 / Source-3 records that describe the same
real-world business. Scored with per-entity **F0.5, macro-averaged** (precision-weighted; a true singleton scores 1 only
when nothing is predicted). The test set adds a country (France) that never appears in training.

The pipeline is **candidate generation in three cheap steps + a supervised matcher**, fully vectorized, and regenerates both
submission files from the raw data on one laptop (8 cores, 15 GB RAM, no GPU, no external data or pre-trained models).

## Results

| | |
|---|---|
| Matcher macro F0.5, 7,500 **hold-out** entities (never used for any decision) | **0.9647** |
| Matcher macro F0.5, 7,500 validation entities | 0.9628 |
| Simple baselines on the same entities (exact name, fuzzy rule, similarity / TF-IDF thresholds) | 0.48 - 0.73 |
| Ceiling if every true pair that is a candidate were predicted | 0.9876 |
| Candidates per Source-1 entity (training sample) / share of **all** true pairs kept | **6.08** / 96.55% |
| Test split: candidate pairs / per entity | 12,288,781 / **7.1** (France 9.3, India 7.3, US 6.0) |
| Test split: predicted matches | 5,671,608 pairs; 93.9% of the 1,732,544 entities get at least one |

The candidate set was cut from 30 to ~6 per entity (retrieval + ranker shortlist -> learned filter) with no loss in F0.5:
0.9625 / 0.9639 (validation / hold-out) with 30 candidates, 0.9628 / 0.9647 with the filtered set.

Full write-up: [`Documentation_template.md`](Documentation_template.md). Detailed reports: [`code/business_entity_resolution/reports/`](code/business_entity_resolution/reports/)
(candidate generation, experiment logs, 26 edge cases with a failure taxonomy, validation / hold-out results).

## Method

```
normalized records (per country partition; country is never a model feature)
   |
   |  1. retrieval: inverted indices over rare name / address tokens, exact name / compact name / postal keys,
   |     address and name bigrams, compact-name prefix -- every key family capped by document frequency
   v
~450 retrieved pool records per entity
   |  2. learned ranker (LightGBM on cheap retrieval evidence): shortlist of the best 30  -> 97.0% of all true pairs
   v
30 per entity
   |  3. learned candidate filter (retrieval evidence + 7 cheap string similarities), cross-fitted,
   |     probability floor chosen on validation entities                                -> 96.5% of all true pairs
   v
~6 per entity  = output/candidate_pairs.tsv = exactly what the matcher scores
   |  4. matcher: LightGBM over ~90 vectorized pairwise features (string similarities, token / IDF overlap,
   |     house number / postal agreement, pool ambiguity statistics, within-entity context)
   |  5. decision: threshold searched on the challenge's own metric (singleton rule included)
   v
output/matching_results.tsv
```

Evaluation mimics the task: S1 entities are split into train / validation / hold-out; validation entities drive every
decision (early stopping, thresholds, rules, candidate floor), hold-out entities are scored once at the end.

## Repository layout

```
Documentation_template.md            methodology write-up (filled from the run's own reports)
output/matching_results.tsv.gz       final matches (the leaderboard file), gzip-compressed
output/candidate_pairs.tsv.gz        final candidate set, gzip-compressed
code/business_entity_resolution/
    src/                             pipeline (run_all.py runs everything; see its README)
    tests/                           unit tests (python -m unittest discover -s code/business_entity_resolution/tests)
    reports/                         blocking / edge-case / validation reports, experiment logs
    artifacts/                       trained models and configs (matcher, ranker, candidate filter, thresholds)
    README.md, requirements.txt      exact run instructions and pinned dependencies
```

The result files are stored compressed because GitHub rejects files over 100 MB. To get the submission files:

```bash
gunzip -k output/matching_results.tsv.gz output/candidate_pairs.tsv.gz
```

## Reproduce

1. Put the challenge's `dataset/` (train / test TSV files) and `utils/validate_submission.py` next to `code/`
   (they are not redistributed here).
2. Create the environment: `python -m venv .venv` and `pip install -r code/business_entity_resolution/requirements.txt`.
3. Run everything (stages are resumable; `--list` shows them, `--from <stage>` resumes):

```bash
python code/business_entity_resolution/src/run_all.py
```

On the reference laptop: normalization ~20 min, test-split retrieval + ranker ~71 min, candidate filter ~13 min, scoring
~15 min, training stages a few minutes each. Stage-by-stage commands are in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
