# Business Entity Resolution at Scale (Amazon ML Challenge 2026)

Given business records from three independent sources with noisy names and addresses (typos, abbreviations, missing
fields, transliterated scripts), find for every Source-1 entity all Source-2 / Source-3 records that describe the same
real-world business. Scored with per-entity **F0.5, macro-averaged** (precision-weighted; a true singleton scores 1 only
when nothing is predicted). The test set adds a country (France) that never appears in training.

The pipeline has four layers:
1. **Candidate generation** on CPU: multi-key blocking, a learned ranker, a learned candidate filter, and second-hop retrieval.
2. **First-stage LightGBM matcher**.
3. **Fine-tuned multilingual cross-encoder**: mDeBERTa-v3-base, MIT licence, trained on a Kaggle GPU.
4. **Graph stacker** that resolves every candidate collectively over the complete candidate graph (exclusivity of pool
   records, entity context, sibling agreement).

No external data or lookup service is used.

## Results

Macro F0.5 on 7,500 **hold-out** S1 entities, which were never used for training, early stopping or any selection. The
last column is the public leaderboard.

| version | what it adds | hold-out F0.5 | public LB |
|---|---|---|---|
| v1 | blocking + ranker + candidate filter + LightGBM matcher | 0.9647 | 0.93 |
| v2 | graph stacker on the full training world | 0.9742 | 0.95 |
| v3 | + fine-tuned cross-encoder (mDeBERTa-v3-base) | 0.9818 | 0.9739 |
| **v4** | + second-hop candidates, cross-encoder on the new pairs | **0.9847** | see portal |

On the same entities, the ceiling with a perfect matcher on the v4 candidate sets is 0.9913. The v4 candidate set holds
97.90% of all true pairs, with 7.4 candidates per entity. Simple baselines score 0.48 – 0.73.

Full write-up: [`Documentation_template.md`](Documentation_template.md). Detailed reports are in
[`code/business_entity_resolution/reports/`](code/business_entity_resolution/reports/).

## Method

```
normalized records (per country partition; country is never a model feature)
   |  1. retrieval: inverted indices over rare name / address tokens, exact keys, address and name bigrams,
   |     compact-name prefix (document-frequency caps)            -> ~450 pool records per entity
   |  2. learned ranker: shortlist of 30                          -> 97.0% of all true pairs
   |  3. learned candidate filter (cross-fitted)                   -> ~6 per entity, 96.5% of all true pairs
   |  4. second-hop retrieval: the entity's confident matches query the indices again,
   |     learned new-pair filter                                   -> ~7.4 per entity, 97.9% of all true pairs
   v
candidate set = output/candidate_pairs.tsv = exactly what the models score
   |  first-stage LightGBM over ~90 pairwise features (p1) for the COMPLETE world
   |  cross-encoder: mDeBERTa-v3-base fine-tuned on 1.17M labelled pairs (p_ce)
   |  graph stacker: LightGBM over pairwise features + p1 + p_ce + collective features
   |  decision: one threshold chosen with the challenge metric on validation entities
   v
output/matching_results.tsv
```

## Repository layout

```
Documentation_template.md            methodology write-up (filled from the run's own reports)
output/matching_results.tsv.gz       final matches (the leaderboard file), gzip-compressed
output/candidate_pairs.tsv.gz        final candidate set, gzip-compressed
code/business_entity_resolution/
    src/                             pipeline (see its README for every stage)
    tests/                           unit tests
    reports/                         blocking / edge-case / validation reports, experiment logs
    artifacts/                       trained models and configs: matcher, ranker, candidate filter, new-pair filter,
                                     stackers, entity groups, fine-tuned cross-encoder (ce/ce_model, Git LFS)
    README.md, requirements.txt, requirements-gpu.txt
```

The result files are stored compressed because GitHub rejects files over 100 MB:

```bash
gunzip -k output/matching_results.tsv.gz output/candidate_pairs.tsv.gz
```

The cross-encoder weights (558 MB) are stored with Git LFS. Run `git lfs install` before cloning, or `git lfs pull` in
an existing clone.

## Reproduce

1. Put the challenge's `dataset/` and `utils/validate_submission.py` next to `code/`. They are not redistributed here.
2. Create a CPU environment: `pip install -r code/business_entity_resolution/requirements.txt`.
3. Run the stages in the order given in
   [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md). `run_all.py` runs the first
   design. The stacker, second-hop and cross-encoder stages have their own commands there. The two GPU scripts
   (`kaggle_ce.py`, `kaggle_dense.py`) are self-contained and run on Kaggle "GPU T4 x2" or any CUDA machine
   (`requirements-gpu.txt`).
