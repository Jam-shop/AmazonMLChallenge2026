# Business Entity Resolution Pipeline

This is a deterministic, offline solution for the Amazon ML Challenge 2026. It uses only the supplied TSV files. No external entity data, geocoder, API, or pretrained model is used. The matcher is scikit-learn's BSD-licensed `HistGradientBoostingClassifier`; it is far below the 8B-parameter limit and is exported to a small text tree ensemble for dependency-free C++ inference.

## Environment

- Linux, Python 3.11, GNU C++ 17 with OpenMP
- Peak measured RAM: about 6.3 GB during all-label final training; allow 12 GB for full test blocking/inference
- Allow at least 5 GB of free working disk in addition to the supplied dataset

Install Python dependencies:

```bash
python3 -m pip install -r code/business_entity_resolution/requirements.txt
```

All commands below are run from `student_resource/`.

## Reproduce end to end

The convenience entry point performs compilation, leakage-safe validation training, all-label final fitting, test inference, and official validation:

```bash
bash code/business_entity_resolution/run_pipeline.sh
```

The major stages are:

1. `candidate_engine.cpp` builds open-country exact, rare-token, individual trigram, and rare-trigram-pair indexes. It ranks and caps the final candidate set before scoring.
2. `train_evaluate.py` uses an S1-disjoint deterministic split (`SplitMix64(entity_id) % 5`) and optimizes the exact entity-level macro-F0.5 threshold.
3. `full_positive_features.cpp` materializes features for every labeled positive relation. `final_train.py` fits on all 7,638,365 positives plus all validation-mined hard negatives, with weights preserving the validated class balance.
4. The exported tree model is scored by the C++ engine over all test candidates, creating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
5. The organizer's `utils/validate_submission.py` runs with `--check-ids`.

Intermediate feature files are large and may be deleted after `analysis/final_model.txt` has been produced. Fixed random seed: 2026.

## Output contract

Both outputs contain exactly one tab-separated row per test S1. Empty values represent singletons. Candidate IDs and match IDs are unique S2/S3 IDs, and every predicted match is drawn from that S1's final candidate set.

## Source files

- `candidate_engine.cpp`: normalization, blocking, candidate ranking, features, exported-model inference, TSV output
- `sample_analysis.py`: deterministic true/non-match string-noise audit
- `audit_data.py`: bounded-memory dataset audit
- `train_evaluate.py`: baselines, exact macro-F0.5 evaluation, model comparison, threshold selection
- `full_positive_features.cpp`: feature extraction for all labeled positive relations
- `final_train.py`: final all-training-label model fit and export

