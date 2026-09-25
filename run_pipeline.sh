#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p analysis output code/business_entity_resolution/bin

CXXFLAGS=(-O3 -std=c++17 -march=native -fopenmp)
g++ "${CXXFLAGS[@]}" -o code/business_entity_resolution/bin/candidate_engine code/business_entity_resolution/src/candidate_engine.cpp
g++ "${CXXFLAGS[@]}" -o code/business_entity_resolution/bin/full_positive_features code/business_entity_resolution/src/full_positive_features.cpp

python3 code/business_entity_resolution/src/audit_data.py --data dataset --output analysis/audit.json
python3 code/business_entity_resolution/src/sample_analysis.py --data dataset --modulus 200 --output analysis/sample_noise.tsv

OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}" code/business_entity_resolution/bin/candidate_engine \
  --data dataset/train --sample-mod 100 --sample-rem 0 --max-df 1000 --max-bucket 5 --top-k 100 \
  --features-out analysis/candidate_features.tsv

python3 code/business_entity_resolution/src/train_evaluate.py \
  --features analysis/candidate_features.tsv --data dataset/train --sample-mod 100 --sample-rem 0 \
  --model-out analysis/validation_model.txt --metrics-out analysis/metrics.json \
  --experiments-out analysis/experiments.csv

OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}" code/business_entity_resolution/bin/full_positive_features \
  dataset/train analysis/full_positive_features.tsv

THRESHOLD="$(python3 -c 'import json; print(json.load(open("analysis/metrics.json"))["decision_threshold_raw"])')"
python3 code/business_entity_resolution/src/final_train.py \
  --positives analysis/full_positive_features.tsv --candidates analysis/candidate_features.tsv \
  --threshold "$THRESHOLD" --output analysis/final_model.txt

OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}" code/business_entity_resolution/bin/candidate_engine \
  --data dataset/test --sample-mod 1 --sample-rem 0 --max-df 80000 --max-bucket 20 --top-k 50 \
  --features-out '' --model analysis/final_model.txt \
  --candidates-out output/candidate_pairs.tsv --matching-out output/matching_results.tsv

python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
