# ML Challenge 2026: Business Entity Resolution — Solution Report

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We use a two-stage, scalable pipeline: a high-recall C++ blocking and candidate-ranking engine to produce a compact set of candidate pairs, followed by learned pairwise matching models (logistic regression and a HistGradientBoosting ensemble). Key innovations are a multi-strategy blocking scheme (token/q-gram LSH + compound keys + exact keys) implemented in C++ for scale, deterministic sampling for reliable validation, and an entity-level macro F0.5 optimization to choose the decision threshold that best serves the competition metric.

---

## 2. Methodology

### 2.1 Problem Analysis
- Input is noisy business registry-style data: missing fields, inconsistent use of punctuation and abbreviations, international postal formats, and company suffixes (Inc/LLC/etc.) that do not help matching.
- Addresses show many local abbreviations and numeric tokens (suite/apt/building numbers, postal codes) that are useful signals when present but frequently missing.
- Entity identifiers use prefixed IDs like `S1-12345` and the dataset is multi-source (S1, S2, S3) with country-coded rows, so blocking keyed by country reduces spurious cross-country comparisons.

### 2.2 Solution Strategy
- Approach Type: Blocking + Classifier (hybrid)
- Pipeline outline:
	1. Preprocess and normalize text (remove diacritics, map `&`→`and`, casefold, remove legal suffixes from names, abbreviate common address tokens).
	2. Build a compact blocking index keyed by country and multiple signal types (exact name, exact address, rare token hashes, q-gram LSH keys and compound q-gram pairs).
	3. For each target row, retrieve candidates via index buckets, compute pairwise evidence counts and a lightweight rank, and keep the top-K candidates per source1 row.
	4. Extract a fixed feature vector per pair (token/q-gram overlaps, edit similarity, containment, numeric/postal agreement, exact flags, retrieval hits, etc.).
	5. Train classifiers (logistic + HistGradientBoosting) on a deterministic training sample; validate on an entity-disjoint fold; optimize an entity-level macro F0.5 threshold by exact scan across score ties.
	6. Final training uses all labeled positives and mined hard negatives (sample-weighted) to produce a final HGB model saved in the engine textual format.

**Core Innovation:** A production-ready, memory-efficient C++ blocking and feature-extraction engine that combines multiple complementary blocking strategies (including a MinHash-style q-gram LSH with multi-band aggregation and compound q-gram pairs) so that the true match set is retained while keeping candidate counts tractable.

---

## 3. Candidate Generation (Blocking)
We reduce the cartesian comparison space via a multi-strategy blocking index implemented in `src/candidate_engine.cpp`.

- **Blocking keys used:**
	- Exact normalized name and normalized address keyed by country.
	- Rare token hashes from tokenized names and addresses (DF-limited using `--max-df`).
	- Q-gram hashes (3-character q-grams) and a 32-band MinHash-style LSH (`lsh_keys`) derived from q-grams.
	- Compound pair hashes formed from pairs of q-gram hashes to detect multi-token co-occurrences.

- **Candidate pairs generated:**
	- For each target row, keys are looked up in a sorted `index` (pairs of `(key, s1_idx)`); buckets are pruned by a `--max-bucket` limit and low-DF tokens to avoid noisy high-frequency signals.
	- A small ranked heap per `s1` entry keeps the top-K candidates (configurable via `--top-k`). This yields a compact list of candidate pairs per `S1` that is typically orders of magnitude smaller than full cross-product.

- **How we ensure true matches are not lost:**
	- Multiple complementary blocking keys (exact, token, q-gram, compound q-gram) increase recall across different kinds of variations.
	- `choose_rare` prefers low-document-frequency tokens to avoid noisy high-frequency tokens while still including discriminative tokens.
	- Exact name/address matches are always considered and low q-gram-dice candidates are filtered only after passing several heuristic evidence checks; this keeps high-recall while trimming low-quality pairs.

---

## 4. Matching Model

**Features used:** (feature vector computed in `candidate_engine.cpp::features` and mirrored in Python feature pipelines)
- Source indicator: `source == 3` flag.
- Exact-match flags: `name_exact`, `address_exact` (boolean if normalized fields are identical and non-empty).
- Name features: token Jaccard, q-gram Dice, normalized edit similarity, containment (substring), length ratio.
- Address features: token Jaccard, q-gram Dice, normalized edit similarity, containment, length ratio.
- Numeric/address-specific: `numeric_agree`, `numeric_conflict`, `numeric_overlap_ratio` (counts of numeric tokens in address), `postal_agree` and `postal_conflict` (postal code overlaps).
- Retrieval / evidence: `retrieval_hits`, and derived `best_field` / `worst_field` summary features (best/worst of name/address similarities), plus a compact `hits` summary used for ranking.

**Model type:** Logistic Regression (baseline) and HistGradientBoostingClassifier (primary candidate). Final production model uses the HGB model trained with all positives and mined hard negatives.

**Training strategy:**
- Training uses all labeled positives and a deterministic sample of negatives for the validation-supported training sample; final training uses every labeled positive plus mined hard negatives and applies sample weighting to preserve effective class mass.

**Threshold selection method:**
- We perform an exact, group-aware scan across score ties (see `train_evaluate.optimize_threshold`) to directly maximize entity-level macro F0.5 (the competition metric). This selects a single score threshold such that the extracted matched set best balances precision and recall per entity.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** reported in `analysis/metrics.json` under the `macro_f0.5` key (generated by `train_evaluate.py`). See `analysis/experiments.csv` for experiment breakdowns.

- **Common false positives (wrong merges):**
	- Short, generic business names (e.g., single-word names like `green`) or highly common tokens that pass token/q-gram overlap thresholds.
	- Different businesses sharing the same postal token or building identifier where address token overlap is strong but registration refers to distinct entities.

- **Common false negatives (missed matches):**
	- Records with heavy textual divergence (abbreviations, reordered tokens, or missing address numbers) that fail to trigger any blocking key despite being true matches.
	- Cases where most discriminative tokens were filtered out by DF thresholds; rare edge cases can be recovered by tuning `--max-df` and LSH parameters.

Detailed per-country and multiplicity breakdowns are included in `analysis/metrics.json` (`country_macro_f0.5`, `multiplicity_macro_f0.5`).

---

## 6. Conclusion
We present a scalable, high-recall candidate-generation engine combined with a compact, strong pairwise classifier whose threshold is optimized directly for the entity-level macro F0.5 metric. The approach balances recall and precision through complementary blocking strategies and careful negative mining, producing a practical system for large-scale entity resolution.

---

## Appendix

### A. Code Artefacts
All code lives in the submission under `code/src/` (see the repository `src/` folder). Key files and entry points:

- `src/candidate_engine.cpp` — high-performance C++ blocking, ranking, and feature extraction engine. Produces candidate heaps and can write `--features-out`, `--candidates-out`, and `--matching-out` when run in final mode.
- `src/train_evaluate.py` — trains logistic and HGB models, optimizes entity-level macro F0.5 threshold, writes `analysis/model.txt`, `analysis/metrics.json`, and `analysis/experiments.csv`.
- `src/final_train.py` — final HGB fit on all positives + mined hard negatives; writes a textual HGB model compatible with the C++ engine.
- `src/audit_data.py` — chunked dataset audit that reports missing values, country distributions, and duplicate statistics.
- `src/sample_analysis.py` — creates deterministic labeled sample pairs (positives/negatives) and writes a tab-separated sample file useful for error analysis.

Reproducible commands (run from the submission root where `code/` lives):

```bash
# Compile the candidate engine (Linux, with OpenMP support):
g++ -O3 -std=c++17 -fopenmp -o candidate_engine code/src/candidate_engine.cpp

# Run blocking / feature extraction (example):
./candidate_engine --data dataset/train --features-out analysis/candidate_features.tsv --top-k 100 --max-df 80

# Train and evaluate (Python, produces model and metrics):
python3 code/src/train_evaluate.py --features analysis/candidate_features.tsv --model-out analysis/model.txt --metrics-out analysis/metrics.json --experiments-out analysis/experiments.csv

# Final training using mined negatives and saved threshold (produces textual HGB model):
python3 code/src/final_train.py --positives analysis/positives.tsv --candidates analysis/candidate_features.tsv --threshold <THRESHOLD_FROM_VALIDATE> --output analysis/final_model.txt

# Run the compiled engine in final mode to write candidate lists and matched outputs
./candidate_engine --data dataset/train --model analysis/final_model.txt --candidates-out output/candidate_pairs.tsv --matching-out output/matching_results.tsv
```

### B. Additional Results
All evaluation artifacts are under `analysis/` after running the above pipeline: `metrics.json`, `experiments.csv`, `model.txt`, and intermediate `candidate_features.tsv`. Use `sample_analysis.py` to produce small deterministic sample pairs for manual inspection: `python3 code/src/sample_analysis.py --data dataset --output analysis/sample_noise.tsv`.

---

**Note:** parameter knobs (e.g., `--max-df`, `--max-bucket`, `--top-k`, and LGB/HGB hyperparameters) are available to trade recall vs compute and were tuned empirically on the validation fold in `train_evaluate.py`.
