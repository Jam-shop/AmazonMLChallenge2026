#!/usr/bin/env python3
"""Fit the pair matcher and optimize the exact entity-level macro F0.5 rule."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


META = ["s1_id", "target_id", "fold", "label"]


def fbeta(tp: int, pred: int, true: int) -> float:
    if true == 0:
        return float(pred == 0)
    if pred == 0 or tp == 0:
        return 0.0
    precision, recall = tp / pred, tp / true
    return 1.25 * precision * recall / (0.25 * precision + recall)


def load_truth(data: Path, sample_mod: int, sample_rem: int):
    truth, country = {}, {}
    with (data / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n = int(row["entity_id"].split("-", 1)[1])
            if n % sample_mod == sample_rem:
                country[row["entity_id"]] = row["country"]
    with (data / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source1_entity_id"] in country:
                truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    return truth, country


def entity_fold(sid: str) -> int:
    # Same SplitMix64 operation as candidate_engine.cpp.
    x = (int(sid.split("-", 1)[1]) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= x >> 31
    return int(x % 5 == 0)


def metric_at_threshold(scores, labels, groups, true_counts, threshold):
    pred = scores >= threshold
    pg = np.bincount(groups[pred], minlength=len(true_counts))
    tg = np.bincount(groups[pred & labels.astype(bool)], minlength=len(true_counts))
    vals = np.array([fbeta(int(t), int(p), int(y)) for t, p, y in zip(tg, pg, true_counts)])
    tp, npred, ntrue = int(tg.sum()), int(pg.sum()), int(true_counts.sum())
    return vals, tp / npred if npred else 1.0, tp / ntrue if ntrue else 1.0


def optimize_threshold(scores, labels, groups, true_counts):
    # Exact scan over score ties. Initial state predicts the empty set.
    order = np.argsort(-scores, kind="mergesort")
    pred = np.zeros(len(true_counts), dtype=np.int32)
    tp = np.zeros(len(true_counts), dtype=np.int32)
    current = np.array([fbeta(0, 0, int(y)) for y in true_counts])
    total = float(current.sum()); best = (total / len(true_counts), float("inf"))
    i = 0
    while i < len(order):
        score = scores[order[i]]; j = i
        touched = set()
        while j < len(order) and scores[order[j]] == score:
            k = order[j]; g = groups[k]
            if g not in touched:
                total -= current[g]; touched.add(int(g))
            pred[g] += 1; tp[g] += int(labels[k]); j += 1
        for g in touched:
            current[g] = fbeta(int(tp[g]), int(pred[g]), int(true_counts[g])); total += current[g]
        value = total / len(true_counts)
        if value > best[0]: best = (value, float(score))
        i = j
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("dataset/train"))
    ap.add_argument("--sample-mod", type=int, default=100)
    ap.add_argument("--sample-rem", type=int, default=0)
    ap.add_argument("--model-out", type=Path, default=Path("analysis/model.txt"))
    ap.add_argument("--metrics-out", type=Path, default=Path("analysis/metrics.json"))
    ap.add_argument("--experiments-out", type=Path, default=Path("analysis/experiments.csv"))
    args = ap.parse_args()
    truth, countries = load_truth(args.data, args.sample_mod, args.sample_rem)
    feature_cols = [c for c in pd.read_csv(args.features, sep="\t", nrows=0).columns if c not in META]

    # Train on all positives and a deterministic 5% sample of retrieved negatives.
    xs, ys = [], []
    for chunk in pd.read_csv(args.features, sep="\t", chunksize=250_000):
        chunk["retrieval_hits"] = 0.5  # neutralized so all-label final fitting has no blocker-score shift
        target_num = chunk.target_id.str.split("-", n=1).str[1].astype(np.int64)
        keep = (chunk.fold == 0) & ((chunk.label == 1) | (target_num % 4 == 0))
        if keep.any():
            xs.append(chunk.loc[keep, feature_cols].to_numpy(np.float32)); ys.append(chunk.loc[keep, "label"].to_numpy(np.int8))
    x = np.concatenate(xs); y = np.concatenate(ys)
    model = LogisticRegression(C=2.0, max_iter=500, solver="lbfgs", random_state=2026)
    model.fit(x, y)
    tree_model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=31,
                                                min_samples_leaf=40, l2_regularization=1.0,
                                                random_state=2026)
    tree_model.fit(x, y)
    print("fit_rows", len(y), "fit_positives", int(y.sum()), "features", len(feature_cols))

    # Retain only entity-disjoint validation pairs.
    frames = []
    for chunk in pd.read_csv(args.features, sep="\t", chunksize=250_000):
        chunk["retrieval_hits"] = 0.5
        v = chunk[chunk.fold == 1]
        if len(v):
            vx = v[feature_cols].to_numpy(np.float32)
            score = model.decision_function(vx); tree_score = tree_model.decision_function(vx)
            frames.append(pd.DataFrame({"s1_id": v.s1_id.to_numpy(), "label": v.label.to_numpy(np.int8), "score": score, "tree_score": tree_score,
                                        "name_exact": v.name_exact.to_numpy(np.int8), "address_exact": v.address_exact.to_numpy(np.int8),
                                        "best_field": v.best_field.to_numpy(np.float32), "worst_field": v.worst_field.to_numpy(np.float32)}))
    val = pd.concat(frames, ignore_index=True)
    val_ids = sorted(s for s in truth if entity_fold(s) == 1)
    gmap = {s: i for i, s in enumerate(val_ids)}
    groups = val.s1_id.map(gmap).to_numpy(np.int32)
    labels = val.label.to_numpy(np.int8); logistic_scores = val.score.to_numpy(np.float64)
    tree_scores = val.tree_score.to_numpy(np.float64)
    true_counts = np.array([len(truth[s]) for s in val_ids], dtype=np.int32)
    logistic_macro, logistic_threshold = optimize_threshold(logistic_scores, labels, groups, true_counts)
    tree_macro, tree_threshold = optimize_threshold(tree_scores, labels, groups, true_counts)
    if tree_macro >= logistic_macro:
        selected_name, scores, threshold = "hist_gradient_boosting", tree_scores, tree_threshold
    else:
        selected_name, scores, threshold = "logistic", logistic_scores, logistic_threshold
    vals, precision, recall = metric_at_threshold(scores, labels, groups, true_counts, threshold)
    singleton = true_counts == 0
    counts = np.bincount(groups, minlength=len(val_ids))
    candidate_recall = float(labels.sum() / true_counts.sum())

    experiments = []
    def assess(name, rule):
        sc = rule.astype(float)
        vv, pp, rr = metric_at_threshold(sc, labels, groups, true_counts, 0.5)
        experiments.append({"experiment": name, "candidate_recall": candidate_recall,
                            "avg_candidates": float(counts.mean()), "precision": pp, "recall": rr,
                            "macro_f0.5": float(vv.mean()), "singleton_accuracy": float(vv[singleton].mean()),
                            "non_singleton_f0.5": float(vv[~singleton].mean())})
    assess("A_exact_name_or_address", ((val.name_exact == 1) | (val.address_exact == 1)).to_numpy())
    assess("B_manual_fuzzy", ((val.best_field >= .72) & (val.worst_field >= .18) | (val.best_field >= .88)).to_numpy())
    assess("C_character_ngram", ((val.best_field >= .82) & (val.worst_field >= .10)).to_numpy())
    lv, lp, lr = metric_at_threshold(logistic_scores, labels, groups, true_counts, logistic_threshold)
    experiments.append({"experiment": "D_logistic_optimized", "candidate_recall": candidate_recall,
                        "avg_candidates": float(counts.mean()), "precision": lp, "recall": lr,
                        "macro_f0.5": float(lv.mean()), "singleton_accuracy": float(lv[singleton].mean()),
                        "non_singleton_f0.5": float(lv[~singleton].mean())})
    tv, tp, tr = metric_at_threshold(tree_scores, labels, groups, true_counts, tree_threshold)
    experiments.append({"experiment": "E_hist_gradient_boosting", "candidate_recall": candidate_recall,
                        "avg_candidates": float(counts.mean()), "precision": tp, "recall": tr,
                        "macro_f0.5": float(tv.mean()), "singleton_accuracy": float(tv[singleton].mean()),
                        "non_singleton_f0.5": float(tv[~singleton].mean())})
    pd.DataFrame(experiments).to_csv(args.experiments_out, index=False)

    if selected_name == "hist_gradient_boosting":
        with args.model_out.open("w") as out:
            out.write(f"type hgb\nthreshold {threshold:.17g}\nbaseline {float(tree_model._baseline_prediction[0,0]):.17g}\ntrees {len(tree_model._predictors)}\n")
            for predictors in tree_model._predictors:
                nodes = predictors[0].nodes
                out.write(f"tree {len(nodes)}\n")
                for n in nodes:
                    out.write("node %d %.17g %d %d %.17g %d %d\n" %
                              (int(n['feature_idx']), float(n['num_threshold']), int(n['left']), int(n['right']),
                               float(n['value']), int(n['is_leaf']), int(n['missing_go_to_left'])))
    else:
        raw_coef = model.coef_[0]
        args.model_out.write_text("type logistic\nintercept %.17g\nthreshold %.17g\ncoefficients %s\n" %
                                  (model.intercept_[0], threshold, " ".join(f"{z:.17g}" for z in raw_coef)))
    by_country = {c: float(vals[[countries[s] == c for s in val_ids]].mean()) for c in sorted(set(countries.values()))}
    by_mult = {str(k): float(vals[true_counts == k].mean()) for k in sorted(set(true_counts))}
    total_targets = sum(1 for src in (2, 3) for _ in open(args.data / f"train_source{src}.tsv", encoding="utf-8")) - 2
    metrics = {
        "validation_s1": len(val_ids), "validation_pairs": len(val), "macro_f0.5": float(vals.mean()),
        "precision": precision, "recall": recall, "singleton_accuracy": float(vals[singleton].mean()),
        "non_singleton_f0.5": float(vals[~singleton].mean()), "candidate_recall": candidate_recall,
        "average_candidates": float(counts.mean()), "median_candidates": float(np.median(counts)),
        "p95_candidates": float(np.percentile(counts, 95)), "max_candidates": int(counts.max()),
        "candidate_reduction_ratio": 1.0 - float(counts.mean()) / total_targets,
        "selected_model": selected_name, "decision_threshold_raw": threshold,
        "logistic_macro_f0.5": logistic_macro, "hgb_macro_f0.5": tree_macro,
        "country_macro_f0.5": by_country, "multiplicity_macro_f0.5": by_mult,
        "fit_rows": len(y), "fit_positives": int(y.sum()), "features": feature_cols,
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
