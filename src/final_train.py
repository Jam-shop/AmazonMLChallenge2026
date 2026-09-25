#!/usr/bin/env python3
"""Final matcher fit using every labeled positive plus all mined hard negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


META = ["s1_id", "target_id", "fold", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cols = [c for c in pd.read_csv(args.positives, sep="\t", nrows=0).columns if c not in META]
    xp = []
    for df in pd.read_csv(args.positives, sep="\t", chunksize=300_000):
        df["retrieval_hits"] = 0.5
        xp.append(df[cols].to_numpy(np.float32))
    xn = []
    for df in pd.read_csv(args.candidates, sep="\t", chunksize=300_000):
        df = df[df.label == 0]
        if len(df):
            df["retrieval_hits"] = 0.5
            xn.append(df[cols].to_numpy(np.float32))
    xp = np.concatenate(xp); xn = np.concatenate(xn)
    x = np.concatenate([xp, xn])
    y = np.r_[np.ones(len(xp), dtype=np.uint8), np.zeros(len(xn), dtype=np.uint8)]
    # Preserve the effective class mass of the validation-supported training sample
    # while still exposing the learner to every labeled pair.
    effective_pos, effective_neg = 58_992.0, 479_879.0 - 58_992.0
    weights = np.r_[np.full(len(xp), effective_pos / len(xp), dtype=np.float32),
                    np.full(len(xn), effective_neg / len(xn), dtype=np.float32)]
    print("final_rows", len(x), "all_positives", len(xp), "hard_negatives", len(xn), flush=True)
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=31,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=2026)
    model.fit(x, y, sample_weight=weights)
    with args.output.open("w") as out:
        out.write(f"type hgb\nthreshold {args.threshold:.17g}\nbaseline {float(model._baseline_prediction[0,0]):.17g}\ntrees {len(model._predictors)}\n")
        for predictors in model._predictors:
            nodes = predictors[0].nodes
            out.write(f"tree {len(nodes)}\n")
            for n in nodes:
                out.write("node %d %.17g %d %d %.17g %d %d\n" %
                          (int(n['feature_idx']), float(n['num_threshold']), int(n['left']), int(n['right']),
                           float(n['value']), int(n['is_leaf']), int(n['missing_go_to_left'])))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
