#!/usr/bin/env python
"""Diagnose, but do not alter, the extreme publication-split Dmax R2 result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    pred = pd.read_csv(args.predictions, encoding="utf-8-sig", low_memory=False)
    subset = pred[(pred["task"] == "T2_dmax_reg") & (pred["split_regime"] == "pub") & (pred["feature"] == "F1_morgan")].copy()
    if subset.empty:
        raise ValueError("no publication-split T2 Morgan predictions found")
    rows = []
    for seed, group in subset.groupby("seed"):
        y = group["y_true"].to_numpy(float)
        p = group["y_pred"].to_numpy(float)
        ss_res = float(np.square(y - p).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        rows.append({
            "seed": int(seed),
            "n_test": int(len(group)),
            "y_mean": float(y.mean()),
            "y_sd": float(y.std(ddof=0)),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "prediction_mean": float(p.mean()),
            "prediction_sd": float(p.std(ddof=0)),
            "prediction_min": float(p.min()),
            "prediction_max": float(p.max()),
            "mae": float(np.abs(y - p).mean()),
            "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot else None,
            "baseline_mean_squared_error": float(np.mean(np.square(y - y.mean()))),
            "worst_absolute_error": float(np.abs(y - p).max()),
        })
    payload = {
        "scope": "Diagnostic only; publication-split T2 predictions are unchanged and no records are removed.",
        "rows": rows,
        "pooled_n": int(len(subset)),
        "pooled_y_sd": float(subset["y_true"].std(ddof=0)),
        "interpretation": [
            "A negative R2 means the frozen model has larger squared error than the test-set mean predictor for that seed.",
            "The diagnosis separates small test-set variance, prediction range and individual extreme errors; it does not identify a causal mechanism.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
