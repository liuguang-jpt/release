# -*- coding: utf-8 -*-
"""Matched-backbone evaluation for exact and one-sided censored pDC50 data.

The primary comparison holds the MLP architecture and optimizer fixed while
changing only how censored training records are handled: drop, substitute the
reported bound, or use a one-sided censored loss.  XGBoost AFT is reported as a
separate standard censored-regression baseline and is not part of the matched
architecture claim.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from benchmark_contract import (
    FEATURE_PATH,
    MANIFEST_PATH,
    PROCESSED_DIR,
    SEEDS,
    assert_group_disjoint,
    group_values,
    load_contract,
    load_dataset,
    load_groups,
    load_manifest,
    role_indices,
)
from pu_pipeline import censored_loss_fit_predict, mlp_regression_fit_predict, regression_metrics

RESULTS_PATH = PROCESSED_DIR / "censored_eval_results_v3.json"
PREDICTIONS_PATH = PROCESSED_DIR / "censored_predictions_v3.csv"
SPLITS = ["random", "scaffold"]
MLP_CONFIG = {"epochs": 60, "batch_size": 256, "learning_rate": 1e-3, "hidden": 128}


def summarize(values: dict[str, list[float]]) -> dict:
    return {
        key: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=0))}
        for key, v in values.items()
        if v
    }


def censored_metrics(pred: np.ndarray, bound: np.ndarray, side: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    bound = np.asarray(bound, dtype=float)
    side = np.asarray(side, dtype=str)
    right = side == "right-censored"  # true pDC50 is below the reported upper bound
    left = side == "left-censored"    # true pDC50 is above the reported lower bound
    violation = np.where(right, pred > bound, pred < bound)
    magnitude = np.where(right, np.maximum(pred - bound, 0), np.maximum(bound - pred, 0))
    out = {
        "n": int(len(pred)),
        "n_right": int(right.sum()),
        "n_left": int(left.sum()),
        "violation_rate": float(violation.mean()),
        "constraint_coverage": float(1.0 - violation.mean()),
        "violation_magnitude": float(magnitude.mean()),
        "violation_magnitude_given_violation": float(magnitude[violation].mean()) if violation.any() else 0.0,
    }
    for label, mask in [("right", right), ("left", left)]:
        if mask.any():
            out[f"{label}_violation_rate"] = float(violation[mask].mean())
            out[f"{label}_constraint_coverage"] = float(1.0 - violation[mask].mean())
            out[f"{label}_violation_magnitude"] = float(magnitude[mask].mean())
            active = violation[mask]
            out[f"{label}_violation_magnitude_given_violation"] = (
                float(magnitude[mask][active].mean()) if active.any() else 0.0
            )
        else:
            out[f"{label}_violation_rate"] = float("nan")
            out[f"{label}_constraint_coverage"] = float("nan")
            out[f"{label}_violation_magnitude"] = float("nan")
            out[f"{label}_violation_magnitude_given_violation"] = float("nan")
    return out


def xgb_aft_fit_predict(
    X_train: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit a standard XGBoost accelerated-failure-time model."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(np.asarray(X_train, dtype=np.float32))
    dtrain.set_float_info("label_lower_bound", np.asarray(lower, dtype=np.float32))
    dtrain.set_float_info("label_upper_bound", np.asarray(upper, dtype=np.float32))
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": int(seed),
        "nthread": max(1, (os.cpu_count() or 2) - 1),
    }
    booster = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
    return np.asarray(booster.predict(xgb.DMatrix(np.asarray(X_test, dtype=np.float32))), dtype=float)


def main() -> None:
    started = time.time()
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    contract = load_contract(validate_hashes=True)
    X = np.load(FEATURE_PATH, mmap_mode="r")
    if X.shape[0] != len(df):
        raise AssertionError("feature rows do not match record-level dataset")

    obs_type = df["dc50_obs_type"].astype(str).to_numpy()
    exact_value = pd.to_numeric(df["pdc50_value"], errors="coerce").to_numpy(float)
    lower_value = pd.to_numeric(df["pdc50_lower"], errors="coerce").to_numpy(float)
    upper_value = pd.to_numeric(df["pdc50_upper"], errors="coerce").to_numpy(float)
    exact_mask = (obs_type == "exact") & np.isfinite(exact_value)
    left_mask = (obs_type == "left-censored") & np.isfinite(lower_value)
    right_mask = (obs_type == "right-censored") & np.isfinite(upper_value)
    censored_mask = left_mask | right_mask
    endpoint_mask = exact_mask | censored_mask

    results = {
        "meta": {
            "script": "censored_eval_pipeline.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "manifest_family": "all_records_2way",
            "splits": SPLITS,
            "seeds": SEEDS,
            "endpoint": "pDC50",
            "n_exact_total": int(exact_mask.sum()),
            "n_left_censored_total": int(left_mask.sum()),
            "n_right_censored_total": int(right_mask.sum()),
            "n_interval_censored_excluded": int(np.sum(obs_type == "interval-censored")),
            "matched_backbone_methods": ["mlp_drop", "mlp_bound", "mlp_censored"],
            "matched_backbone": {
                "architecture": "2048 -> 128 ReLU Dropout(0.2) -> 64 ReLU -> 1",
                "optimizer": "Adam, weight_decay=1e-5",
                **MLP_CONFIG,
            },
            "separate_standard_baseline": "xgb_aft",
            "interpretation": "Only the three MLP methods support a loss-handling comparison with architecture held fixed.",
            "censor_semantics": {
                "right-censored": "true pDC50 <= pdc50_upper; prediction above upper bound is a violation",
                "left-censored": "true pDC50 >= pdc50_lower; prediction below lower bound is a violation",
            },
        },
        "splits": {},
    }
    prediction_rows: list[dict] = []

    for split in SPLITS:
        aggregate: dict[str, list[float]] = defaultdict(list)
        per_seed = []
        group_ids = group_values(df, groups, split)
        for seed in SEEDS:
            roles = role_indices(df, manifest, "all_records_2way", split, seed)
            assert_group_disjoint(df, groups, roles, split)
            tr_all, te_all = roles["train"], roles["test"]
            tr_endpoint = tr_all[endpoint_mask[tr_all]]
            tr_exact = tr_all[exact_mask[tr_all]]
            tr_censored = tr_all[censored_mask[tr_all]]
            te_exact = te_all[exact_mask[te_all]]
            te_censored = te_all[censored_mask[te_all]]
            if min(len(tr_exact), len(tr_censored), len(te_exact), len(te_censored)) == 0:
                raise AssertionError(f"empty required endpoint subset: {split}/{seed}")
            if set(tr_endpoint) & (set(te_exact) | set(te_censored)):
                raise AssertionError(f"endpoint train/test leakage: {split}/{seed}")

            train_bound = np.where(
                right_mask[tr_censored], upper_value[tr_censored], lower_value[tr_censored]
            )
            train_censor_code = np.where(right_mask[tr_censored], 1.0, -1.0)
            combined_train = np.concatenate([tr_exact, tr_censored])
            combined_y = np.concatenate([exact_value[tr_exact], train_bound])
            combined_censor = np.concatenate([np.zeros(len(tr_exact)), train_censor_code])
            eval_idx = np.concatenate([te_exact, te_censored])

            mlp_args = {
                "epochs": MLP_CONFIG["epochs"],
                "bs": MLP_CONFIG["batch_size"],
                "lr": MLP_CONFIG["learning_rate"],
                "hidden": MLP_CONFIG["hidden"],
            }
            pred_drop = mlp_regression_fit_predict(
                X[tr_exact], exact_value[tr_exact], X[eval_idx], seed=seed, **mlp_args
            )
            pred_bound = mlp_regression_fit_predict(
                X[combined_train], combined_y, X[eval_idx], seed=seed, **mlp_args
            )
            pred_censored = censored_loss_fit_predict(
                X[combined_train], combined_y, combined_censor, X[eval_idx], seed=seed, **mlp_args
            )

            aft_lower = np.concatenate(
                [
                    exact_value[tr_exact],
                    np.where(left_mask[tr_censored], lower_value[tr_censored], -np.inf),
                ]
            )
            aft_upper = np.concatenate(
                [
                    exact_value[tr_exact],
                    np.where(right_mask[tr_censored], upper_value[tr_censored], np.inf),
                ]
            )
            pred_aft = xgb_aft_fit_predict(X[combined_train], aft_lower, aft_upper, X[eval_idx], seed)

            methods = {
                "mlp_drop": pred_drop,
                "mlp_bound": pred_bound,
                "mlp_censored": pred_censored,
                "xgb_aft": pred_aft,
            }
            n_exact_test = len(te_exact)
            test_bound = np.where(
                right_mask[te_censored], upper_value[te_censored], lower_value[te_censored]
            )
            test_side = obs_type[te_censored]
            seed_entry = {
                "seed": int(seed),
                "n_train_exact": int(len(tr_exact)),
                "n_train_censored": int(len(tr_censored)),
                "n_train_left": int(left_mask[tr_censored].sum()),
                "n_train_right": int(right_mask[tr_censored].sum()),
                "n_test_exact": int(len(te_exact)),
                "n_test_censored": int(len(te_censored)),
                "n_test_left": int(left_mask[te_censored].sum()),
                "n_test_right": int(right_mask[te_censored].sum()),
                "metrics": {},
            }
            for method, pred_all in methods.items():
                exact_pred = np.asarray(pred_all[:n_exact_test], dtype=float)
                cens_pred = np.asarray(pred_all[n_exact_test:], dtype=float)
                exact_metrics = regression_metrics(exact_value[te_exact], exact_pred)
                constraint_metrics = censored_metrics(cens_pred, test_bound, test_side)
                seed_entry["metrics"][method] = {
                    "exact_test": exact_metrics,
                    "censored_test": constraint_metrics,
                }
                for metric, value in exact_metrics.items():
                    aggregate[f"{method}::exact::{metric}"].append(value)
                for metric, value in constraint_metrics.items():
                    if metric not in {"n", "n_right", "n_left"}:
                        aggregate[f"{method}::censored::{metric}"].append(value)

                for idx, yt, yp in zip(te_exact, exact_value[te_exact], exact_pred):
                    prediction_rows.append(
                        {
                            "split_regime": split,
                            "seed": int(seed),
                            "method": method,
                            "evaluation_set": "exact",
                            "record_id": int(df.iloc[idx]["record_id"]),
                            "group_id": str(group_ids[idx]),
                            "obs_type": "exact",
                            "y_true": float(yt),
                            "bound": np.nan,
                            "censor_side": "",
                            "y_pred": float(yp),
                        }
                    )
                for idx, bound, side, yp in zip(te_censored, test_bound, test_side, cens_pred):
                    prediction_rows.append(
                        {
                            "split_regime": split,
                            "seed": int(seed),
                            "method": method,
                            "evaluation_set": "censored",
                            "record_id": int(df.iloc[idx]["record_id"]),
                            "group_id": str(group_ids[idx]),
                            "obs_type": str(side),
                            "y_true": np.nan,
                            "bound": float(bound),
                            "censor_side": "right" if side == "right-censored" else "left",
                            "y_pred": float(yp),
                        }
                    )
            per_seed.append(seed_entry)
            print(
                f"{split} seed={seed}: exact MAE drop/censored="
                f"{seed_entry['metrics']['mlp_drop']['exact_test']['mae']:.3f}/"
                f"{seed_entry['metrics']['mlp_censored']['exact_test']['mae']:.3f}; "
                f"censored violation drop/censored="
                f"{seed_entry['metrics']['mlp_drop']['censored_test']['violation_rate']:.3f}/"
                f"{seed_entry['metrics']['mlp_censored']['censored_test']['violation_rate']:.3f}"
            )
        split_entry = summarize(aggregate)
        split_entry["per_seed"] = per_seed
        results["splits"][split] = split_entry

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {PREDICTIONS_PATH} ({len(prediction_rows):,} predictions)")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
