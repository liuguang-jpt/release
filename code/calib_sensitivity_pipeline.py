# -*- coding: utf-8 -*-
"""Calibration and class-prior sensitivity under the frozen v3 benchmark.

All train/calibration/test roles come from split_manifest_v3.csv.  No local
record or group split is permitted in this script.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from benchmark_contract import (
    FEATURE_PATH,
    LABEL_COL,
    MANIFEST_PATH,
    PROCESSED_DIR,
    SEEDS,
    SPLITS,
    assert_group_disjoint,
    group_values,
    load_contract,
    load_dataset,
    load_groups,
    load_manifest,
    role_indices,
)
from pu_pipeline import classification_metrics, nnpu_fit_predict

CALIB_RESULTS_PATH = PROCESSED_DIR / "calib_results_v3.json"
CALIB_PREDICTIONS_PATH = PROCESSED_DIR / "calib_predictions_v3.csv"
PRIOR_RESULTS_PATH = PROCESSED_DIR / "prior_sensitivity_v3.json"
PRIORS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]


def logits_from_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(prob / (1 - prob))


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.2, hi: float = 5.0) -> float:
    """Fit a scalar temperature by calibration-set negative log likelihood."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)
    best_t, best_nll = 1.0, float("inf")
    for temp in np.linspace(lo, hi, 97):
        prob = expit(logits / temp)
        nll = -np.mean(
            y * np.log(np.clip(prob, 1e-7, 1 - 1e-7))
            + (1 - y) * np.log(np.clip(1 - prob, 1e-7, 1 - 1e-7))
        )
        if nll < best_nll:
            best_t, best_nll = float(temp), float(nll)
    return best_t


def summarize(values: dict[str, list[float]]) -> dict:
    return {
        key: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=0))}
        for key, v in values.items()
        if v
    }


def role_counts(labels: np.ndarray, idx: np.ndarray) -> dict[str, int]:
    return {label.lower(): int(np.sum(labels[idx] == label)) for label in ["P", "N", "A", "U"]}


def main() -> None:
    started = time.time()
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    contract = load_contract(validate_hashes=True)
    X = np.load(FEATURE_PATH, mmap_mode="r")
    if X.shape[0] != len(df):
        raise AssertionError("feature rows do not match record-level dataset")

    labels = df[LABEL_COL].astype(str).to_numpy()
    p_mask = labels == "P"
    n_mask = labels == "N"
    group_cache = {split: group_values(df, groups, split) for split in SPLITS}

    calibration = {
        "meta": {
            "script": "calib_sensitivity_pipeline.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "manifest_family": "all_records_3way",
            "protocol": (
                "P_train plus all train-role non-P as the PU unlabeled pool; "
                "oracle calibration uses calibration-role P/N only; evaluation uses test-role P/N only"
            ),
            "method_semantics": {
                "raw": "uncalibrated nnPU score",
                "positive_only_temp_exploratory": "temperature selected using positive calibration records only; exploratory, not deployment calibration",
                "temp_oracle_pn": "temperature selected using oracle P/N calibration labels",
                "iso_oracle_pn": "isotonic calibration using oracle P/N calibration labels",
            },
            "prior_note": "raw nnPU prior is the disclosed in-sample logistic SCAR heuristic q/c; not cross-fitted",
            "seeds": SEEDS,
        },
        "splits": {},
    }
    predictions: list[dict] = []

    print("===== Calibration under all_records_3way =====")
    for split in SPLITS:
        aggregate: dict[str, list[float]] = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            roles = role_indices(df, manifest, "all_records_3way", split, seed)
            assert_group_disjoint(df, groups, roles, split)
            tr_all = roles["train"]
            cal_all = roles["calibration"]
            te_all = roles["test"]

            p_train = tr_all[p_mask[tr_all]]
            u_pool = tr_all[~p_mask[tr_all]]
            p_cal = cal_all[p_mask[cal_all]]
            n_cal = cal_all[n_mask[cal_all]]
            p_test = te_all[p_mask[te_all]]
            n_test = te_all[n_mask[te_all]]
            if min(len(p_train), len(u_pool), len(p_cal), len(n_cal), len(p_test), len(n_test)) == 0:
                raise AssertionError(f"empty required calibration subset: {split}/{seed}")

            train_set, cal_set, test_set = set(tr_all), set(cal_all), set(te_all)
            if train_set & cal_set or train_set & test_set or cal_set & test_set:
                raise AssertionError(f"record leakage in calibration roles: {split}/{seed}")
            if set(u_pool) & (cal_set | test_set):
                raise AssertionError(f"U pool leakage: {split}/{seed}")

            X_train = np.vstack([X[p_train], X[u_pool]])
            s_train = np.concatenate(
                [np.ones(len(p_train), dtype=int), np.zeros(len(u_pool), dtype=int)]
            )
            cal_idx = np.concatenate([p_cal, n_cal])
            y_cal = np.concatenate(
                [np.ones(len(p_cal), dtype=int), np.zeros(len(n_cal), dtype=int)]
            )
            test_idx = np.concatenate([p_test, n_test])
            y_test = np.concatenate(
                [np.ones(len(p_test), dtype=int), np.zeros(len(n_test), dtype=int)]
            )

            # Fit one deterministic nnPU model and obtain calibration and test scores together.
            both_idx = np.concatenate([cal_idx, test_idx])
            raw_both, prior_info = nnpu_fit_predict(
                X_train, s_train, X[both_idx], seed=seed, return_info=True
            )
            raw_cal = raw_both[: len(cal_idx)]
            raw_test = raw_both[len(cal_idx) :]
            logits_cal = logits_from_prob(raw_cal)
            logits_test = logits_from_prob(raw_test)

            positive_only_t = fit_temperature(logits_cal[: len(p_cal)], np.ones(len(p_cal), dtype=int))
            oracle_t = fit_temperature(logits_cal, y_cal)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(raw_cal, y_cal)

            method_predictions = {
                "raw": raw_test,
                "positive_only_temp_exploratory": expit(logits_test / positive_only_t),
                "temp_oracle_pn": expit(logits_test / oracle_t),
                "iso_oracle_pn": np.asarray(iso.predict(raw_test), dtype=float),
            }
            seed_entry = {
                "seed": int(seed),
                "n_train_all": int(len(tr_all)),
                "n_train": role_counts(labels, tr_all),
                "n_u_pool": int(len(u_pool)),
                "u_pool_composition": role_counts(labels, u_pool),
                "n_calibration_all": int(len(cal_all)),
                "n_calibration_p": int(len(p_cal)),
                "n_calibration_n": int(len(n_cal)),
                "n_test_all": int(len(te_all)),
                "n_test_p": int(len(p_test)),
                "n_test_n": int(len(n_test)),
                "nnpu_prior": prior_info,
                "positive_only_temperature": float(positive_only_t),
                "oracle_pn_temperature": float(oracle_t),
                "metrics": {},
            }
            for method, pred in method_predictions.items():
                metrics = classification_metrics(y_test, pred)
                if metrics is None:
                    raise AssertionError(f"single-class test set: {split}/{seed}")
                seed_entry["metrics"][method] = metrics
                for metric, value in metrics.items():
                    aggregate[f"{method}::{metric}"].append(value)
                for rid, gid, yt, yp in zip(
                    df.iloc[test_idx]["record_id"], group_cache[split][test_idx], y_test, pred
                ):
                    predictions.append(
                        {
                            "split_regime": split,
                            "seed": int(seed),
                            "method": method,
                            "record_id": int(rid),
                            "group_id": str(gid),
                            "y_true": int(yt),
                            "y_pred": float(yp),
                        }
                    )
            per_seed.append(seed_entry)
            print(
                f"  {split} seed={seed}: train P={len(p_train)} U={len(u_pool)}; "
                f"cal P/N={len(p_cal)}/{len(n_cal)}; test P/N={len(p_test)}/{len(n_test)}; "
                f"raw Brier={seed_entry['metrics']['raw']['brier']:.4f} -> "
                f"oracle-temp={seed_entry['metrics']['temp_oracle_pn']['brier']:.4f}"
            )
        split_entry = summarize(aggregate)
        split_entry["per_seed"] = per_seed
        calibration["splits"][split] = split_entry

    CALIB_RESULTS_PATH.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(CALIB_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"wrote {CALIB_RESULTS_PATH}")
    print(f"wrote {CALIB_PREDICTIONS_PATH} ({len(predictions):,} predictions)")

    print("===== nnPU prior sensitivity on the same train/test roles =====")
    sensitivity = {
        "meta": {
            "script": "calib_sensitivity_pipeline.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "manifest_family": "all_records_3way",
            "protocol": "same train-role P/non-P pool and test-role P/N as calibration experiment; calibration records are unused",
            "priors": PRIORS,
            "seeds": SEEDS,
        },
        "splits": {},
    }
    for split in SPLITS:
        aggregate: dict[str, list[float]] = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            roles = role_indices(df, manifest, "all_records_3way", split, seed)
            assert_group_disjoint(df, groups, roles, split)
            tr_all, cal_all, te_all = roles["train"], roles["calibration"], roles["test"]
            p_train = tr_all[p_mask[tr_all]]
            u_pool = tr_all[~p_mask[tr_all]]
            p_test = te_all[p_mask[te_all]]
            n_test = te_all[n_mask[te_all]]
            if set(tr_all) & set(cal_all) or set(tr_all) & set(te_all) or set(cal_all) & set(te_all):
                raise AssertionError(f"role leakage in prior sensitivity: {split}/{seed}")
            X_train = np.vstack([X[p_train], X[u_pool]])
            s_train = np.concatenate(
                [np.ones(len(p_train), dtype=int), np.zeros(len(u_pool), dtype=int)]
            )
            test_idx = np.concatenate([p_test, n_test])
            y_test = np.concatenate(
                [np.ones(len(p_test), dtype=int), np.zeros(len(n_test), dtype=int)]
            )
            seed_entry = {
                "seed": int(seed),
                "n_train_p": int(len(p_train)),
                "n_u_pool": int(len(u_pool)),
                "u_pool_composition": role_counts(labels, u_pool),
                "n_calibration_excluded": int(len(cal_all)),
                "n_test_p": int(len(p_test)),
                "n_test_n": int(len(n_test)),
                "metrics_by_prior": {},
            }
            for prior in PRIORS:
                pred = nnpu_fit_predict(
                    X_train, s_train, X[test_idx], seed=seed, prior=prior, epochs=40
                )
                metrics = classification_metrics(y_test, pred)
                if metrics is None:
                    raise AssertionError(f"single-class test set: {split}/{seed}")
                key = f"pi={prior:.2f}"
                seed_entry["metrics_by_prior"][key] = metrics
                for metric, value in metrics.items():
                    aggregate[f"{key}::{metric}"].append(value)
            per_seed.append(seed_entry)
            print(f"  {split} seed={seed}: {len(PRIORS)} priors complete")
        split_entry = summarize(aggregate)
        split_entry["per_seed"] = per_seed
        sensitivity["splits"][split] = split_entry

    PRIOR_RESULTS_PATH.write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {PRIOR_RESULTS_PATH}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
