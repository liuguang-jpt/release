# -*- coding: utf-8 -*-
"""Internal post-cutoff temporal stress test and frozen matched controls."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from benchmark_contract import (
    FEATURE_PATH,
    LABEL_COL,
    MANIFEST_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEEDS,
    load_contract,
    load_dataset,
    load_manifest,
    role_indices,
    sha256_file,
)
from pu_pipeline import classification_metrics, regression_metrics

MATCHED_CONTROL_PATH = PROCESSED_DIR / "temporal_matched_controls_v3.csv"
MATCHED_AUDIT_PATH = PROCESSED_DIR / "temporal_matched_controls_v3_audit.json"
RESULTS_PATH = PROCESSED_DIR / "external_validation_v3.json"
PREDICTIONS_PATH = PROCESSED_DIR / "external_validation_predictions_v3.csv"
REPORT_PATH = REPORTS_DIR / "TEMPORAL_STRESS_TEST_V3.md"


def summarize(values: dict[str, list[float]]) -> dict:
    return {
        key: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=0))}
        for key, v in values.items()
        if v
    }


def fit_classifier(X: np.ndarray, train_idx: np.ndarray, y_train: np.ndarray, test_idx: np.ndarray, seed: int) -> np.ndarray:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X[train_idx], y_train)
    return model.predict_proba(X[test_idx])[:, 1]


def fit_regressor(X: np.ndarray, train_idx: np.ndarray, y_train: np.ndarray, test_idx: np.ndarray, seed: int) -> np.ndarray:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=seed,
        verbosity=0,
    )
    model.fit(X[train_idx], y_train)
    return model.predict(X[test_idx])


def main() -> None:
    started = time.time()
    df = load_dataset()
    manifest = load_manifest(validate=True)
    contract = load_contract(validate_hashes=True)
    control = pd.read_csv(MATCHED_CONTROL_PATH, encoding="utf-8-sig")
    control_audit = json.loads(MATCHED_AUDIT_PATH.read_text(encoding="utf-8"))
    if control_audit["assignments_sha256"] != sha256_file(MATCHED_CONTROL_PATH):
        raise AssertionError("matched-control assignment hash mismatch")
    X = np.load(FEATURE_PATH, mmap_mode="r")
    if X.shape[0] != len(df):
        raise AssertionError("feature rows do not match record-level dataset")

    temporal_roles = role_indices(df, manifest, "temporal_postcutoff", "temporal", 0)
    tr_temporal, te_temporal = temporal_roles["train"], temporal_roles["test"]
    if set(tr_temporal) & set(te_temporal):
        raise AssertionError("temporal record overlap")
    train_doi = set(df.iloc[tr_temporal]["article_doi"].dropna().astype(str))
    test_doi = set(df.iloc[te_temporal]["article_doi"].dropna().astype(str))
    if train_doi & test_doi:
        raise AssertionError("temporal publication overlap")

    id_to_pos = pd.Series(np.arange(len(df)), index=df["record_id"])
    labels = df[LABEL_COL].astype(str).to_numpy()
    exact_mask = (df["dc50_obs_type"].astype(str).to_numpy() == "exact") & df["pdc50_value"].notna().to_numpy()
    pdc50 = pd.to_numeric(df["pdc50_value"], errors="coerce").to_numpy(float)
    prediction_rows: list[dict] = []

    result = {
        "meta": {
            "script": "external_validation.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "matched_control": str(MATCHED_CONTROL_PATH),
            "matched_control_sha256": sha256_file(MATCHED_CONTROL_PATH),
            "split_schema": contract["split_schema"],
            "temporal_protocol": "train DOI year <= 2022; test DOI year >= 2024; year 2023 is a buffer; roles consumed from temporal_postcutoff manifest",
            "scope": "internal post-cutoff stress test from the same source snapshot; not an independent external validation dataset",
            "matched_control_scope": (
                "controls exactly match train/test sizes and classification P/N counts and approximately match coarse E3, source_has_doi, evidence-grade, and target-frequency covariates; "
                "they do not isolate a pure causal time effect or fully match scaffold/publication novelty"
            ),
            "prohibited_claim": "results must not be described as proving that sample size has been excluded as the cause of temporal degradation",
            "seeds": SEEDS,
        },
        "temporal_counts": {
            "n_train_all": int(len(tr_temporal)),
            "n_test_all": int(len(te_temporal)),
            "n_buffer_or_missing_year": int(len(temporal_roles["excluded"])),
            "train_year_min": float(df.iloc[tr_temporal]["year_doi"].min()),
            "train_year_max": float(df.iloc[tr_temporal]["year_doi"].max()),
            "test_year_min": float(df.iloc[te_temporal]["year_doi"].min()),
            "test_year_max": float(df.iloc[te_temporal]["year_doi"].max()),
            "publication_overlap": 0,
        },
        "tasks": {},
        "matched_control_audit": control_audit["audits"],
    }

    # P/N classification.
    task_results = {}
    for experiment in ["temporal_postcutoff", "matched_random_control"]:
        aggregate: dict[str, list[float]] = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            if experiment == "temporal_postcutoff":
                train_idx = tr_temporal[np.isin(labels[tr_temporal], ["P", "N"])]
                test_idx = te_temporal[np.isin(labels[te_temporal], ["P", "N"])]
            else:
                sub = control[(control["task"] == "pn_clf") & (control["match_seed"] == seed)]
                train_idx = id_to_pos.loc[sub.loc[sub["role"] == "train", "record_id"]].to_numpy(dtype=int)
                test_idx = id_to_pos.loc[sub.loc[sub["role"] == "test", "record_id"]].to_numpy(dtype=int)
            if set(train_idx) & set(test_idx):
                raise AssertionError(f"classification train/test overlap: {experiment}/{seed}")
            y_train = (labels[train_idx] == "P").astype(int)
            y_test = (labels[test_idx] == "P").astype(int)
            pred = fit_classifier(X, train_idx, y_train, test_idx, seed)
            metrics = classification_metrics(y_test, pred)
            if metrics is None:
                raise AssertionError(f"single-class classification test: {experiment}/{seed}")
            for metric, value in metrics.items():
                aggregate[metric].append(value)
            per_seed.append(
                {
                    "seed": int(seed),
                    "n_train": int(len(train_idx)),
                    "n_train_p": int(y_train.sum()),
                    "n_train_n": int((1 - y_train).sum()),
                    "n_test": int(len(test_idx)),
                    "n_test_p": int(y_test.sum()),
                    "n_test_n": int((1 - y_test).sum()),
                    "metrics": metrics,
                }
            )
            for idx, yt, yp in zip(test_idx, y_test, pred):
                prediction_rows.append(
                    {
                        "task": "pn_clf",
                        "experiment": experiment,
                        "seed": int(seed),
                        "record_id": int(df.iloc[idx]["record_id"]),
                        "y_true": int(yt),
                        "y_pred": float(yp),
                    }
                )
        task_results[experiment] = {**summarize(aggregate), "per_seed": per_seed}
        print(f"P/N {experiment}: AUC={task_results[experiment]['roc_auc']['mean']:.4f}")
    task_results["descriptive_difference_matched_minus_temporal"] = {
        metric: float(task_results["matched_random_control"][metric]["mean"] - task_results["temporal_postcutoff"][metric]["mean"])
        for metric in ["roc_auc", "pr_auc", "mcc", "balanced_acc", "brier", "ece"]
    }
    result["tasks"]["pn_classification"] = task_results

    # Exact pDC50 regression.
    task_results = {}
    for experiment in ["temporal_postcutoff", "matched_random_control"]:
        aggregate: dict[str, list[float]] = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            if experiment == "temporal_postcutoff":
                train_idx = tr_temporal[exact_mask[tr_temporal]]
                test_idx = te_temporal[exact_mask[te_temporal]]
            else:
                sub = control[(control["task"] == "pdc50_reg") & (control["match_seed"] == seed)]
                train_idx = id_to_pos.loc[sub.loc[sub["role"] == "train", "record_id"]].to_numpy(dtype=int)
                test_idx = id_to_pos.loc[sub.loc[sub["role"] == "test", "record_id"]].to_numpy(dtype=int)
            if set(train_idx) & set(test_idx):
                raise AssertionError(f"regression train/test overlap: {experiment}/{seed}")
            pred = fit_regressor(X, train_idx, pdc50[train_idx], test_idx, seed)
            metrics = regression_metrics(pdc50[test_idx], pred)
            for metric, value in metrics.items():
                aggregate[metric].append(value)
            per_seed.append(
                {
                    "seed": int(seed),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_y_mean": float(np.mean(pdc50[train_idx])),
                    "test_y_mean": float(np.mean(pdc50[test_idx])),
                    "metrics": metrics,
                }
            )
            for idx, yt, yp in zip(test_idx, pdc50[test_idx], pred):
                prediction_rows.append(
                    {
                        "task": "pdc50_reg",
                        "experiment": experiment,
                        "seed": int(seed),
                        "record_id": int(df.iloc[idx]["record_id"]),
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    }
                )
        task_results[experiment] = {**summarize(aggregate), "per_seed": per_seed}
        print(f"pDC50 {experiment}: R2={task_results[experiment]['r2']['mean']:.4f}")
    task_results["descriptive_difference_matched_minus_temporal"] = {
        metric: float(task_results["matched_random_control"][metric]["mean"] - task_results["temporal_postcutoff"][metric]["mean"])
        for metric in ["mae", "rmse", "r2", "spearman"]
    }
    result["tasks"]["pdc50_regression"] = task_results

    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    clf = result["tasks"]["pn_classification"]
    reg = result["tasks"]["pdc50_regression"]
    lines = [
        "# Internal post-cutoff temporal stress test (v3)",
        "",
        f"> Generated: {result['meta']['generated']}  ",
        "> Temporal roles are read from the frozen `temporal_postcutoff` manifest.",
        "",
        "## Scope and claim boundary",
        "",
        "This is an **internal post-cutoff stress test** from the same source snapshot, not independent external validation. Training uses DOI year <=2022, testing uses DOI year >=2024, and 2023 is a buffer.",
        "",
        "The matched controls exactly reproduce train/test sample sizes and P/N class counts and approximately match coarse E3 ligase, DOI availability, evidence-grade, and target-frequency covariates. They do **not** fully match scaffold/publication novelty and therefore do not prove that a pure time effect—or the absence of a sample-size effect—has been identified.",
        "",
        "## P/N classification",
        "",
        "| Experiment | Train P/N | Test P/N | ROC-AUC | PR-AUC | MCC | Balanced accuracy | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment, label in [("temporal_postcutoff", "Post-cutoff"), ("matched_random_control", "Matched random control")]:
        x = clf[experiment]
        first = x["per_seed"][0]
        lines.append(
            f"| {label} | {first['n_train_p']}/{first['n_train_n']} | {first['n_test_p']}/{first['n_test_n']} | "
            f"{x['roc_auc']['mean']:.3f} | {x['pr_auc']['mean']:.3f} | {x['mcc']['mean']:.3f} | "
            f"{x['balanced_acc']['mean']:.3f} | {x['brier']['mean']:.3f} | {x['ece']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Exact pDC50 regression",
        "",
        "| Experiment | Train n | Test n | MAE | RMSE | R2 | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment, label in [("temporal_postcutoff", "Post-cutoff"), ("matched_random_control", "Matched random control")]:
        x = reg[experiment]
        first = x["per_seed"][0]
        lines.append(
            f"| {label} | {first['n_train']} | {first['n_test']} | {x['mae']['mean']:.3f} | "
            f"{x['rmse']['mean']:.3f} | {x['r2']['mean']:.3f} | {x['spearman']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Matched-control diagnostics",
        "",
        "The frozen control assignment and feature-level standardized mean differences are saved in:",
        f"- `{MATCHED_CONTROL_PATH}`",
        f"- `{MATCHED_AUDIT_PATH}`",
        "",
        "The appropriate manuscript conclusion is: performance differences are **consistent with temporal and dataset-composition shift**, but the present internal controls do not identify time as the sole cause.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {PREDICTIONS_PATH} ({len(prediction_rows):,} predictions)")
    print(f"wrote {REPORT_PATH}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
