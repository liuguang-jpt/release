# -*- coding: utf-8 -*-
"""Summarize frozen-model external validation metrics and exploratory T4 cohort composition."""
from __future__ import annotations
import hashlib, json, math, warnings
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("PROTAC_EXTERNAL_ROOT", str(REPO_ROOT / "external_data")))
PROC = BASE / "processed"
RESULTS = BASE / "results"
COHORT = PROC / "external_validation_cohort.csv"
PRED = RESULTS / "external_predictions.csv"
OUT_T3 = RESULTS / "external_T3_metrics_by_split_seed.csv"
OUT_T1 = RESULTS / "external_T1_metrics_by_split_seed.csv"
OUT_T2 = RESULTS / "external_T2_metrics_by_split_seed.csv"
OUT_SUMMARY = RESULTS / "external_validation_summary.json"


def finite(v):
    return None if v is None or not np.isfinite(v) else float(v)


def ece_binary(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    if total == 0:
        return np.nan
    value = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            value += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def classification_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n_records": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1-y).sum()),
        "threshold": 0.5,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "balanced_accuracy": np.nan,
        "mcc": np.nan,
        "brier": np.nan,
        "ece_10bin": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "accuracy": np.nan,
    }
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = finite(roc_auc_score(y, p))
        out["pr_auc"] = finite(average_precision_score(y, p))
    out["balanced_accuracy"] = finite(balanced_accuracy_score(y, pred))
    out["mcc"] = finite(matthews_corrcoef(y, pred))
    out["brier"] = finite(brier_score_loss(y, p))
    out["ece_10bin"] = finite(ece_binary(y, p, 10))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = finite(tp/(tp+fn)) if (tp+fn) else np.nan
    out["specificity"] = finite(tn/(tn+fp)) if (tn+fp) else np.nan
    out["accuracy"] = finite((tp+tn)/len(y)) if len(y) else np.nan
    return out


def regression_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n_records": int(len(y)), "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan}
    out["mae"] = finite(mean_absolute_error(y, p))
    out["rmse"] = finite(math.sqrt(mean_squared_error(y, p)))
    if len(y) > 1 and np.nanstd(y) > 0:
        out["r2"] = finite(r2_score(y, p))
    if len(y) > 1 and np.nanstd(y) > 0 and np.nanstd(p) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["spearman"] = finite(spearmanr(y, p).statistic)
    return out


def summarize_task(df, task, kind):
    rows = []
    for (split, seed), g in df[df.task.eq(task)].groupby(["split_regime", "seed"], sort=True):
        y = g.y_true.to_numpy(float)
        p = g.y_pred.to_numpy(float)
        m = classification_metrics(y, p) if kind == "clf" else regression_metrics(y, p)
        rows.append({"task": task, "split_regime": split, "seed": int(seed), **m})
    result = pd.DataFrame(rows)
    metrics = (["roc_auc", "pr_auc", "balanced_accuracy", "mcc", "brier", "ece_10bin", "sensitivity", "specificity", "accuracy"]
               if kind == "clf" else ["mae", "rmse", "r2", "spearman"])
    agg = []
    for metric in metrics:
        vals = pd.to_numeric(result[metric], errors="coerce").dropna()
        agg.append({"task": task, "summary": "mean_sd_across_12_models", "metric": metric,
                    "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "n_models": int(vals.size)})
    result.to_csv(RESULTS / f"external_{task.split('_')[0]}_metrics_by_split_seed.csv", index=False, encoding="utf-8-sig")
    return result, agg


def main():
    pred = pd.read_csv(PRED, low_memory=False)
    cohort = pd.read_csv(COHORT, low_memory=False)
    t3, t3agg = summarize_task(pred, "T3_pn_clf", "clf")
    t1, t1agg = summarize_task(pred, "T1_pdc50_reg", "reg")
    t2, t2agg = summarize_task(pred, "T2_dmax_reg", "reg")
    # Rename outputs to the stable filenames promised in the implementation plan.
    t3.to_csv(OUT_T3, index=False, encoding="utf-8-sig")
    t1.to_csv(OUT_T1, index=False, encoding="utf-8-sig")
    t2.to_csv(OUT_T2, index=False, encoding="utf-8-sig")

    # T4 is intentionally exploratory in scheme B: composition and label availability only.
    t4 = cohort[cohort["t4_eligible"].astype(str).str.lower().eq("true")].copy()
    pnu_counts = t4["activity_evidence_external_v1"].value_counts().to_dict()
    t4_by_target = t4.groupby("target_uniprot")["activity_evidence_external_v1"].value_counts().unstack(fill_value=0)
    t4_by_target_path = RESULTS / "external_T4_exploratory_label_distribution_by_target.csv"
    t4_by_target.to_csv(t4_by_target_path, encoding="utf-8-sig")

    summary = {
        "protocol": {
            "external_cohort": str(COHORT),
            "predictions": str(PRED),
            "n_models": int(pred[["task", "split_regime", "seed"]].drop_duplicates().shape[0]),
            "split_regimes": sorted(pred.split_regime.unique().tolist()),
            "seeds": sorted(pred.seed.unique().astype(int).tolist()),
            "classification_threshold": 0.5,
            "ece_bins": 10,
        },
        "task_metrics": {
            "T3_pn_clf": t3agg,
            "T1_pdc50_reg": t1agg,
            "T2_dmax_reg": t2agg,
        },
        "T4_exploratory": {
            "n_records": int(len(t4)),
            "label_counts": {str(k): int(v) for k, v in pnu_counts.items()},
            "label_proportions": {str(k): float(v/len(t4)) for k, v in pnu_counts.items()},
            "not_used_for_primary_inference": True,
            "by_target_file": str(t4_by_target_path),
        },
        "input_hashes": {
            "cohort_sha256": hashlib.sha256(COHORT.read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256(PRED.read_bytes()).hexdigest(),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"T3": t3.to_dict(orient="records"), "T1": t1.to_dict(orient="records"), "T2": t2.to_dict(orient="records"), "T4": summary["T4_exploratory"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
