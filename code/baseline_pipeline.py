# -*- coding: utf-8 -*-
"""Morgan/XGBoost baselines driven exclusively by split_manifest_v3.csv."""
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
    SEEDS,
    SPLITS,
    assert_group_disjoint,
    group_values,
    load_contract,
    load_dataset,
    load_groups,
    load_manifest,
    role_indices,
    task_eligibility,
)

RESULTS_JSON = PROCESSED_DIR / "baseline_results_v3.json"
PREDICTIONS_CSV = PROCESSED_DIR / "baseline_predictions_v3.csv"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.any():
            value += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(value)


def classification_metrics(y_true, y_prob):
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def regression_metrics(y_true, y_pred):
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "spearman": float(rho),
    }


def context_features(df: pd.DataFrame) -> np.ndarray:
    blocks = []
    for col in ["target", "e3_ligase", "source_has_doi"]:
        blocks.append(pd.get_dummies(df[col].fillna("UNKNOWN"), prefix=col, dtype=np.float32).to_numpy())
    return np.hstack(blocks).astype(np.float32)


def main():
    t0 = time.time()
    contract = load_contract(validate_hashes=True)
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    X_morgan = np.load(FEATURE_PATH).astype(np.float32)
    if X_morgan.shape[0] != len(df):
        raise AssertionError("Morgan feature cache is not aligned to the record-level dataset")
    X_context = context_features(df)
    features = {
        "F1_morgan": X_morgan,
        "F2_morgan_ctx": np.hstack([X_morgan, X_context]).astype(np.float32),
    }

    lab = df[LABEL_COL].to_numpy()
    task_defs = {
        "T1_pdc50_reg": {
            "contract_task": "baseline_pdc50",
            "kind": "reg",
            "y": pd.to_numeric(df["pdc50_value"], errors="coerce").to_numpy(),
            "desc": "exact pDC50 regression",
        },
        "T2_dmax_reg": {
            "contract_task": "baseline_dmax",
            "kind": "reg",
            "y": pd.to_numeric(df["dmax_value"], errors="coerce").to_numpy(),
            "desc": "exact Dmax regression",
        },
        "T3_pn_clf": {
            "contract_task": "baseline_pn",
            "kind": "clf",
            "y": (lab == "P").astype(int),
            "desc": "P/N classification",
        },
        "T4_diag_uan_clf": {
            "contract_task": "baseline_uan",
            "kind": "clf",
            "y": (lab == "P").astype(int),
            "desc": "diagnostic P versus N/U classification",
        },
    }

    import xgboost as xgb

    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    results = {
        "meta": {
            "script": "baseline_pipeline.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "seeds": SEEDS,
            "features": {"F1_morgan": "Morgan radius=2, 2048 bits", "F2_morgan_ctx": "Morgan + POI/E3/source"},
            "label_col": LABEL_COL,
        },
        "tasks": {},
        "splits": {},
    }
    prediction_rows = []

    for task_id, spec in task_defs.items():
        eligible = task_eligibility(df, spec["contract_task"])
        results["tasks"][task_id] = {
            "desc": spec["desc"],
            "n_records": int(eligible.sum()),
            "pos_rate": float(spec["y"][eligible].mean()) if spec["kind"] == "clf" else None,
            "eligibility": contract["tasks"][spec["contract_task"]]["eligibility"],
        }
        results["splits"][task_id] = {}
        for split in SPLITS:
            agg = defaultdict(list)
            per_seed = []
            for seed in SEEDS:
                roles = role_indices(df, manifest, "all_records_2way", split, seed)
                assert_group_disjoint(df, groups, roles, split)
                tr = roles["train"][eligible[roles["train"]]]
                te = roles["test"][eligible[roles["test"]]]
                if len(tr) == 0 or len(te) == 0:
                    continue
                y_tr = spec["y"][tr]
                y_te = spec["y"][te]
                if spec["kind"] == "clf" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2):
                    continue
                seed_entry = {"seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)), "metrics": {}}
                test_groups = group_values(df, groups, split)[te]
                for feature_name, X in features.items():
                    if spec["kind"] == "reg":
                        model = xgb.XGBRegressor(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=n_jobs,
                            random_state=seed,
                            verbosity=0,
                        )
                        model.fit(X[tr], y_tr)
                        pred = model.predict(X[te])
                        metrics = regression_metrics(y_te, pred)
                    else:
                        model = xgb.XGBClassifier(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=n_jobs,
                            random_state=seed,
                            eval_metric="logloss",
                            verbosity=0,
                        )
                        model.fit(X[tr], y_tr)
                        pred = model.predict_proba(X[te])[:, 1]
                        metrics = classification_metrics(y_te, pred)
                        if metrics is None:
                            continue
                    seed_entry["metrics"][feature_name] = metrics
                    for metric, value in metrics.items():
                        agg[f"{feature_name}::{metric}"].append(value)
                    for rid, yt, yp, gid in zip(df.iloc[te]["record_id"], y_te, pred, test_groups):
                        prediction_rows.append(
                            {
                                "task": task_id,
                                "split_regime": split,
                                "seed": seed,
                                "feature": feature_name,
                                "record_id": int(rid),
                                "group_id": str(gid),
                                "y_true": float(yt),
                                "y_pred": float(yp),
                            }
                        )
                per_seed.append(seed_entry)
            entry = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=0))} for k, v in agg.items()}
            entry["per_seed"] = per_seed
            entry["manifest_family"] = "all_records_2way"
            results["splits"][task_id][split] = entry
            key = "F1_morgan::roc_auc" if spec["kind"] == "clf" else "F1_morgan::r2"
            if key in entry:
                print(f"{task_id}/{split}: {key.split('::')[1]}={entry[key]['mean']:.4f}")

    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {RESULTS_JSON}")
    print(f"wrote {PREDICTIONS_CSV} ({len(prediction_rows):,} predictions)")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
