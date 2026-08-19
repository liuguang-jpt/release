# -*- coding: utf-8 -*-
"""Manifest-driven shortcut and label-permutation controls for P/N prediction.

All train/test roles come from split_manifest_v3.csv. Encoders are fitted only
on the manifest training role. Label permutations are restricted to training
records and never alter frozen record roles.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from baseline_pipeline import classification_metrics
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

RESULTS_JSON = PROCESSED_DIR / "shortcut_controls_v3.json"
PREDICTIONS_CSV = PROCESSED_DIR / "shortcut_predictions_v3.csv"


def encode_categories(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    x_train = encoder.fit_transform(train[columns].fillna("UNKNOWN").astype(str))
    x_test = encoder.transform(test[columns].fillna("UNKNOWN").astype(str))
    return x_train, x_test


def metadata_features(df: pd.DataFrame, groups: pd.DataFrame, tr: np.ndarray, te: np.ndarray):
    base = df[["target", "e3_ligase", "source_has_doi", "year_doi"]].copy()
    base["pub_group"] = groups["pub_group"].to_numpy()
    feature_pairs = {}
    for name, columns in {
        "target_only": ["target"],
        "e3_only": ["e3_ligase"],
        "source_only": ["source_has_doi"],
        "publication_only": ["pub_group"],
    }.items():
        feature_pairs[name] = encode_categories(base.iloc[tr], base.iloc[te], columns)

    cat_train, cat_test = encode_categories(
        base.iloc[tr], base.iloc[te], ["target", "e3_ligase", "source_has_doi"]
    )
    train_year = pd.to_numeric(base.iloc[tr]["year_doi"], errors="coerce")
    test_year = pd.to_numeric(base.iloc[te]["year_doi"], errors="coerce")
    median_year = float(train_year.median()) if train_year.notna().any() else 0.0
    train_numeric = np.column_stack(
        [train_year.fillna(median_year).to_numpy(np.float32), train_year.isna().to_numpy(np.float32)]
    )
    test_numeric = np.column_stack(
        [test_year.fillna(median_year).to_numpy(np.float32), test_year.isna().to_numpy(np.float32)]
    )
    feature_pairs["metadata_only"] = (
        sparse.hstack([cat_train, sparse.csr_matrix(train_numeric)], format="csr"),
        sparse.hstack([cat_test, sparse.csr_matrix(test_numeric)], format="csr"),
    )
    return feature_pairs


def permute_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    permuted = y.copy()
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) > 1:
            permuted[positions] = rng.permutation(permuted[positions])
    return permuted


def fit_predict(x_train, y_train, x_test, seed):
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
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    t0 = time.time()
    contract = load_contract(validate_hashes=True)
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    x_morgan = np.load(FEATURE_PATH).astype(np.float32)
    if x_morgan.shape != (len(df), 2048):
        raise AssertionError("Morgan features are not aligned to the record-level dataset")

    labels = df[LABEL_COL].to_numpy()
    y_all = (labels == "P").astype(int)
    eligible = task_eligibility(df, "shortcut_pn")
    results = {
        "meta": {
            "script": "shortcut_controls.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "manifest_family": "all_records_2way",
            "seeds": SEEDS,
            "label": LABEL_COL,
            "encoder_policy": "fit on manifest train role only; unseen test categories ignored",
            "permutation_policy": "labels permuted only inside the manifest train role; roles remain frozen",
            "primary_permutation_control": "structure_perm_full",
            "within_group_permutation_note": "descriptive only; singleton-heavy groups may leave many labels unchanged",
        },
        "splits": {},
    }
    prediction_rows: list[dict] = []

    for split in SPLITS:
        aggregate = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            roles = role_indices(df, manifest, "all_records_2way", split, seed)
            assert_group_disjoint(df, groups, roles, split)
            tr = roles["train"][eligible[roles["train"]]]
            te = roles["test"][eligible[roles["test"]]]
            y_train = y_all[tr]
            y_test = y_all[te]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                raise AssertionError(f"single-class P/N role in {split}/{seed}")

            train_groups = group_values(df, groups, split)[tr]
            test_groups = group_values(df, groups, split)[te]
            feature_pairs = {"structure_only": (x_morgan[tr], x_morgan[te])}
            feature_pairs.update(metadata_features(df, groups, tr, te))
            rng = np.random.RandomState(seed + 701)
            permuted = {
                "structure_perm_full": rng.permutation(y_train),
                "structure_perm_within_group": permute_within_groups(y_train, train_groups, rng),
            }

            seed_entry = {
                "seed": seed,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_train_p": int(y_train.sum()),
                "n_test_p": int(y_test.sum()),
                "metrics": {},
            }
            for method, (x_train, x_test) in feature_pairs.items():
                probability = fit_predict(x_train, y_train, x_test, seed)
                metrics = classification_metrics(y_test, probability)
                if metrics is None:
                    raise AssertionError(f"metrics unavailable for {split}/{seed}/{method}")
                seed_entry["metrics"][method] = metrics
                for metric, value in metrics.items():
                    aggregate[f"{method}::{metric}"].append(value)
                for rid, truth, pred, gid in zip(df.iloc[te]["record_id"], y_test, probability, test_groups):
                    prediction_rows.append(
                        {
                            "split_regime": split,
                            "seed": seed,
                            "method": method,
                            "record_id": int(rid),
                            "group_id": str(gid),
                            "y_true": int(truth),
                            "y_prob": float(pred),
                        }
                    )

            for method, permuted_y in permuted.items():
                probability = fit_predict(x_morgan[tr], permuted_y, x_morgan[te], seed)
                metrics = classification_metrics(y_test, probability)
                if metrics is None:
                    raise AssertionError(f"permutation metrics unavailable for {split}/{seed}/{method}")
                seed_entry["metrics"][method] = metrics
                seed_entry.setdefault("permutation", {})[method] = {
                    "changed_fraction": float(np.mean(permuted_y != y_train))
                }
                for metric, value in metrics.items():
                    aggregate[f"{method}::{metric}"].append(value)
                for rid, truth, pred, gid in zip(df.iloc[te]["record_id"], y_test, probability, test_groups):
                    prediction_rows.append(
                        {
                            "split_regime": split,
                            "seed": seed,
                            "method": method,
                            "record_id": int(rid),
                            "group_id": str(gid),
                            "y_true": int(truth),
                            "y_prob": float(pred),
                        }
                    )
            per_seed.append(seed_entry)
            print(
                f"{split}/{seed}: structure={seed_entry['metrics']['structure_only']['roc_auc']:.3f}; "
                f"metadata={seed_entry['metrics']['metadata_only']['roc_auc']:.3f}; "
                f"perm={seed_entry['metrics']['structure_perm_full']['roc_auc']:.3f}"
            )

        summary = {
            key: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=0))}
            for key, values in aggregate.items()
        }
        results["splits"][split] = {"summary": summary, "per_seed": per_seed}

    predictions = pd.DataFrame(prediction_rows)
    expected_methods = {
        "structure_only",
        "target_only",
        "e3_only",
        "source_only",
        "publication_only",
        "metadata_only",
        "structure_perm_full",
        "structure_perm_within_group",
    }
    if set(predictions["method"]) != expected_methods:
        raise AssertionError("shortcut prediction methods are incomplete")
    if predictions.duplicated(["split_regime", "seed", "method", "record_id"]).any():
        raise AssertionError("duplicate shortcut predictions")

    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {RESULTS_JSON}")
    print(f"wrote {PREDICTIONS_CSV} ({len(predictions):,} predictions)")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
