# -*- coding: utf-8 -*-
"""Train frozen internal split models and evaluate them once on the external cohort."""
from __future__ import annotations
import os
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get("PROTAC_ROOT", str(Path(__file__).resolve().parents[1])))
BASE = Path(os.environ.get("PROTAC_EXTERNAL_ROOT", str(PROJECT / "external_data")))
PROC_EXT = BASE / "processed"
RESULTS = BASE / "results"
MODELS = RESULTS / "models"
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
from benchmark_contract import load_contract, load_dataset, load_manifest, role_indices, SEEDS, SPLITS, FEATURE_PATH

import xgboost as xgb

INTERNAL_FEATURE = FEATURE_PATH
COHORT_PATH = PROC_EXT / "external_validation_cohort.csv"
EXTERNAL_FEATURE = PROC_EXT / "external_morgan_fp_2048.npy"
PREDICTIONS = RESULTS / "external_predictions.csv"
MODEL_MANIFEST = RESULTS / "external_model_manifest.csv"

PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model(kind: str, seed: int):
    common = dict(**PARAMS, n_jobs=max(1, (os.cpu_count() or 2) - 1), random_state=seed, verbosity=0)
    if kind == "reg":
        return xgb.XGBRegressor(**common)
    return xgb.XGBClassifier(**common, eval_metric="logloss")


def bool_series(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().map({"true": True, "false": False}).fillna(False).to_numpy(bool)


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    internal = load_dataset()
    manifest = load_manifest(validate=True)
    contract = load_contract(validate_hashes=True)
    X_internal = np.load(INTERNAL_FEATURE, mmap_mode="r")
    if X_internal.shape[0] != len(internal) or X_internal.shape[1] != 2048:
        raise AssertionError("internal feature cache shape mismatch")
    external = pd.read_csv(COHORT_PATH, encoding="utf-8-sig", low_memory=False)
    X_external = np.load(EXTERNAL_FEATURE, mmap_mode="r")
    if X_external.shape != (len(external), 2048):
        raise AssertionError("external feature cache shape mismatch")

    task_specs = {
        "T1_pdc50_reg": {
            "kind": "reg",
            "internal_mask": (internal["dc50_obs_type"].astype(str).eq("exact") & pd.to_numeric(internal["pdc50_value"], errors="coerce").notna()).to_numpy(),
            "external_mask": bool_series(external["t1_eligible"]),
            "internal_y": pd.to_numeric(internal["pdc50_value"], errors="coerce").to_numpy(float),
            "external_y": pd.to_numeric(external["pdc50_value"], errors="coerce").to_numpy(float),
        },
        "T2_dmax_reg": {
            "kind": "reg",
            "internal_mask": (internal["dmax_obs_type"].astype(str).eq("exact") & pd.to_numeric(internal["dmax_value"], errors="coerce").notna()).to_numpy(),
            "external_mask": bool_series(external["t2_eligible"]),
            "internal_y": pd.to_numeric(internal["dmax_value"], errors="coerce").to_numpy(float),
            "external_y": pd.to_numeric(external["dmax_value"], errors="coerce").to_numpy(float),
        },
        "T3_pn_clf": {
            "kind": "clf",
            "internal_mask": internal["activity_evidence_v2"].isin(["P", "N"]).to_numpy(),
            "external_mask": external["activity_evidence_external_v1"].isin(["P", "N"]).to_numpy(),
            "internal_y": internal["activity_evidence_v2"].map({"N": 0, "P": 1}).to_numpy(float),
            "external_y": external["activity_evidence_external_v1"].map({"N": 0, "P": 1}).to_numpy(float),
        },
    }
    prediction_rows = []
    manifest_rows = []
    for split in SPLITS:
        for seed in SEEDS:
            roles = role_indices(internal, manifest, "all_records_3way", split, int(seed))
            train_idx_all = roles["train"]
            for task, spec in task_specs.items():
                train_idx = train_idx_all[spec["internal_mask"][train_idx_all]]
                ext_idx = np.flatnonzero(spec["external_mask"])
                y_train = spec["internal_y"][train_idx]
                y_ext = spec["external_y"][ext_idx]
                if len(train_idx) == 0 or len(ext_idx) == 0:
                    raise AssertionError(f"empty train/external set for {task}/{split}/{seed}")
                if spec["kind"] == "clf" and len(np.unique(y_train)) < 2:
                    raise AssertionError(f"single-class train set for {task}/{split}/{seed}")
                model = make_model(spec["kind"], int(seed))
                model.fit(X_internal[train_idx], y_train)
                if spec["kind"] == "clf":
                    pred = model.predict_proba(X_external[ext_idx])[:, 1]
                else:
                    pred = model.predict(X_external[ext_idx])
                model_path = MODELS / f"{task}__{split}__seed{seed}.json"
                model.save_model(str(model_path))
                for pos, y_true, y_pred in zip(ext_idx, y_ext, pred):
                    prediction_rows.append({
                        "task": task,
                        "split_regime": split,
                        "seed": int(seed),
                        "external_record_id": int(external.iloc[pos]["external_record_id"]),
                        "external_row": int(pos),
                        "y_true": float(y_true),
                        "y_pred": float(y_pred),
                        "inchikey": str(external.iloc[pos]["inchikey"]),
                        "source_patent_id": str(external.iloc[pos]["source_patent_id"]),
                        "target_uniprot": str(external.iloc[pos]["target_uniprot"]),
                        "e3_normalized": str(external.iloc[pos]["e3_normalized"]),
                    })
                manifest_rows.append({
                    "task": task,
                    "kind": spec["kind"],
                    "split_regime": split,
                    "seed": int(seed),
                    "split_family": "all_records_3way",
                    "training_role": "train",
                    "n_train": int(len(train_idx)),
                    "n_external": int(len(ext_idx)),
                    "n_train_positive": int((y_train == 1).sum()) if spec["kind"] == "clf" else None,
                    "n_train_negative": int((y_train == 0).sum()) if spec["kind"] == "clf" else None,
                    "model_path": str(model_path),
                    "model_sha256": sha256(model_path),
                    "internal_dataset_sha256": contract["artifacts"]["dataset_sha256"],
                    "internal_feature_sha256": contract["artifacts"]["feature_sha256"],
                    "manifest_sha256": contract["artifacts"]["manifest_sha256"],
                    "external_cohort_sha256": sha256(COHORT_PATH),
                    "external_feature_sha256": sha256(EXTERNAL_FEATURE),
                    "threshold_primary": 0.5 if spec["kind"] == "clf" else None,
                    **PARAMS,
                })
                print(f"completed {task}/{split}/{seed}: train={len(train_idx):,}, external={len(ext_idx):,}")
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(PREDICTIONS, index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(MODEL_MANIFEST, index=False, encoding="utf-8-sig")
    print(f"wrote {PREDICTIONS} ({len(pred_df):,} rows)")
    print(f"wrote {MODEL_MANIFEST} ({len(manifest_rows):,} model rows)")
    print(f"elapsed_seconds={time.time()-started:.1f}")


if __name__ == "__main__":
    main()
