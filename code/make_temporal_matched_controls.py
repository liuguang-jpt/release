# -*- coding: utf-8 -*-
"""Freeze sample-size/class/covariate-matched controls for temporal stress tests."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from benchmark_contract import PROCESSED_DIR, SEEDS, load_dataset, load_manifest, role_indices, sha256_file

ASSIGNMENT_PATH = PROCESSED_DIR / "temporal_matched_controls_v3.csv"
AUDIT_PATH = PROCESSED_DIR / "temporal_matched_controls_v3_audit.json"


def build_covariates(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    target = df["target"].fillna("__MISSING__").astype(str)
    target_frequency = target.map(target.value_counts()).to_numpy(dtype=float)
    target_bin = pd.cut(
        target_frequency,
        bins=[-np.inf, 5, 20, 100, np.inf],
        labels=["target_freq_1_5", "target_freq_6_20", "target_freq_21_100", "target_freq_gt100"],
    ).astype(str)
    cov = pd.DataFrame(
        {
            "source_with_doi": df["source_has_doi"].astype(str).eq("with_doi").astype(float),
            "evidence_grade": df["evidence_grade"].fillna("__MISSING__").astype(str),
            "e3_ligase": df["e3_ligase"].fillna("__MISSING__").astype(str),
            "target_frequency_bin": target_bin,
        }
    )
    categorical = pd.get_dummies(
        cov[["evidence_grade", "e3_ligase", "target_frequency_bin"]],
        prefix=["grade", "e3", "targetfreq"],
        dtype=float,
    )
    matrix = pd.concat([cov[["source_with_doi"]], categorical], axis=1)
    return matrix.to_numpy(dtype=np.float32), matrix.columns.tolist()


def greedy_match(
    reference_idx: np.ndarray,
    candidate_idx: np.ndarray,
    covariates: np.ndarray,
    rng: np.random.Generator,
    nearest_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_idx = np.asarray(reference_idx, dtype=int)
    available = list(np.asarray(candidate_idx, dtype=int))
    if len(available) < len(reference_idx):
        raise ValueError(f"insufficient candidates: need {len(reference_idx)}, have {len(available)}")
    matched = []
    matched_ref = []
    distances = []
    for ref in rng.permutation(reference_idx):
        candidates = np.asarray(available, dtype=int)
        distance = np.sum((covariates[candidates] - covariates[ref]) ** 2, axis=1)
        k = min(nearest_k, len(candidates))
        nearest_positions = np.argpartition(distance, k - 1)[:k]
        selected_position = int(rng.choice(nearest_positions))
        selected = int(candidates[selected_position])
        matched.append(selected)
        matched_ref.append(int(ref))
        distances.append(float(np.sqrt(distance[selected_position])))
        available.remove(selected)
    return np.asarray(matched), np.asarray(matched_ref), np.asarray(distances)


def balance_diagnostics(
    reference_idx: np.ndarray,
    matched_idx: np.ndarray,
    covariates: np.ndarray,
    names: list[str],
) -> dict:
    ref_mean = covariates[reference_idx].mean(axis=0)
    mat_mean = covariates[matched_idx].mean(axis=0)
    ref_var = covariates[reference_idx].var(axis=0)
    mat_var = covariates[matched_idx].var(axis=0)
    pooled = np.sqrt((ref_var + mat_var) / 2)
    smd = np.divide(ref_mean - mat_mean, pooled, out=np.zeros_like(ref_mean), where=pooled > 0)
    details = {
        name: {
            "reference_mean": float(ref_mean[i]),
            "matched_mean": float(mat_mean[i]),
            "standardized_mean_difference": float(smd[i]),
        }
        for i, name in enumerate(names)
    }
    return {
        "max_abs_standardized_mean_difference": float(np.max(np.abs(smd))),
        "mean_abs_standardized_mean_difference": float(np.mean(np.abs(smd))),
        "features": details,
    }


def main() -> None:
    df = load_dataset()
    manifest = load_manifest(validate=True)
    roles = role_indices(df, manifest, "temporal_postcutoff", "temporal", 0)
    temporal_reference = np.concatenate([roles["train"], roles["test"]])
    temporal_reference_set = set(temporal_reference.tolist())
    covariates, covariate_names = build_covariates(df)
    labels = df["activity_evidence_v2"].astype(str).to_numpy()
    exact_mask = (df["dc50_obs_type"].astype(str).to_numpy() == "exact") & df["pdc50_value"].notna().to_numpy()

    rows: list[dict] = []
    audits: dict[str, dict] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # Classification: match train/test sizes and P/N counts exactly, with no reused control records.
        available = set(np.flatnonzero(np.isin(labels, ["P", "N"])).tolist()) - temporal_reference_set
        task_selected: dict[str, list[int]] = {"train": [], "test": []}
        task_reference: dict[str, list[int]] = {"train": [], "test": []}
        for role in ["test", "train"]:
            role_idx = roles[role]
            for label in ["N", "P"]:
                reference = role_idx[labels[role_idx] == label]
                candidates = np.asarray(sorted(i for i in available if labels[i] == label), dtype=int)
                matched, matched_ref, distance = greedy_match(reference, candidates, covariates, rng)
                for idx, ref, dist in zip(matched, matched_ref, distance):
                    rows.append(
                        {
                            "task": "pn_clf",
                            "match_seed": int(seed),
                            "role": role,
                            "record_id": int(df.iloc[idx]["record_id"]),
                            "matched_to_record_id": int(df.iloc[ref]["record_id"]),
                            "class_label": label,
                            "covariate_distance": float(dist),
                        }
                    )
                available -= set(matched.tolist())
                task_selected[role].extend(matched.tolist())
                task_reference[role].extend(matched_ref.tolist())
        audits[f"pn_clf::{seed}"] = {
            "n_train": len(task_selected["train"]),
            "n_test": len(task_selected["test"]),
            "train_class_counts": {x: int(np.sum(labels[task_selected["train"]] == x)) for x in ["P", "N"]},
            "test_class_counts": {x: int(np.sum(labels[task_selected["test"]] == x)) for x in ["P", "N"]},
            "train_balance": balance_diagnostics(np.asarray(task_reference["train"]), np.asarray(task_selected["train"]), covariates, covariate_names),
            "test_balance": balance_diagnostics(np.asarray(task_reference["test"]), np.asarray(task_selected["test"]), covariates, covariate_names),
            "record_overlap": len(set(task_selected["train"]) & set(task_selected["test"])),
            "reference_overlap": len((set(task_selected["train"]) | set(task_selected["test"])) & temporal_reference_set),
        }

        # Regression: match train/test sizes and coarse covariates; outcome values are not used for matching.
        available = set(np.flatnonzero(exact_mask).tolist()) - temporal_reference_set
        task_selected = {"train": [], "test": []}
        task_reference = {"train": [], "test": []}
        for role in ["test", "train"]:
            reference = roles[role][exact_mask[roles[role]]]
            candidates = np.asarray(sorted(available), dtype=int)
            matched, matched_ref, distance = greedy_match(reference, candidates, covariates, rng)
            for idx, ref, dist in zip(matched, matched_ref, distance):
                rows.append(
                    {
                        "task": "pdc50_reg",
                        "match_seed": int(seed),
                        "role": role,
                        "record_id": int(df.iloc[idx]["record_id"]),
                        "matched_to_record_id": int(df.iloc[ref]["record_id"]),
                        "class_label": "",
                        "covariate_distance": float(dist),
                    }
                )
            available -= set(matched.tolist())
            task_selected[role].extend(matched.tolist())
            task_reference[role].extend(matched_ref.tolist())
        audits[f"pdc50_reg::{seed}"] = {
            "n_train": len(task_selected["train"]),
            "n_test": len(task_selected["test"]),
            "train_balance": balance_diagnostics(np.asarray(task_reference["train"]), np.asarray(task_selected["train"]), covariates, covariate_names),
            "test_balance": balance_diagnostics(np.asarray(task_reference["test"]), np.asarray(task_selected["test"]), covariates, covariate_names),
            "record_overlap": len(set(task_selected["train"]) & set(task_selected["test"])),
            "reference_overlap": len((set(task_selected["train"]) | set(task_selected["test"])) & temporal_reference_set),
        }

    assignment = pd.DataFrame(rows)
    if assignment.duplicated(["task", "match_seed", "role", "record_id"]).any():
        raise AssertionError("duplicate matched-control assignment")
    assignment.to_csv(ASSIGNMENT_PATH, index=False, encoding="utf-8-sig")
    payload = {
        "meta": {
            "script": "make_temporal_matched_controls.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS,
            "reference_manifest_family": "temporal_postcutoff",
            "matching": "greedy without replacement; random choice among five nearest coarse-covariate candidates",
            "covariates": covariate_names,
            "constraints": "classification train/test sizes and P/N counts exactly match temporal roles; regression train/test sizes exactly match; outcome values are not matched",
            "scope": "diagnostic matched controls; they do not isolate a pure temporal causal effect",
        },
        "assignments_sha256": sha256_file(ASSIGNMENT_PATH),
        "audits": audits,
    }
    AUDIT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, audit in audits.items():
        if audit["record_overlap"] or audit["reference_overlap"]:
            raise AssertionError(f"matched-control leakage: {key}")
        print(
            f"{key}: train={audit['n_train']} test={audit['n_test']} "
            f"max|SMD| train/test={audit['train_balance']['max_abs_standardized_mean_difference']:.3f}/"
            f"{audit['test_balance']['max_abs_standardized_mean_difference']:.3f}"
        )
    print(f"wrote {ASSIGNMENT_PATH} ({len(assignment):,} rows)")
    print(f"wrote {AUDIT_PATH}")


if __name__ == "__main__":
    main()
