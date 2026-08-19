# -*- coding: utf-8 -*-
"""Generate the frozen benchmark split manifest and normalized contract.

Outputs
-------
data/derived/split_manifest_v3.csv
    One row per record and split configuration. This is the sole source of
    train/calibration/test roles.
data/derived/benchmark_contract_v3.json
    Task eligibility, endpoint/label semantics, hashes, group policy, and
    provenance metadata.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from benchmark_contract import (
    CONTRACT_PATH,
    DATA_PATH,
    FEATURE_INDEX_PATH,
    FEATURE_META_PATH,
    FEATURE_PATH,
    GROUP_PATH,
    LABEL_COL,
    MANIFEST_PATH,
    SEEDS,
    SPLITS,
    SPLIT_SCHEMA,
    audit_manifest,
    git_state,
    group_values,
    load_dataset,
    load_groups,
    sha256_file,
    task_eligibility,
)


def two_way_roles(groups: np.ndarray, seed: int, test_size: float = 0.20) -> np.ndarray:
    idx = np.arange(len(groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(idx, groups=groups))
    roles = np.full(len(groups), "excluded", dtype=object)
    roles[tr] = "train"
    roles[te] = "test"
    return roles


def three_way_roles(groups: np.ndarray, seed: int) -> np.ndarray:
    idx = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_cal, test = next(outer.split(idx, groups=groups))
    inner_groups = groups[train_cal]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 100003)
    train_rel, cal_rel = next(inner.split(train_cal, groups=inner_groups))
    train = train_cal[train_rel]
    calibration = train_cal[cal_rel]
    roles = np.full(len(groups), "excluded", dtype=object)
    roles[train] = "train"
    roles[calibration] = "calibration"
    roles[test] = "test"
    return roles


def add_config(rows: list[dict], df: pd.DataFrame, family: str, split: str, seed: int, roles: np.ndarray, groups: np.ndarray):
    for record_id, role, group_id in zip(df["record_id"].to_numpy(), roles, groups):
        rows.append(
            {
                "record_id": int(record_id),
                "split_family": family,
                "split_regime": split,
                "fold": 1,
                "seed": int(seed),
                "role": str(role),
                "group_id": str(group_id),
            }
        )


def main():
    t0 = time.time()
    df = load_dataset()
    groups_df = load_groups(df)
    rows: list[dict] = []

    for split in SPLITS:
        gv = group_values(df, groups_df, split)
        for seed in SEEDS:
            add_config(rows, df, "all_records_2way", split, seed, two_way_roles(gv, seed), gv)
            add_config(rows, df, "all_records_3way", split, seed, three_way_roles(gv, seed), gv)

    # Fixed internal post-cutoff stress test. Missing years and the 2023 buffer are excluded.
    temporal_roles = np.full(len(df), "excluded", dtype=object)
    year = pd.to_numeric(df["year_doi"], errors="coerce")
    temporal_roles[(year <= 2022).fillna(False).to_numpy()] = "train"
    temporal_roles[(year >= 2024).fillna(False).to_numpy()] = "test"
    temporal_groups = np.asarray(
        [f"doi:{x}" if pd.notna(x) and str(x).strip() else f"record:{rid}" for x, rid in zip(df["article_doi"], df["record_id"])],
        dtype=object,
    )
    add_config(rows, df, "temporal_postcutoff", "temporal", 0, temporal_roles, temporal_groups)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    artifacts = {
        "dataset_path": "data/derived/protac_clean_record_level.csv",
        "dataset_sha256": sha256_file(DATA_PATH),
        "group_path": "data/derived/protac_split_groups.csv",
        "group_sha256": sha256_file(GROUP_PATH),
        "feature_path": "data/derived/morgan_fp_2048.npy",
        "feature_sha256": sha256_file(FEATURE_PATH) if FEATURE_PATH.exists() else None,
        "feature_index_sha256": sha256_file(FEATURE_INDEX_PATH) if FEATURE_INDEX_PATH.exists() else None,
        "feature_meta_sha256": sha256_file(FEATURE_META_PATH) if FEATURE_META_PATH.exists() else None,
        "manifest_path": "data/derived/split_manifest_v3.csv",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }
    task_specs = {
        "baseline_pdc50": {
            "split_family": "all_records_2way",
            "eligibility": "dc50_obs_type == exact and pdc50_value is finite",
            "endpoint": "pDC50 exact regression",
            "label_version": None,
        },
        "baseline_dmax": {
            "split_family": "all_records_2way",
            "eligibility": "dmax_obs_type == exact and dmax_value is finite",
            "endpoint": "Dmax exact regression",
            "label_version": None,
        },
        "baseline_pn": {
            "split_family": "all_records_2way",
            "eligibility": f"{LABEL_COL} in {{P,N}}",
            "endpoint": "P/N classification",
            "label_version": LABEL_COL,
        },
        "baseline_uan": {
            "split_family": "all_records_2way",
            "eligibility": f"{LABEL_COL} in {{P,N,U}}",
            "endpoint": "diagnostic P versus N/U classification",
            "label_version": LABEL_COL,
        },
        "pu": {
            "split_family": "all_records_2way",
            "eligibility": "all records assigned before label filtering",
            "endpoint": "P versus hidden non-P training; P/N test",
            "label_version": LABEL_COL,
        },
        "calibration": {
            "split_family": "all_records_3way",
            "eligibility": "all records assigned before label filtering",
            "endpoint": "nnPU probability calibration; oracle P/N calibration explicitly separated",
            "label_version": LABEL_COL,
        },
        "censored_pdc50": {
            "split_family": "all_records_2way",
            "eligibility": "pDC50 exact or one-sided censored endpoint",
            "endpoint": "pDC50 regression and censoring-constraint evaluation",
            "label_version": None,
        },
        "shortcut_pn": {
            "split_family": "all_records_2way",
            "eligibility": f"{LABEL_COL} in {{P,N}}",
            "endpoint": "shortcut controls for P/N classification",
            "label_version": LABEL_COL,
        },
        "temporal_postcutoff": {
            "split_family": "temporal_postcutoff",
            "eligibility": "year_doi <= 2022 train; year_doi >= 2024 test; 2023/missing excluded",
            "endpoint": "internal temporal stress test",
            "label_version": LABEL_COL,
        },
    }
    eligibility_counts = {name: int(task_eligibility(df, name).sum()) for name in task_specs}
    contract = {
        "split_schema": SPLIT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": "PROTAC-DB-3.0-derived-record-level-v0.5",
        "label_version": LABEL_COL,
        "group_version": "protac_split_groups.csv sha256",
        "seeds": SEEDS,
        "split_regimes": SPLITS,
        "missing_group_policy": "missing scaffold/publication/POI values receive record-specific groups",
        "role_policy": {
            "all_records_2way": "80% train / 20% test by frozen groups",
            "all_records_3way": "60% train / 20% calibration / 20% test by frozen groups",
            "temporal_postcutoff": "year<=2022 train, year>=2024 test, 2023 buffer",
        },
        "tasks": task_specs,
        "eligibility_counts": eligibility_counts,
        "artifacts": artifacts,
        "git": git_state(),
    }
    with CONTRACT_PATH.open("w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)

    audit_rows = audit_manifest(df, groups_df, manifest)
    print(f"wrote {MANIFEST_PATH} ({len(manifest):,} rows)")
    print(f"wrote {CONTRACT_PATH}")
    print(f"audited {len(audit_rows)} split configurations; all record/group overlaps = 0")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


