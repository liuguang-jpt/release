# -*- coding: utf-8 -*-
"""Shared benchmark contract utilities.

The frozen split manifest is the only source of train/calibration/test roles.
Experiment scripts may filter records by task eligibility, but may not create
new random/group splits locally.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(os.environ.get("PROTAC_ROOT", str(PROJECT_DIR)))
PROCESSED_DIR = ROOT_DIR / "data" / "derived"
REPORTS_DIR = ROOT_DIR / "reports"
MANIFEST_PATH = PROCESSED_DIR / "split_manifest_v3.csv"
CONTRACT_PATH = PROCESSED_DIR / "benchmark_contract_v3.json"
DATA_PATH = PROCESSED_DIR / "protac_clean_record_level.csv"
GROUP_PATH = PROCESSED_DIR / "protac_split_groups.csv"
FEATURE_PATH = PROCESSED_DIR / "morgan_fp_2048.npy"
FEATURE_INDEX_PATH = PROCESSED_DIR / "morgan_fp_2048_index.csv"
FEATURE_META_PATH = PROCESSED_DIR / "morgan_fp_2048_meta.json"
LABEL_COL = "activity_evidence_v2"
SEEDS = [20260815, 20260816, 20260817]
SPLITS = ["random", "scaffold", "pub", "poi"]
SPLIT_SCHEMA = "protac-benchmark-split-v3.0.0"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_state() -> Dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "status", "--porcelain"],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except Exception:
        return {"commit": "unknown", "working_tree_dirty": None}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if not df["record_id"].is_unique:
        raise AssertionError("record_id must be unique in the record-level dataset")
    return df


def load_groups(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_dataset()
    groups = pd.read_csv(GROUP_PATH, encoding="utf-8-sig").set_index("record_id")
    missing = set(df["record_id"]) - set(groups.index)
    if missing:
        raise AssertionError(f"split-group table misses {len(missing)} record_ids")
    return groups.loc[df["record_id"]].copy()


def group_values(df: pd.DataFrame, groups: pd.DataFrame, split_regime: str) -> np.ndarray:
    """Return stable grouping values aligned to df rows.

    Missing scientific groups are record-specific rather than being collapsed
    into one artificial mega-group.
    """
    if split_regime == "random":
        return np.asarray([f"record:{rid}" for rid in df["record_id"]], dtype=object)
    col = {"scaffold": "scaffold", "pub": "pub_group", "poi": "poi"}[split_regime]
    values = groups[col].astype(object).to_numpy(copy=True)
    missing = pd.isna(values) | (pd.Series(values).astype(str).str.strip().to_numpy() == "")
    rid = df["record_id"].to_numpy()
    values[missing] = np.asarray([f"__MISSING__:{col}:{x}" for x in rid[missing]], dtype=object)
    return values.astype(str)


def load_contract(validate_hashes: bool = True) -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as f:
        contract = json.load(f)
    if validate_hashes:
        current = {
            "dataset_sha256": sha256_file(DATA_PATH),
            "group_sha256": sha256_file(GROUP_PATH),
            "feature_sha256": sha256_file(FEATURE_PATH) if FEATURE_PATH.exists() else None,
            "feature_index_sha256": sha256_file(FEATURE_INDEX_PATH) if FEATURE_INDEX_PATH.exists() else None,
            "feature_meta_sha256": sha256_file(FEATURE_META_PATH) if FEATURE_META_PATH.exists() else None,
        }
        for key, value in current.items():
            recorded = contract["artifacts"].get(key)
            if recorded != value:
                raise AssertionError(
                    f"benchmark contract hash mismatch for {key}: recorded={recorded}, current={value}. "
                    "Regenerate and freeze the manifest before running experiments."
                )
    return contract


def load_manifest(validate: bool = True) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", dtype={"group_id": str})
    required = {"record_id", "split_family", "split_regime", "fold", "seed", "role", "group_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise AssertionError(f"manifest missing columns: {sorted(missing)}")
    if manifest.duplicated(["record_id", "split_family", "split_regime", "seed"]).any():
        raise AssertionError("manifest has duplicate assignment keys")
    if validate:
        contract = load_contract(validate_hashes=True)
        if contract["split_schema"] != SPLIT_SCHEMA:
            raise AssertionError("unexpected split schema")
    return manifest


def assignment_frame(
    manifest: pd.DataFrame,
    split_family: str,
    split_regime: str,
    seed: int,
) -> pd.DataFrame:
    sub = manifest[
        (manifest["split_family"] == split_family)
        & (manifest["split_regime"] == split_regime)
        & (manifest["seed"] == seed)
    ].copy()
    if sub.empty:
        raise KeyError(f"manifest assignment not found: {split_family}/{split_regime}/{seed}")
    return sub


def role_indices(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    split_family: str,
    split_regime: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    sub = assignment_frame(manifest, split_family, split_regime, seed)
    if len(sub) != len(df) or set(sub["record_id"]) != set(df["record_id"]):
        raise AssertionError(f"manifest does not cover the full dataset for {split_family}/{split_regime}/{seed}")
    pos = pd.Series(np.arange(len(df)), index=df["record_id"])
    out: Dict[str, np.ndarray] = {}
    for role in ["train", "calibration", "test", "excluded"]:
        ids = sub.loc[sub["role"] == role, "record_id"]
        out[role] = pos.loc[ids].to_numpy(dtype=int) if len(ids) else np.asarray([], dtype=int)
    nonempty = [set(v.tolist()) for k, v in out.items() if k != "excluded" and len(v)]
    for i in range(len(nonempty)):
        for j in range(i + 1, len(nonempty)):
            if nonempty[i] & nonempty[j]:
                raise AssertionError("record overlap across manifest roles")
    return out


def assert_group_disjoint(
    df: pd.DataFrame,
    groups: pd.DataFrame,
    roles: Dict[str, np.ndarray],
    split_regime: str,
) -> None:
    gv = group_values(df, groups, split_regime)
    role_names = [r for r in ["train", "calibration", "test"] if len(roles.get(r, []))]
    role_groups = {r: set(gv[roles[r]]) for r in role_names}
    for i, left in enumerate(role_names):
        for right in role_names[i + 1 :]:
            overlap = role_groups[left] & role_groups[right]
            if overlap:
                raise AssertionError(
                    f"group overlap in {split_regime}: {left}/{right}, n={len(overlap)}"
                )


def task_eligibility(df: pd.DataFrame, task_name: str) -> np.ndarray:
    if task_name == "baseline_pdc50":
        return ((df["dc50_obs_type"] == "exact") & df["pdc50_value"].notna()).to_numpy()
    if task_name == "baseline_dmax":
        return ((df["dmax_obs_type"] == "exact") & df["dmax_value"].notna()).to_numpy()
    if task_name in {"baseline_pn", "shortcut_pn"}:
        return df[LABEL_COL].isin(["P", "N"]).to_numpy()
    if task_name == "baseline_uan":
        return df[LABEL_COL].isin(["P", "N", "U"]).to_numpy()
    if task_name in {"pu", "calibration"}:
        return np.ones(len(df), dtype=bool)
    if task_name == "censored_pdc50":
        return (
            df["dc50_obs_type"].isin(["exact", "left-censored", "right-censored"])
            & (df["pdc50_value"].notna() | df["pdc50_lower"].notna() | df["pdc50_upper"].notna())
        ).to_numpy()
    if task_name == "temporal_postcutoff":
        return df["year_doi"].notna().to_numpy()
    raise KeyError(f"unknown task: {task_name}")


def audit_manifest(df: pd.DataFrame, groups: pd.DataFrame, manifest: pd.DataFrame) -> list[dict]:
    rows = []
    configs = manifest[["split_family", "split_regime", "seed"]].drop_duplicates()
    for cfg in configs.itertuples(index=False):
        roles = role_indices(df, manifest, cfg.split_family, cfg.split_regime, int(cfg.seed))
        if cfg.split_regime in SPLITS:
            assert_group_disjoint(df, groups, roles, cfg.split_regime)
        rows.append(
            {
                "split_family": cfg.split_family,
                "split_regime": cfg.split_regime,
                "seed": int(cfg.seed),
                "n_train": int(len(roles["train"])),
                "n_calibration": int(len(roles["calibration"])),
                "n_test": int(len(roles["test"])),
                "n_excluded": int(len(roles["excluded"])),
                "record_overlap": 0,
                "group_overlap": 0,
            }
        )
    return rows

