# -*- coding: utf-8 -*-
"""Build and audit the frozen Morgan fingerprint cache.

The feature matrix row order is explicitly bound to record_id through a sidecar
index and metadata file. Raw source data are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_contract import DATA_PATH, FEATURE_PATH, PROCESSED_DIR, sha256_file

INDEX_PATH = PROCESSED_DIR / "morgan_fp_2048_index.csv"
META_PATH = PROCESSED_DIR / "morgan_fp_2048_meta.json"
RADIUS = 2
N_BITS = 2048


def validate_cache(df: pd.DataFrame) -> tuple[bool, str]:
    required = [FEATURE_PATH, INDEX_PATH, META_PATH]
    if not all(path.exists() for path in required):
        return False, "one or more cache artifacts are missing"
    try:
        x = np.load(FEATURE_PATH, mmap_mode="r")
        index = pd.read_csv(INDEX_PATH, encoding="utf-8-sig")
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"cache cannot be read: {exc}"
    if x.shape != (len(df), N_BITS) or x.dtype != np.float32:
        return False, f"unexpected feature shape/dtype: {x.shape}/{x.dtype}"
    if list(index.columns) != ["row_index", "record_id"]:
        return False, "sidecar columns are not row_index, record_id"
    if not np.array_equal(index["row_index"].to_numpy(), np.arange(len(df))):
        return False, "sidecar row_index is not contiguous"
    if not np.array_equal(index["record_id"].to_numpy(), df["record_id"].to_numpy()):
        return False, "sidecar record_id order differs from the dataset"
    if meta.get("dataset_sha256") != sha256_file(DATA_PATH):
        return False, "dataset hash differs from feature metadata"
    if meta.get("feature_sha256") != sha256_file(FEATURE_PATH):
        return False, "feature hash differs from feature metadata"
    if meta.get("radius") != RADIUS or meta.get("n_bits") != N_BITS:
        return False, "fingerprint parameters differ from the frozen protocol"
    return True, "cache, record_id sidecar, and hashes are aligned"


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, int, str]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import ConvertToNumpyArray

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    x = np.zeros((len(df), N_BITS), dtype=np.float32)
    invalid = 0
    for i, smiles in enumerate(df["smiles"].fillna("").astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        ConvertToNumpyArray(generator.GetFingerprint(mol), x[i])
    return x, invalid, rdBase.rdkitVersion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if the audited cache is valid")
    args = parser.parse_args()

    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if not df["record_id"].is_unique:
        raise AssertionError("record_id must be unique before feature construction")

    valid, reason = validate_cache(df)
    if valid and not args.force:
        print(f"feature cache valid: {reason}")
        return
    print(f"rebuilding Morgan cache: {reason}")

    x, invalid_smiles, rdkit_version = build_features(df)
    tmp_feature = FEATURE_PATH.with_suffix(".tmp.npy")
    np.save(tmp_feature, x)
    os.replace(tmp_feature, FEATURE_PATH)

    sidecar = pd.DataFrame({"row_index": np.arange(len(df), dtype=int), "record_id": df["record_id"].to_numpy()})
    sidecar.to_csv(INDEX_PATH, index=False, encoding="utf-8-sig")
    metadata = {
        "schema": "morgan-feature-cache-v1.0.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": "data/derived/protac_clean_record_level.csv",
        "dataset_sha256": sha256_file(DATA_PATH),
        "feature_path": "data/derived/morgan_fp_2048.npy",
        "feature_sha256": sha256_file(FEATURE_PATH),
        "index_path": "data/derived/morgan_fp_2048_index.csv",
        "index_sha256": sha256_file(INDEX_PATH),
        "n_records": int(len(df)),
        "n_bits": N_BITS,
        "radius": RADIUS,
        "dtype": "float32",
        "smiles_column": "smiles",
        "invalid_smiles_zero_vectors": int(invalid_smiles),
        "rdkit_version": rdkit_version,
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    valid, reason = validate_cache(df)
    if not valid:
        raise AssertionError(f"new feature cache failed validation: {reason}")
    print(f"wrote {FEATURE_PATH} with shape={x.shape}, invalid_smiles={invalid_smiles}")
    print(f"wrote {INDEX_PATH} and {META_PATH}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

