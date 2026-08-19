# -*- coding: utf-8 -*-
"""Build external Morgan fingerprint cache under the frozen internal protocol."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import os
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("PROTAC_EXTERNAL_ROOT", str(REPO_ROOT / "external_data")))
PROC = BASE / "processed"
COHORT = PROC / "external_validation_cohort.csv"
FEATURE = PROC / "external_morgan_fp_2048.npy"
INDEX = PROC / "external_morgan_fp_2048_index.csv"
META = PROC / "external_morgan_fp_2048_meta.json"
RADIUS = 2
N_BITS = 2048


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    started = time.time()
    df = pd.read_csv(COHORT, encoding="utf-8-sig", low_memory=False)
    if not df["external_record_id"].is_unique:
        raise AssertionError("external_record_id must be unique")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    X = np.zeros((len(df), N_BITS), dtype=np.float32)
    invalid = []
    for i, smi in enumerate(df["smiles_canonical"].fillna("").astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(int(df.iloc[i]["external_record_id"]))
            continue
        ConvertToNumpyArray(generator.GetFingerprint(mol), X[i])
    if invalid:
        raise AssertionError(f"invalid external SMILES: {len(invalid)}")
    if X.shape != (len(df), N_BITS) or X.dtype != np.float32:
        raise AssertionError(f"unexpected feature shape/dtype: {X.shape}/{X.dtype}")
    if np.any(np.sum(X, axis=1) == 0):
        raise AssertionError("zero Morgan vectors found")
    np.save(FEATURE, X)
    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=int),
        "external_record_id": df["external_record_id"].to_numpy(dtype=int),
        "inchikey": df["inchikey"].astype(str).to_numpy(),
    }).to_csv(INDEX, index=False, encoding="utf-8-sig")
    meta = {
        "schema": "external-morgan-feature-cache-v1.0.0",
        "cohort_path": str(COHORT),
        "cohort_sha256": sha256(COHORT),
        "feature_path": str(FEATURE),
        "feature_sha256": sha256(FEATURE),
        "index_path": str(INDEX),
        "index_sha256": sha256(INDEX),
        "n_records": int(len(df)),
        "n_bits": N_BITS,
        "radius": RADIUS,
        "dtype": str(X.dtype),
        "smiles_column": "smiles_canonical",
        "invalid_smiles": len(invalid),
        "zero_vectors": int(np.sum(np.sum(X, axis=1) == 0)),
        "rdkit_version": rdBase.rdkitVersion,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
