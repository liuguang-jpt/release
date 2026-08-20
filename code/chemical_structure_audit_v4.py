#!/usr/bin/env python
"""Add machine-readable chemical identifiers without altering the frozen v3 table."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifiers(smiles: object) -> tuple[str, str, str, str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return "", "", "", "missing_smiles"
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "", "", "", "invalid_smiles"
    try:
        return (
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
            Chem.MolToInchi(molecule),
            Chem.MolToInchiKey(molecule),
            "ok",
        )
    except Exception as exc:  # Keep an explicit failure category in the data product.
        return "", "", "", f"conversion_error:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    required = {"record_id", "smiles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not frame["record_id"].is_unique:
        raise ValueError("record_id must be unique")

    derived = [identifiers(value) for value in frame["smiles"]]
    frame[["canonical_smiles", "inchi", "inchikey", "structure_parse_status"]] = pd.DataFrame(
        derived, index=frame.index
    )
    ok = frame["structure_parse_status"].eq("ok")
    key_counts = Counter(frame.loc[ok, "inchikey"])
    duplicate_keys = {key for key, count in key_counts.items() if count > 1}
    frame["structure_duplicate_record_count"] = frame["inchikey"].map(key_counts).fillna(0).astype(int)
    frame["is_duplicate_inchikey"] = frame["inchikey"].isin(duplicate_keys)

    status_counts = frame["structure_parse_status"].value_counts(dropna=False).to_dict()
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "rdkit_version": rdBase.rdkitVersion,
        "rows": int(len(frame)),
        "unique_record_ids": int(frame["record_id"].nunique()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "valid_structures": int(ok.sum()),
        "unique_inchikeys": int(frame.loc[ok, "inchikey"].nunique()),
        "duplicate_inchikey_groups": int(len(duplicate_keys)),
        "records_in_duplicate_inchikey_groups": int(frame["is_duplicate_inchikey"].sum()),
        "new_columns": [
            "canonical_smiles",
            "inchi",
            "inchikey",
            "structure_parse_status",
            "structure_duplicate_record_count",
            "is_duplicate_inchikey",
        ],
        "scope": "Derived FAIR identifiers only. The frozen v3 analysis table, split manifest and feature cache are unchanged.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    audit["output_sha256"] = sha256(args.output)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Chemical structure audit v4",
        "",
        f"Generated (UTC): {audit['generated_at_utc']}",
        "",
        "## Results",
        f"- Input rows: {audit['rows']:,}",
        f"- Valid parsed structures: {audit['valid_structures']:,}",
        f"- Unique InChIKeys: {audit['unique_inchikeys']:,}",
        f"- Duplicate InChIKey groups: {audit['duplicate_inchikey_groups']:,}",
        f"- Records in duplicate InChIKey groups: {audit['records_in_duplicate_inchikey_groups']:,}",
        f"- Parse-status counts: {audit['status_counts']}",
        "",
        "## Scope",
        audit["scope"],
        "",
        "The v4 table adds canonical SMILES, InChI and InChIKey for machine-readable chemical identity. It does not replace the v3 frozen analysis input or alter previously reported features, splits or results.",
        "",
        "## Integrity",
        f"- Input SHA-256: `{audit['input_sha256']}`",
        f"- Output SHA-256: `{audit['output_sha256']}`",
        f"- RDKit: `{audit['rdkit_version']}`",
    ]
    args.audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
