#!/usr/bin/env python
"""Create v4 data dictionary and contract from the frozen v3 metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v3-dictionary", type=Path, required=True)
    p.add_argument("--v3-contract", type=Path, required=True)
    p.add_argument("--v4-data", type=Path, required=True)
    p.add_argument("--out-dictionary", type=Path, required=True)
    p.add_argument("--out-contract", type=Path, required=True)
    args = p.parse_args()

    dictionary = pd.read_csv(args.v3_dictionary, encoding="utf-8-sig")
    additions = pd.DataFrame([
        ["canonical_smiles", "RDKit canonical isomeric SMILES derived from smiles", "", "derived from smiles", "CCO"],
        ["inchi", "InChI identifier derived from smiles", "", "derived from smiles", "InChI=1S/..."],
        ["inchikey", "InChIKey identifier derived from smiles", "", "derived from smiles", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        ["structure_parse_status", "Structure conversion status", "", "RDKit", "ok"],
        ["structure_duplicate_record_count", "Number of records sharing the same InChIKey", "", "derived", "2"],
        ["is_duplicate_inchikey", "Whether the record belongs to a repeated InChIKey group", "", "derived", "True"],
    ], columns=dictionary.columns)
    dictionary = pd.concat([dictionary, additions], ignore_index=True)
    args.out_dictionary.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(args.out_dictionary, index=False, encoding="utf-8-sig")

    contract = json.loads(args.v3_contract.read_text(encoding="utf-8"))
    contract["dataset_version"] = "PROTAC-DB-3.0-derived-record-level-v0.5-plus-structure-identifiers-v4"
    contract["structure_identifier_version"] = "structure-identifiers-v4"
    contract["artifacts"]["dataset_v3_path"] = contract["artifacts"].pop("dataset_path")
    contract["artifacts"]["dataset_v4_path"] = "data/derived/protac_clean_record_level_v4.csv"
    contract["artifacts"]["dataset_v4_sha256"] = sha256(args.v4_data)
    contract["artifacts"]["data_dictionary_v4_path"] = "data/derived/data_dictionary_v4.csv"
    contract["artifacts"]["data_dictionary_v4_sha256"] = sha256(args.out_dictionary)
    contract["notes"] = [
        "The v3 table, frozen split manifest and feature cache remain the source of reported model results.",
        "The v4 table adds derived chemical identifiers for FAIR reuse and structure auditing only.",
    ]
    args.out_contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_v4_sha256": contract["artifacts"]["dataset_v4_sha256"], "dictionary_v4_sha256": contract["artifacts"]["data_dictionary_v4_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
