# -*- coding: utf-8 -*-
"""Freeze the eligible external validation cohort and emit auditable summaries."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import os
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("PROTAC_EXTERNAL_ROOT", str(REPO_ROOT / "external_data")))
PROC = BASE / "processed"
AUDITS = BASE / "audits"
REPORTS = BASE / "reports"
COHORT = PROC / "external_validation_cohort.csv"
SUMMARY = PROC / "external_validation_cohort_summary.csv"
EXCLUSION = PROC / "external_validation_exclusion_log.csv"
META = REPORTS / "external_validation_cohort_metadata.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def main() -> None:
    clean_path = PROC / "external_record_level_clean.csv"
    overlap_path = AUDITS / "external_overlap_audit.csv"
    clean = pd.read_csv(clean_path, encoding="utf-8-sig", low_memory=False)
    overlap = pd.read_csv(overlap_path, encoding="utf-8-sig", low_memory=False)
    required = {
        "external_record_id", "final_external_eligibility", "activity_evidence_external_v1",
        "t1_eligible", "t2_eligible", "t3_eligible", "t4_eligible", "inchikey",
        "inchikey_connectivity", "source_patent_id", "smiles_canonical",
    }
    missing = required - set(clean.columns)
    if missing:
        raise AssertionError(f"external clean table missing columns: {sorted(missing)}")
    if not clean["external_record_id"].is_unique:
        raise AssertionError("external_record_id must be unique")
    if len(overlap) != len(clean):
        raise AssertionError("overlap audit row count does not match clean table")
    if (overlap["match_level"].eq("L1_structure") & overlap["final_external_eligibility"].eq("eligible_nonoverlap")).any():
        raise AssertionError("L1 structure overlap leaked into eligible_nonoverlap")
    if (overlap["match_level"].eq("L3_experiment") & overlap["final_external_eligibility"].eq("eligible_nonoverlap")).any():
        raise AssertionError("L3 experiment overlap leaked into eligible_nonoverlap")
    if (overlap["match_level"].eq("L4_potential") & overlap["final_external_eligibility"].eq("eligible_nonoverlap")).any():
        raise AssertionError("L4 potential overlap leaked into eligible_nonoverlap")

    keep = clean["final_external_eligibility"].eq("eligible_nonoverlap")
    cohort = clean.loc[keep].copy().reset_index(drop=True)
    cohort["external_cohort_row"] = range(len(cohort))
    cohort.to_csv(COHORT, index=False, encoding="utf-8-sig")

    ex = clean.loc[~keep, ["external_record_id", "tpd_id", "final_external_eligibility", "activity_evidence_external_v1", "inchikey", "source_patent_id"]].copy()
    ex["exclusion_reason"] = ex["final_external_eligibility"]
    ex.to_csv(EXCLUSION, index=False, encoding="utf-8-sig")

    summary_rows = []
    def add(scope: str, task: str, subset: pd.DataFrame) -> None:
        summary_rows.append({
            "scope": scope,
            "task": task,
            "n_records": int(len(subset)),
            "n_structures": int(subset["inchikey"].nunique(dropna=True)),
            "n_connectivity_structures": int(subset["inchikey_connectivity"].nunique(dropna=True)),
            "n_patents": int(subset["source_patent_id"].nunique(dropna=True)),
            "n_P": int(subset["activity_evidence_external_v1"].eq("P").sum()),
            "n_N": int(subset["activity_evidence_external_v1"].eq("N").sum()),
            "n_U": int(subset["activity_evidence_external_v1"].eq("U").sum()),
            "n_A": int(subset["activity_evidence_external_v1"].eq("A").sum()),
        })
    add("eligible_nonoverlap", "all", cohort)
    add("eligible_nonoverlap", "T1_pdc50", cohort.loc[as_bool(cohort["t1_eligible"])])
    add("eligible_nonoverlap", "T2_dmax", cohort.loc[as_bool(cohort["t2_eligible"])])
    add("eligible_nonoverlap", "T3_pn", cohort.loc[cohort["activity_evidence_external_v1"].isin(["P", "N"])])
    add("eligible_nonoverlap", "T4_pnu", cohort.loc[cohort["activity_evidence_external_v1"].isin(["P", "N", "U"])])
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False, encoding="utf-8-sig")

    t3 = cohort.loc[cohort["activity_evidence_external_v1"].isin(["P", "N"])].copy()
    conflict_counts = t3.groupby("inchikey")["activity_evidence_external_v1"].nunique()
    conflict_structures = conflict_counts[conflict_counts > 1]
    cohort_meta = {
        "source_clean_sha256": sha256(clean_path),
        "overlap_audit_sha256": sha256(overlap_path),
        "cohort_sha256": sha256(COHORT),
        "exclusion_log_sha256": sha256(EXCLUSION),
        "summary_sha256": sha256(SUMMARY),
        "n_clean_records": int(len(clean)),
        "n_cohort_records": int(len(cohort)),
        "n_exact_overlap_excluded": int(clean["final_external_eligibility"].eq("exact_overlap").sum()),
        "n_probable_overlap_excluded": int(clean["final_external_eligibility"].eq("probable_overlap").sum()),
        "n_l1_in_cohort": int(((overlap["match_level"] == "L1_structure") & (overlap["final_external_eligibility"] == "eligible_nonoverlap")).sum()),
        "t1_records": int(as_bool(cohort["t1_eligible"]).sum()),
        "t2_records": int(as_bool(cohort["t2_eligible"]).sum()),
        "t3_records": int(len(t3)),
        "t4_records": int(cohort["activity_evidence_external_v1"].isin(["P", "N", "U"]).sum()),
        "t3_unique_structures": int(t3["inchikey"].nunique()),
        "t3_conflict_structures": int(len(conflict_structures)),
        "t3_conflict_records": int(t3[t3["inchikey"].isin(conflict_structures.index)].shape[0]),
        "dmax_rule_expected": "P if Dmax >= 70%; N if Dmax <= 20%",
        "eligibility_rule": "final_external_eligibility == eligible_nonoverlap",
    }
    META.write_text(json.dumps(cohort_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cohort_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
