# External Validation Data — Index and Rebuild Guide (TPDdb)

## Why record-level external data is NOT in this release

The external validation cohort is derived from **TPDdb** (*Nucleic Acids Research*, 2026). At release time, the upstream licence of TPDdb is **UNKNOWN** — the compliance audit (see `docs/COMPLIANCE_AUDIT_EXTERNAL_DATA.md`, item: "TPDdb license = UNKNOWN … this is a blocking item for data release") concluded that **public redistribution of TPDdb-derived record-level data must not proceed** until the upstream terms are confirmed.

Therefore this release ships, for the external validation:

- ✅ **Aggregated metrics only** (`external_validation_summary.json`, `external_T1/T2/T3_metrics_by_split_seed.csv`, `external_T4_exploratory_label_distribution_by_target.csv`, `external_structure_cluster_bootstrap.csv`, `external_patent_cluster_bootstrap.csv`, `external_conflict_label_sensitivity.csv`, `external_model_manifest.csv`)
- ✅ **Full rebuild scripts** (`external_code/`)
- ❌ No cohort table, no per-record predictions, no per-record labels, no TPDdb-derived SMILES or endpoints

## What the external cohort is

- 16,808 `eligible_nonoverlap` records retained after exact/probable overlap exclusion against the main PROTAC-DB-derived training set
- L1 (structure) leakage into the retained cohort: zero
- The cohort was never used for model fitting, tuning, threshold selection or calibration (frozen-model evaluation)
- Reported as **structure-disjoint, TPDdb-derived external evaluation** — not prospective, not laboratory-independent, not fully time-independent

## How to rebuild the cohort yourself

1. Obtain TPDdb from its official source (see the TPDdb publication in *Nucleic Acids Research* 2026 for the access URL) and place the raw export in your working directory.
2. Run, in order:

```bash
python external_code/external_data_preprocessing.py      # normalise, pair DC50/Dmax, map labels per PNU annotation spec v2.0
python external_code/build_external_features.py          # Morgan fingerprints + feature assembly
python external_code/build_external_validation_cohort.py # overlap exclusion vs. main dataset, structure-disjoint filtering
python external_code/run_external_frozen_models.py       # frozen-model evaluation
python external_code/external_cluster_bootstrap.py       # cluster bootstrap CIs
python external_code/summarize_external_validation.py    # reproduce the aggregated metrics in data/external/
```

3. Cross-check your aggregates against `data/external/external_validation_summary.json`.

## Known limitations (inherited from the audit, verify on rebuild)

1. DC50/Dmax pairing keys on (compound, cell line, target, suffix) — sampling re-check recommended
2. Free-text cell-line normalisation needs manual review before cross-cell-line analysis
3. P/N composition of the 16,808 retained records must be re-tabulated before use as a standalone cohort

