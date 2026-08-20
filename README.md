# Bias-Aware Learning from Censored and Incompletely Observed PROTAC Degradation Data

Public GitHub release **v1.0.2** accompanying the v3.2 manuscript revision of the same title. The corresponding versioned Zenodo record will be cited here after its publication is verified.

**Authors:** Guanglu Liu; Shuang Wang (corresponding author)  
**Affiliation:** College of Computer Science and Technology / Qingdao Institute of Software, China University of Petroleum (East China), Qingdao, China  
**Contact:** wangshuang@upc.edu.cn

## Release status

This repository is the public source release. The versioned Zenodo archive remains the permanent archival record and must be checked for version consistency before citation.

- GitHub repository: `https://github.com/liuguang-jpt/release`
- Release tag: `v1.0.2`
- Earlier Zenodo archive DOI: `10.5281/zenodo.22015283` (v1.0.0; do not cite as the v1.0.2 archive DOI)
- Required final check: the GitHub repository, `v1.0.2` tag and the v1.0.2 Zenodo DOI must resolve publicly before manuscript availability statements cite this release.

## What this release contains

| Path | Content | Licence / redistribution boundary |
|---|---|---|
| `code/` | Main ETL, labelling, grouped-split, baseline, PU, calibration, censoring and sensitivity-analysis scripts | Project code: MIT |
| `external_code/` | Scripts for rebuilding the TPDdb-derived external cohort from a user-obtained upstream copy and reproducing frozen-model evaluation | Project code: MIT |
| `data/derived/` | Record-level PROTAC-DB-derived analysis table, dictionaries, split manifests, feature metadata/index files, calibration annotations and the adjudicated internal-consistency sample | Subject to PROTAC-DB upstream terms; this project grants no additional rights over upstream record content. The 121 MB Morgan cache is distributed in the Zenodo archive, not GitHub, and can be rebuilt locally. |
| `data/derived/gold_set_annotations/` | Historical file path containing the 132-record adjudicated internal-consistency sample, sampling design and annotator templates | Not an independent expert gold standard; record-level content remains subject to upstream terms |
| `data/raw/` | SHA-256 manifest of the source snapshot; the raw PROTAC-DB snapshot is not redistributed | Manifest only |
| `data/external/` | Aggregate external-evaluation metrics and model manifest only | TPDdb-derived record-level cohort, labels and predictions are not redistributed |
| `docs/` | Project-authored protocols, data-version notes and compliance documentation | Project-authored text: CC BY 4.0; embedded/upstream record content remains subject to upstream terms |

See `LICENSE.md` for the complete mixed-licence statement.

## Scientific scope

The main table contains 15,535 records with frozen evidence-availability labels: P = 2,484, N = 534, A = 655 and U = 11,862. The labels are operational derivatives of database fields, not condition-complete biological ground truth.

The adjudicated internal-consistency sample contains 132 records: 121 frozen quota-sampled records plus 11 rare-censor safeguards. Annotators A and B agreed on 127/132 records (96.2%; Cohen's κ = 0.9483). The five disagreements were adjudicated by the first author through literal application of the frozen rule, and all five final labels equal annotator B's label. This sample assesses protocol consistency; it is not independent expert or third-party validation.

The TPDdb-derived evaluation is structure-disjoint at the retained exact-structure level and provides bounded external evidence. It is not prospective, laboratory-independent or fully time-independent validation. Because TPDdb redistribution rights remain unresolved, only aggregate outputs and rebuild scripts are included.

## Installation

Recommended: Python 3.11–3.13 in a clean virtual environment.

```bash
pip install -r requirements.txt
```

PyTorch is optional and is needed only for the neural PU and censoring-aware models. Install the CPU or CUDA wheel appropriate for the local operating system and CUDA version, then install the optional constraint:

```bash
pip install -r requirements-optional-torch.txt
```

Do not reuse the original machine-specific CPU wheel specification on another platform.

## Reproduction entry points

The repository root is used as the default project root. Override it with `PROTAC_ROOT` only when using a different layout.

```bash
# Inspect the ordered workflow without running experiments
python code/run_all.py --dry-run --skip-etl --skip-shortcut-controls --skip-slow-bootstrap

# Rebuild labels and core derived tables
python code/etl_protac.py
python code/relabel_semantics.py
python code/build_split_groups.py
python code/build_morgan_features.py
python code/make_split_manifest.py

# Run the main experiment families
python code/baseline_pipeline.py
python code/pu_pipeline.py
python code/calib_sensitivity_pipeline.py
python code/censored_eval_pipeline.py

# Run the v4 reviewer-requested comparisons
python code/matched_backbone_pu.py
python code/head_to_head_reimplementations.py
```

The raw PROTAC-DB snapshot is not included. Place an authorised local copy in the location expected by the ETL command or provide the relevant command-line/environment configuration.

To rebuild the external evaluation, obtain TPDdb directly under its current access terms and place the required input files under `external_data/`:

```bash
python external_code/external_data_preprocessing.py
python external_code/build_external_validation_cohort.py
python external_code/build_external_features.py
python external_code/run_external_frozen_models.py
python external_code/summarize_external_validation.py
python external_code/build_external_validation_report.py
```

## Key files

- `data/derived/protac_clean_record_level.csv` — main record-level analysis table.
- `data/derived/protac_clean_record_level_v4.csv` — derived table with canonical SMILES, InChI and InChIKey; it does not replace the v3 analysis input.
- `data/derived/split_manifest_v3.csv` and `split_manifest_v3_audit.json` — frozen split roles and leakage audit.
- `data/derived/data_dictionary.csv` — field-level dictionary.
- `data/derived/data_dictionary_v4.csv` and `benchmark_contract_v4.json` — v4 structure-identifier metadata and hashes.
- `data/derived/morgan_fp_2048.npy` — large feature cache included in the Zenodo archive; excluded from GitHub because it exceeds GitHub's 100 MB file limit and can be rebuilt with `code/build_morgan_features.py`.
- `data/derived/gold_set_annotations/gold_sample_132.csv` — frozen 132-record sampling file.
- `data/derived/gold_set_annotations/gold_final.csv` — adjudicated internal-consistency labels.
- `data/external/external_validation_summary.json` — aggregate TPDdb-derived evaluation summary.
- `reports/pu_prior_multiplier_sensitivity_v4.json` — 0.8x/1.0x/1.2x prior sensitivity.
- `reports/dmax_publication_extreme_diagnostic_v4.json` — diagnostic of the extreme publication-split Dmax R2.
- `reports/matched_backbone_pu_results_v4.json` and `matched_backbone_pu_predictions_v4.csv` — common-MLP supervised/PU comparison under the frozen manifest.
- `reports/head_to_head_reimplementations_v4.json` and `head_to_head_reimplementations_v4.csv` — DeepPROTACs- and DegradeMaster-aligned protocol-constrained comparisons; these are not official-weight reproductions.
- `docs/HEAD_TO_HEAD_PROTOCOL_V4.md` — source commits, missing inputs, feature mapping and comparison boundaries for the two representative models.
- `docs/DATA_VERSION_README.md` — data lineage and execution notes.
- `ZENODO_GITHUB_UPLOAD_GUIDE.md` — exact GitHub and Zenodo publication fields.

## Citation

Use `CITATION.cff`. Cite PROTAC-DB 3.0 for the upstream database and this release for the project code, protocols and frozen benchmark artefacts. Add the version-specific Zenodo DOI to `CITATION.cff` only after the v1.0.1 record is published and verified.
