# Bias-Aware Learning from Censored and Incompletely Observed PROTAC Degradation Data

Local release candidate **v1.0.0** accompanying the manuscript of the same title.

**Authors:** Guanglu Liu; Shuang Wang (corresponding author)  
**Affiliation:** College of Computer Science and Technology / Qingdao Institute of Software, China University of Petroleum (East China), Qingdao, China  
**Contact:** wangshuang@upc.edu.cn

## Release status

This directory is ready for public deposition, but public availability must not be claimed until both records have been activated and checked.

- Intended GitHub repository: `https://github.com/liuguang-jpt/release`
- Planned manuscript tag: `v1.0-paper`
- Author-provided Zenodo DOI: `10.5281/zenodo.22015283`
- Required final check: the GitHub repository, GitHub Release and DOI must resolve publicly before the manuscript availability statements are upgraded from pending to public.

## What this release contains

| Path | Content | Licence / redistribution boundary |
|---|---|---|
| `code/` | Main ETL, labelling, grouped-split, baseline, PU, calibration, censoring and sensitivity-analysis scripts | Project code: MIT |
| `external_code/` | Scripts for rebuilding the TPDdb-derived external cohort from a user-obtained upstream copy and reproducing frozen-model evaluation | Project code: MIT |
| `data/derived/` | Record-level PROTAC-DB-derived analysis table, dictionaries, split manifests, feature metadata/index files, calibration annotations and the adjudicated internal-consistency sample | Subject to PROTAC-DB upstream terms; this project grants no additional rights over upstream record content |
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
- `data/derived/split_manifest_v3.csv` and `split_manifest_v3_audit.json` — frozen split roles and leakage audit.
- `data/derived/data_dictionary.csv` — field-level dictionary.
- `data/derived/gold_set_annotations/gold_sample_132.csv` — frozen 132-record sampling file.
- `data/derived/gold_set_annotations/gold_final.csv` — adjudicated internal-consistency labels.
- `data/external/external_validation_summary.json` — aggregate TPDdb-derived evaluation summary.
- `docs/DATA_VERSION_README.md` — data lineage and execution notes.
- `ZENODO_GITHUB_UPLOAD_GUIDE.md` — exact GitHub and Zenodo publication fields.

## Citation

Use `CITATION.cff`. Cite PROTAC-DB 3.0 for the upstream database and this release for the project code, protocols and frozen benchmark artefacts. The DOI and repository URL must be verified after public activation.
