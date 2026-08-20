# Head-to-head protocol for representative PROTAC models (v4)

## Scope

This document defines the additional comparison requested by the v4 review. The comparison is performed under the frozen `all_records_2way` manifest (`split_manifest_v3.csv`), the frozen `activity_evidence_v2` labels and the common P/N test subset. It uses four split regimes (`random`, `scaffold`, `pub`, `poi`) and seeds `20260815`, `20260816` and `20260817`.

## Selected representative papers

1. Li et al., **DeepPROTACs**, *Nature Communications* 13, 7133 (2022), DOI `10.1038/s41467-022-34807-3`; official repository `https://github.com/fenglei104/DeepPROTACs`, commit `dfb62c4d137b7133d5ea4834a2615779630f2a52`.
2. Liu et al., **DegradeMaster**, *Bioinformatics* 41 (Suppl. 1), i342-i351 (2025), DOI `10.1093/bioinformatics/btaf191`; official repository `https://github.com/ABILiLab/DegradeMaster`, commit `aa149beaf067a051e070b3b281f7dba43c2f3e90`.

These papers were selected because DeepPROTACs represents an early supervised deep PROTAC predictor, whereas DegradeMaster represents a recent 3-D equivariant and semi-supervised predictor.

## Input and comparability boundary

The released record-level table contains canonical PROTAC SMILES, target and E3 identifiers, endpoints and labels, but it does not contain the protein-pocket structures, ligand files, linker segmentation and 3-D coordinates required by the official implementations. No official weights are therefore applied to the frozen benchmark.

The added results are explicitly labelled **paper-aligned reimplementation** (DeepPROTACs) and **protocol-constrained reimplementation** (DegradeMaster). Both use available Morgan radius-2 PROTAC fingerprints, target/E3 identity indicators and RDKit molecular descriptors. DeepPROTACs-aligned uses supervised P/N training. DegradeMaster-aligned uses a fixed confidence-threshold teacher/student pseudo-label enrichment step on the train-role U pool; the 3-D E(3)-equivariant encoder is not reproduced.

The Morgan-XGBoost row is the existing frozen baseline retrained on the same train/test roles for paired comparison. Results must not be interpreted as reproductions of the numerical results in either source paper, nor as evidence that one architecture is universally superior.

## Metrics and outputs

For every split and seed, report ROC-AUC, PR-AUC, MCC, balanced accuracy, Brier score and ten-bin ECE. The machine-readable outputs are:

- `reports/head_to_head_reimplementations_v4.json`
- `reports/head_to_head_reimplementations_v4.csv`
- `code/head_to_head_reimplementations.py`

The output JSON records the manifest hash, feature construction, official repository commits, missing inputs, deviations, per-seed sample counts and pseudolabel counts. Test-role labels are used only for evaluation.
