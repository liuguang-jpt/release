# Adjudicated internal-consistency annotation-set validation (v3)

> Generated: 2026-08-17 12:45:05  
> Frozen file: `data/derived/gold_set_annotations/gold_final.csv`  
> Record bootstrap: 10,000 replicates, percentile 95% CI.

## 1. Annotator agreement

- A/B agreement: **127/132 = 96.2%**.
- Cohen's κ: **0.948**.
- A/B disagreements: **5**.
- Final equals annotator A: 127/132; final equals annotator B: **132/132**.
- Seven non-empty notes do not mean seven annotation disagreements: five notes correspond to A/B disagreements and two document source-data corrections on records already labeled P/P.

| Annotator A \ Annotator B | P | N | A | U |
|---|---:|---:|---:|---:|
| P | 42 | 0 | 2 | 0 |
| N | 0 | 19 | 0 | 0 |
| A | 0 | 3 | 36 | 0 |
| U | 0 | 0 | 0 | 30 |

## 2. Automatic rule versus final four-class label

- Agreement/accuracy: **118/132 = 0.894 (95% CI 0.841–0.947)**.
- Macro-F1: **0.899 (95% CI 0.843–0.946)**.

| Final label \ Automatic rule | P | N | A | U |
|---|---:|---:|---:|---:|
| P | 41 | 0 | 0 | 1 |
| N | 0 | 19 | 0 | 3 |
| A | 10 | 0 | 28 | 0 |
| U | 0 | 0 | 0 | 30 |

### Per-class performance

| Class | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Support |
|---|---:|---:|---:|---:|
| P | 0.804 (95% CI 0.690–0.906) | 0.976 (95% CI 0.919–1.000) | 0.882 (95% CI 0.804–0.944) | 42 |
| N | 1.000 (95% CI 1.000–1.000) | 0.864 (95% CI 0.700–1.000) | 0.927 (95% CI 0.824–1.000) | 22 |
| A | 1.000 (95% CI 1.000–1.000) | 0.737 (95% CI 0.585–0.872) | 0.848 (95% CI 0.738–0.932) | 38 |
| U | 0.882 (95% CI 0.767–0.973) | 1.000 (95% CI 1.000–1.000) | 0.938 (95% CI 0.868–0.986) | 30 |

### Failure taxonomy (14 mismatches)

| Failure type | Count | Interpretation |
|---|---:|---|
| `interval_overcalling` | 8 | An interval endpoint was treated as sufficiently positive although the expert label remained ambiguous. |
| `pctdeg_signal_ignored` | 4 | DC50/Dmax were absent, but percent-degradation evidence supported P or N while the rule returned U. |
| `borderline_or_combined_evidence_overcalling` | 1 | Moderate/borderline combined evidence was overcalled as P. |
| `cross_endpoint_conflict` | 1 | Strong DC50/Dmax and percent-degradation signals conflicted, but the OR rule returned P. |

The complete record-level failure table is a locally generated diagnostic output (`data/derived/gold_rule_failures_v3.csv`) and is not distributed in this release; the aggregate taxonomy is reproduced above.

## 3. Frozen-model transfer check on final P/N records

- Training records: 2,948 (2,433 P; 515 N), with all 132 internal-consistency record IDs excluded.
- Test records: 64 (42 P; 22 N).
- Model and threshold were fixed: Morgan-2048 + XGBoost; decision threshold = 0.5; no tuning on the internal-consistency sample.

| Metric | Estimate (95% record-bootstrap CI) |
|---|---:|
| roc_auc | 0.887 (95% CI 0.761–0.982) |
| pr_auc | 0.896 (95% CI 0.775–0.992) |
| accuracy_at_0.5 | 0.906 (95% CI 0.828–0.969) |
| macro_f1_at_0.5 | 0.888 (95% CI 0.789–0.964) |
| balanced_accuracy_at_0.5 | 0.864 (95% CI 0.762–0.952) |
| mcc_at_0.5 | 0.798 (95% CI 0.648–0.930) |
| brier | 0.107 (95% CI 0.053–0.173) |

This is a **record-held-out annotation-set transfer check**, not external validation: the annotations were sampled from the same underlying database, and only record IDs—not all related scaffolds/publications/targets—were excluded.

## 4. Reporting items requiring author confirmation

> **状态（2026-08-17）：已闭合** —— 全部四项已由 `GOLD_ANNOTATION_PROTOCOL.md`（P1 闭合文档）回答。摘要如下：

1. ~~Supply the actual annotator and adjudicator roles/qualifications, independence, blinding procedure, and frozen guideline version~~ → 协议 §1–§4。A=第一作者（规则开发者）；B=AI 标注助手（完全盲）；仲裁=第一作者本人，**非独立第三人**；指南 v2.0 冻结于 2026-08-16。
2. ~~Explain why all five adjudicated disagreement labels equal annotator B's labels~~ → 协议 §5。127 条共识 + 5 条按 v2.0 规范字面裁定（3 条 N 阈值触发、2 条信号冲突）全部落向 B；2 条数据修正后 A/B 一致。final≡B 是共识+规范仲裁的必然，非复制 B 列。
3. ~~Explain why the planned sample size of 200 resulted in 132 completed records~~ → 协议 §7。配额公式每格 cap=8（min(8, max(2, round(200·n/N))))，配额 121 + 稀有删失保底 11 = 132；"200" 仅存于文件名，68 条从未进入模板。
4. State that the stratified internal-consistency aggregate is not automatically a prevalence-weighted estimate for the full database. → 仍为论文 Limitation 8，不变。

## 5. Claim boundary

The 96.2% agreement and κ=0.948 validate **annotator concordance**. The 89.4% rule accuracy validates the **automatic rule against final labels**. The frozen P/N experiment evaluates **model transfer on the final P/N subset**. These are separate questions and must not be combined into one 'independent validation' claim.

## 6. Rule agreement computed separately against A, B, and final labels (P1 Q9)

Computed by `code/gold_rule_agreement.py`; machine-readable copy: `data/derived/gold_rule_agreement_by_annotator.json (locally generated; aggregate values are reported here)`. Automatic rule = `activity_evidence_v2` (v2.0), n = 132.

| Label source | Accuracy | Cohen's κ | Macro-F1 |
|---|---:|---:|---:|
| Annotator A | 0.8712 | 0.8234 | 0.8674 |
| Annotator B | 0.8939 | 0.8554 | 0.8986 |
| Final label | 0.8939 | 0.8554 | 0.8986 |

- final ≡ B, so rule-vs-final equals rule-vs-B by construction (not a duplicated report).
- The A/B gap (0.0227 accuracy, 3 records) mirrors the five adjudicated disagreements: on four of them the rule agrees with B (three N-threshold triggers, one endpoint conflict), and on 11650 the rule (P) agrees with A while B/final are A.
- All three viewpoints remain in the strong-agreement range (κ ≥ 0.82); conclusions are robust to the label-source choice.
- Full annotation-chain documentation (roles, qualifications, independence, blinding, guideline version, adjudication records, 200→132 explanation, model identity for the frozen AUC) is in `GOLD_ANNOTATION_PROTOCOL.md`.


