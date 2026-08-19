# 外部数据处理合规审查报告（TPDdb）

> 审查日期: 2026-08-18 ｜ 审查对象: `06_外部数据独立检验处理/` 全链路（脚本 + 产物 + 报告）
> 审查基准: `EXTERNAL_DATA_PREPROCESSING_SPEC.md` v1.0、主体标注规范 `PNU删失标注规范.md` v2.0
> 文档性质：本报告保留修复前 P0/P1/P2 的**历史审计发现**，用于说明问题、修复依据和重跑过程；第 9 节及文末记录修复后的状态。读者不得把第 1–4 节的修复前数字当作当前外部队列定义。
> 当前结论: **修复后通过 —— `eligible_nonoverlap = 16,808`，其中结构重复为 0 条，可进入冻结模型的结构独立候选外部评估。许可证、人工抽样复核和聚类敏感性分析仍为开放事项。**

---

## 0. 总判定（修复后）

| 级别 | 数量 | 当前状态 |
|---|---|---|
| P0（阻断） | 0 | 已修复 |
| P1（重要） | 0 | 已修复 |
| P2（报告缺陷） | 0 | 已修复 |
| 合规通过项 | 17 | 通过 |

---

## 1. P0-1（阻断）：2,124 条与主体结构完全相同的记录被标记为"独立不重叠"

### 事实（实测交叉表）

`audits/external_overlap_audit.csv` 的 `match_level × final_external_eligibility`：

| match_level | eligible_nonoverlap | exact_overlap | probable_overlap |
|---|---:|---:|---:|
| L1_structure（结构精确重复） | **2,124** | 0 | 0 |
| L3_experiment（结构+POI+E3+细胞+终点） | 0 | 36 | 0 |
| L4_potential（连接层潜在重复） | 0 | 0 | 43 |
| no_match | 16,808 | 0 | 0 |

即：`eligible_nonoverlap = 16,808（无匹配）+ 2,124（L1 结构重复但实验条件不同）= 18,932`。

### 根因

`scripts/external_data_preprocessing.py` line 904–909：

```python
elif match_level == "L1_structure":
    # L1 匹配但非 L3: 同化合物不同实验条件 -> 可进入外部验证
    if same_poi and same_e3 and same_cell and same_ep:
        elig = "exact_overlap"
    else:
        elig = "eligible_nonoverlap"
```

脚本把 **InChIKey / canonical SMILES 与主体完全相同的记录**（L1），只要 POI/E3/细胞系/终点不一致，就放行进入外部验证集。

### 为何不合规

- 违反 SPEC §10.4："正式外部主分析只保留 `eligible_nonoverlap`"，而 L1 结构精确重复本身即应视为 `exact_overlap`。
- 外部验证的独立性核心是**结构**不重叠：冻结模型学的是 Morgan fingerprint（结构）。结构相同的记录即使来自不同专利/不同实验，模型预测的也是"训练集里见过的结构"。
- 后果：若用当前 18,932 条 eligible 做外部验证，其中 **2,124 条（11.2%）是主体训练集见过的结构 → 结构泄漏，"独立外部验证"前提被破坏**。

### 修复

L1 结构精确重复一律判 `exact_overlap`（或 `structure_overlap`），无论实验条件是否相同；绝不混入主 eligible。可另建敏感性分析报告"含结构重复的外部结果"，但不得作为独立验证主结果。

---

## 2. P0-2（阻断）：Dmax 标签阈值与主体不一致（80/10 vs 70/20）

### 事实

`scripts/external_data_preprocessing.py` line 44–45：

```python
DMAX_P = 80.0     # Dmax >= 80% -> P
DMAX_N = 10.0     # Dmax <= 10% -> N
DC50_P = 100.0    # (与主体一致)
DC50_N = 1000.0   # (与主体一致)
```

标签映射 line 353–356 据此判定 Dmax≥80→P、Dmax≤10→N。

### 主体规则

`PNU删失标注规范.md` v2.0 §5.2：**Dmax ≥ 70% → P 候选，Dmax ≤ 20% → N 候选**。

80/10 是 **v0.1 已废弃阈值**（规范 §7 修订历史：v0.1 抄自 DeepPROTACs 惯例的 Dmax 80/10，已于 2026-08-16 废弃）。

### 为何不合规

- 违反 SPEC §9.3："外部数据预处理阶段不得擅自重新发明一套与主体数据完全不同的标签规则"。
- 后果：Dmax ∈ [70, 80) 的记录主体判 P、外部判 A/U；Dmax ∈ (10, 20] 的记录主体判 N、外部判 A/U。系统性改变 P/N 构成，破坏"端点对齐"，外部 T3/T4 的 P/N 标签与主体不可比。

### 修复

`DMAX_P = 70.0`、`DMAX_N = 20.0`，与主体 v2.0 完全对齐后重跑标签映射。

---

## 3. P1-1（重要）：obs_type 命名不一致导致报告/metadata 统计错误

### 事实

- `parse_dc50/parse_dmax` 对空值返回 `obs_type="endpoint-missing"`（**连字符**，line 128/133/213）。
- 端点审计表 `audits/external_endpoint_audit.csv` 用连字符统计：**正确**（DC50 endpoint-missing = 11,356 = 59.73%；Dmax = 13,272 = 69.81%）。
- 但 `external_processing_metadata.json` 的 `endpoint_quality` 与报告 §6 用 **`"endpoint_missing"`（下划线）** 比较（line 1063–1068），恒为 0。

### 后果

- 报告 §6.1/§6.2 显示 `endpoint_missing = 0`，而报告自己引用的 endpoint_audit.csv 说 59.7%/69.8% —— **报告自相矛盾**；
- DC50 各 obs_type 之和（7,655）远小于总数 19,011，占比失真。

### 修复

统一 obs_type 命名（建议全下划线或全连字符），修正 endpoint_quality 统计与报告 §6。

---

## 4. P2（报告缺陷，非阻断）

| # | 问题 | 位置 | 修复 |
|---|---|---|---|
| P2-1 | 报告 §1 "原始活性表 19,011" 错误，真实 raw_activity_table = **23,320** | `generate_report` line 1213（stats 缺 `row_counts`，回退到 total_records） | 把 row_counts 传入 stats |
| P2-2 | 报告 §3 "样本构建日志"与"数据合约 5 检查"为**空表** | 同上，`sample_construction_log`/`data_contract_checks` 在 metadata 顶层、不在 stats | 传入 stats |
| P2-3 | 报告署名"数据工程师(洗澄明)"为脚本写死的占位人名 | line 1467 | 改为实际执行者或移除 |
| P2-4 | 报告 §5 "重复结构 9,249" 表述易误导（是外部内部重复，非与主体重复） | 报告 §5 | 改为"外部记录级内部重复（同一化合物多条件）" |

---

## 5. 合规通过项（如实记录，不代表整体通过）

1. ✅ 原始文件原样保存，SHA-256 + manifest 齐全（`external_raw_manifest.csv`）；
2. ✅ 来源/版本/下载日期如实记录，license=UNKNOWN 如实填写（不猜测，符合 SPEC §5.2）；
3. ✅ 比较符号（> < ≥ ≤）正确拆分为 censored，未转 exact（line 162–176）；
4. ✅ 缺失值未当 N（"No Degradation/NDE → not_parseable，不自动标 N"，line 137–139）；
5. ✅ 非数值端点（等级字母 A/B/C/D、定性符号 +/++/…）→ not_parseable，不强行转数值；
6. ✅ T1–T4 资格字段布尔化（line 755–759）；
7. ✅ 单位换算（uM/pM/M→nM；比例→%）（line 110–122, 284）；
8. ✅ 本脚本不训练模型、不调参（符合 SPEC §12.5 禁令）；
9. ✅ `external_endpoint_audit.csv` 端点统计正确（连字符口径）；
10. ✅ 处理元数据 JSON 完整（SHA-256、row_counts、overlap_counts、task_eligibility 等）。

---

## 6. 需人工决策项（非脚本可自动判定）

1. **TPDdb license = UNKNOWN**：上游许可不明。论文若发布派生数据（clean 表、gold 标签映射），需先确认 TPDdb 使用条款；这是"数据发布"的阻塞项，非处理错误。
2. **DC50/Dmax 配对**：报告已知局限 §13.4 已声明"按(化合物+细胞系+靶标+后缀)配对，可能存在配对错误"，需人工抽样复核。
3. **细胞系规范化**：部分自由文本细胞系（报告 §13.3），需人工复核后才可用于跨细胞系分析。

---

## 7. 修复优先级与顺序

| 顺序 | 动作 | 级别 |
|---|---|---|
| 1 | 修正 L1 结构重复 → exact_overlap（重跑去重资格） | P0-1 |
| 2 | DMAX_P=70 / DMAX_N=20（对齐主体 v2.0，重跑标签映射） | P0-2 |
| 3 | 统一 obs_type 命名 + 修 endpoint_quality 统计 | P1-1 |
| 4 | 修 generate_report 的 stats 传递（row_counts/log/contract）+ 移除占位署名 | P2 |
| 5 | 重跑 `external_data_preprocessing.py`，重新生成全部产物与报告，重新核对交叉表与标签分布 | — |

修复完成后应复核的两个硬指标：
- `L1_structure` 记录在 `final_external_eligibility` 中 **0 条**落入 eligible_nonoverlap；
- 外部 P/N 标签分布与主体 v2.0（70/20）口径可比。

---

## 8. 结合主体 PROTAC-DB 3.0 数据集的实证复核（2026-08-18 二次确认）

复核方法：直接读取主体 `data/derived/protac_clean_record_level.csv`（15,535 条，`activity_evidence_v2` 分布 P=2,484 / N=534 / A=655 / U=11,862），验证 Dmax 阈值的**实际落盘行为**，并量化外部数据改阈值后的差异。

### 8.1 主体 Dmax 阈值的实证锚点（决定性证据）

只看"DC50 无信号（既不 ≤100 也不 ≥1000）+ Dmax 有精确值"的记录，按 Dmax 区间交叉 `activity_evidence_v2`：

| Dmax 区间 | 主体判 P | 主体判 N | 主体判 A | 结论 |
|---|---:|---:|---:|---|
| ≤10 | 0 | 37 | 0 | N 阈值含 ≤10 |
| **(10, 20]** | 0 | **36** | 0 | **N 阈值 = 20（非 10）** |
| (20, 30] | 0 | 10 | 41 | 20 为边界，之上转入 A |
| (30, 70) | 5 | 45 | 187 | 灰色区 |
| **[70, 80)** | **71** | 0 | 1 | **P 阈值 = 70（非 80）** |
| [80, 150] | 181 | 0 | 0 | P 阈值含 ≥80 |

**结论**：主体 v2.0 的 Dmax 阈值实为 **P≥70% / N≤20%**（与 `PNU删失标注规范.md` §5.2 完全一致）。外部脚本的 80/10 是 v0.1 废弃阈值，**确认必须改为 70/20**。

### 8.2 外部数据改 70/20 后的标签差异量化

对 19,011 条外部记录重放 `assign_evidence_external`（仅改 Dmax 阈值）：

| 口径 | P | N | A | U |
|---|---:|---:|---:|---:|
| 当前（80/10，错误） | 2,863 | 405 | 888 | 14,855 |
| **主体（70/20，正确）** | **2,994** | **481** | **681** | 14,855 |
| 差异 | **+131** | **+76** | **−207** | 0 |

共 **223 条**标签与主体口径不符，涉及区间：Dmax∈[70,80) 117 条、Dmax∈(10,20] 84 条、其余 22 条为删失/边界记录。这些记录在"独立外部验证"中若按 80/10 计，会把主体口径下的 P 少算 131 条、N 少算 76 条，系统性偏置外部 P/N 构成。

### 8.3 复核确认：哪些不用改

1. **DC50 阈值 100/1000**：外部脚本 `DC50_P=100.0`、`DC50_N=1000.0` 与主体 v2.0 §5.1 一致，**无需修改**；
2. **结构规范化口径**：主体表无原生 InChIKey，外部脚本用同一 RDKit `standardize_structure` 对主体 `smiles` 现场重算 canonical_smiles/InChIKey，与外部记录同一口径，**去重键可比，无需修改**（问题仅在资格判定 line 904–909 把 L1 结构重复放行）；
3. **P0-1 去重修复方向不变**：L1（canonical SMILES 或 InChIKey 相同）一律 `exact_overlap`，不因 POI/E3/细胞系/终点不同而放行。

### 8.4 最终修改清单（精确到脚本行）

| 位置 | 现值 | 改为 | 依据 |
|---|---|---|---|
| `external_data_preprocessing.py` line 44 | `DMAX_P = 80.0` | `DMAX_P = 70.0` | 主体 §5.2 + 8.1 实证 |
| `external_data_preprocessing.py` line 45 | `DMAX_N = 10.0` | `DMAX_N = 20.0` | 主体 §5.2 + 8.1 实证 |
| line 904–909（L1 分支） | 实验条件不同 → eligible_nonoverlap | 一律 `exact_overlap` | SPEC §10.4 + 结构泄漏 |
| line 1063–1068（endpoint_quality 统计） | `"endpoint_missing"`（下划线） | `"endpoint-missing"`（连字符，与 parse 一致） | P1-1 统计自洽 |
| `generate_report` stats 传递 | 缺 row_counts/log/contract | 补传 | P2-1/P2-2 |
| line 1467 署名 | "数据工程师(洗澄明)" | 移除或改实际执行者 | P2-3 |

修改后重跑 `external_data_preprocessing.py`，复核 8.2 表（P/N 应为 2,994/481）与 L1 结构重复 0 条落入 eligible_nonoverlap。

---

## 9. 修复执行与复核结果（2026-08-18 重跑后）

已按 §8.4 清单修改 `scripts/external_data_preprocessing.py` 并重跑，全部修复验证通过：

| 硬指标 | 预期 | 实测 | 判定 |
|---|---|---|---|
| 标签 P | 2,994 | **2,994** | ✅ |
| 标签 N | 481 | **481** | ✅ |
| 标签 A / U | 681 / 14,855 | **681 / 14,855** | ✅ |
| `exact_overlap` | 2,160（L1 结构重复全部排除） | **2,160** | ✅ |
| `eligible_nonoverlap` | 16,808（0 条结构重复） | **16,808** | ✅ |
| `probable_overlap` | 43 | **43** | ✅ |
| L1 结构重复落入 eligible_nonoverlap | 0 | **0** | ✅ |
| metadata `dc50_endpoint_missing` | 11,356 | **11,356** | ✅ |
| metadata `dmax_endpoint_missing` | 13,272 | **13,272** | ✅ |
| metadata `row_counts.raw_activity_table` | 23,320 | **23,320** | ✅ |
| 报告含占位署名"洗澄明" | 移除 | **已移除** | ✅ |
| 报告 §6.1 endpoint-missing | 11,356 (59.7%) | **11,356 (59.7%)** | ✅ |

### 修复后最终资格分布

| final_external_eligibility | 数量 | 含义 |
|---|---:|---|
| `eligible_nonoverlap` | 16,808 | 结构层面与主体无重叠，可进入正式外部验证 |
| `exact_overlap` | 2,160 | 与主体 InChIKey/canonical SMILES 相同，**排除**（结构泄漏） |
| `probable_overlap` | 43 | 连接层潜在重复，需人工复核 |

### 合规状态更新

- P0-1（结构泄漏）→ **已修复**：L1 结构精确重复一律 `exact_overlap`，0 条泄漏。
- P0-2（Dmax 阈值 80/10）→ **已修复**：改 70/20，标签分布对齐主体 v2.0。
- P1-1（obs_type 命名）→ **已修复**：`endpoint-missing` 统一连字符，统计自洽。
- P2-1/2/3/4（报告缺陷）→ **已修复**：row_counts 传参、样本构建日志非空、署名移除、表述修正。

### 剩余人工决策项（不变）

1. TPDdb `license = UNKNOWN`：论文发布派生数据前须确认上游许可；
2. DC50/Dmax 配对（化合物+细胞系+靶标+后缀）需人工抽样复核；
3. 细胞系自由文本规范化需人工复核；
4. 16,808 条 `eligible_nonoverlap` 中 P/N 构成（T3 资格）需在主分析流程前单独统计，并确认"结构独立"后剩余 P/N 的统计力。

**结论：外部数据预处理现已合规，`eligible_nonoverlap = 16,808` 条可作为结构独立的候选外部验证集，进入下一阶段冻结模型外部检验。**




## Final scheme B execution status (2026-08-18)

The approved frozen-model external evaluation is complete. The retained TPDdb-derived cohort contains 16,808 `eligible_nonoverlap` records; exact/probable overlap exclusion is complete, L1 structure leakage into the retained cohort is zero, and the external cohort was not used for model fitting, tuning, threshold selection or calibration. Formal results are in `reports/EXTERNAL_VALIDATION_ANALYSIS_REPORT.md`; machine-readable outputs are in `results/`. The result is reported as structure-disjoint, TPDdb-derived external evaluation, not prospective, laboratory-independent or fully time-independent validation. TPDdb licensing remains `UNKNOWN` and must be resolved before public redistribution.

