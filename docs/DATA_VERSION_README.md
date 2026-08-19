# 数据版本与复现说明 — release v1.0.0

## 1. 数据来源与边界

- 上游来源：PROTAC-DB 3.0，项目访问日期为 2026-08-15。
- 本发布包包含 PROTAC-DB 派生的记录级分析材料，但不包含原始数据库快照。
- 派生记录仍受上游数据库条款约束；本项目不对上游记录授予额外权利。
- TPDdb 外部检验只发布聚合指标和重建脚本，不发布 TPDdb 记录级队列、标签或逐记录预测。

## 2. 当前冻结数据

| 文件 | 内容 |
|---|---|
| `data/derived/protac_clean_record_level.csv` | 15,535 行记录级主表；包含冻结的 `activity_evidence_v2` 标签 |
| `data/derived/data_dictionary.csv` | 字段字典 |
| `data/derived/protac_split_groups.csv` | scaffold、publication、POI 等分组键 |
| `data/derived/split_manifest_v3.csv` | 三个固定种子、四种划分制度下的冻结角色 |
| `data/derived/split_manifest_v3_audit.json` | 角色和组重叠审计 |
| `data/derived/benchmark_contract_v3.json` | 任务资格、划分、种子和 artefact 哈希契约 |
| `data/derived/morgan_fp_2048_index.csv` | Morgan 特征记录索引 |
| `data/derived/morgan_fp_2048_meta.json` | Morgan 特征参数和哈希元数据；`.npy` 特征矩阵由脚本重建 |
| `data/derived/protac_annotation_150.csv` | 150 条校准标注 |
| `data/derived/gold_set_annotations/gold_sample_132.csv` | 132 条冻结抽样清单（目录名 `gold_set_annotations` 为历史路径） |
| `data/derived/gold_set_annotations/gold_final.csv` | 132 条仲裁后内部一致性标签 |
| `data/raw/raw_data_manifest.csv` | 原始源文件哈希清单，不含原始数据 |
| `data/external/` | TPDdb-derived 聚合外部检验指标和模型清单 |

## 3. 标签语义

冻结标签列为 `activity_evidence_v2`：

```text
P = 2,484
N = 534
A = 655
U = 11,862
总计 = 15,535
```

规则概要：

- DC50 ≤ 100 nM 或 Dmax ≥ 70% 提供 P 信号；
- DC50 ≥ 1,000 nM 或 Dmax ≤ 20% 提供 N 信号；
- 强 P/N 信号冲突时标记 A；
- 没有可用判定信号时标记 U；
- Dmax > 150 或 DC50 ≤ 0 的异常值只使对应终点失效，不使整条记录自动失效；
- exact、left-censored、right-censored、interval-censored 和 endpoint-missing 状态分别保存。

这些标签是数据库字段的操作性派生结果，不是条件完整的生物学真值。

## 4. 内部一致性样本

原计划目标为 200 条，但冻结配额公式的每层上限使实际配额抽样为 121 条；再加入 11 条稀有删失保护记录，最终得到 132 条。不存在“漏标的 68 条”。

- A/B 一致：127/132 = 96.2%；
- Cohen's κ = 0.9483；
- A/B 分歧：5 条；
- 仲裁者：第一作者本人，不是独立第三方；
- 五条分歧均按冻结规则裁定为 B 标签。

因此正确名称是“132-record adjudicated internal-consistency sample”，不是独立专家 gold standard。

## 5. 任务资格数

| 任务 | 资格条件 | 冻结数量 |
|---|---|---:|
| T1 | exact pDC50 | 2,625 |
| T2 | exact Dmax | 1,801 |
| T3 | P/N | 3,018 |
| T4 | P/N/U | 14,880 |
| censoring-aware pDC50 | exact 或单侧删失 | 3,444 |
| temporal year-observed set | DOI 年份可用于 cutoff | 6,244 |

角色分配先于任务过滤，避免按终点可用性重新划分数据。

## 6. TPDdb-derived 外部检验

外部处理保留 16,808 条 `eligible_nonoverlap` 记录，并排除 2,160 条 exact overlap 和 43 条 probable overlap。T3 有 3,240 条 P/N 记录。十二个冻结 T3 模型的均值为：ROC-AUC 0.586、PR-AUC 0.887、balanced accuracy 0.517、MCC 0.063、specificity 0.057。T1 MAE 0.813、R² = -0.435；T2 MAE 24.658、R² = -2.479。

这属于 structure-disjoint、TPDdb-derived 的有边界外部证据；不是前瞻性、实验室独立或完全时间独立验证。

## 7. 复现顺序

在仓库根目录建立干净环境：

```bash
pip install -r requirements.txt
```

如需运行神经 PU 或删失感知模型，再按本机平台安装 CPU/CUDA 版 PyTorch，并执行：

```bash
pip install -r requirements-optional-torch.txt
```

主流程：

```bash
python code/etl_protac.py
python code/relabel_semantics.py
python code/build_split_groups.py
python code/build_morgan_features.py
python code/make_split_manifest.py
python code/audit_split_manifest.py
python code/baseline_pipeline.py
python code/pu_pipeline.py
python code/calib_sensitivity_pipeline.py
python code/censored_eval_pipeline.py
```

只检查计划、不实际运行：

```bash
python code/run_all.py --dry-run --skip-etl --skip-shortcut-controls --skip-slow-bootstrap
```

外部流程要求用户自行取得符合上游条款的 TPDdb 文件，并放在 `external_data/`：

```bash
python external_code/external_data_preprocessing.py
python external_code/build_external_validation_cohort.py
python external_code/build_external_features.py
python external_code/run_external_frozen_models.py
python external_code/summarize_external_validation.py
python external_code/build_external_validation_report.py
```

## 8. 已知限制

1. 细胞系和处理时间大量来自自由文本解析，存在缺失和抽取误差。
2. 定量终点稀疏；任务样本数不是独立分子数。
3. 随机划分可能受相关化学系列影响，应优先结合 scaffold、publication 和 POI 分组结果解读。
4. DOI-derived 年份只覆盖 40.2% 记录，内部 post-cutoff 分析不能视为外部验证。
5. Morgan `.npy` 特征矩阵未随包固定分发，但其参数、索引与预期哈希已记录，可由脚本重建。
6. 发布前必须再次核验 GitHub、Zenodo、上游许可和目标期刊政策。

