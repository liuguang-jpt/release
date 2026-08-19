# -*- coding: utf-8 -*-
from pathlib import Path
import os
import json, hashlib
import pandas as pd
import numpy as np

BASE=Path(__file__).resolve().parents[1]
PROC=BASE/'processed'; RESULTS=BASE/'results'; REPORTS=BASE/'reports'
REPORTS.mkdir(exist_ok=True)

def fmt(x, d=3):
    if pd.isna(x): return 'NA'
    return f'{float(x):.{d}f}'

def mean_sd(df, metric):
    s=pd.to_numeric(df[metric], errors='coerce').dropna()
    return f'{s.mean():.3f} ± {s.std(ddof=1):.3f}'

def split_table(df, metrics):
    lines=[]
    for split,g in df.groupby('split_regime', sort=True):
        vals=[f'{split}']+[mean_sd(g,m) for m in metrics]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def ci_span(df, task, metric, unit):
    x=df[(df.task==task)&(df.metric==metric)&(df.cluster_unit==unit)]
    return f"median point={x.point_estimate.median():.3f}; model-specific 95% bootstrap interval span [{x.q025.min():.3f}, {x.q975.max():.3f}]"

def conflict_summary(c, protocol, metric):
    x=c[(c.protocol==protocol)&(c.metric==metric)].value
    return f'{x.mean():.3f} ± {x.std(ddof=1):.3f}'

meta=json.loads((BASE/'reports'/'external_validation_cohort_metadata.json').read_text(encoding='utf-8'))
summary=pd.read_csv(PROC/'external_validation_cohort_summary.csv')
t3=pd.read_csv(RESULTS/'external_T3_metrics_by_split_seed.csv')
t1=pd.read_csv(RESULTS/'external_T1_metrics_by_split_seed.csv')
t2=pd.read_csv(RESULTS/'external_T2_metrics_by_split_seed.csv')
struct=pd.read_csv(RESULTS/'external_structure_cluster_bootstrap.csv')
patent=pd.read_csv(RESULTS/'external_patent_cluster_bootstrap.csv')
conf=pd.read_csv(RESULTS/'external_conflict_label_sensitivity.csv')
cohort=pd.read_csv(PROC/'external_validation_cohort.csv', low_memory=False)

def row(task): return summary[(summary.scope=='eligible_nonoverlap')&(summary.task==task)].iloc[0]
r1,r2,r3,r4=[row(x) for x in ['T1_pdc50','T2_dmax','T3_pn','T4_pnu']]
t4=cohort[cohort.t4_eligible.astype(str).str.lower()=='true'].activity_evidence_external_v1.value_counts()

out=[]
out.append('# TPDdb 外部独立检验方案 B：最终分析报告')
out.append('')
out.append('**报告日期：2026-08-18**  ') 
out.append('**分析状态：已完成冻结模型外部评估、稳健性分析和报告汇总；论文同步更新待最终审计确认。**')
out.append('')
out.append('## 1. 执行摘要')
out.append('')
out.append(f'- 按已批准的方案 B，使用 **random / scaffold / publication / POI 四种内部拆分协议 × 20260815、20260816、20260817 三个固定 seed**，共训练并冻结 **36 个模型（T1/T2/T3 各 12 个）**，外部数据只用于一次性评估。')
out.append(f'- 外部队列来自 TPDdb 快照（版本标记 `2025-08-31`），清洗记录 {meta["n_clean_records"]:,} 条；排除 exact overlap {meta["n_exact_overlap_excluded"]:,} 条、probable overlap {meta["n_probable_overlap_excluded"]:,} 条后，得到 {meta["n_cohort_records"]:,} 条 `eligible_nonoverlap` 主队列。L1 结构重复进入主队列为 {meta["n_l1_in_cohort"]} 条。')
out.append(f'- **T3 主结果：**12 个冻结模型 ROC-AUC **{t3.roc_auc.mean():.3f} ± {t3.roc_auc.std(ddof=1):.3f}**，PR-AUC **{t3.pr_auc.mean():.3f} ± {t3.pr_auc.std(ddof=1):.3f}**。然而在固定阈值 0.5 下 specificity 仅 **{t3.specificity.mean():.3f}**，balanced accuracy **{t3.balanced_accuracy.mean():.3f}**，说明排序能力高于阈值分类能力，且受外部队列 P/N 比例（{int(r3.n_P)}:{int(r3.n_N)}）影响明显。')
out.append(f'- **T1 补充结果：**MAE **{t1.mae.mean():.3f} ± {t1.mae.std(ddof=1):.3f}**，RMSE **{t1.rmse.mean():.3f} ± {t1.rmse.std(ddof=1):.3f}**，R² **{t1.r2.mean():.3f} ± {t1.r2.std(ddof=1):.3f}**，Spearman **{t1.spearman.mean():.3f} ± {t1.spearman.std(ddof=1):.3f}**。')
out.append(f'- **T2 补充结果：**MAE **{t2.mae.mean():.3f} ± {t2.mae.std(ddof=1):.3f}**，RMSE **{t2.rmse.mean():.3f} ± {t2.rmse.std(ddof=1):.3f}**，R² **{t2.r2.mean():.3f} ± {t2.r2.std(ddof=1):.3f}**，Spearman **{t2.spearman.mean():.3f} ± {t2.spearman.std(ddof=1):.3f}**。整体未显示可迁移的定量回归性能，且 T2 在 scaffold/publication 协议下波动很大。')
out.append('- T4 不作为主要推断，仅报告 P/N/U 构成；U 占比约 79.9%，因此不把 T4 当作已完成的三分类外部模型验证。')
out.append('')
out.append('## 2. 数据来源、版本与合规边界')
out.append('')
out.append('- **来源：**TPDdb；源文件为 `PROTAC_main_table.txt` 与 `PROTAC_activity.txt`，下载地址由原始 manifest 记录。')
out.append('- **快照版本：**`2025-08-31`（按外部预处理文件中的 `source_version` 字段记录）。')
out.append('- **许可证：**当前项目记录为 `UNKNOWN`。因此本报告支持研究内部可复现分析，但不自动授予数据再分发权；对外发布前仍需确认 TPDdb 的许可、引用和再分发条款。')
out.append('- **外部性的准确表述：**本分析是 `TPDdb-derived`, `source-independent` and `structure-disjoint` external evaluation cohort；不能称为前瞻性验证、实验室独立验证或完全时间独立验证。原因包括 TPDdb 的底层来源以专利为主、出版年份覆盖不完整、与主体数据库的来源级重合无法完全排除。')
out.append('')
out.append('## 3. 外部队列构建与去重')
out.append('')
out.append('| 阶段 | 记录数 | 说明 |')
out.append('|---|---:|---|')
out.append(f'| 清洗后记录 | {meta["n_clean_records"]:,} | 保留可解析结构、基本活动字段和来源追踪字段 |')
out.append(f'| exact overlap 排除 | {meta["n_exact_overlap_excluded"]:,} | 与主体数据按严格结构/键匹配排除 |')
out.append(f'| probable overlap 排除 | {meta["n_probable_overlap_excluded"]:,} | 按重叠审计规则标记为可能重复并排除 |')
out.append(f'| 正式主队列 | {meta["n_cohort_records"]:,} | `final_external_eligibility == eligible_nonoverlap` |')
out.append('')
out.append('任务资格：')
out.append('')
out.append('| 任务 | 外部记录 | 唯一结构 | connectivity 结构 | 专利数 | P | N | U | A |')
out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for rr in [r1,r2,r3,r4]:
    out.append(f'| {rr.task} | {int(rr.n_records):,} | {int(rr.n_structures):,} | {int(rr.n_connectivity_structures):,} | {int(rr.n_patents):,} | {int(rr.n_P):,} | {int(rr.n_N):,} | {int(rr.n_U):,} | {int(rr.n_A):,} |')
out.append('')
out.append(f'T3 队列包含 {meta["t3_unique_structures"]:,} 个唯一结构，其中 {meta["t3_conflict_structures"]} 个结构同时出现 P/N 记录，共涉及 {meta["t3_conflict_records"]} 条记录；因此冲突标签敏感性分析被预先纳入。')
out.append('')
out.append('## 4. 特征与冻结模型协议')
out.append('')
out.append('- 分子表示：canonical full-PROTAC SMILES 的 Morgan fingerprint，radius=2，2048 bits，`float32`；RDKit `2026.03.5`；外部记录 invalid SMILES=0、zero vectors=0。')
out.append('- 模型：XGBoost，`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`。')
out.append('- 模型训练只使用主体数据集的 `all_records_3way` manifest 中 `train` 角色；外部队列未参与训练、调参、特征选择、阈值选择或概率校准。')
out.append('- T3 主阈值固定为 0.5；报告 ROC-AUC、PR-AUC、balanced accuracy、MCC、Brier、10-bin ECE、sensitivity、specificity。T1/T2 报告 MAE、RMSE、R² 和 Spearman。')
out.append('')
out.append('## 5. T3 P/N 外部主结果')
out.append('')
out.append('| 拆分协议 | ROC-AUC | PR-AUC | balanced accuracy | MCC | Brier | ECE | sensitivity | specificity |')
out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
out.append(split_table(t3,['roc_auc','pr_auc','balanced_accuracy','mcc','brier','ece_10bin','sensitivity','specificity']))
out.append(f'| **总体（12模型）** | **{t3.roc_auc.mean():.3f} ± {t3.roc_auc.std(ddof=1):.3f}** | **{t3.pr_auc.mean():.3f} ± {t3.pr_auc.std(ddof=1):.3f}** | {t3.balanced_accuracy.mean():.3f} ± {t3.balanced_accuracy.std(ddof=1):.3f} | {t3.mcc.mean():.3f} ± {t3.mcc.std(ddof=1):.3f} | {t3.brier.mean():.3f} ± {t3.brier.std(ddof=1):.3f} | {t3.ece_10bin.mean():.3f} ± {t3.ece_10bin.std(ddof=1):.3f} | {t3.sensitivity.mean():.3f} ± {t3.sensitivity.std(ddof=1):.3f} | {t3.specificity.mean():.3f} ± {t3.specificity.std(ddof=1):.3f} |')
out.append('')
out.append('解释：外部 T3 的 ROC-AUC 约为 0.59，明显低于主体数据中随机拆分的 0.919、scaffold 的 0.905、publication 的 0.742 和 POI 的 0.792（这些内部数字来自当前论文稿的既有结果表）。外部 PR-AUC 仍较高，主要因为外部 P/N 不平衡且 P 占 85.4%；因此不能用 PR-AUC 单独宣称强外部分类器。固定阈值结果显示模型几乎偏向预测 P，specificity 很低，提示外部标签基线、阈值迁移和支持域差异共同影响应用性能。')
out.append('')
out.append('## 6. T1 pDC50 与 T2 Dmax 补充结果')
out.append('')
out.append('| 任务/拆分 | MAE | RMSE | R² | Spearman |')
out.append('|---|---:|---:|---:|---:|')
for name,df in [('T1 pDC50',t1),('T2 Dmax',t2)]:
    for split,g in df.groupby('split_regime',sort=True):
        out.append(f'| {name} / {split} | {g.mae.mean():.3f} ± {g.mae.std(ddof=1):.3f} | {g.rmse.mean():.3f} ± {g.rmse.std(ddof=1):.3f} | {g.r2.mean():.3f} ± {g.r2.std(ddof=1):.3f} | {g.spearman.mean():.3f} ± {g.spearman.std(ddof=1):.3f} |')
    out.append(f'| **{name} / 总体** | **{df.mae.mean():.3f} ± {df.mae.std(ddof=1):.3f}** | **{df.rmse.mean():.3f} ± {df.rmse.std(ddof=1):.3f}** | **{df.r2.mean():.3f} ± {df.r2.std(ddof=1):.3f}** | **{df.spearman.mean():.3f} ± {df.spearman.std(ddof=1):.3f}** |')
out.append('')
out.append('结论：T1/T2 在结构独立外部队列上均未达到稳定的正向 R²；Spearman 仅为弱相关，T2 的 RMSE/R² 在 publication/scaffold 协议下尤为不稳定。外部检验因此不支持把当前 Morgan+XGBoost 回归基线表述为可直接迁移的定量预测器。')
out.append('')
out.append('## 7. T4 P/N/U 探索性结果')
out.append('')
out.append(f'- T4 eligible 记录：{len(cohort[cohort.t4_eligible.astype(str).str.lower()=="true"]):,}。')
out.append(f'- P={int(t4.get("P",0)):,}（{t4.get("P",0)/len(cohort[cohort.t4_eligible.astype(str).str.lower()=="true"]):.1%}），N={int(t4.get("N",0)):,}（{t4.get("N",0)/len(cohort[cohort.t4_eligible.astype(str).str.lower()=="true"]):.1%}），U={int(t4.get("U",0)):,}（{t4.get("U",0)/len(cohort[cohort.t4_eligible.astype(str).str.lower()=="true"]):.1%}）。')
out.append('- 方案 B 未训练 T4 三分类冻结模型；T4 仅用于展示外部队列中“未知/未充分观测”状态的规模，不纳入主要推断。')
out.append('')
out.append('## 8. Cluster bootstrap 与冲突标签敏感性')
out.append('')
out.append('Bootstrap 次数为 1,000；每个 split/seed/task 单独进行，以 `inchikey` 或 `source_patent_id` 为重采样单位。下表的区间范围是 12 个模型各自 95% cluster-bootstrap 区间的最小下限至最大上限，不应误读为单一 pooled confidence interval。')
out.append('')
out.append('| 任务 | 聚类单位 | 指标 | 12模型 point median | 95%区间下限范围 | 95%区间上限范围 |')
out.append('|---|---|---|---:|---:|---:|')
for task,metric in [('T3_pn_clf','roc_auc'),('T3_pn_clf','pr_auc'),('T1_pdc50_reg','mae'),('T1_pdc50_reg','r2'),('T2_dmax_reg','mae'),('T2_dmax_reg','r2')]:
    for unit, label in [('inchikey','structure'),('source_patent_id','patent')]:
        d=(struct if unit=='inchikey' else patent); x=d[(d.task==task)&(d.metric==metric)]
        out.append(f'| {task} | {label} | {metric} | {x.point_estimate.median():.3f} | {x.q025.min():.3f} | {x.q975.max():.3f} |')
out.append('')
out.append('T3 冲突敏感性（12 个模型均值 ± SD）：')
out.append('')
out.append('| 标签协议 | 记录数/结构数 | ROC-AUC | PR-AUC | balanced accuracy | MCC |')
out.append('|---|---:|---:|---:|---:|---:|')
for p,name in [('record_level_all','记录级全部'),('exclude_conflict_structures','排除冲突结构'),('structure_majority_label','结构级多数标签')]:
    x=conf[conf.protocol==p].iloc[0]
    out.append(f'| {name} | {int(x.n_records):,} / {int(x.n_structures):,} | {conflict_summary(conf,p,"roc_auc")} | {conflict_summary(conf,p,"pr_auc")} | {conflict_summary(conf,p,"balanced_accuracy")} | {conflict_summary(conf,p,"mcc")} |')
out.append('')
out.append(f'冲突结构共 {meta["t3_conflict_structures"]} 个。排除冲突结构后 T3 ROC-AUC 均值由 {conflict_summary(conf,"record_level_all","roc_auc")} 变为 {conflict_summary(conf,"exclude_conflict_structures","roc_auc")}；结构级多数标签协议在排除 75 个平票结构后，ROC-AUC 为 {conflict_summary(conf,"structure_majority_label","roc_auc")}。结论方向没有改变：外部分类排序能力有限，且标签冲突不是唯一解释。')
out.append('')
out.append('## 9. 与主体数据结果的比较和论证方式')
out.append('')
out.append('1. **论证链第一步：**主体数据内部四种拆分下性能差异，说明评估契约会改变表观性能。')
out.append('2. **第二步：**外部队列经 exact/probable overlap 排除且结构层面无 L1 泄漏，使用同一 Morgan+XGBoost 协议直接评估，检验模型能否跨数据库支持域迁移。')
out.append('3. **第三步：**固定 12 个模型而非选取外部最佳模型，避免外部调参造成乐观偏差；通过 structure/patent bootstrap 检查重复结构和专利来源聚类对不确定性的影响。')
out.append('4. **第四步：**通过冲突标签敏感性检验排除“少数矛盾结构完全驱动结论”的解释；通过 T4 分布说明大量记录缺乏可映射到 P/N 的数值证据。')
out.append('5. **最终结论：**当前模型在主体数据内部的高表现不能直接外推到 TPDdb-derived 外部队列；T3 仍保留有限排序信号，但阈值分类、T1 和 T2 定量迁移均不稳定。')
out.append('')
out.append('## 10. 论文中可使用的结论边界')
out.append('')
out.append('- 可以写："In a structure-disjoint, TPDdb-derived external evaluation cohort, the frozen Morgan–XGBoost classifier retained modest ranking ability, whereas fixed-threshold discrimination and quantitative endpoint transfer were limited."')
out.append('- 不应写："The model was prospectively validated", "laboratory-independent validation", "fully independent external validation" 或 "the model generalizes across all PROTAC datasets"。')
out.append('- T3 应同时报告 ROC-AUC、PR-AUC、balanced accuracy、MCC、sensitivity/specificity，并明确外部队列 P/N 不平衡和阈值固定条件。')
out.append('- T1/T2 应报告为补充外部检验，不应把负 R² 解释成“模型反向预测”或因单次异常 RMSE 作因果归因。')
out.append('')
out.append('## 11. 可复现文件和哈希')
out.append('')
out.append('| 文件 | 用途 |')
out.append('|---|---|')
out.append('| `processed/external_validation_cohort.csv` | 冻结外部主队列 |')
out.append('| `processed/external_morgan_fp_2048.npy` | 外部 Morgan 特征 |')
out.append('| `results/external_predictions.csv` | 88,896 条冻结模型预测 |')
out.append('| `results/external_model_manifest.csv` | 36 个模型的训练与哈希记录 |')
out.append('| `results/external_T3_metrics_by_split_seed.csv` | T3 逐 split/seed 指标 |')
out.append('| `results/external_T1_metrics_by_split_seed.csv` | T1 逐 split/seed 指标 |')
out.append('| `results/external_T2_metrics_by_split_seed.csv` | T2 逐 split/seed 指标 |')
out.append('| `results/external_structure_cluster_bootstrap.csv` | inchikey cluster bootstrap |')
out.append('| `results/external_patent_cluster_bootstrap.csv` | source_patent_id cluster bootstrap |')
out.append('| `results/external_conflict_label_sensitivity.csv` | T3 冲突标签敏感性 |')
out.append('| `scripts/build_external_validation_cohort.py` | 外部队列构建 |')
out.append('| `scripts/build_external_features.py` | 外部特征构建 |')
out.append('| `scripts/run_external_frozen_models.py` | 冻结模型训练和预测 |')
out.append('| `scripts/summarize_external_validation.py` | 指标汇总 |')
out.append('| `scripts/external_cluster_bootstrap.py` | 聚类 bootstrap 和冲突分析 |')
out.append('')
out.append('关键哈希：')
out.append('')
out.append(f'- cohort SHA-256: `{meta["cohort_sha256"]}`')
out.append(f'- exclusion log SHA-256: `{meta["exclusion_log_sha256"]}`')
out.append(f'- Morgan feature SHA-256: `215535f3f29e036c8c262bebb35464ba4b6a605f971ec9b6069da3686ecb7114`')
out.append(f'- predictions SHA-256: `{hashlib.sha256((RESULTS/"external_predictions.csv").read_bytes()).hexdigest()}`')
out.append('')
out.append('## 12. 当前剩余事项')
out.append('')
out.append('- TPDdb 许可证和再分发权限需要在开放科学发布包前获得明确证据。')
out.append('- 论文中应把本报告的外部验证结果与内部结果分开呈现，不能把外部队列当作主体数据库的普通测试折。')
out.append('- 若后续加入新外部数据集，应在本报告协议基础上新增队列版本、去重日志、特征哈希、模型 manifest 和预注册阈值，不得回溯调参。')
(REPORTS/'EXTERNAL_VALIDATION_ANALYSIS_REPORT.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print(REPORTS/'EXTERNAL_VALIDATION_ANALYSIS_REPORT.md')
