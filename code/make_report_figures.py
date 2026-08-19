# -*- coding: utf-8 -*-
"""
make_report_figures.py — PROTAC-DB 3.0 初步审计报告 + 6 张图 (300dpi)
依赖: etl_protac.py 产出的 protac_clean_record_level.csv 与 protac_clean_audit_stats.json
输出: reports/DATA_AUDIT_REPORT.md  +  reports/figures/*.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")
REP  = os.path.join(BASE, "reports")
FIG  = os.path.join(REP, "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(OUT, "protac_clean_record_level.csv"), encoding="utf-8-sig")
st = json.load(open(os.path.join(OUT, "protac_clean_audit_stats.json"), encoding="utf-8"))

N = len(df)
DPI = 300
C = {"P": "#2ca02c", "N": "#d62728", "A": "#ff7f0e", "U": "#7f7f7f",
     "dc": "#1f77b4", "dm": "#9467bd"}

# ---------------- Fig1: 删失类型分布 ----------------
order = ["exact", "left-censored", "right-censored", "interval-censored", "endpoint-missing"]
dc = [st["dc50_obs_types"].get(k, 0) for k in order]
dm = [st["dmax_obs_types"].get(k, 0) for k in order]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, dc, w, label="DC50", color=C["dc"])
ax.bar(x + w/2, dm, w, label="Dmax", color=C["dm"])
ax.set_xticks(x); ax.set_xticklabels([o.replace("-censored", "\n-censored") for o in order], fontsize=8)
ax.set_ylabel("records"); ax.set_title("Endpoint observation / censoring types (DC50 vs Dmax)")
for i, v in enumerate(dc): ax.text(i - w/2, v + 30, str(v), ha="center", fontsize=7)
for i, v in enumerate(dm): ax.text(i + w/2, v + 30, str(v), ha="center", fontsize=7)
ax.legend(); ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_censoring_distribution.png"), dpi=DPI); plt.close(fig)

# ---------------- Fig2: activity_evidence 分布 ----------------
ae = st["activity_default"]
order2 = ["P", "N", "A", "U"]
vals = [ae.get(k, 0) for k in order2]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(order2, vals, color=[C[k] for k in order2])
ax.set_ylabel("records"); ax.set_title(f"activity_evidence distribution (provisional, CP=100nM)\nP={vals[0]} N={vals[1]} A={vals[2]} U={vals[3]}")
for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+50, f"{v}\n{v/N*100:.1f}%", ha="center", fontsize=8)
ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_activity_evidence.png"), dpi=DPI); plt.close(fig)

# ---------------- Fig3: Top POI & Top E3 ----------------
top_t = df["target"].value_counts().head(12)
top_e = df["e3_ligase"].value_counts().head(12)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.barh(top_t.index[::-1], top_t.values[::-1], color=C["dc"])
a1.set_title(f"Top POI (Target) — top1 share {top_t.iloc[0]/N*100:.1f}%"); a1.set_xlabel("records")
a2.barh(top_e.index[::-1], top_e.values[::-1], color=C["dm"])
a2.set_title(f"Top E3 ligase — top1 share {top_e.iloc[0]/N*100:.1f}%"); a2.set_xlabel("records")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_top_poi_e3.png"), dpi=DPI); plt.close(fig)

# ---------------- Fig4: 观测率 2x2 (POI/E3/source/year) ----------------
def obs_rate(series): return (series != "endpoint-missing").mean()
tr = df.assign(_o=(df.dc50_obs_type != "endpoint-missing")).groupby("target")["_o"].mean().sort_values(ascending=False).head(10)
er = df.assign(_o=(df.dc50_obs_type != "endpoint-missing")).groupby("e3_ligase")["_o"].mean().sort_values(ascending=False).head(10)
sr = df.assign(_o=(df.dc50_obs_type != "endpoint-missing")).groupby("source_has_doi")["_o"].mean()
yr = df.dropna(subset=["year_doi"]).assign(_o=(df.dropna(subset=["year_doi"]).dc50_obs_type != "endpoint-missing")).groupby("year_doi")["_o"].mean().sort_index()
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0,0].barh(tr.index[::-1], tr.values[::-1]); axes[0,0].set_title("DC50 obs-rate by Top POI"); axes[0,0].set_xlabel("obs rate")
axes[0,1].barh(er.index[::-1], er.values[::-1]); axes[0,1].set_title("DC50 obs-rate by Top E3"); axes[0,1].set_xlabel("obs rate")
axes[1,0].bar(sr.index, sr.values, color=[C["P"] if i=="with_doi" else C["N"] for i in sr.index])
axes[1,0].set_title("DC50 obs-rate by source (DOI)"); axes[1,0].set_ylim(0,1); axes[1,0].set_ylabel("obs rate")
axes[1,1].plot(yr.index, yr.values, "o-", color=C["dc"]); axes[1,1].set_title(f"DC50 obs-rate by year (cov {st['year_coverage']*100:.0f}%)")
axes[1,1].set_ylim(0,1); axes[1,1].set_ylabel("obs rate"); axes[1,1].set_xlabel("year")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_obs_rate_panels.png"), dpi=DPI); plt.close(fig)

# ---------------- Fig5: 阈值敏感性 ----------------
sens = st["activity_sensitivity"]
cuts = [int(k) for k in sens.keys()]
Pv = [sens[k]["P"] for k in sens.keys()]
Nv = [sens[k]["N"] for k in sens.keys()]
Av = [sens[k]["A"] for k in sens.keys()]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(cuts, Pv, "o-", color=C["P"], label="P")
ax.plot(cuts, Nv, "s-", color=C["N"], label="N")
ax.plot(cuts, Av, "^-", color=C["A"], label="A (ambiguous)")
ax.set_xscale("log"); ax.set_xlabel("DC50 P-cutoff (nM, log)"); ax.set_ylabel("records")
ax.set_title("Threshold sensitivity of activity_evidence\n(N-cutoff = 10 x P-cutoff)")
for x0, y0 in zip(cuts, Pv): ax.text(x0, y0+40, str(y0), ha="center", fontsize=8, color=C["P"])
for x0, y0 in zip(cuts, Nv): ax.text(x0, y0-90, str(y0), ha="center", fontsize=8, color=C["N"])
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_threshold_sensitivity.png"), dpi=DPI); plt.close(fig)

# ---------------- Fig6: 化学系列集中度 (scaffold) ----------------
sc = df["scaffold"].dropna()
top_s = sc.value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(range(len(top_s)), top_s.values, color=C["dc"])
ax.set_xticks(range(len(top_s))); ax.set_xticklabels([f"S{i+1}" for i in range(len(top_s))], fontsize=7)
ax.set_ylabel("records"); ax.set_title(f"Top-15 Murcko scaffolds (of {st['n_unique_scaffolds']} unique)\n"
                                        f"top-1 share {st['top1_scaffold_share']*100:.2f}%  |  "
                                        f"{st['n_scaffolds_shared']} scaffolds shared across >=2 records")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_chemical_series.png"), dpi=DPI); plt.close(fig)
print("figures written")

# ================= 报告 markdown =================
def share(series, topn):
    vc = series.value_counts()
    return vc.head(topn).sum() / len(series) * 100, vc

top1_t, vc_t = share(df["target"], 1)
top5_t, _ = share(df["target"], 5)
top10_t, _ = share(df["target"], 10)
top1_e, vc_e = share(df["e3_ligase"], 1)
top5_e, _ = share(df["e3_ligase"], 5)
top10_e, _ = share(df["e3_ligase"], 10)
top1_c, vc_c = share(df["cell_line"].dropna(), 1)
top5_c, _ = share(df["cell_line"].dropna(), 5)
top1_s, vc_s = share(sc, 1)
top5_s, _ = share(sc, 5)

ov = st["overlap"]
dcobs = st["dc50_obs_share"]; dmobs = st["dmax_obs_share"]
ae_d = st["activity_default"]
sens_tbl = " | ".join([f"CP={c}nM" for c in sens.keys()])
sens_rows = ""
for k in sens.keys():
    s = sens[k]
    sens_rows += f"| CP={k}nM (N={int(float(k)*10)}nM) | {s['P']} | {s['N']} | {s['A']} | {s['U']} |\n"
_first_k, _last_k = list(sens.keys())[0], list(sens.keys())[-1]
_p0, _p1 = sens[_first_k]["P"], sens[_last_k]["P"]
_n0, _n1 = sens[_first_k]["N"], sens[_last_k]["N"]
_a_min = min(sens[k]["A"] for k in sens); _a_max = max(sens[k]["A"] for k in sens)

def md_table(d, topn=10):
    items = list(d.items())[:topn]
    return "\n".join(f"| {k} | {v} | {v/N*100:.2f}% |" for k, v in items)

report = f"""# DATA AUDIT REPORT — PROTAC Bias-Aware PNU 数据集（初步版, v0.1-provisional）

> 生成: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  |  数据工程师: 洗澄明
> 数据源: PROTAC-DB 3.0 (http://cadd.zju.edu.cn/protacdb/, 下载 2026-08-15)
> 对应执行计划 8/20 清单 + PNU 标注规范 §4/§5。本审计为**初步版**，活动证据标签为 provisional。

## 0. 数据合约（5 检查，适配记录级横截面数据）
| 检查 | 结果 |
|---|---|
| 形状 (raw) | ✅ 15,502 × 89 |
| 关键变量 dtype | ✅ 全字符串读取后受控解析 |
| 最大缺失率 < 1 (整列) | ✅ 通过 |
| Compound ID 非全局唯一 | ✅ 预期 (10,725 唯一); 记录身份=SMILES+Target+E3+实验条件 |
| 面板平衡 | N/A — 本数据为横截面记录级, 非时间序列面板 |

样本构建日志: 15,502 raw → 解析+inline 展开 → **15,535 记录级行**（+33 来自 Percent degradation 列 inline 细胞系展开）。

## 1. 删失/观测类型分布
DC50 观测率 **{dcobs*100:.1f}%**, Dmax 观测率 **{dmobs*100:.1f}%**（其余为 endpoint-missing）。

### DC50
| 类型 | 记录数 | 占比 |
|---|---|---|
{md_table(st['dc50_obs_types'])}

### Dmax
| 类型 | 记录数 | 占比 |
|---|---|---|
{md_table(st['dmax_obs_types'])}

> 关键发现: 定量端点**高度稀疏**。约 3/4 记录无 DC50、5/6 无 Dmax。删失以右删失(>X)与左删失(<X)为主, 区间与复合比较较少。解析规则见 §7。

## 2. DC50 与 Dmax 重叠
| 组合 | 记录数 | 占比 |
|---|---|---|
| 两者皆有 | {ov['both']} | {ov['both']/N*100:.1f}% |
| 仅 DC50 | {ov['dc50_only']} | {ov['dc50_only']/N*100:.1f}% |
| 仅 Dmax | {ov['dmax_only']} | {ov['dmax_only']/N*100:.1f}% |
| 两者皆无 | {ov['neither']} | {ov['neither']/N*100:.1f}% |

> 仅 {ov['both']/N*100:.1f}% 记录同时拥有 DC50 与 Dmax —— 多任务(回归 vs 分类)样本量差异显著, 需分别建模。

## 3. activity_evidence（临时阈值, provisional）
默认规则: DC50≤100nM→P候选, DC50≥1000nM→N候选, Dmax≥80%→P, Dmax≤10%→N; 组合冲突/临界→A; 两终点皆缺→U。
(标注规范 §5 未定稿, **必须做敏感性分析与人审**。)

| 类别 | 记录数 | 占比 |
|---|---|---|
| P (阳性候选) | {ae_d.get('P',0)} | {ae_d.get('P',0)/N*100:.1f}% |
| N (阴性候选) | {ae_d.get('N',0)} | {ae_d.get('N',0)/N*100:.1f}% |
| A (ambiguous) | {ae_d.get('A',0)} | {ae_d.get('A',0)/N*100:.1f}% |
| U (endpoint-missing) | {ae_d.get('U',0)} | {ae_d.get('U',0)/N*100:.1f}% |

### 阈值敏感性扫描（N-cutoff = 10 × P-cutoff）
| 阈值 | P | N | A | U |
|---|---|---|---|---|
{sens_rows}
> P 随 cutoff 上升（{_p0}→{_p1}）, N 随 cutoff 下降（{_n0}→{_n1}）; A 在 {_a_min}–{_a_max} 间波动。说明标签对阈值**中度敏感**, 下游应报告多 cutoff 下的稳健性。

## 4. 不平衡分析（Top-N 占比）
### POI (Target)
- top-1 占比 **{top1_t:.1f}%** ({vc_t.index[0]}), top-5 **{top5_t:.1f}%**, top-10 **{top10_t:.1f}%**
| POI | 记录数 | 占比 |
|---|---|---|
{md_table(vc_t, 10)}

### E3 ligase
- top-1 占比 **{top1_e:.1f}%** ({vc_e.index[0]}), top-5 **{top5_e:.1f}%**, top-10 **{top10_e:.1f}%**
| E3 | 记录数 | 占比 |
|---|---|---|
{md_table(vc_e, 10)}

### Cell line（文本挖掘, 覆盖率有限）
- 非空 cell_line 记录 {int(df.cell_line.notna().sum())} 条; top-1 **{top1_c:.1f}%** ({vc_c.index[0]})
| cell_line | 记录数 | 占比 |
|---|---|---|
{md_table(vc_c, 10)}

### Source（DOI 有无）
| source | 记录数 | 占比 |
|---|---|---|
| with_doi | {st['source'].get('with_doi',0)} | {st['source'].get('with_doi',0)/N*100:.1f}% |
| no_doi | {st['source'].get('no_doi',0)} | {st['source'].get('no_doi',0)/N*100:.1f}% |

### Year（自 DOI 正则提取, 覆盖率 {st['year_coverage']*100:.1f}%）
- 有年份记录 {int(df.year_doi.notna().sum())} 条; 详见图 fig4。

> **结论**: 数据在 **E3 维度高度集中**（CRBN 占 ~70%, VHL ~25%）, POI 维度中等集中（ERalpha/BRD4/BTK 居前）。cell_line 因依赖文本挖掘, 大量缺失。

## 5. 重复与冲突
| 指标 | 比例 |
|---|---|
| is_duplicate（完全重复记录级副本） | {st['dup_rate']*100:.2f}% ({int(st['dup_rate']*N)} 条) |
| is_conflict（同键不同值） | {st['conflict_rate']*100:.2f}% ({int(st['conflict_rate']*N)} 条) |
| has_replicates（含重复测量） | {st['has_replicates_rate']*100:.2f}% |
| is_dose_series（长串序列） | {st['dose_series_rate']*100:.2f}% |

## 6. 观测概率随 POI / E3 / source / year
- 见 **fig4_obs_rate_panels.png**。总体 DC50 观测率 {dcobs*100:.1f}%。
- 按 source: with_doi {st['obs_rate_by_source'].get('with_doi',0)*100:.1f}% vs no_doi {st['obs_rate_by_source'].get('no_doi',0)*100:.1f}%。
- 按 year: 见 fig4 右下（覆盖率 {st['year_coverage']*100:.0f}%, 仅作近似）。
- 观测概率在不同 POI/E3 间差异明显（fig4 上排），提示**非随机缺失(MNAR)风险**——活性化合物更可能被测定并报告，构成 PNU 框架需显式建模的偏差。

## 7. 化学系列集中度 & 随机划分泄漏风险
- 唯一 Murcko scaffold 数: **{st['n_unique_scaffolds']}**（占记录 {st['n_unique_scaffolds']/N*100:.1f}%）。
- top-1 scaffold 占比 **{st['top1_scaffold_share']*100:.2f}%**, top-5 **{top5_s:.2f}%** → 在**全分子 scaffold 层面数据高度多样**, 并未集中于少数 scaffold。
- 但 **{st['frac_records_shared_scaffold']*100:.1f}%** 的记录其 scaffold 与 ≥1 条其他记录共享（{st['n_scaffolds_shared']} 个共享 scaffold）。
- **泄漏风险**: 若随机划分训练/测试, 同 scaffold 化合物会跨集出现 → 药化系列泄漏。建议按 scaffold（或 warhead+linker）**分组划分(group/split by scaffold)**, 而非随机行划分。见 fig6。

## 8. 关键决策与歧义处理（供主理人/标注者复核）
1. **斜杠消歧**:
   - 同单位两值（如 `490/500`）→ 重复测量, `dc50_replicates`=原串, 主值取**首值**（标注规范）。
   - inline 细胞系（`PANC-1:0/0;K562:0/37.7`, 位于 Percent degradation 列）→ **按细胞系展开多行**, 每块内斜杠为 replicate, 取均值作 `pctdeg_value`。
   - ≥4 数字长串（如 `98/99/97/92/65/33/13`）→ `is_dose_series=True`; DC50 取首值, **Dmax 取最大值**作候选。
2. **逗号**: `>10,000` 判为千分位→10000; `6,1`/`1,15` 判为欧洲小数→6.1/1.15（极少数 `6,1` 可能为 "6/1" 误写, 已记录为局限）。
3. **全角/括号**: `＞`→`>`, `(n/a)`/`(nm)` 去除; 日期串（脏数据）→ endpoint-missing。
4. **OCR 错误**: 数字上下文中 `O.3`→`0.3`。
5. **复合比较**: `>5 and <50`→ interval-censored [5,50]; `>=`/`<=` 并入比较方向（右/左删失）。
6. **pDC50 符号反转**（标注规范 §4）: DC50>X → pDC50<9-log10(X)（上界）; DC50<X → pDC50>9-log10(X)（下界）。已对全部删失记录校验一致性（0 误差）。
7. **activity_evidence 仅基于 DC50/Dmax**: Percent degradation 单列数据（无 DC50/Dmax）记为 U（pctdeg 单独保留于 `pctdeg_value`）。
8. **inline 细胞系展开**: 仅 Percent degradation 列出现（33 行），展开后 +33 行; 其余细胞系/时间来自 Assay 文本正则挖掘（大量 NaN）。

## 9. 交付物清单
- `data/derived/protac_clean_record_level.csv`（15,535 × 42）
- `data/derived/data_dictionary.csv`
- `data/derived/protac_clean_audit_stats.json`
- `data/raw/raw_data_manifest.csv`（sha256）
- `templates/raw_data_manifest_template.csv`, `templates/protac_annotation_template.csv`
- `reports/PILOT_20_EXAMPLES.csv`
- `reports/DATA_AUDIT_REPORT.md` + `reports/figures/`（fig1–fig6）
- `README_data_version.md`

## 10. 局限与下一步
- 无独立阴性标签列; activity_evidence 临时阈值需人审定稿（标注规范 §5）。
- cell_line/treatment_time 多为文本挖掘, 缺失率高, 建议结合组件表与人工补标。
- 年份仅自 DOI 近似; 建议接入文献元数据补全。
- 下一步: 150 条人工标注（基于 PILOT_20 模板）→ 校准活动证据阈值 → 设计 scaffold 分组划分。
"""
# fix with_doi row formatting
with open(os.path.join(REP, "DATA_AUDIT_REPORT.md"), "w", encoding="utf-8") as f:
    f.write(report)
print("wrote DATA_AUDIT_REPORT.md")
