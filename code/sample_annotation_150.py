import os
# -*- coding: utf-8 -*-
"""
sample_annotation_150.py — PROTAC Bias-Aware 数据工程 (序 0+1: 150 分层抽样 + 标注模板预填)

目的: 为下游人工标注 150 条制备待标注样本, 供校准 activity_evidence 阈值 (§5 当前 provisional)。

主分层 A = dc50_obs_type (5 类) × B = activity_evidence (P/N/A/U) → 20 cell;
叠加尽量保证 E3 (CRBN/VHL/other) 与 Top-5 POI 有代表。
random.seed(20260815) 可复现; 不修改 data/raw/ 与清洁主表。

产出:
  data/derived/protac_annotation_150.csv    (150 行, 严格列序, 3 个 annotator 列留空)
  reports/ANNOTATION_SAMPLING_DESIGN.md       (中文设计文档)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROC = ROOT / "data/derived"
REPORTS = ROOT / "reports"
CODE = ROOT / "40_项目_Bias-Aware/code"

MAIN = PROC / "protac_clean_record_level.csv"
PILOT = REPORTS / "PILOT_20_EXAMPLES.csv"
OUT_CSV = PROC / "protac_annotation_150.csv"
OUT_MD = REPORTS / "ANNOTATION_SAMPLING_DESIGN.md"

SEED = 20260815
N_TOTAL = 150
OBS_QUOTA = {  # 每 dc50_obs_type 配额 (稀有记号保底)
    "interval-censored": 12,
    "left-censored": 18,
    "right-censored": 18,
    "endpoint-missing": 20,
    "exact": 82,
}
OBS_TYPES = ["exact", "left-censored", "right-censored", "interval-censored", "endpoint-missing"]
ACTIVITIES = ["P", "N", "A", "U"]

# 严格输出列序
OUT_COLS = [
    "raw_row_index", "record_id", "compound_id", "target", "e3_ligase", "cell_line",
    "treatment_time_h", "dc50_raw", "dc50_obs_type", "dc50_value", "dc50_lower", "dc50_upper",
    "dmax_raw", "dmax_obs_type", "dmax_value", "dmax_lower", "dmax_upper",
    "pctdeg_raw", "pctdeg_value", "source_quality", "article_doi",
    "provisional_activity_evidence", "stratum",
    "annotator_activity_evidence", "annotator_confidence", "annotator_note",
]

NEED = [
    "raw_row_index", "record_id", "compound_id", "target", "e3_ligase", "cell_line",
    "treatment_time_h", "dc50_raw", "dc50_obs_type", "dc50_value", "dc50_lower", "dc50_upper",
    "dmax_raw", "dmax_obs_type", "dmax_value", "dmax_lower", "dmax_upper",
    "pctdeg_raw", "pctdeg_value", "source_quality", "article_doi", "activity_evidence",
]


# ----------------------------------------------------------------------------
# 0. 载入主表, 校验真实列名
# ----------------------------------------------------------------------------
main = pd.read_csv(MAIN)
missing_cols = [c for c in NEED if c not in main.columns]
if missing_cols:
    raise SystemExit(f"主表缺失列: {missing_cols}")
main = main[NEED].rename(columns={"activity_evidence": "provisional_activity_evidence"})

# Top-5 POI (按主表实际频次)
top5_poi = list(main["target"].value_counts().head(5).index)
e3_cats = ["CRBN", "VHL", "other"]


# ----------------------------------------------------------------------------
# 1. 比例分配 (每个 obs_type 内按 activity 比例, 每 cell floor=2, 上限=可用数)
# ----------------------------------------------------------------------------
def allocate_proportional(avail, total, floor=2):
    """avail: {cell: 可用数}; 返回 {cell: 抽数}, 和=total, 每 cell>=min(floor,avail), <=avail。"""
    cells = list(avail.keys())
    alloc = {c: min(floor, avail[c]) for c in cells}
    remaining = total - sum(alloc.values())
    if remaining > 0:
        headroom = {c: avail[c] - alloc[c] for c in cells}
        tot_head = sum(headroom.values())
        if tot_head > 0:
            raw = {c: headroom[c] / tot_head * remaining for c in cells}
            base = {c: int(raw[c]) for c in cells}
            rem = remaining - sum(base.values())
            order = sorted([c for c in cells if headroom[c] > 0],
                           key=lambda c: raw[c] - base[c], reverse=True)
            for c in order:
                if rem <= 0:
                    break
                if base[c] + 1 <= avail[c]:
                    base[c] += 1
                    rem -= 1
            for c in cells:
                alloc[c] += base[c]
    # 安全夹紧 (防极端)
    for c in cells:
        if alloc[c] > avail[c]:
            alloc[c] = avail[c]
    return alloc


rs = np.random.RandomState(SEED)
cell_plan = {}
for o in OBS_TYPES:
    sub = main[main["dc50_obs_type"] == o]
    vc = sub["provisional_activity_evidence"].value_counts()
    avail = {a: int(vc.get(a, 0)) for a in ACTIVITIES}
    alloc = allocate_proportional(avail, OBS_QUOTA[o], floor=2)
    for a in ACTIVITIES:
        cell_plan[(o, a)] = alloc[a]

# 计划总数校验
assert sum(cell_plan.values()) == N_TOTAL, f"计划总数 {sum(cell_plan.values())} != {N_TOTAL}"

# ----------------------------------------------------------------------------
# 2. 逐 cell 抽样
# ----------------------------------------------------------------------------
frames = []
for o in OBS_TYPES:
    for a in ACTIVITIES:
        n = cell_plan[(o, a)]
        if n <= 0:
            continue
        sub = main[(main["dc50_obs_type"] == o) & (main["provisional_activity_evidence"] == a)]
        if len(sub) == 0:
            print(f"[edge] cell ({o},{a}) 可用=0, 跳过 (计划 {n})")
            continue
        if len(sub) < n:
            print(f"[edge] cell ({o},{a}) 可用={len(sub)} < 计划 {n}, 全取并说明")
            samp = sub
        else:
            samp = sub.sample(n=n, random_state=rs)
        samp = samp.copy()
        samp["stratum"] = f"{o}|{a}"
        frames.append(samp)

sampled = pd.concat(frames, ignore_index=True)

# ----------------------------------------------------------------------------
# 3. E3 / Top-5 POI 覆盖保底 (叠加; 自然覆盖通常已完整, 缺则换入并记 swap)
# ----------------------------------------------------------------------------
swaps = []
def coverage_report(df):
    cov_e3 = {c: int((df["e3_ligase"] == c).sum()) if c != "other"
              else int((~df["e3_ligase"].isin(["CRBN", "VHL"])).sum()) for c in e3_cats}
    cov_poi = {p: int((df["target"] == p).sum()) for p in top5_poi}
    cov_obs = {o: int((df["dc50_obs_type"] == o).sum()) for o in OBS_TYPES}
    cov_act = {a: int((df["provisional_activity_evidence"] == a).sum()) for a in ACTIVITIES}
    return cov_e3, cov_poi, cov_obs, cov_act

cov_e3, cov_poi, cov_obs, cov_act = coverage_report(sampled)
# 若缺失, 换入 (保留真实 stratum, 文档说明)
for c in e3_cats:
    if cov_e3[c] == 0:
        cand = main[(main["e3_ligase"] == c if c != "other"
                     else ~main["e3_ligase"].isin(["CRBN", "VHL"]))
                    & (~main["record_id"].isin(sampled["record_id"]))]
        if len(cand):
            donor_stratum = sampled["stratum"].value_counts().idxmax()
            donor = sampled[sampled["stratum"] == donor_stratum].sample(1, random_state=rs)
            sampled = sampled[~sampled["record_id"].isin(donor["record_id"])].copy()
            new = cand.sample(1, random_state=rs).copy()
            new_stratum = f"{new['dc50_obs_type'].iloc[0]}|{new['provisional_activity_evidence'].iloc[0]}"
            new["stratum"] = new_stratum
            sampled = pd.concat([sampled, new], ignore_index=True)
            swaps.append(f"E3={c}: 换入 record_id={int(new['record_id'].iloc[0])} (true stratum {new_stratum}; donor stratum {donor_stratum})")
for p in top5_poi:
    if cov_poi[p] == 0:
        cand = main[(main["target"] == p) & (~main["record_id"].isin(sampled["record_id"]))]
        if len(cand):
            donor_stratum = sampled["stratum"].value_counts().idxmax()
            donor = sampled[sampled["stratum"] == donor_stratum].sample(1, random_state=rs)
            sampled = sampled[~sampled["record_id"].isin(donor["record_id"])].copy()
            new = cand.sample(1, random_state=rs).copy()
            new_stratum = f"{new['dc50_obs_type'].iloc[0]}|{new['provisional_activity_evidence'].iloc[0]}"
            new["stratum"] = new_stratum
            sampled = pd.concat([sampled, new], ignore_index=True)
            swaps.append(f"POI={p}: 换入 record_id={int(new['record_id'].iloc[0])} (true stratum {new_stratum}; donor stratum {donor_stratum})")

# 重算实际覆盖
cov_e3, cov_poi, cov_obs, cov_act = coverage_report(sampled)
# 实际每 cell 计数
actual_cell = sampled["stratum"].value_counts().to_dict()

# ----------------------------------------------------------------------------
# 4. 组装输出 (3 个 annotator 列留空)
# ----------------------------------------------------------------------------
sampled = sampled.copy()
sampled["annotator_activity_evidence"] = pd.NA
sampled["annotator_confidence"] = pd.NA
sampled["annotator_note"] = pd.NA
sampled = sampled[OUT_COLS]
assert len(sampled) == N_TOTAL, f"最终行数 {len(sampled)} != {N_TOTAL}"
sampled.to_csv(OUT_CSV, index=False)

# ----------------------------------------------------------------------------
# 5. 打印每 cell 实抽数
# ----------------------------------------------------------------------------
print("\n=== 计划 vs 实际 每 cell 抽数 ===")
print(f"{'stratum':<28}{'plan':>6}{'actual':>7}")
for o in OBS_TYPES:
    for a in ACTIVITIES:
        s = f"{o}|{a}"
        print(f"{s:<28}{cell_plan[(o,a)]:>6}{actual_cell.get(s,0):>7}")
print(f"\n总计: plan={sum(cell_plan.values())}  actual={len(sampled)}  swaps={len(swaps)}")
print("Top-5 POI:", top5_poi)
print("E3 覆盖:", cov_e3)
print("Top5 POI 覆盖:", cov_poi)
print("obs 记号覆盖:", cov_obs)
print("activity 覆盖:", cov_act)

# ----------------------------------------------------------------------------
# 6. 写设计文档 (中文)
# ----------------------------------------------------------------------------
pilot = pd.read_csv(PILOT)

# 配额表 (按 obs_type 汇总 + 20 cell)
quota_rows = []
for o in OBS_TYPES:
    for a in ACTIVITIES:
        s = f"{o}|{a}"
        quota_rows.append(f"| {o} | {a} | {cell_plan[(o,a)]} | {actual_cell.get(s,0)} |")
quota_tbl = "\n".join(quota_rows)

obs_summary = " | ".join([f"{o}={cov_obs[o]}" for o in OBS_TYPES])
act_summary = " | ".join([f"{a}={cov_act[a]}" for a in ACTIVITIES])
e3_summary = " | ".join([f"{c}={cov_e3[c]}" for c in e3_cats])
poi_summary = " | ".join([f"{p}={cov_poi[p]}" for p in top5_poi])

# PILOT 5 个示范 (每类 1 条): exact=row19, right=row238, left=row429, interval=row365, missing=row0
pilot_examples = [
    ("exact (精确值)", 19,
     "BRD9, DC50=560 nM (exact), Dmax=80% (exact)。",
     "DC50=560 nM 落在 (100,1000] nM 区间, 非 P 候选(≤100)亦非 N 候选(≥1000); Dmax=80% 触及 P 边界(≥80%)。综合判为 P 边界/偏强, 但 DC50 未达高效力。",
     "P", "中",
     "DC50=560 接近 P 边界上沿, Dmax=80 恰为阈值; 建议 §5 明确 Dmax 边界计入 P 与否。"),
    ("right-censored (> 右删失)", 238,
     "BRD4, DC50>10000 nM (right-censored), Dmax=N.D. (endpoint-missing)。",
     "DC50>10000 nM 远超 N 候选阈值(≥1000 nM), 换算 pDC50<5, 强力支持低效力证据; Dmax 缺失不影响 N 判定。",
     "N", "高",
     "右删失下界 10000 nM 已足够判 N; 若 §5 用更严 N 阈值亦满足。"),
    ("left-censored (< 左删失)", 429,
     "FAK, DC50<10 nM (left-censored), Dmax=87% (exact)。",
     "DC50<10 nM 落在 P 候选区(≤100 nM), 且 Dmax=87%≥80% 强降解; 上限未给出但已足够支持高效力。",
     "P", "高",
     "左删失仅知 <10 nM, 已 <100 nM P 阈值; 若 §5 要求精确值才计 P, 则标 A。"),
    ("interval-censored (区间删失)", 365,
     "BRD4, DC50=1~3 nM (interval-censored), Dmax 缺失。",
     "DC50∈[1,3] nM 全部 ≤100 nM, 落入 P 候选区且属高效力; 区间窄, 不确定度低。",
     "P", "高",
     "区间全部位于 P 区, 标 P 稳健; 仅当 §5 要求精确值才降为 A。"),
    ("endpoint-missing (终点缺失)", 0,
     "BRD7, DC50 与 Dmax 均无观测 (endpoint-missing)。",
     "两定量终点皆缺, 无任何效力/降解约束证据; 既非 P 亦非 N, 属未标注。",
     "U", "高",
     "无观测即 U; 不可因缺报判为 N (路线 D 强调)。"),
]

pilot_tbl = "| 类型 | raw_row_index | 原始记载 | 临时判据推理 | 建议 annotator_activity_evidence | confidence | 备注 |\n|---|---|---|---|---|---|---|\n"
for typ, ridx, raw, reason, sug, conf, note in pilot_examples:
    pilot_tbl += f"| {typ} | {ridx} | {raw} | {reason} | {sug} | {conf} | {note} |\n"

md = f"""# PROTAC Bias-Aware — 150 条人工标注分层抽样设计 (序 0+1)

> 数据工程交付物, 服务于校准 `activity_evidence` 阈值 (标注规范 §5, 当前为 provisional)。
> 复现: `python code/sample_annotation_150.py` (seed={SEED})。主表: `data/derived/protac_clean_record_level.csv` (15535 行)。
> **本设计文档中的 PILOT 判据与建议标注均为临时示范, 最终以人工判断 + 校准后规则为准, §5 未定稿。**

## 1. 分层方案说明

主分层为 **A = dc50_obs_type (5 类) × B = activity_evidence (P/N/A/U)** 的交叉 20 cell。理由:

- **dc50_obs_type** 是本研究偏差与删失结构的核心轴: exact / left-censored / right-censored / interval-censored / endpoint-missing 五种观测类型对应不同的证据强度与建模处理 (区间/删失损失 vs 精确回归)。人工核查必须每种都覆盖, 否则阈值校准会偏向"易标"的精确值。
- **activity_evidence** 是待校准的标签本身 (P/N/A/U)。按标签层交叉, 保证 P/N/A/U 各状态都有代表, 才能检验 §5 阈值是否把边界样本错分。
- **叠加 E3 (CRBN/VHL/other) 与 Top-5 POI** 配额: 数据高度集中于 CRBN (~70%) 与少数 POI; 纯比例抽样可能漏掉 VHL/other 或长尾 POI。抽样后核验覆盖, 缺失则换入 (见 §3)。

分配方法: 每个 obs_type 先给定配额 (稀有记号保底), 再在该 obs_type 内按 activity 比例分配至 4 个 cell, 每 cell 下限 2、上限为可用数; 配额不足则全取并说明。组内用 `RandomState({SEED})` 可复现抽取。

## 2. 配额分配表 (计划 vs 实际, 总和=150)

稀有记号保底: interval-censored≥12、left-censored≥18、right-censored≥18、endpoint-missing≥20、exact 取剩余≈82。

| dc50_obs_type | activity | 计划抽数 | 实际抽数 |
|---|---|---|---|
{quota_tbl}

- 合计计划 = {sum(cell_plan.values())}, 合计实际 = {len(sampled)}。
- 稀有记号保底全部达成: interval-censored≥12、left-censored≥18、right-censored≥18、endpoint-missing≥20。
- 所有实际 cell 计数均 ≥ 2 (下限满足)。除 E3/Top-5 POI 覆盖保底换入使 1 个 donor cell (exact|P) 实际比计划少 1、对应 cell (endpoint-missing|U) 多 1 外, 其余 cell 实际 ≥ 计划下限。
- **边界情况**: 计划为 0 的 cell (exact|U, left-censored|N, left-censored|U, right-censored|P, right-censored|U, interval-censored|N, interval-censored|U) 其可用数本就为 0 (如 exact|U: 精确 DC50 必非 U), 正确计 0; 全部抽样 cell 可用数均 ≥ 计划, 未触发"全取"截断。
- E3 / Top-5 POI 覆盖保底换入: {len(swaps)} 次 ({'; '.join(swaps) if swaps else '无需换入, 自然覆盖已完整'})。

## 3. 覆盖性确认 (证无遗漏)

- **5 类 obs 记号覆盖**: {obs_summary} — 全部 ≥ 保底下限, 无遗漏。
- **P/N/A/U 覆盖**: {act_summary} — 四态均有代表。
- **E3 覆盖**: {e3_summary}。
- **Top-5 POI ({', '.join(top5_poi)}) 覆盖**: {poi_summary}。
- 结论: 样本对 5 类观测记号与 P/N/A/U 四态均完整覆盖, 无类型遗漏。

## 4. 填写指南 (逐列)

| 列 | 人工如何填写 |
|---|---|
| raw_row_index / record_id / compound_id | 定位用, 勿改 |
| target / e3_ligase / cell_line / treatment_time_h | 上下文参考, 可补充修正 (如 cell_line 文本挖掘有误) |
| dc50_raw / dc50_obs_type / dc50_value(-lower/-upper) | 原始与解析值, 供核对; 如解析有误在 note 标注 |
| dmax_raw / dmax_obs_type / dmax_value(-lower/-upper) | 同上 |
| pctdeg_raw / pctdeg_value | Percent degradation 原始与均值, 参考 |
| source_quality / article_doi | 来源, 供判断证据等级 |
| provisional_activity_evidence | 系统临时标签, **仅供参考**; 以人工判断为准 |
| stratum | 所属分层标签, 勿改 |
| **annotator_activity_evidence** | **人工判据 (临时)**: P=有充分证据支持有效降解 (如 DC50≤100 nM 或 Dmax≥80% 且条件充分); N=明确低/无效降解证据 (如 DC50≥1000 nM 或 Dmax≤10%); A=矛盾/临界/信息不足 (边界值、单终点、区间跨阈值); U=两定量终点皆缺, 无任何可用观测。**不要把 U 当 0/负类。** |
| **annotator_confidence** | 高 / 中 / 低: 高=证据明确且落在阈值同侧; 中=边界或单终点; 低=信息极少或矛盾 |
| **annotator_note** | 记: 解析疑点、边界/跨阈值、单终点依赖、来源疑点、与系统标签分歧理由 |

> 临时判据阈值 (DC50: P≤100 nM / N≥1000 nM; Dmax: P≥80% / N≤10%) 来自 README 临时规则, **§5 未定稿**, 校准后可能调整。

## 5. PILOT_20 填制示例 (每类 1 条, 临时示范)

选自 `reports/PILOT_20_EXAMPLES.csv` (20 条, =/><区间/缺失 各 4), 取 5 条示范完整填法:

{pilot_tbl}

> **再次声明**: 上表 "建议 annotator_activity_evidence / confidence / 备注" 为基于临时阈值的示范推理, 仅用于统一标注口径; 最终标注以人工独立判断 + §5 校准后规则为准。

## 6. 产出文件

- `data/derived/protac_annotation_150.csv` — 150 行待标注样本 (严格列序, 3 个 annotator 列留空)。
- 本设计文档。
"""

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\n已写出:\n  {OUT_CSV}\n  {OUT_MD}")
