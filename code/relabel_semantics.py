# -*- coding: utf-8 -*-
"""
relabel_semantics.py — U/N 语义拆分 + 任务标签/证据质量分离 (审稿 03 号 P2-1~P2-4)
==================================================================================
背景: 六份审稿意见指出当前 P/N/A/U 是"任务标签"而非"证据等级", 且 U 类混入
      20.2% 有 Percent degradation 的记录, N 类统一称 experimental negative 过强。

本脚本在主表上新增两组派生列 (不改动 activity_evidence_v2 原标签, 保留对照):

1. U 语义拆分 (u_subtype):
   - U0       : 无任何降解观测 (无 pctdeg, 无 dc50/dmax)
   - O-PD     : 有 Percent degradation 但无 DC50/Dmax 端点
   - A-context: 有数值但条件不足 (按审稿建议的语义占位; 当前 ETL 无独立条件完整度字段,
                以 pctdeg 存在 + 无 cell/time 作为弱条件缺失代理, 详见局限)

2. N 语义拆分 (n_subtype):
   - N-confirmed   : 有 dmax exact 且 <=20 (明确低降解深度) 或 dc50 exact 且 >=1000 + dmax exact
   - N-low-potency : dc50 exact >=1000 或 dc50 右删失 (低效力, 无 dmax 佐证)
   - N-low-depth   : dmax exact <=20 或 dmax 左删失 (深度不足, dc50 缺)
   (规则从审稿 03 号 P2-3 的 N-confirmed/N-low-potency/N-low-depth 落地为可操作规则)

3. 证据质量等级 (evidence_grade, 任务标签与证据质量分离):
   综合: 端点是否拟合/直接读数, 删失与否, 条件完整度, 来源可溯性
   - A: 双端点或单端点 exact + 条件完整 (cell+time 或 DOI)
   - B: 单端点 exact 但条件部分缺失
   - C: 删失/区间端点
   - D: 仅 pctdeg 或无端点
   (注: 完整证据等级需原始文献回溯, 本列为基于现有字段的代理等级, 论文中须如实说明)

输出:
  data/derived/protac_clean_record_level.csv 新增列: u_subtype, n_subtype, evidence_grade, label_semantics_note
运行: .../python.exe relabel_semantics.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")

def main():
    path = os.path.join(OUT, "protac_clean_record_level.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    lab = df["activity_evidence_v2"].values

    # ---------- 1. U 语义拆分 ----------
    u_sub = np.full(len(df), "", dtype=object)
    U_mask = lab == "U"
    has_pct = df["pctdeg_value"].notna().values
    has_cond = df["cell_line"].notna().values | df["treatment_time_h"].notna().values
    u_sub[U_mask & ~has_pct] = "U0"
    u_sub[U_mask & has_pct & has_cond] = "O-PD"      # 有降解读数且有部分条件
    u_sub[U_mask & has_pct & ~has_cond] = "A-context"  # 有读数但条件缺失(代理)
    df["u_subtype"] = u_sub

    # ---------- 2. N 语义拆分 ----------
    n_sub = np.full(len(df), "", dtype=object)
    N_mask = lab == "N"
    dc50_exact_hi = (df["dc50_obs_type"] == "exact") & (df["dc50_value"] >= 1000)
    dc50_right = df["dc50_obs_type"] == "right-censored"
    dmax_exact_lo = (df["dmax_obs_type"] == "exact") & (df["dmax_value"] <= 20)
    # ⚠️ 修复 v1 (2026-08-16 彻查): left-censored 必须检查上界 <=20 才算低深度,
    #    不能把所有 left-censored 都当作低深度 (如 '<50'/'<=25' 上界 >20 不构成低深度证据)
    dmax_left_lo = (df["dmax_obs_type"] == "left-censored") & (df["dmax_upper"] <= 20)
    dmax_lo = dmax_exact_lo | dmax_left_lo
    # N-confirmed: 有 dmax 低深度佐证 (exact<=20 或 left 上界<=20) 且 dc50 高效力(高DC50)
    confirmed = dmax_lo & (dc50_exact_hi | dc50_right)
    # N-low-potency: dc50 高效力(高 DC50)但无 dmax 低深度佐证
    low_potency = (dc50_exact_hi | dc50_right) & ~dmax_lo
    # N-low-depth: 仅 dmax 低深度 (dc50 缺或未达高 DC50)
    low_depth = dmax_lo & ~(dc50_exact_hi | dc50_right)
    n_sub[N_mask & confirmed] = "N-confirmed"
    n_sub[N_mask & low_potency] = "N-low-potency"
    n_sub[N_mask & low_depth] = "N-low-depth"
    n_sub[N_mask & ~confirmed & ~low_potency & ~low_depth] = "N-low-potency"  # 兜底(无明确信号)
    df["n_subtype"] = n_sub

    # ---------- 3. 证据质量等级 (任务标签 vs 证据质量分离) ----------
    dc50_obs = df["dc50_obs_type"] != "endpoint-missing"
    dmax_obs = df["dmax_obs_type"] != "endpoint-missing"
    has_any_endpoint = dc50_obs | dmax_obs
    exact_endpoint = ((df["dc50_obs_type"] == "exact") | (df["dmax_obs_type"] == "exact"))
    censored_endpoint = has_any_endpoint & ~exact_endpoint
    cond_ok = df["cell_line"].notna() | df["treatment_time_h"].notna()
    has_doi = df["source_has_doi"] == "with_doi"
    grade = np.full(len(df), "D", dtype=object)
    # A: 有 exact 端点 + (条件完整或 DOI 可溯)
    grade[exact_endpoint & (cond_ok | has_doi)] = "A"
    # B: 有 exact 端点但条件缺失且无 DOI
    grade[exact_endpoint & ~(cond_ok | has_doi)] = "B"
    # C: 仅删失/区间端点
    grade[censored_endpoint] = "C"
    # D: 仅 pctdeg 或无端点
    df["evidence_grade"] = grade

    df["label_semantics_note"] = (
        "v2.2 语义拆分: u_subtype(U0/O-PD/A-context), n_subtype(N-confirmed/N-low-potency/N-low-depth), "
        "evidence_grade(A/B/C/D 代理等级, 基于现有字段, 非完整文献回溯等级)"
    )

    # 统计
    print("=== u_subtype ===")
    print(df[df["u_subtype"] != ""]["u_subtype"].value_counts().to_dict())
    print("\n=== n_subtype ===")
    print(df[df["n_subtype"] != ""]["n_subtype"].value_counts().to_dict())
    print("\n=== evidence_grade (全部) ===")
    print(df["evidence_grade"].value_counts().to_dict())
    print("\n=== 交叉: activity_evidence_v2 x evidence_grade ===")
    print(pd.crosstab(df["activity_evidence_v2"], df["evidence_grade"]))

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n已写回主表, 新增 u_subtype/n_subtype/evidence_grade/label_semantics_note")

    # 数据字典更新
    dd_path = os.path.join(OUT, "data_dictionary.csv")
    dd = pd.read_csv(dd_path, encoding="utf-8-sig")
    dd = dd[~dd["column_name"].isin(["u_subtype", "n_subtype", "evidence_grade", "label_semantics_note"])]
    new_rows = pd.DataFrame([
        {"column_name": "u_subtype", "description": "U 语义拆分: U0(无观测)/O-PD(有pctdeg)/A-context(有读数但条件不足)", "unit": "", "source_raw_column": "derived(v2.2)", "example_value": "O-PD"},
        {"column_name": "n_subtype", "description": "N 语义拆分: N-confirmed(双证据)/N-low-potency(高DC50无佐证)/N-low-depth(仅Dmax低)", "unit": "", "source_raw_column": "derived(v2.2)", "example_value": "N-low-potency"},
        {"column_name": "evidence_grade", "description": "证据质量等级 A/B/C/D (现有字段代理, 非完整文献回溯)", "unit": "", "source_raw_column": "derived(v2.2)", "example_value": "A"},
        {"column_name": "label_semantics_note", "description": "语义拆分说明", "unit": "", "source_raw_column": "fixed", "example_value": "v2.2 语义拆分..."},
    ])
    dd = pd.concat([dd, new_rows], ignore_index=True)
    dd.to_csv(dd_path, index=False, encoding="utf-8-sig")
    print("数据字典已更新")

if __name__ == "__main__":
    main()
