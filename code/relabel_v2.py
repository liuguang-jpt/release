# -*- coding: utf-8 -*-
"""
relabel_v2.py — 修正 activity_evidence 标签规则 (OR 逻辑 + 精细脏值防护)
==========================================================================
背景: v1 标签用了"双终点一致优先"逻辑, 但在 150 条人工标注上一致率仅 82.7%,
      低于"任一信号 OR"逻辑的 94.7%。本脚本回退为 OR 逻辑, 并修正脏值防护阈值。

v2 规则 (校准后, 一致率应达 94.7%):
  P 信号: DC50<=100nM 或 Dmax>=70%
  N 信号: DC50>=1000nM 或 Dmax<=20%
  冲突(p&n同时): A
  仅 p: P; 仅 n: N; 双缺: U; 有观测但不触发: A
  脏值防护: Dmax>150 或 Dmax<0 或 DC50<=0 视为该终点无效(不参与信号), 不整体判 U
   (Dmax 101/106/107/112 为测量波动, 保留; 仅 2000 这类明显错误才防护)

输出: 主表新增 activity_evidence_v2 列 (保留 v1 作对照)
运行: .../python.exe relabel_v2.py
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")

def rule_v2(r, cp=100.0, dp=70.0, dn=20.0):
    """OR 逻辑 + 精细脏值防护。返回 (标签, 信号说明)。"""
    dc, dm = r["dc50_obs_type"], r["dmax_obs_type"]
    dv, dl, du = r["dc50_value"], r["dc50_lower"], r["dc50_upper"]
    mv, ml, mu = r["dmax_value"], r["dmax_lower"], r["dmax_upper"]
    cn = 10.0 * cp
    # 脏值防护: 单终点脏值 → 该终点无效 (视为缺失), 不整体判 U
    dc_ok = not (dc == "exact" and pd.notna(dv) and dv <= 0)
    dm_ok = not (dm == "exact" and pd.notna(mv) and (mv > 150 or mv < 0))
    p = n = 0; sig = []
    if dc_ok:
        if dc == "exact" and pd.notna(dv):
            if dv <= cp: p += 1; sig.append(f"DC50={dv:g}<=100")
            elif dv >= cn: n += 1; sig.append(f"DC50={dv:g}>=1000")
        elif dc == "left-censored" and pd.notna(du):
            if du <= cp: p += 1; sig.append(f"DC50<{du:g}")
        elif dc == "right-censored" and pd.notna(dl):
            if dl >= cn: n += 1; sig.append(f"DC50>{dl:g}")
        elif dc == "interval-censored" and pd.notna(dl) and pd.notna(du):
            if du <= cp: p += 1; sig.append("DC50_int<=100")
            elif dl >= cn: n += 1; sig.append("DC50_int>=1000")
    if dm_ok:
        if dm == "exact" and pd.notna(mv):
            if mv >= dp: p += 1; sig.append(f"Dmax={mv:g}>=70")
            elif mv <= dn: n += 1; sig.append(f"Dmax={mv:g}<=20")
        elif dm == "right-censored" and pd.notna(ml):
            if ml >= dp: p += 1; sig.append(f"Dmax>{ml:g}")
        elif dm == "left-censored" and pd.notna(mu):
            if mu <= dn: n += 1; sig.append(f"Dmax<{mu:g}")
    if p > 0 and n > 0:
        return "A", ";".join(sig)
    if p > 0:
        return "P", ";".join(sig)
    if n > 0:
        return "N", ";".join(sig)
    if dc == "endpoint-missing" and dm == "endpoint-missing":
        return "U", ";".join(sig)
    return "A", ";".join(sig)

def main():
    path = os.path.join(OUT, "protac_clean_record_level.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")

    # 全量打标
    ev = df.apply(lambda r: rule_v2(r), axis=1)
    df["activity_evidence_v2"] = [e[0] for e in ev]
    df["activity_signal_v2"] = [e[1] for e in ev]
    df["activity_evidence_v2_note"] = "v2 定稿: OR 逻辑, CP=100/N=1000, Dmax P>=70%/N<=20%; 脏值 Dmax>150/DC50<=0 单终点失效; 见规范 §5 v2"

    # 150 条上验证一致率
    ann = pd.read_excel(os.path.join(OUT, "protac_annotation_150_raw_submission.xlsx"))
    merged = ann.merge(df[["record_id", "activity_evidence_v2"]], on="record_id", how="left")
    acc = float((merged["activity_evidence_v2"] == merged["annotator_activity_evidence"]).mean()) * 100
    print(f"全量 v2 分布: {df['activity_evidence_v2'].value_counts().to_dict()}")
    print(f"150 条人工标注一致率: {acc:.1f}%")

    # 敏感性: 多 cutoff
    print("\n=== v2 敏感性 (OR 逻辑, 多 CP) ===")
    for cp in [50, 100, 300, 1000]:
        def rule_cp(r):
            return rule_v2(r, cp=cp)[0]
        labs = df.apply(rule_cp, axis=1)
        print(f"CP={cp} (N={cp*10}): {labs.value_counts().to_dict()}")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已写回主表, 新增 activity_evidence_v2 列")

    # 更新数据字典
    dd_path = os.path.join(OUT, "data_dictionary.csv")
    dd = pd.read_csv(dd_path, encoding="utf-8-sig")
    # 移除旧 v1 行(若有), 添加 v2 行
    dd = dd[~dd["column_name"].isin(["activity_evidence_v2", "activity_signal_v2", "activity_evidence_v2_note"])]
    new_rows = pd.DataFrame([
        {"column_name": "activity_evidence_v2", "description": "活动证据 v2 定稿: OR 逻辑 P/N/A/U (CP=100/N=1000, Dmax 70/20, 脏值单终点失效)", "unit": "", "source_raw_column": "rule(v2 §5)", "example_value": "P"},
        {"column_name": "activity_signal_v2", "description": "触发 v2 标签的规则明细", "unit": "", "source_raw_column": "rule", "example_value": "DC50=45<=100"},
        {"column_name": "activity_evidence_v2_note", "description": "v2 规则说明", "unit": "", "source_raw_column": "fixed", "example_value": "v2 定稿规则..."},
    ])
    dd = pd.concat([dd, new_rows], ignore_index=True)
    dd.to_csv(dd_path, index=False, encoding="utf-8-sig")
    print("数据字典已更新")

if __name__ == "__main__":
    main()
