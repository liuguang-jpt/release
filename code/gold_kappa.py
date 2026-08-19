# -*- coding: utf-8 -*-
"""
gold_kappa.py — 内部一致性样本双标注一致性计算 (Cohen's kappa + 逐类一致率)
========================================================================
输入: data/derived/gold_set_annotations/template_annotator_A.csv
      data/derived/gold_set_annotations/template_annotator_B.csv
      (两份均由标注者填写完成后回传)
输出: reports/GOLD_SET_AGREEMENT.md

用法: 两名标注者完成盲标后运行本脚本。
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GDIR = os.path.join(BASE, "data", "derived", "gold_set_annotations")
REP  = os.path.join(BASE, "reports")

def cohen_kappa(y1, y2):
    y1 = np.asarray(y1); y2 = np.asarray(y2)
    classes = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return float("nan")
    # 混淆矩阵
    conf = pd.crosstab(pd.Series(y1), pd.Series(y2))
    conf = conf.reindex(index=classes, columns=classes).fillna(0).values.astype(float)
    po = np.trace(conf) / n
    row_sums = conf.sum(axis=1); col_sums = conf.sum(axis=0)
    pe = (row_sums @ col_sums) / (n * n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)

def main():
    fa = os.path.join(GDIR, "template_annotator_A.csv")
    fb = os.path.join(GDIR, "template_annotator_B.csv")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print("标注未完成: 需要 template_annotator_A.csv 与 template_annotator_B.csv 均已填写。")
        print("先运行 sample_gold_set.py 生成模板, 由两名标注者独立盲标后回传。")
        return
    a = pd.read_csv(fa, encoding="utf-8-sig")
    b = pd.read_csv(fb, encoding="utf-8-sig")
    merged = a[["record_id", "annotator_activity_evidence", "annotator_confidence"]] \
        .merge(b[["record_id", "annotator_activity_evidence"]], on="record_id", suffixes=("_A", "_B"))
    merged = merged.dropna(subset=["annotator_activity_evidence_A", "annotator_activity_evidence_B"])
    if len(merged) == 0:
        print("无有效标注对, 请检查填写。")
        return

    yA = merged["annotator_activity_evidence_A"].values
    yB = merged["annotator_activity_evidence_B"].values
    kappa = cohen_kappa(yA, yB)
    agree = (yA == yB).mean()

    lines = []
    lines.append("# 内部一致性样本双标注一致性报告\n")
    lines.append(f"> 生成: 2026-08-16 ｜ 样本对: {len(merged)} ｜ 脚本: `gold_kappa.py`\n")
    lines.append(f"## 结果\n")
    lines.append(f"- **Cohen's κ = {kappa:.3f}**")
    lines.append(f"- 总体一致率 = {agree*100:.1f}%")
    lines.append("")
    lines.append("## 混淆矩阵 (标注者A x B)\n")
    conf = pd.crosstab(pd.Series(yA, name="A"), pd.Series(yB, name="B"))
    lines.append("```")
    lines.append(conf.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## 逐类一致率 (仅对双方都标注的样本)\n")
    classes = sorted(set(yA) | set(yB))
    for c in classes:
        m = (yA == c) | (yB == c)
        if m.sum() == 0:
            continue
        both = (yA == c) & (yB == c)
        lines.append(f"- {c}: 双方一致 {both.sum()}/{m.sum()} ({both.sum()/m.sum()*100:.1f}%)")
    lines.append("")
    lines.append("## 解读")
    lines.append("- κ ≥ 0.8: 良好一致, 可合并仲裁; 0.6-0.8: 中等, 需审查分歧; <0.6: 标注指南需修订。")
    lines.append("- 本研究的分歧由第一作者（标注者 A）按冻结规则复核，属于非独立仲裁；结果写入 gold_final.csv。")
    lines.append("")
    with open(os.path.join(REP, "GOLD_SET_AGREEMENT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"κ = {kappa:.3f}, 一致率 = {agree*100:.1f}%")
    print("wrote reports/GOLD_SET_AGREEMENT.md")

if __name__ == "__main__":
    main()

