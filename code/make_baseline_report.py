# -*- coding: utf-8 -*-
"""
make_baseline_report.py — 从 baseline_results_v2.json 渲染基线报告与图
========================================================================
输出:
  reports/BASELINE_REPORT.md    (人类可读基线报告, 中文)
  reports/figures/fig8_baseline_*.png  (图编号延续审计 fig1-7)
依赖 baseline_pipeline.py 产出的 data/derived/baseline_results_v2.json
运行: python make_baseline_report.py
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
DPI = 300

res = json.load(open(os.path.join(OUT, "baseline_results_v2.json"), encoding="utf-8"))

TASK_ZH = {
    "T1_pdc50_reg": "T1 pDC50 回归（DC50 exact）",
    "T2_dmax_reg": "T2 Dmax 回归（Dmax exact）",
    "T3_pn_clf": "T3 活性分类 P vs N（U/A 排除）",
    "T4_diag_uan_clf": "T4 诊断：U 当作 N（错误基线对照）",
}
SPLIT_ZH = {
    "random": "random", "scaffold": "scaffold", "pub": "publication",
    "poi": "cold-POI", "temporal": "temporal",
}
METRIC_ZH = {
    "mae": "MAE", "rmse": "RMSE", "r2": "R²", "spearman": "Spearman",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC",
    "balanced_acc": "Bal.Acc", "brier": "Brier", "ece": "ECE",
}

def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else "—"

# ---------------- 报告文本 ----------------
lines = []
lines.append("# BASELINE REPORT — PROTAC Bias-Aware PNU 项目：第一版简单强基线\n")
lines.append(f"> 生成: {res['meta']['generated']}  |  脚本: `{res['meta']['script']}`")
lines.append(f"> 数据: `protac_clean_record_level.csv`（{res['meta']['records_total']:,} 条记录）+ `protac_split_groups.csv`")
lines.append(f"> 种子: {res['meta']['seeds']}  |  划分比: {res['meta']['split_ratio']}")
lines.append(f"> 特征: {res['meta']['features']['F1_morgan']}（F1）；{res['meta']['features']['F2_morgan_ctx']}（F2）")
lines.append(f"> 标签: `{res['meta']['label_col']}` — {res['meta']['label_note']}\n")
lines.append("> 本报告目的：检验管线可运行性、任务难度与划分泄漏；结果供路线决策与论文引用（正式投稿前须随标注/阈值版本冻结重跑）。\n")

lines.append("## 1. 任务规模与正类率\n")
lines.append("| 任务 | 说明 | 记录数 | 正类率 |")
lines.append("|---|---|---|---|")
for tid, t in res["tasks"].items():
    pr = f"{t['pos_rate']*100:.1f}%" if t["pos_rate"] is not None else "—"
    lines.append(f"| {TASK_ZH.get(tid, tid)} | {t['desc']} | {t['n_records']:,} | {pr} |")
lines.append("")

lines.append("## 2. 主结果：F1（纯 Morgan）跨划分\n")
lines.append("> 回归看 MAE/RMSE/R²/Spearman；分类看 ROC-AUC/PR-AUC/MCC/Bal.Acc；校准看 Brier/ECE（越小越好）。mean±sd 跨 3 种子。\n")

for tid in ["T1_pdc50_reg", "T2_dmax_reg", "T3_pn_clf", "T4_diag_uan_clf"]:
    if tid not in res["splits"]:
        continue
    metrics = {"T1_pdc50_reg": ["mae", "rmse", "r2", "spearman"],
               "T2_dmax_reg": ["mae", "rmse", "r2", "spearman"],
               "T3_pn_clf": ["roc_auc", "pr_auc", "mcc", "balanced_acc", "brier", "ece"],
               "T4_diag_uan_clf": ["roc_auc", "pr_auc", "mcc", "balanced_acc", "brier", "ece"]}[tid]
    lines.append(f"### {TASK_ZH.get(tid, tid)}\n")
    lines.append("| 划分 | " + " | ".join(METRIC_ZH[m] for m in metrics) + " | 泄漏率(s/p/poi) |")
    lines.append("|---|" + "---|" * len(metrics) + "---|")
    for sname in ["random", "scaffold", "pub", "poi", "temporal"]:
        if sname not in res["splits"][tid]:
            continue
        e = res["splits"][tid][sname]
        cells = []
        for m in metrics:
            v = e.get(f"F1_morgan::{m}")
            if v:
                cells.append(f"{fmt(v['mean'])}±{fmt(v['sd'])}")
            else:
                cells.append("—")
        lk = e.get("leakage", {})
        lks = f"{lk.get('scaffold', float('nan'))*100:.0f}/{lk.get('pub', float('nan'))*100:.0f}/{lk.get('poi', float('nan'))*100:.0f}%" if lk else "—"
        lines.append(f"| {SPLIT_ZH[sname]} | " + " | ".join(cells) + f" | {lks} |")
    lines.append("")

lines.append("## 3. 关键解读（供项目决策）\n")
lines.append("1. **管线可运行性**：本报告证明 特征→划分→训练→指标 全链路可复现运行，后续换模型只改训练层。")
lines.append("2. **泄漏量化**：各划分下的 scaffold/pub/POI 泄漏率应接近 0%（分组划分）或高（random），据此量化随机划分的乐观偏差。")
lines.append("3. **任务难度**：pDC50 与 Dmax 的 R²/Spearman 可判断定量任务是否有可学信号；P/N 分类的 PR-AUC 在 P:N≈82:18 不平衡下的水平决定下游模型空间。")
lines.append("4. **U-as-N 诊断（T4）**：与 T3 对比可量化'把 U 当负样本'造成的性能虚高/偏倚，是论文'U≠N'论点的直接证据。")
lines.append("5. **F1 vs F2**：若 F2（含 POI/E3 上下文）在 cold-POI 划分下相对 F1 无提升，说明模型主要记忆靶点而非学习结构规律——与 Ribes et al. 的观察一致。")
lines.append("")
lines.append("## 4. 局限\n")
lines.append("- 标签为规则派生（v1.0，经 150 条人工校准，一致率 94.7% 最优参数）；阈值敏感性（CP 50/100/300/1000）须在论文中报告。")
lines.append("- temporal 划分仅覆盖有 DOI 年份的记录（约 40.2%），其样本构成与其他划分不同，比较时注意口径。")
lines.append("- XGBoost 超参为固定默认档（300 树 / depth 6 / lr 0.05），未调优；作为基线足够。")
lines.append("- 标签版本历史：provisional（80/10 OR，`baseline_results.json`）→ v1（双终点一致 70/20，已废弃，一致率仅 82.7%）→ **v2（OR 70/20 + 脏值防护，一致率 94.7%，当前定稿）**。")
lines.append("")

with open(os.path.join(REP, "BASELINE_REPORT.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote reports/BASELINE_REPORT.md")

# ---------------- 图 ----------------
def plot_task_bars(tid, metrics, fname, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    if len(metrics) == 1:
        axes = [axes]
    splits = [s for s in ["random", "scaffold", "pub", "poi", "temporal"] if s in res["splits"].get(tid, {})]
    for ax, m in zip(axes, metrics):
        means, sds = [], []
        for s in splits:
            v = res["splits"][tid][s].get(f"F1_morgan::{m}")
            means.append(v["mean"] if v else np.nan)
            sds.append(v["sd"] if v else 0.0)
        x = np.arange(len(splits))
        ax.bar(x, means, yerr=sds, capsize=3, color="#1f77b4", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([SPLIT_ZH[s] for s in splits], fontsize=8)
        ax.set_title(METRIC_ZH[m], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for xi, (me, sd) in enumerate(zip(means, sds)):
            ax.text(xi, me + sd + 0.01 * (max(means) - min(means) + 1e-9),
                    f"{me:.3f}", ha="center", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=DPI)
    plt.close(fig)
    print("wrote", fname)

plot_task_bars("T1_pdc50_reg", ["mae", "rmse", "r2", "spearman"],
               "fig8_baseline_t1_pdc50.png", "T1 pDC50 regression — F1 Morgan across splits (mean±sd, 3 seeds)")
plot_task_bars("T2_dmax_reg", ["mae", "rmse", "r2", "spearman"],
               "fig9_baseline_t2_dmax.png", "T2 Dmax regression — F1 Morgan across splits (mean±sd, 3 seeds)")
plot_task_bars("T3_pn_clf", ["roc_auc", "pr_auc", "mcc", "balanced_acc", "brier", "ece"],
               "fig10_baseline_t3_pnclf.png", "T3 P/N classification — F1 Morgan across splits (mean±sd, 3 seeds)")
plot_task_bars("T4_diag_uan_clf", ["roc_auc", "pr_auc", "mcc", "balanced_acc", "brier", "ece"],
               "fig11_baseline_t4_uasn.png", "T4 diagnostic U-as-N — F1 Morgan across splits (mean±sd, 3 seeds)")
print("done")
