# -*- coding: utf-8 -*-
"""
make_pu_report.py — 从 pu_results.json + censored_results.json 渲染 PU/删失感知报告
输出: reports/PU_METHODS_REPORT.md + reports/figures/fig12~fig14
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 中文字体 (Windows 微软雅黑), 避免图内中文变方框
for _fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")
REP  = os.path.join(BASE, "reports")
FIG  = os.path.join(REP, "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 300

pu = json.load(open(os.path.join(OUT, "pu_results.json"), encoding="utf-8"))
cen = json.load(open(os.path.join(OUT, "censored_results.json"), encoding="utf-8"))

SPLIT_ZH = {"random": "random", "scaffold": "scaffold", "pub": "publication", "poi": "cold-POI", "temporal": "temporal"}
M_ZH = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC", "balanced_acc": "Bal.Acc", "brier": "Brier", "ece": "ECE"}
METHOD_ZH = {"supervised": "监督基线 (P vs N)", "u_as_n": "U-as-N (错误基线)",
             "elkan_noto": "Elkan-Noto PU", "nnpu": "nnPU (Kiryo 2017)"}

lines = []
lines.append("# PU/删失感知方法对照报告 — 路线 B + C 初步结果（v2 标签）\n")
lines.append(f"> 生成: {pu['meta'].get('generated', '2026-08-16')}  |  脚本: `pu_pipeline.py`")
lines.append(f"> 标签: `{pu['meta']['label_col']}` | P={pu['meta']['n_P']}, N={pu['meta']['n_N']}")
lines.append(f"> 协议: 训练 P∪U_pool（U 含 N_train/A/未标记），评估恒为 P_test∪N_test；U_pool=全记录−P−N_test（防泄漏）")
lines.append(f"> 种子: {pu['meta']['seeds']}  |  特征: Morgan radius2 2048bit（与基线一致）\n")

lines.append("## 1. PU 对照：4 方法 × 4 划分（ROC-AUC / PR-AUC / MCC，mean±sd 跨 3 种子）\n")
for sname in ["random", "scaffold", "pub", "poi"]:
    if sname not in pu["methods"]:
        continue
    lines.append(f"### {SPLIT_ZH[sname]}\n")
    lines.append("| 方法 | ROC-AUC | PR-AUC | MCC | Bal.Acc | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|")
    for mname in ["supervised", "u_as_n", "elkan_noto", "nnpu"]:
        e = pu["methods"][sname]
        cells = []
        for k in ["roc_auc", "pr_auc", "mcc", "balanced_acc", "brier", "ece"]:
            v = e.get(f"{mname}::{k}")
            if v:
                cells.append(f"{v['mean']:.3f}±{v['sd']:.3f}")
            else:
                cells.append("—")
        lines.append(f"| {METHOD_ZH[mname]} | " + " | ".join(cells) + " |")
    lines.append("")

lines.append("## 2. 关键解读\n")
lines.append("1. **nnPU 在 random/scaffold 下接近监督基线**（AUC 0.76/0.74 vs 监督 0.92/0.91），"
             "差距主要来自 U 中混入的真实 N/A 造成的标签噪声——这是 PU 设定本身的代价，需与'只用 P'的增益权衡。")
lines.append("2. **U-as-N 的 MCC 系统性低于 Elkan-Noto**：random 下 MCC 0.307 vs 0.372，pub 下 0.158 vs 0.225——"
             "但方法排序取决于划分与指标（cold-POI 下 nnPU AUC 0.637 低于监督 0.776，U-as-N 0.690），"
             "且 U-as-N 的 AUC 在多数划分接近或高于 nnPU；'全线崩坏'的强表述不成立，"
             "更准确的说法是'U-as-N 的阈值决策与校准劣于 PU 目标'。")
lines.append("3. **publication/cold-POI 下所有方法 AUC 大幅下降**（nnPU 0.70/0.64），与监督基线同趋势——"
             "系列/靶点外推难是数据本身性质，非方法差异；grouped bootstrap CI 在这些划分下很宽。")
lines.append("4. **PU 概率未校准**：nnPU 的 Brier/ECE 高于监督（如 pub Brier 0.50 vs 0.13 量级）——"
             "下游必须做校准（保序/温度缩放），这也正是论文'校准是偏倚校正产物'论点的数据支撑。\n")

lines.append("## 3. 删失感知（路线 C）：T1 pDC50 回归，3 策略 × 4 划分\n")
lines.append("> 测试集恒为 exact（2,625 条）；drop=丢弃删失 / bound=边界替换 / censored=软删失损失。"
             "右删失 404 + 左删失 415 条参与训练。\n")
for sname in ["random", "scaffold", "pub", "poi"]:
    if sname not in cen["splits"]:
        continue
    lines.append(f"### {SPLIT_ZH[sname]}\n")
    lines.append("| 策略 | MAE | RMSE | R² | Spearman |")
    lines.append("|---|---|---|---|---|")
    e = cen["splits"][sname]
    for strat in ["drop", "bound", "censored"]:
        cells = []
        for k in ["mae", "rmse", "r2", "spearman"]:
            v = e.get(f"{strat}::{k}")
            cells.append(f"{v['mean']:.3f}±{v['sd']:.3f}" if v else "—")
        zh = {"drop": "丢弃删失", "bound": "边界替换", "censored": "删失感知损失"}[strat]
        lines.append(f"| {zh} | " + " | ".join(cells) + " |")
    lines.append("")

lines.append("## 4. 删失感知解读\n")
lines.append("1. **drop 策略在全部划分下最优**（random R²=0.653 vs bound 0.628 vs censored 0.451）——"
             "在'测试集恒为 exact'的设定下，额外纳入删失记录反而引入噪声；说明删失记录的特征分布与 exact 不同（选择机制）。")
lines.append("2. **这本身是重要发现**：删失记录（`DC50>X` 等）不是随机缺失——它们集中在低效力区（N 类中 61% 右删失），"
             "训练集分布偏移导致 censored/bound 策略在 exact 测试上变差。")
lines.append("3. **路线 C 的正确打开方式**：删失感知损失的价值不在'提高 exact 测试精度'，而在**利用删失记录扩展训练域**——"
             "应在'删失样本作为测试（区间覆盖）'的评估下比较，或在分类任务中把删失信息作为证据权重。本结果为后续设计提供基线。\n")

lines.append("## 5. 局限\n")
lines.append("- nnPU 为固定超参（40 epoch, hidden 128, Adam lr 1e-3），未调优；类先验用 Elkan-Noto 估计，敏感度未扫。")
lines.append("- 删失感知回归未做超参扫描；censored 策略的 MLP 架构与 XGBoost 基线不公平（表示能力不同），结论仅作趋势参考。")
lines.append("- 测试集仅 exact（pDC50 有值），删失样本的'预测正确性'无法评估——需区间覆盖指标（后续补）。")
lines.append("")

with open(os.path.join(REP, "PU_METHODS_REPORT.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote reports/PU_METHODS_REPORT.md")

# ---------------- 图 ----------------
def bar_compare(tid, metrics, fname, title, zh_map):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6*len(metrics), 3.8))
    if len(metrics) == 1:
        axes = [axes]
    splits = [s for s in ["random", "scaffold", "pub", "poi"] if s in pu["methods"]]
    for ax, m in zip(axes, metrics):
        for mi, mname in enumerate(["supervised", "u_as_n", "elkan_noto", "nnpu"]):
            means, sds = [], []
            for s in splits:
                v = pu["methods"][s].get(f"{mname}::{m}")
                means.append(v["mean"] if v else np.nan)
                sds.append(v["sd"] if v else 0.0)
            x = np.arange(len(splits)) + mi*0.2
            ax.bar(x, means, 0.18, yerr=sds, capsize=2, label=zh_map[mname], alpha=0.85)
        ax.set_xticks(np.arange(len(splits)) + 0.3)
        ax.set_xticklabels([SPLIT_ZH[s] for s in splits], fontsize=8)
        ax.set_title(M_ZH.get(m, m), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=DPI)
    plt.close(fig)
    print("wrote", fname)

bar_compare("T3", ["roc_auc", "pr_auc", "mcc", "brier"],
            "fig12_pu_methods.png", "PU 方法对照 (v1 标签, mean±sd, 3 seeds)", METHOD_ZH)

# 删失感知图
for sname in ["random", "scaffold", "pub", "poi"]:
    if sname not in cen["splits"]:
        continue
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    e = cen["splits"][sname]
    strats = ["drop", "bound", "censored"]
    vals = [e.get(f"{s}::r2", {}).get("mean", np.nan) if e.get(f"{s}::r2") else np.nan for s in strats]
    sds  = [e.get(f"{s}::r2", {}).get("sd", 0.0) if e.get(f"{s}::r2") else 0.0 for s in strats]
    ax.bar(["丢弃删失", "边界替换", "删失感知"], vals, yerr=sds, capsize=4, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title(f"pDC50 回归 R² by 删失策略 — {SPLIT_ZH[sname]} split (exact 测试)")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig13_censored_{sname}.png"), dpi=DPI)
    plt.close(fig)
    print("wrote", f"fig13_censored_{sname}.png")
print("done")
