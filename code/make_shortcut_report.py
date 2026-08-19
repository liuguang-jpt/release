# -*- coding: utf-8 -*-
"""
make_shortcut_report.py — 渲染 shortcut 负对照报告
输出: reports/SHORTCUT_CONTROLS.md + reports/figures/fig17_shortcut_controls.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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

res = json.load(open(os.path.join(OUT, "shortcut_controls.json"), encoding="utf-8"))

SPLIT_ZH = {"random": "random", "scaffold": "scaffold", "pub": "publication", "poi": "cold-POI"}
FEAT_ZH = {
    "structure_only": "结构 (Morgan)",
    "metadata_only": "元数据 (target+E3+来源+年份)",
    "target_only": "仅靶点",
    "e3_only": "仅E3",
    "source_only": "仅来源",
    "publication_only": "仅论文系列",
    "structure_perm_full": "结构+完全置换标签",
}

lines = []
lines.append("# 捷径负对照报告（shortcut negative controls）\n")
lines.append(f"> 生成: 2026-08-16  |  脚本: `shortcut_controls.py` ｜ 协议: 监督 P/N, group-first split, 泄漏断言")
lines.append(f"> 目的: 检验分类器高分来自「可迁移化学规律」还是「靶点/E3/论文/元数据捷径」(反方审稿 Major 1)\n")

lines.append("## 1. 各特征集 ROC-AUC（mean±sd，3 seeds）\n")
lines.append("| 特征集 | random | scaffold | publication | cold-POI |")
lines.append("|---|---|---|---|---|")
for f in ["structure_only", "metadata_only", "target_only", "e3_only", "source_only", "publication_only"]:
    cells = []
    for s in ["random", "scaffold", "pub", "poi"]:
        e = res["splits"][s].get(f"{f}::roc_auc")
        cells.append(f"{e['mean']:.3f}±{e['sd']:.3f}" if e else "—")
    lines.append(f"| {FEAT_ZH[f]} | " + " | ".join(cells) + " |")
lines.append("")

lines.append("## 2. 标签置换检验（structure 特征，ROC-AUC）\n")
lines.append("| 划分 | 原标签 | 组内置换 | 完全置换 |")
lines.append("|---|---|---|---|")
for s in ["random", "scaffold", "pub", "poi"]:
    e = res["splits"][s]
    cells = [f"{e['structure_only::roc_auc']['mean']:.3f}",
             f"{e['structure_perm_grp::roc_auc']['mean']:.3f}" if "structure_perm_grp::roc_auc" in e else "—",
             f"{e['structure_perm_full::roc_auc']['mean']:.3f}"]
    lines.append(f"| {SPLIT_ZH[s]} | " + " | ".join(cells) + " |")
lines.append("")

lines.append("## 3. 关键解读\n")
lines.append("1. **完全标签置换（perm_full）全部回落到 0.47–0.53 ≈ 随机** → 协议无泄漏、训练-测试无信息污染，学习信号真实；")
lines.append("2. **靶点捷径在 cold-POI 下被完全切断**：target-only AUC 0.500（测试 POI 未见于训练 → 无信息）→ 冷靶点划分有效；")
lines.append("3. **论文捷径在 publication 下被完全切断**：publication-only AUC 0.500 → 论文划分有效；")
lines.append("4. **捷径确实存在，但结构有增量**：random/scaffold 下 metadata-only 达 structure 的 89%/91%，但结构仍领先 0.09–0.10；")
lines.append("5. **严格划分下结构增量更大**：publication 下 structure 0.681 vs meta 0.621；cold-POI 下 0.793 vs 0.647 → 切断捷径后结构信息价值凸显；")
lines.append("6. ⚠️ 组内置换（perm_grp）在 random 下未打乱（无组键，代码分支不生效）属实现缺陷；scaffold 下单例组占 85.7% 使组内置换近似无效——**以 perm_full 为准**。\n")

lines.append("## 4. 对论文的意义\n")
lines.append("- 支持「结构模型学到部分真实化学规律」：完全置换标签后 AUC≈0.5，且结构在严格划分下保持增量；")
lines.append("- 但元数据捷径贡献显著（random 下达 89%）：正文必须加入本负对照，并报告「结构相对元数据的增量」而非绝对 AUC；")
lines.append("- 冷靶点/论文捷径被切断的结果，同时验证了 group split 协议本身的有效性。\n")

lines.append("## 5. 局限\n")
lines.append("- 组内置换实现缺陷（random 未打乱、scaffold 单例组问题）→ 泄漏检验以完全置换为准；")
lines.append("- 元数据特征未含实验室/团队级变量（protac.csv 无此字段）；")
lines.append("- publication-only 的 NO_DOI 巨组（1,632 条）可能扭曲 publication split 的捷径估计。\n")

with open(os.path.join(REP, "SHORTCUT_CONTROLS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote reports/SHORTCUT_CONTROLS.md")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
# 左: 特征集对比
splits = ["random", "scaffold", "pub", "poi"]
feats = ["structure_only", "metadata_only", "target_only", "publication_only"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ax = axes[0]
x = np.arange(len(splits))
for f, c in zip(feats, colors):
    means = [res["splits"][s][f"{f}::roc_auc"]["mean"] for s in splits]
    sds = [res["splits"][s][f"{f}::roc_auc"]["sd"] for s in splits]
    ax.errorbar(x, means, yerr=sds, fmt="o-", color=c, label=FEAT_ZH[f], capsize=3)
ax.axhline(0.5, ls="--", color="gray", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([SPLIT_ZH[s] for s in splits])
ax.set_ylim(0.4, 1.0); ax.set_ylabel("ROC-AUC"); ax.set_title("负对照: 捷径 vs 结构")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
# 右: 置换检验
ax2 = axes[1]
perm_splits = ["random", "scaffold", "pub", "poi"]
orig = [res["splits"][s]["structure_only::roc_auc"]["mean"] for s in perm_splits]
full = [res["splits"][s]["structure_perm_full::roc_auc"]["mean"] for s in perm_splits]
x2 = np.arange(len(perm_splits))
ax2.plot(x2, orig, "o-", color="#1f77b4", label="原标签")
ax2.plot(x2, full, "s--", color="#d62728", label="完全置换标签")
ax2.axhline(0.5, ls=":", color="gray", lw=0.8)
ax2.set_xticks(x2); ax2.set_xticklabels([SPLIT_ZH[s] for s in perm_splits])
ax2.set_ylim(0.4, 1.0); ax2.set_ylabel("ROC-AUC"); ax2.set_title("标签置换检验 (泄漏检查)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.suptitle("Shortcut 负对照 (3 seeds, group-first split, 泄漏断言)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig17_shortcut_controls.png"), dpi=DPI)
plt.close(fig)
print("wrote fig17_shortcut_controls.png")
print("done")
