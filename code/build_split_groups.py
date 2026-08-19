import os
# -*- coding: utf-8 -*-
"""
build_split_groups.py — PROTAC Bias-Aware 数据工程 (Step 5: 划分分组键)

在已落地的记录级清洁主表 (protac_clean_record_level.csv, 15535 x 42) 之上,
为建模阶段 (研究执行计划 §8.4) 构建"防泄漏"分组键:
  - scaffold            : Murcko scaffold SMILES (scaffold / chemical-series split 的 group key)
  - pub_group           : 归一化 DOI (publication / patent-family split)
  - temporal_group/year : 文献年份 (temporal split; 仅 ~40% 有年份, 其余 UNKNOWN)
  - poi / e3 / poi_e3   : 靶标 / E3 / 组合 (cold-POI / cold-E3 / cold-combination split)

同时独立复现 5-check 数据合约, 确认清洁主表完整可复现。

输出:
  data/derived/protac_split_groups.csv
  data/derived/split_group_summary.json
  reports/figures/fig7_split_groups.png

不修改 data/raw/ 与既有清洁主表。
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROC = ROOT / "data/derived"
CODE = ROOT / "40_项目_Bias-Aware/code"
REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MAIN = PROC / "protac_clean_record_level.csv"
EXPECTED_SHAPE = (15535, 42)

# ----------------------------------------------------------------------------
# 0. 载入
# ----------------------------------------------------------------------------
df = pd.read_csv(MAIN)
n_rows, n_cols = df.shape

# ----------------------------------------------------------------------------
# 1. 5-check 数据合约 (独立复现)
# ----------------------------------------------------------------------------
contract = {}
contract["shape_ok"] = (n_rows, n_cols) == EXPECTED_SHAPE
contract["record_id_dtype_int"] = str(df["record_id"].dtype).startswith("int")
contract["record_id_unique"] = bool(df["record_id"].is_unique)
contract["max_col_missing"] = float(df.isna().mean().max())
contract["max_missing_lt_1"] = contract["max_col_missing"] < 1.0
contract["scaffold_nonnull_rate"] = float(df["scaffold"].notna().mean())
contract["panel_balance"] = "N/A (cross-sectional record-level, not a panel)"
contract["all_checks_pass"] = all(
    [contract["shape_ok"], contract["record_id_dtype_int"],
     contract["record_id_unique"], contract["max_missing_lt_1"]]
)

# 最稀缺列 (供建模阶段特征可用性参考)
miss = df.isna().mean().sort_values(ascending=False)
contract["top10_missing_columns"] = [
    {"column": c, "missing_rate": round(float(r), 4)} for c, r in miss.head(10).items()
]

# ----------------------------------------------------------------------------
# 2. 构建分组键
# ----------------------------------------------------------------------------
out = pd.DataFrame()
out["record_id"] = df["record_id"]
out["compound_id"] = df["compound_id"]
out["poi"] = df["target"].astype("string")
out["e3"] = df["e3_ligase"].astype("string")

# 2a. scaffold (已是 Murcko scaffold SMILES)
out["scaffold"] = df["scaffold"].astype("string")

# 2b. publication / patent-family: 归一化 DOI
def norm_doi(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip(".").strip()
    return s if s else None

out["pub_group"] = df["article_doi"].map(norm_doi).astype("object")
out["pub_group"] = out["pub_group"].fillna("NO_DOI").astype("string")

# 2c. temporal: 文献年份
year_int = pd.to_numeric(df["year_doi"], errors="coerce").astype("Int64")
out["year_int"] = year_int
out["temporal_group"] = year_int.astype("string").fillna("UNKNOWN")

# 2d. cold 组合
out["poi_e3"] = (out["poi"].fillna("NA") + "||" + out["e3"].fillna("NA")).astype("string")

# ----------------------------------------------------------------------------
# 3. 汇总: 组规模 / 冷划分可行性 / 泄漏估计
# ----------------------------------------------------------------------------
summary = {}
summary["n_records"] = int(n_rows)
summary["n_expected"] = EXPECTED_SHAPE[0]

n_scaffold_groups = int(out["scaffold"].nunique())
n_pub_groups = int(out["pub_group"].nunique())
n_poi = int(out["poi"].nunique())
n_e3 = int(out["e3"].nunique())
n_poi_e3 = int(out["poi_e3"].nunique())
n_year_known = int((out["temporal_group"] != "UNKNOWN").sum())

summary["n_scaffold_groups"] = n_scaffold_groups
summary["n_pub_groups"] = n_pub_groups
summary["n_poi"] = n_poi
summary["n_e3"] = n_e3
summary["n_poi_e3_combos"] = n_poi_e3
summary["n_year_known"] = n_year_known
summary["year_coverage"] = round(n_year_known / n_rows, 4)

# 单记录 scaffold 组占比 (即随机划分时会被"同 scaffold 跨折"泄漏的比例)
scaff_counts = out["scaffold"].value_counts()
shared_mask = out["scaffold"].map(scaff_counts) > 1
summary["frac_records_shared_scaffold"] = round(float(shared_mask.mean()), 4)
summary["note_scaffold_leakage"] = (
    "随机划分会把同 scaffold 记录同时放入训练/测试, 估计泄漏率≈此比例; "
    "须用 GroupShuffleSplit(scaffold) 或留系列划分。"
)

# 冷划分可行性: 各 POI / E3 / 组合 的测试折最小样本阈值
def cold_feasible(series, min_test=20):
    vc = series.value_counts()
    return {
        "n_groups": int(len(vc)),
        "n_groups_ge_%d" % min_test: int((vc >= min_test).sum()),
        "median_group_size": int(vc.median()),
        "min_group_size": int(vc.min()),
        "max_group_size": int(vc.max()),
    }

summary["cold_poi_feasible_min20"] = cold_feasible(out["poi"])
summary["cold_e3_feasible_min20"] = cold_feasible(out["e3"])
summary["cold_poi_e3_feasible_min20"] = cold_feasible(out["poi_e3"])

# 组规模分布 (供图)
scaff_sizes = scaff_counts.values
pub_sizes = out["pub_group"].value_counts().values
summary["scaffold_group_size"] = {
    "max": int(scaff_sizes.max()),
    "median": int(np.median(scaff_sizes)),
    "top10_counts": [int(x) for x in scaff_counts.head(10).values],
}
summary["pub_group_size"] = {
    "max": int(pub_sizes.max()),
    "median": int(np.median(pub_sizes)),
    "n_singleton": int((pub_sizes == 1).sum()),
}

# 推荐的划分配置 (供建模阶段直接引用)
summary["recommended_splits"] = {
    "random": "仅作与旧工作对照; record_id 直接 shuffle",
    "scaffold": "GroupShuffleSplit(group=scaffold); 主结果之一",
    "chemical_series": "同 scaffold 整体同侧 (group=scaffold 即可)",
    "publication": "GroupShuffleSplit(group=pub_group); 阻止同来源近邻泄漏",
    "temporal": "year_int 升序; 例: train<=2022 / test>=2023; 仅 %d 条有年份" % n_year_known,
    "cold_poi": "留一/留多 POI; 需 POI 组规模>=阈值",
    "cold_e3": "留一/留多 E3; 需 E3 组规模>=阈值",
    "cold_combination": "留 (POI,E3) 组合; 最难且有价值",
}

summary["data_contract"] = contract

# ----------------------------------------------------------------------------
# 4. 写出
# ----------------------------------------------------------------------------
out.to_csv(PROC / "protac_split_groups.csv", index=False)
with open(PROC / "split_group_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 5. 图: 组规模分布 + 组数概览
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) scaffold group-size distribution (log-x histogram)
axes[0, 0].hist(scaff_sizes, bins=np.logspace(0, np.log10(scaff_sizes.max() + 1), 30))
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlabel("scaffold group size (n records)")
axes[0, 0].set_ylabel("number of groups")
axes[0, 0].set_title("Scaffold group-size distribution (log-x)")
axes[0, 0].axvline(1, color="red", ls="--", lw=1, label="singleton = 1")
axes[0, 0].legend()

# (b) top-15 scaffold groups
top = scaff_counts.head(15)[::-1]
axes[0, 1].barh(range(len(top)), top.values, color="#4C72B0")
axes[0, 1].set_yticks(range(len(top)))
axes[0, 1].set_yticklabels([f"g{i}" for i in range(len(top))])
axes[0, 1].set_xlabel("n records")
axes[0, 1].set_title("Top-15 scaffold groups (largest leakage source)")

# (c) pub group-size distribution
axes[1, 0].hist(pub_sizes, bins=np.logspace(0, np.log10(pub_sizes.max() + 1), 30))
axes[1, 0].set_xscale("log")
axes[1, 0].set_xlabel("publication group size (n records)")
axes[1, 0].set_ylabel("number of groups")
axes[1, 0].set_title("Publication group-size distribution (log-x)")
axes[1, 0].axvline(1, color="red", ls="--", lw=1, label="singleton")
axes[1, 0].legend()

# (d) group-count overview
labels = ["scaffold", "pub", "POI", "E3", "POIxE3", "year_known"]
vals = [n_scaffold_groups, n_pub_groups, n_poi, n_e3, n_poi_e3, n_year_known]
axes[1, 1].bar(labels, vals, color="#55A868")
axes[1, 1].set_ylabel("number of unique groups")
axes[1, 1].set_title("Unique group counts by split dimension (n=%d)" % n_rows)
for i, v in enumerate(vals):
    axes[1, 1].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

fig.suptitle("PROTAC Bias-Aware — Leakage-free split grouping keys", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIGDIR / "fig7_split_groups.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. 控制台摘要
# ----------------------------------------------------------------------------
print("=== 数据合约 5-check ===")
for k, v in contract.items():
    print(f"  {k}: {v}")
print("\n=== 划分分组键 ===")
print(f"  records           : {n_rows}")
print(f"  scaffold groups   : {n_scaffold_groups}")
print(f"  pub groups        : {n_pub_groups}")
print(f"  POI               : {n_poi}")
print(f"  E3                : {n_e3}")
print(f"  POI x E3 combos   : {n_poi_e3}")
print(f"  year known        : {n_year_known} ({summary['year_coverage']*100:.1f}%)")
print(f"  shared-scaffold   : {summary['frac_records_shared_scaffold']*100:.1f}% (random-split 泄漏估计)")
print("\n已写出:")
print("  ", PROC / "protac_split_groups.csv")
print("  ", PROC / "split_group_summary.json")
print("  ", FIGDIR / "fig7_split_groups.png")
