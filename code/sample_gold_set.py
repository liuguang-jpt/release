# -*- coding: utf-8 -*-
"""
sample_gold_set.py — 132 条内部一致性样本抽样与双模板生成
===============================================================

该样本与 150 条开发/校准标注不重叠，用于检查冻结协议的一致性。
它不是独立专家 gold standard，也不是外部验证。

输出位于 data/derived/gold_set_annotations/：
  gold_sample_132.csv
  template_annotator_A.csv
  template_annotator_B.csv
  GOLD_SAMPLING_DESIGN.md

历史名义目标为 200；冻结配额公式实际生成 121 条，再补入 11 条稀有删失保护记录，最终为 132 条。
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")
GDIR = os.path.join(OUT, "gold_set_annotations")
os.makedirs(GDIR, exist_ok=True)
SEED = 20260817

def main():
    df = pd.read_csv(os.path.join(OUT, "protac_clean_record_level.csv"), encoding="utf-8-sig")
    ann = pd.read_csv(os.path.join(OUT, "protac_annotation_150.csv"), encoding="utf-8-sig")
    dev_ids = set(ann["record_id"].tolist())   # 原 150 条开发集

    # 排除开发集
    cand = df[~df["record_id"].isin(dev_ids)].copy()
    print(f"候选记录 (排除原150条): {len(cand)}")

    # 分层键
    cand["stratum_obs"] = cand["dc50_obs_type"]
    cand["stratum_ev"] = cand["activity_evidence_v2"]
    cand["stratum_e3"] = cand["e3_ligase"].map(lambda x: x if x in ("CRBN", "VHL") else "other")
    cand["stratum_src"] = cand["source_has_doi"]
    cand["stratum_key"] = (cand["stratum_obs"] + "|" + cand["stratum_ev"] + "|" +
                           cand["stratum_e3"] + "|" + cand["stratum_src"])

    # 分层配额: 目标 200 条 (obs 5 类 x ev 4 类, 稀有层保底)
    n_target = 200
    quotas = {}
    strata_counts = cand["stratum_key"].value_counts()
    # 优先覆盖所有 obs x ev 组合 (5x4=20 cell)
    for key, cnt in strata_counts.items():
        quotas[key] = min(8, max(2, int(round(n_target * cnt / len(cand)))))
    # 稀有 obs 类型保底
    obs_counts = cand["stratum_obs"].value_counts()
    for ot in ["interval-censored", "left-censored", "right-censored"]:
        sub = cand[cand["stratum_obs"] == ot]
        need = 15 - quotas.get(f"{ot}|", 0)  # 粗略, 后面按 cell 补
    # 简化: 直接按 cell 抽样, 不足的补稀有 obs
    sampled = []
    rng = np.random.RandomState(SEED)
    for key, q in quotas.items():
        sub = cand[cand["stratum_key"] == key]
        k = min(q, len(sub))
        if k > 0:
            sampled.append(sub.sample(n=k, random_state=SEED))
    gold = pd.concat(sampled, ignore_index=True)
    # 补稀有 obs (interval/left 至少 15)
    for ot, min_n in [("interval-censored", 15), ("left-censored", 18), ("right-censored", 18)]:
        have = (gold["stratum_obs"] == ot).sum()
        if have < min_n:
            extra = cand[(cand["stratum_obs"] == ot) & (~cand["record_id"].isin(gold["record_id"]))]
            add = extra.sample(n=min(min_n - have, len(extra)), random_state=SEED)
            gold = pd.concat([gold, add], ignore_index=True)
    # 确保不与开发集重叠 + 无重复
    assert not gold["record_id"].isin(dev_ids).any(), "内部一致性样本与开发集重叠!"
    assert gold["record_id"].nunique() == len(gold), "内部一致性样本有重复记录"
    gold = gold.reset_index(drop=True)

    print(f"internal-consistency sample: {len(gold)} 条 (与原150条记录ID不重叠)")
    print("\n分层覆盖 (obs x ev):")
    print(pd.crosstab(gold["stratum_obs"], gold["stratum_ev"]))
    print("\nE3 覆盖:", gold["stratum_e3"].value_counts().to_dict())
    print("来源覆盖:", gold["stratum_src"].value_counts().to_dict())

    # ---------- 盲标模板: 只含原始证据字段, 不含任何派生标签/阈值 ----------
    gold_cols = ["record_id", "raw_row_index", "smiles", "target", "e3_ligase",
                 "cell_line", "treatment_time_h", "dc50_raw", "dmax_raw", "pctdeg_raw",
                 "article_doi"]
    gold_out = gold[gold_cols].copy()
    gold_out.to_csv(os.path.join(GDIR, "gold_sample_132.csv"), index=False, encoding="utf-8-sig")

    for who in ["A", "B"]:
        tpl = gold_out.copy()
        tpl["annotator_activity_evidence"] = ""   # P/N/A/U (盲标: 不提供任何阈值/示例)
        tpl["annotator_confidence"] = ""          # 高/中/低
        tpl["annotator_note"] = ""
        tpl.to_csv(os.path.join(GDIR, f"template_annotator_{who}.csv"), index=False, encoding="utf-8-sig")

    # ---------- 抽样设计说明 ----------
    design = f"""# 132 条仲裁内部一致性样本的抽样设计（历史 Gold 路径）

**生成日期：** 2026-08-16  
**脚本：** `code/sample_gold_set.py`  
**固定种子：** {SEED}

## 目的与边界

该样本用于检查冻结标注协议的内部一致性，并与原 150 条开发/校准标注保持记录 ID 不重叠。它不是独立专家 gold standard，也不是外部验证。

角色：A 为第一作者和规则开发者；B 为 AI 辅助标注者；仲裁者为第一作者本人，不是独立第三人。

## 抽样结果

配额公式 `min(8, max(2, round(200 × n_cell / N)))` 将每个分层单元上限设为 8，故名义目标 200 不可达：配额抽样 121 条，稀有删失保护补充 11 条，最终 {len(gold)} 条。不存在未完成的 68 条。

- E3 覆盖：{gold['stratum_e3'].value_counts().to_dict()}
- 来源覆盖：{gold['stratum_src'].value_counts().to_dict()}

## 盲法

模板仅包含原始证据字段，不包含派生标签、阈值、示例、模型预测或对方答案。A 是字段盲但不是知识盲；B 的 AI 辅助身份必须按期刊政策披露。

该样本不得用于阈值选择、先验估计、模型训练或校准。A/B 一致性和后续仲裁只支持协议执行的一致性，不支持独立专家真值或独立第三方裁定。
"""
    with open(os.path.join(GDIR, "GOLD_SAMPLING_DESIGN.md"), "w", encoding="utf-8") as f:
        f.write(design)
    print("内部一致性样本模板已生成:", GDIR)

if __name__ == "__main__":
    main()




