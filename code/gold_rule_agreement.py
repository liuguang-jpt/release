# -*- coding: utf-8 -*-
"""
gold_rule_agreement.py — P1 闭合：Gold 标注链可审计性
======================================================
回答盲审 C3 的两项量化问题:
  Q9: 分别用标注者 A、标注者 B、最终标签 计算"自动规则 vs 人工标签"一致率;
  Q7/8: 重放 sample_gold_set.py 的分层配额逻辑, 验证 200 目标 vs 132 实际产出的来源。

只读: 不修改任何模板/gold 文件。输出:
  data/derived/gold_rule_agreement_by_annotator.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "data", "derived")
GDIR = os.path.join(OUT, "gold_set_annotations")
LABELS = ["P", "N", "A", "U"]
SEED = 20260817


def multiclass(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            lab: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])}
            for i, lab in enumerate(LABELS)
        },
        "confusion_gold_rows_rule_columns": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def replay_sampling_quotas():
    """重放 sample_gold_set.py 的配额逻辑(只读), 解释 132 的来源。"""
    df = pd.read_csv(os.path.join(OUT, "protac_clean_record_level.csv"), encoding="utf-8-sig")
    ann = pd.read_csv(os.path.join(OUT, "protac_annotation_150.csv"), encoding="utf-8-sig")
    dev_ids = set(ann["record_id"].tolist())
    cand = df[~df["record_id"].isin(dev_ids)].copy()
    cand["stratum_obs"] = cand["dc50_obs_type"]
    cand["stratum_ev"] = cand["activity_evidence_v2"]
    cand["stratum_e3"] = cand["e3_ligase"].map(lambda x: x if x in ("CRBN", "VHL") else "other")
    cand["stratum_src"] = cand["source_has_doi"]
    cand["stratum_key"] = (cand["stratum_obs"] + "|" + cand["stratum_ev"] + "|" +
                           cand["stratum_e3"] + "|" + cand["stratum_src"])
    n_target = 200
    strata_counts = cand["stratum_key"].value_counts()
    quotas = {}
    for key, cnt in strata_counts.items():
        quotas[key] = min(8, max(2, int(round(n_target * cnt / len(cand)))))
    sampled = []
    rng = np.random.RandomState(SEED)
    for key, q in quotas.items():
        sub = cand[cand["stratum_key"] == key]
        k = min(q, len(sub))
        if k > 0:
            sampled.append(sub.sample(n=k, random_state=SEED))
    gold = pd.concat(sampled, ignore_index=True)
    quota_sum = len(gold)
    # 保底补抽 (interval>=15, left>=18, right>=18)
    for ot, min_n in [("interval-censored", 15), ("left-censored", 18), ("right-censored", 18)]:
        have = (gold["stratum_obs"] == ot).sum()
        if have < min_n:
            extra = cand[(cand["stratum_obs"] == ot) & (~cand["record_id"].isin(gold["record_id"]))]
            add = extra.sample(n=min(min_n - have, len(extra)), random_state=SEED)
            gold = pd.concat([gold, add], ignore_index=True)
    obs_before = quota_sum
    obs_after = len(gold)
    # 理论配额上限: 5 obs x 4 ev = 20 格, 每格 cap=8 -> 160; 实际由各层数量决定
    n_cells_capped = sum(1 for k, q in quotas.items() if q == 8)
    n_cells_limited = sum(1 for k, q in quotas.items() if q < min(8, strata_counts.get(k, 0)))
    return {
        "n_target_named": n_target,
        "n_quota_cells": len(quotas),
        "n_cells_at_cap_8": int(n_cells_capped),
        "n_cells_limited_by_stratum_size": int(n_cells_limited),
        "n_after_quota_sampling": int(obs_before),
        "n_after_rare_censored_topup": int(obs_after),
        "topup_added": int(obs_after - obs_before),
        "explanation": ("quota formula min(8, max(2, round(200*n/N))) caps every stratum cell at 8; "
                        "cells whose stratum has fewer records are limited by stratum size; "
                        "target of 200 is unreachable under this formula; actual sample is the "
                        "formula output (132). The public output filename is gold_sample_132.csv; "
                        "the historical target of 200 is retained only as design context."),
    }


def main():
    gold = pd.read_csv(os.path.join(GDIR, "gold_final.csv"), encoding="utf-8-sig")
    df = pd.read_csv(os.path.join(OUT, "protac_clean_record_level.csv"), encoding="utf-8-sig")
    merged = gold.merge(df[["record_id", "activity_evidence_v2"]], on="record_id", how="left",
                        validate="one_to_one")
    assert len(merged) == 132
    y_rule = merged["activity_evidence_v2"].astype(str).to_numpy()
    y_a = merged["annotator_A_label"].astype(str).to_numpy()
    y_b = merged["annotator_B_label"].astype(str).to_numpy()
    y_final = merged["final_evidence"].astype(str).to_numpy()
    for name, y in [("vs_annotator_A", y_a), ("vs_annotator_B", y_b), ("vs_final", y_final)]:
        assert np.array_equal(y_final, y_b), "gold_final should equal B column (consistency check)"
    result = {
        "meta": {
            "script": "gold_rule_agreement.py",
            "n_gold": int(len(merged)),
            "note": "automatic rule = activity_evidence_v2 (v2.0, 2026-08-16); labels P/N/A/U",
            "final_equals_B": bool(np.array_equal(y_final, y_b)),
            "final_equals_A": bool(np.array_equal(y_final, y_a)),
            "n_final_equals_A": int((y_final == y_a).sum()),
            "n_final_equals_B": int((y_final == y_b).sum()),
        },
        "rule": {name: multiclass(y, y_rule) for name, y in
                 [("vs_annotator_A", y_a), ("vs_annotator_B", y_b), ("vs_final", y_final)]},
        "sample_size_explanation": replay_sampling_quotas(),
    }
    out_path = os.path.join(OUT, "gold_rule_agreement_by_annotator.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for name in ["vs_annotator_A", "vs_annotator_B", "vs_final"]:
        m = result["rule"][name]
        print(f"{name}: accuracy={m['accuracy']:.4f} kappa={m['cohen_kappa']:.4f} macro-F1={m['macro_f1']:.4f}")
    s = result["sample_size_explanation"]
    print(f"quota: {s['n_after_quota_sampling']} -> topup -> {s['n_after_rare_censored_topup']} "
          f"(cells at cap=8: {s['n_cells_at_cap_8']}, limited by size: {s['n_cells_limited_by_stratum_size']})")
    print("wrote", out_path)


if __name__ == "__main__":
    main()


