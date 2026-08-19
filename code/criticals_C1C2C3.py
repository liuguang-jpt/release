# -*- coding: utf-8 -*-
"""
criticals_C1C2C3.py — 论文 v2 评审 Critical 项修复实验
================================================================
C1: gold-standard holdout — 150 条人工标注从训练中隔离, 只作评估
    协议: 训练 = 全量记录 − 150 条人工标注 (规则标签 v2)
          评估 = 150 条人工标注上的 P/N 分类 (人工标签为金标准)
C2: architecture-matched PU — 补 MLP 监督 / MLP U-as-N 基线 (与 nnPU 同架构)
    消除"nnPU=MLP vs 监督=XGBoost"的架构混淆
C3: label-observability model — 回答 RQ1
    拟合 P(S=1|X) 的 LR/GBM 分类器 (S = 该记录是否有可用活性标签),
    报告 AUC + 特征重要性 (SHAP/permutation), 检验标签可用性是否与
    POI/E3/source/year/Morgan 结构特征相关 (MNAR 证据)。

输出:
  data/derived/c1_gold_holdout.json
  data/derived/c2_arch_matched.json
  data/derived/c3_label_obs.json
运行: .../python.exe criticals_C1C2C3.py
"""
import os, json, time
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")
SEEDS = [20260815, 20260816, 20260817]

import sys
sys.path.insert(0, os.path.join(BASE, "40_项目_Bias-Aware", "code"))
import importlib.util
_spec = importlib.util.spec_from_file_location("pu", os.path.join(BASE, "40_项目_Bias-Aware", "code", "pu_pipeline.py"))
pu = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pu)

from sklearn.model_selection import GroupShuffleSplit, train_test_split
import xgboost as xgb

LABEL = "activity_evidence_v2"

# ---------------- MLP 监督 (BCE) 与 nnPU 同架构 ----------------
def mlp_fit_predict(X_train, y_train, X_test, seed, epochs=40, bs=256, lr=1e-3, hidden=128):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    d = X_train.shape[1]
    model = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            pred = model(Xt[idx]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(1)).numpy()

def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(OUT, "protac_clean_record_level.csv"), encoding="utf-8-sig")
    ann = pd.read_csv(os.path.join(OUT, "protac_annotation_150.csv"), encoding="utf-8-sig")
    X = np.load(os.path.join(OUT, "morgan_fp_2048.npy"))
    print(f"X {X.shape}, df {len(df)}")

    # gold = 150 条人工标注的 record_id
    gold_ids = set(ann["record_id"].tolist())
    gold_mask = df["record_id"].isin(gold_ids).values
    print(f"gold 标注记录: {int(gold_mask.sum())}")

    lab = df[LABEL].values
    P_mask = lab == "P"; N_mask = lab == "N"

    # ============ C1: gold-standard holdout (P/N 分类) ============
    print("\n===== C1: gold holdout — 训练排除 150 条, 评估用人工标签 =====")
    # 训练集: 非 gold 且规则标签为 P/N
    tr_mask = (~gold_mask) & (P_mask | N_mask)
    tr_idx = np.where(tr_mask)[0]
    y_tr = (lab[tr_idx] == "P").astype(int)
    # 评估集: gold 且人工标签为 P/N
    gold_pn = ann["annotator_activity_evidence"].isin(["P", "N"]).values
    # ⚠️ 修复 v1 bug (2026-08-16 审稿发现): 必须按 record_id 对齐, 不能 ann 序 vs df 序混用
    gold_df = (
        ann.loc[gold_pn, ["record_id", "annotator_activity_evidence"]]
           .merge(df[["record_id"]].reset_index(names="row_idx"), on="record_id", how="inner", validate="one_to_one")
    )
    gold_df = gold_df.sort_values("record_id").reset_index(drop=True)
    gold_row_idx = gold_df["row_idx"].to_numpy()
    y_gold = (gold_df["annotator_activity_evidence"].to_numpy() == "P").astype(int)
    X_gold = X[gold_row_idx]
    # 自动断言: 对齐后顺序一致
    assert np.array_equal(df["record_id"].iloc[gold_row_idx].to_numpy(), gold_df["record_id"].to_numpy()), "record_id 对齐失败"
    print(f"训练 P/N: {len(tr_idx)} (P {int(y_tr.sum())}) | gold 评估 P/N: {len(gold_idx)} (P {int(y_gold.sum())})")

    c1 = {}
    for seed in SEEDS:
        r1 = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=max(1, os.cpu_count()-1), random_state=seed,
                               eval_metric="logloss", verbosity=0)
        r1.fit(X[tr_idx], y_tr)
        p = r1.predict_proba(X_gold)[:, 1]
        m = pu.classification_metrics(y_gold, p)
        for k, v in m.items():
            c1.setdefault(k, []).append(v)
        print(f"  seed={seed}: AUC={m['roc_auc']:.3f} MCC={m['mcc']:.3f} Brier={m['brier']:.3f}")
    c1 = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in c1.items()}
    c1["n_train"] = int(len(tr_idx)); c1["n_gold_eval"] = int(len(gold_idx))
    with open(os.path.join(OUT, "c1_gold_holdout.json"), "w", encoding="utf-8") as f:
        json.dump(c1, f, ensure_ascii=False, indent=2)
    print("c1_gold_holdout.json 已写入")

    # ============ C2: architecture-matched MLP 基线 ============
    print("\n===== C2: 架构匹配 MLP (监督/U-as-N vs nnPU 同架构) =====")
    c2 = {"meta": {"note": "MLP 监督与 MLP U-as-N 与 nnPU 同架构(hidden128/40ep); 评估 P_test∪N_test"}, "splits": {}}
    P_idx = np.where(P_mask)[0]; N_idx = np.where(N_mask)[0]
    all_idx = np.arange(len(df))
    scaffold = pd.read_csv(os.path.join(OUT, "protac_split_groups.csv"), encoding="utf-8-sig")["scaffold"].fillna("__NAN__").values
    for sname, gcol in [("random", None), ("scaffold", scaffold)]:
        agg = defaultdict(list)
        for seed in SEEDS:
            pn = np.concatenate([P_idx, N_idx])
            if gcol is None:
                tr, te = train_test_split(np.arange(len(pn)), test_size=0.2, random_state=seed)
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                tr, te = next(iter(gss.split(np.arange(len(pn)), groups=gcol[pn])))
            P_train = pn[tr][P_mask[pn[tr]]]; P_test = pn[te][P_mask[pn[te]]]
            N_train = pn[tr][N_mask[pn[tr]]]; N_test = pn[te][N_mask[pn[te]]]
            U_pool = np.setdiff1d(all_idx, np.concatenate([P_idx, N_test]))
            te_idx = np.concatenate([P_test, N_test])
            y_te = np.concatenate([np.ones(len(P_test)), np.zeros(len(N_test))]).astype(int)
            X_te = X[te_idx]
            # MLP 监督 (P vs N)
            X_sup = np.vstack([X[P_train], X[N_train]])
            y_sup = np.concatenate([np.ones(len(P_train)), np.zeros(len(N_train))]).astype(int)
            p_mlp_sup = mlp_fit_predict(X_sup, y_sup, X_te, seed=seed)
            # MLP U-as-N (P vs U)
            X_uan = np.vstack([X[P_train], X[U_pool]])
            y_uan = np.concatenate([np.ones(len(P_train)), np.zeros(len(U_pool))]).astype(int)
            p_mlp_uan = mlp_fit_predict(X_uan, y_uan, X_te, seed=seed)
            # nnPU (与 pu_pipeline 相同)
            X_nn = np.vstack([X[P_train], X[U_pool]])
            s_nn = np.concatenate([np.ones(len(P_train)), np.zeros(len(U_pool))]).astype(int)
            p_nnpu = pu.nnpu_fit_predict(X_nn, s_nn, X_te, seed=seed)
            for name, p in [("mlp_supervised", p_mlp_sup), ("mlp_u_as_n", p_mlp_uan), ("nnpu", p_nnpu)]:
                m = pu.classification_metrics(y_te, p)
                for k in ["roc_auc", "mcc", "brier", "ece"]:
                    agg[f"{name}::{k}"].append(m[k])
            print(f"  {sname} seed={seed}: sup={agg['mlp_supervised::roc_auc'][-1]:.3f} uan={agg['mlp_u_as_n::roc_auc'][-1]:.3f} nnpu={agg['nnpu::roc_auc'][-1]:.3f}")
        c2["splits"][sname] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in agg.items()}
    with open(os.path.join(OUT, "c2_arch_matched.json"), "w", encoding="utf-8") as f:
        json.dump(c2, f, ensure_ascii=False, indent=2)
    print("c2_arch_matched.json 已写入")

    # ============ C3: label-observability (RQ1) ============
    print("\n===== C3: 标签可用性模型 P(S=1|X) (RQ1) =====")
    # S = 该记录是否有可用活性标签 (非 U)
    S = (lab != "U").astype(int)
    print(f"S 观测率: {S.mean()*100:.1f}% (有活性标签 vs U)")
    # 特征: Morgan + POI/E3/source one-hot (与实验一致)
    ctx = pd.DataFrame(index=df.index)
    for col in ["target", "e3_ligase", "source_has_doi"]:
        d = pd.get_dummies(df[col].fillna("UNKNOWN"), prefix=col, dtype=np.float32)
        ctx = pd.concat([ctx, d], axis=1)
    X_full = np.hstack([X, ctx.values.astype(np.float32)])
    feat_names = [f"fp_{i}" for i in range(2048)] + list(ctx.columns)

    c3 = {"meta": {"S_definition": "S=1 if activity_evidence_v2 != U (has usable label)", "S_rate": float(S.mean())}}
    rows = defaultdict(list)
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=seed, stratify=S)
        for name, clf in [
            ("logistic", __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=2000)),
            ("gbm", xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, n_jobs=max(1, os.cpu_count()-1),
                                      random_state=seed, eval_metric="logloss", verbosity=0)),
        ]:
            clf.fit(X_full[tr], S[tr])
            p = clf.predict_proba(X_full[te])[:, 1]
            m = pu.classification_metrics(S[te], p)
            rows[f"{name}::roc_auc"].append(m["roc_auc"])
            rows[f"{name}::brier"].append(m["brier"])
            # permutation importance (top 15, 采样特征加速)
            rng = np.random.RandomState(seed)
            if name == "gbm":
                base_auc = m["roc_auc"]
                imp = []
                sub_idx = rng.choice(np.arange(len(te)), size=min(2000, len(te)), replace=False)
                Xsub = X_full[te][sub_idx]; ysub = S[te][sub_idx]
                for j in rng.choice(np.arange(X_full.shape[1]), size=300, replace=False):
                    Xp = Xsub.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                    pp = clf.predict_proba(Xp)[:, 1]
                    imp.append((j, base_auc - pu.classification_metrics(ysub, pp)["roc_auc"]))
                imp.sort(key=lambda x: -x[1])
                top = [(feat_names[j], float(v)) for j, v in imp[:15]]
                rows[f"gbm::top_imp_seed{seed}"].append(top)
    c3["results"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)) if len(v) > 1 else 0.0} if isinstance(v[0], float) else v[0]
                     for k, v in rows.items()}
    # 单独存 top-importance
    c3["gbm_top_features_by_seed"] = {f"seed{s}": rows[f"gbm::top_imp_seed{s}"][0] for s in SEEDS}
    with open(os.path.join(OUT, "c3_label_obs.json"), "w", encoding="utf-8") as f:
        json.dump(c3, f, ensure_ascii=False, indent=2, default=str)
    print("c3_label_obs.json 已写入")
    print(f"总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
