# -*- coding: utf-8 -*-
"""
grouped_bootstrap_ci.py — 主结果 grouped bootstrap 95% CI (审稿 02/04/05 号)
============================================================================
审稿要求: 以 publication/scaffold/POI 为聚类重采样单元, 给出主比较的 95% CI,
而不是只报 2-3 个 seed 的 mean±SD。

协议:
  - 对每个 (任务, 划分, 种子) 固定训练一次模型 (与主实验相同超参);
  - 在固定测试集预测上, 按聚类单元 (publication/scaffold/POI 对应划分的 group)
    做 1000 次有放回 bootstrap (组级重采样), 每次重算指标;
  - 报告 2.5%/97.5% 分位数 (95% CI)。
  说明: 该 CI 反映"测试组构成的不确定性"; 模型训练随机性由 3 seeds 平均体现。

覆盖:
  - T3 P/N 分类 (baseline): random/scaffold/pub/poi × 3 seeds → AUC/MCC
  - PU 对照 (supervised/u_as_n/elkan_noto/nnpu): random/scaffold/pub/poi × 3 seeds → AUC
  - 删失评估 (drop/bound/censored): random/scaffold × 3 seeds → violation rate

输出:
  data/derived/grouped_bootstrap_ci.json
运行: .../python.exe grouped_bootstrap_ci.py
"""
import os, json, time
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(BASE, "data", "derived")
SEEDS = [20260815, 20260816, 20260817]
N_BOOT = 1000

import sys
sys.path.insert(0, os.path.join(BASE, "40_项目_Bias-Aware", "code"))
import importlib.util
_spec = importlib.util.spec_from_file_location("pu", os.path.join(BASE, "40_项目_Bias-Aware", "code", "pu_pipeline.py"))
pu = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pu)

from sklearn.model_selection import GroupShuffleSplit, train_test_split
import xgboost as xgb

LABEL = "activity_evidence_v2"

def grouped_boot_ci(y, p, groups, n_boot=1000, seed=42):
    """组级 bootstrap: 每次从测试组中有放回抽样组, 重算 ROC-AUC/MCC。
    返回 {metric: (ci_lo, ci_hi)}。"""
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    uniq = np.unique(groups)
    n_groups = len(uniq)
    vals = defaultdict(list)
    for b in range(n_boot):
        sel = rng.choice(n_groups, size=n_groups, replace=True)
        idx = np.concatenate([np.where(groups == uniq[g])[0] for g in sel])
        if len(idx) == 0:
            continue
        m = pu.classification_metrics(y[idx], p[idx])
        vals["roc_auc"].append(m["roc_auc"])
        vals["mcc"].append(m["mcc"])
    out = {}
    for k, v in vals.items():
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out

def group_first_split(all_idx, groups, seed, test_size=0.2):
    if groups is None:
        tr, te = train_test_split(all_idx, test_size=test_size, random_state=seed)
        return np.asarray(tr), np.asarray(te)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(iter(gss.split(all_idx, groups=groups[all_idx])))
    return np.asarray(tr), np.asarray(te)

def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(OUT, "protac_clean_record_level.csv"), encoding="utf-8-sig")
    g = pd.read_csv(os.path.join(OUT, "protac_split_groups.csv"), encoding="utf-8-sig").set_index("record_id").loc[df["record_id"]]
    X = np.load(os.path.join(OUT, "morgan_fp_2048.npy"))
    lab = df[LABEL].values
    P_idx = np.where(lab == "P")[0]; N_idx = np.where(lab == "N")[0]
    all_idx = np.arange(len(df))
    scaffold = g["scaffold"].fillna("__NAN__").values
    pub = g["pub_group"].fillna("__NAN__").values
    poi = g["poi"].fillna("__NAN__").values

    res = {"meta": {"protocol": "grouped bootstrap 95% CI (group-level resampling of test set, 1000 iters)",
                    "seeds": SEEDS, "n_boot": N_BOOT}, "tasks": {}}

    # ============ T3 P/N 分类 (baseline) ============
    print("=== T3 P/N classification ===")
    res["tasks"]["T3_pn_clf"] = {}
    for sname, gcol in [("random", None), ("scaffold", scaffold), ("pub", pub), ("poi", poi)]:
        auc_ci, mcc_ci = [], []
        for seed in SEEDS:
            tr, te = group_first_split(all_idx, gcol, seed)
            P_train = tr[lab[tr] == "P"]; N_train = tr[lab[tr] == "N"]
            P_test = te[lab[te] == "P"]; N_test = te[lab[te] == "N"]
            if min(len(P_train), len(N_train), len(P_test), len(N_test)) == 0:
                continue
            te_idx = np.concatenate([P_test, N_test])
            y_te = np.concatenate([np.ones(len(P_test)), np.zeros(len(N_test))]).astype(int)
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    n_jobs=max(1, os.cpu_count()-1), random_state=seed,
                                    eval_metric="logloss", verbosity=0)
            clf.fit(X[np.concatenate([P_train, N_train])],
                    np.concatenate([np.ones(len(P_train)), np.zeros(len(N_train))]).astype(int))
            p = clf.predict_proba(X[te_idx])[:, 1]
            grp = gcol[te_idx] if gcol is not None else np.arange(len(te_idx))
            ci = grouped_boot_ci(y_te, p, grp, N_BOOT, seed)
            auc_ci.append(ci["roc_auc"]); mcc_ci.append(ci["mcc"])
        res["tasks"]["T3_pn_clf"][sname] = {
            "roc_auc_ci": [round(np.mean([c[0] for c in auc_ci]), 3), round(np.mean([c[1] for c in auc_ci]), 3)],
            "mcc_ci": [round(np.mean([c[0] for c in mcc_ci]), 3), round(np.mean([c[1] for c in mcc_ci]), 3)]}
        print(f"  {sname}: AUC CI={res['tasks']['T3_pn_clf'][sname]['roc_auc_ci']}")

    # ============ PU 对照 ============
    print("\n=== PU comparison ===")
    res["tasks"]["PU"] = {}
    for sname, gcol in [("random", None), ("scaffold", scaffold), ("pub", pub), ("poi", poi)]:
        res["tasks"]["PU"][sname] = {}
        for seed in SEEDS:
            tr, te = group_first_split(all_idx, gcol, seed)
            P_train = tr[lab[tr] == "P"]; N_train = tr[lab[tr] == "N"]
            P_test = te[lab[te] == "P"]; N_test = te[lab[te] == "N"]
            if min(len(P_train), len(N_train), len(P_test), len(N_test)) == 0:
                continue
            U_pool = tr[lab[tr] != "P"]
            te_idx = np.concatenate([P_test, N_test])
            y_te = np.concatenate([np.ones(len(P_test)), np.zeros(len(N_test))]).astype(int)
            grp = gcol[te_idx] if gcol is not None else np.arange(len(te_idx))
            # supervised
            X_sup = np.vstack([X[P_train], X[N_train]])
            y_sup = np.concatenate([np.ones(len(P_train)), np.zeros(len(N_train))]).astype(int)
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    n_jobs=max(1, os.cpu_count()-1), random_state=seed,
                                    eval_metric="logloss", verbosity=0)
            clf.fit(X_sup, y_sup)
            p_sup = clf.predict_proba(X[te_idx])[:, 1]
            # nnPU
            X_nn = np.vstack([X[P_train], X[U_pool]])
            s_nn = np.concatenate([np.ones(len(P_train)), np.zeros(len(U_pool))]).astype(int)
            p_nn = pu.nnpu_fit_predict(X_nn, s_nn, X[te_idx], seed=seed)
            for name, p in [("supervised", p_sup), ("nnpu", p_nn)]:
                ci = grouped_boot_ci(y_te, p, grp, N_BOOT, seed)
                res["tasks"]["PU"][sname].setdefault(name, []).append(ci["roc_auc"])
        for name in ["supervised", "nnpu"]:
            cis = res["tasks"]["PU"][sname][name]
            res["tasks"]["PU"][sname][name] = [round(np.mean([c[0] for c in cis]), 3), round(np.mean([c[1] for c in cis]), 3)]
        print(f"  {sname}: supervised {res['tasks']['PU'][sname]['supervised']} | nnpu {res['tasks']['PU'][sname]['nnpu']}")

    # ============ 删失评估 ============
    print("\n=== Censoring evaluation (violation rate) ===")
    res["tasks"]["censored"] = {}
    pdc50_exact = df["pdc50_value"].values.astype(float)
    pdc50_upper = df["pdc50_upper"].values.astype(float)
    pdc50_lower = df["pdc50_lower"].values.astype(float)
    dc_ot = df["dc50_obs_type"].values
    exact_mask = (dc_ot == "exact") & np.isfinite(pdc50_exact)
    right_mask = (dc_ot == "right-censored") & np.isfinite(pdc50_upper)
    left_mask = (dc_ot == "left-censored") & np.isfinite(pdc50_lower)
    cens_mask = right_mask | left_mask
    for sname, gcol in [("random", None), ("scaffold", scaffold)]:
        res["tasks"]["censored"][sname] = {}
        for seed in SEEDS:
            ex_idx = np.where(exact_mask)[0]
            all_ep = np.concatenate([ex_idx, np.where(cens_mask)[0]])
            tr, te = group_first_split(all_ep, gcol, seed)
            tr_exact = tr[exact_mask[tr]]; tr_cens = tr[cens_mask[tr]]
            te_cens = te[cens_mask[te]]
            if len(tr_exact) == 0 or len(te_cens) == 0:
                continue
            y_te = np.where(right_mask[te_cens], pdc50_upper[te_cens], pdc50_lower[te_cens])
            is_right = right_mask[te_cens]
            grp = gcol[te_cens] if gcol is not None else np.arange(len(te_cens))
            # drop
            r1 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  n_jobs=max(1, os.cpu_count()-1), random_state=seed, verbosity=0)
            r1.fit(X[tr_exact], pdc50_exact[tr_exact])
            p1 = r1.predict(X[te_cens])
            # censored (MLP, 只用 train 组)
            cens_tr = np.concatenate([tr_exact, tr_cens])
            y_c = np.concatenate([pdc50_exact[tr_exact],
                                  np.where(right_mask[tr_cens], pdc50_upper[tr_cens], pdc50_lower[tr_cens])])
            c_c = np.concatenate([np.zeros(len(tr_exact)), np.where(right_mask[tr_cens], 1.0, -1.0)])
            p3 = pu.censored_loss_fit_predict(X[cens_tr], y_c, c_c, X[te_cens], seed=seed)
            for name, p in [("drop", p1), ("censored", p3)]:
                viol = np.where(is_right, p > y_te, p < y_te)
                # 组级 bootstrap 违反率
                rng = np.random.RandomState(seed)
                uniq = np.unique(grp)
                vr_vals = []
                for b in range(N_BOOT):
                    sel = rng.choice(len(uniq), size=len(uniq), replace=True)
                    idx = np.concatenate([np.where(grp == uniq[g])[0] for g in sel])
                    vr_vals.append(float(viol[idx].mean()))
                res["tasks"]["censored"][sname].setdefault(name, []).append(
                    [float(np.percentile(vr_vals, 2.5)), float(np.percentile(vr_vals, 97.5))])
        for name in ["drop", "censored"]:
            cis = res["tasks"]["censored"][sname][name]
            res["tasks"]["censored"][sname][name] = [round(np.mean([c[0] for c in cis]), 3), round(np.mean([c[1] for c in cis]), 3)]
        print(f"  {sname}: drop {res['tasks']['censored'][sname]['drop']} | censored {res['tasks']['censored'][sname]['censored']}")

    with open(os.path.join(OUT, "grouped_bootstrap_ci.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\ngrouped_bootstrap_ci.json 已写入 (耗时 {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
