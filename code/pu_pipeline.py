# -*- coding: utf-8 -*-
"""PU benchmark driven exclusively by the frozen v3 split manifest."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from benchmark_contract import (
    FEATURE_PATH,
    LABEL_COL,
    MANIFEST_PATH,
    PROCESSED_DIR,
    SEEDS,
    SPLITS,
    assert_group_disjoint,
    group_values,
    load_contract,
    load_dataset,
    load_groups,
    load_manifest,
    role_indices,
)

RESULTS_PATH = PROCESSED_DIR / "pu_results_v3.json"
PREDICTIONS_PATH = PROCESSED_DIR / "pu_predictions_v3.csv"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.any():
            value += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(value)


def classification_metrics(y_true, y_prob):
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def regression_metrics(y_true, y_pred):
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def estimate_scar_prior(X_train, s_train):
    """SCAR heuristic π=q/c using an in-sample logistic observation model.

    This is explicitly not described as cross-fitted or selection-aware.
    """
    from sklearn.linear_model import LogisticRegression

    s_train = np.asarray(s_train, dtype=int)
    q = float(s_train.mean())
    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=0)
    model.fit(X_train, s_train)
    g = model.predict_proba(X_train)[:, 1]
    c = float(np.mean(g[s_train == 1]))
    c = float(np.clip(c, 1e-4, 0.999))
    prior = float(np.clip(q / c, 1e-3, 0.9))
    return prior, {"q": q, "c": c, "method": "in-sample logistic SCAR heuristic q/c"}


def elkan_noto_pu(X_train, s_train, X_test, X_known_p, seed=42):
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, s_train)
    g_test = clf.predict_proba(X_test)[:, 1]
    c = float(np.mean(clf.predict_proba(X_known_p)[:, 1]))
    return np.clip(g_test / max(c, 1e-6), 0, 1), c


def nnpu_fit_predict(X_train, s_train, X_test, seed, prior=None, epochs=40, bs=256, lr=1e-3, hidden=128, return_info=False):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    X_train = np.asarray(X_train, dtype=np.float32)
    s_train = np.asarray(s_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if prior is None:
        prior, prior_info = estimate_scar_prior(X_train, s_train.astype(int))
    else:
        prior_info = {"q": float(s_train.mean()), "c": None, "method": "user-specified", "prior": float(prior)}
    Xt = torch.tensor(X_train)
    st = torch.tensor(s_train)
    Xte = torch.tensor(X_test)
    d = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, bs):
            idx = permutation[start : start + bs]
            x = Xt[idx]
            s = st[idx]
            positive = s == 1
            unlabeled = s == 0
            if not positive.any() or not unlabeled.any():
                continue
            prob = torch.sigmoid(model(x).squeeze(1)).clamp(1e-7, 1 - 1e-7)
            risk_p_pos = -torch.log(prob[positive]).mean()
            risk_p_neg = -torch.log(1 - prob[positive]).mean()
            risk_u_neg = -torch.log(1 - prob[unlabeled]).mean()
            risk = prior * risk_p_pos + torch.clamp(risk_u_neg - prior * risk_p_neg, min=0.0)
            optimizer.zero_grad()
            risk.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(Xte).squeeze(1)).cpu().numpy()
    info = {**prior_info, "prior": float(prior)}
    return (np.clip(pred, 0, 1), info) if return_info else np.clip(pred, 0, 1)


def _regression_network(input_dim, hidden=128):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )


def mlp_regression_fit_predict(X_train, y_train, X_test, seed, epochs=60, bs=256, lr=1e-3, hidden=128):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    Xt = torch.tensor(np.asarray(X_train, dtype=np.float32))
    yt = torch.tensor(np.asarray(y_train, dtype=np.float32))
    Xte = torch.tensor(np.asarray(X_test, dtype=np.float32))
    model = _regression_network(Xt.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(Xt))
        for start in range(0, len(Xt), bs):
            idx = permutation[start : start + bs]
            pred = model(Xt[idx]).squeeze(1)
            loss = 0.5 * ((pred - yt[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(Xte).squeeze(1).cpu().numpy()


def censored_loss_fit_predict(X_train, y_train, cens_train, X_test, seed, epochs=60, bs=256, lr=1e-3, hidden=128):
    """Same MLP backbone as mlp_regression_fit_predict with one-sided losses."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    Xt = torch.tensor(np.asarray(X_train, dtype=np.float32))
    yt = torch.tensor(np.asarray(y_train, dtype=np.float32))
    ct = torch.tensor(np.asarray(cens_train, dtype=np.float32))
    Xte = torch.tensor(np.asarray(X_test, dtype=np.float32))
    model = _regression_network(Xt.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(Xt))
        for start in range(0, len(Xt), bs):
            idx = permutation[start : start + bs]
            pred = model(Xt[idx]).squeeze(1)
            y = yt[idx]
            c = ct[idx]
            losses = []
            exact = c == 0
            right = c > 0
            left = c < 0
            if exact.any():
                losses.append(0.5 * ((pred[exact] - y[exact]) ** 2).mean())
            if right.any():
                losses.append(0.5 * (torch.clamp(pred[right] - y[right], min=0) ** 2).mean())
            if left.any():
                losses.append(0.5 * (torch.clamp(y[left] - pred[left], min=0) ** 2).mean())
            loss = sum(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(Xte).squeeze(1).cpu().numpy()


def main():
    t0 = time.time()
    contract = load_contract(validate_hashes=True)
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    X = np.load(FEATURE_PATH).astype(np.float32)
    if X.shape[0] != len(df):
        raise AssertionError("Morgan feature cache is not aligned")
    lab = df[LABEL_COL].to_numpy()
    p_mask = lab == "P"
    n_mask = lab == "N"
    group_cache = {split: group_values(df, groups, split) for split in SPLITS}

    import xgboost as xgb

    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    results = {
        "meta": {
            "script": "pu_pipeline.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "label_col": LABEL_COL,
            "protocol": "all-record group-first roles; U_pool=train-role non-P; evaluation=test-role P/N",
            "prior_note": "nnPU prior uses an explicitly disclosed in-sample logistic SCAR heuristic q/c; not cross-fitted",
            "seeds": SEEDS,
        },
        "methods": {},
    }
    prediction_rows = []

    for split in SPLITS:
        agg = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            roles = role_indices(df, manifest, "all_records_2way", split, seed)
            assert_group_disjoint(df, groups, roles, split)
            tr_all = roles["train"]
            te_all = roles["test"]
            p_train = tr_all[p_mask[tr_all]]
            n_train = tr_all[n_mask[tr_all]]
            u_pool = tr_all[~p_mask[tr_all]]
            p_test = te_all[p_mask[te_all]]
            n_test = te_all[n_mask[te_all]]
            if min(len(p_train), len(n_train), len(p_test), len(n_test)) == 0:
                continue
            if np.intersect1d(u_pool, te_all).size or np.intersect1d(tr_all, te_all).size:
                raise AssertionError("PU train/test leakage")
            te = np.concatenate([p_test, n_test])
            y_te = np.concatenate([np.ones(len(p_test)), np.zeros(len(n_test))]).astype(int)
            X_pu = np.vstack([X[p_train], X[u_pool]])
            s_pu = np.concatenate([np.ones(len(p_train)), np.zeros(len(u_pool))]).astype(int)

            supervised = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, n_jobs=n_jobs, random_state=seed, eval_metric="logloss", verbosity=0,
            )
            supervised.fit(np.vstack([X[p_train], X[n_train]]), np.concatenate([np.ones(len(p_train)), np.zeros(len(n_train))]))
            p_supervised = supervised.predict_proba(X[te])[:, 1]

            u_as_n = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, n_jobs=n_jobs, random_state=seed, eval_metric="logloss", verbosity=0,
            )
            u_as_n.fit(X_pu, s_pu)
            p_uan = u_as_n.predict_proba(X[te])[:, 1]
            p_en, en_c = elkan_noto_pu(X_pu, s_pu, X[te], X[p_train], seed=seed)
            p_nnpu, prior_info = nnpu_fit_predict(X_pu, s_pu, X[te], seed=seed, return_info=True)

            methods = {"supervised": p_supervised, "u_as_n": p_uan, "elkan_noto": p_en, "nnpu": p_nnpu}
            seed_entry = {
                "seed": seed,
                "n_train_all": int(len(tr_all)),
                "n_train_p": int(len(p_train)),
                "n_train_hidden_n": int(len(n_train)),
                "n_train_a": int(np.sum(lab[u_pool] == "A")),
                "n_train_u": int(np.sum(lab[u_pool] == "U")),
                "n_test_p": int(len(p_test)),
                "n_test_n": int(len(n_test)),
                "nnpu_prior": prior_info,
                "elkan_noto_c": float(en_c),
                "metrics": {},
            }
            for method, pred in methods.items():
                metrics = classification_metrics(y_te, pred)
                if metrics is None:
                    continue
                seed_entry["metrics"][method] = metrics
                for metric, value in metrics.items():
                    agg[f"{method}::{metric}"].append(value)
                for rid, yt, yp, gid in zip(df.iloc[te]["record_id"], y_te, pred, group_cache[split][te]):
                    prediction_rows.append(
                        {
                            "split_regime": split,
                            "seed": seed,
                            "method": method,
                            "record_id": int(rid),
                            "group_id": str(gid),
                            "y_true": int(yt),
                            "y_pred": float(yp),
                        }
                    )
            per_seed.append(seed_entry)
        entry = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=0))} for k, v in agg.items()}
        entry["per_seed"] = per_seed
        entry["manifest_family"] = "all_records_2way"
        results["methods"][split] = entry
        if "nnpu::roc_auc" in entry:
            print(f"PU/{split}: nnPU AUC={entry['nnpu::roc_auc']['mean']:.4f}")

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {PREDICTIONS_PATH} ({len(prediction_rows):,} predictions)")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
