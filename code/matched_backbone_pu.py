# -*- coding: utf-8 -*-
"""Matched-backbone supervised MLP versus PU controls.

This script uses the frozen all_records_2way manifest and the same MLP
backbone/training budget for supervised P/N, U-as-N, Elkan-Noto and nnPU.
It is intentionally separate from pu_pipeline.py so the historical XGBoost
versus MLP results remain traceable and are not overwritten.
"""
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
    REPORTS_DIR,
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

RESULTS_PATH = REPORTS_DIR / "matched_backbone_pu_results_v4.json"
PREDICTIONS_PATH = REPORTS_DIR / "matched_backbone_pu_predictions_v4.csv"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.any():
            value += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
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


def set_seed(seed):
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def runtime_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(input_dim, hidden=128):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )


def mlp_fit_predict(X_train, y_train, X_test, seed, epochs=40, bs=256, lr=1e-3, hidden=128):
    import torch

    set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    device = runtime_device()
    model = make_model(X_train.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device)
    Xte = torch.tensor(X_test, device=device)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xt))
        for start in range(0, len(order), bs):
            idx = order[start : start + bs]
            logits = model(Xt[idx]).squeeze(1)
            loss = loss_fn(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(1)).cpu().numpy()


def estimate_prior(X_train, s_train):
    from sklearn.linear_model import LogisticRegression

    s_train = np.asarray(s_train, dtype=int)
    q = float(s_train.mean())
    clf = LogisticRegression(max_iter=500, solver="liblinear", random_state=0)
    clf.fit(X_train, s_train)
    g = clf.predict_proba(X_train)[:, 1]
    c = float(np.mean(g[s_train == 1]))
    prior = float(np.clip(q / np.clip(c, 1e-4, 0.999), 1e-3, 0.9))
    return prior, {"q": q, "c": c, "method": "in-sample logistic SCAR heuristic q/c"}


def nnpu_fit_predict(X_train, s_train, X_test, seed, prior=None, epochs=40, bs=256, lr=1e-3, hidden=128):
    import torch

    set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    s_train = np.asarray(s_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if prior is None:
        prior, prior_info = estimate_prior(X_train, s_train.astype(int))
    else:
        prior_info = {"method": "user-specified", "prior": float(prior)}
    device = runtime_device()
    model = make_model(X_train.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(X_train, device=device)
    st = torch.tensor(s_train, device=device)
    Xte = torch.tensor(X_test, device=device)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xt))
        for start in range(0, len(order), bs):
            idx = order[start : start + bs]
            x, s = Xt[idx], st[idx]
            positive, unlabeled = s == 1, s == 0
            if not positive.any() or not unlabeled.any():
                continue
            prob = torch.sigmoid(model(x).squeeze(1)).clamp(1e-7, 1 - 1e-7)
            risk_p_pos = -torch.log(prob[positive]).mean()
            risk_p_neg = -torch.log(1 - prob[positive]).mean()
            risk_u_neg = -torch.log(1 - prob[unlabeled]).mean()
            risk = float(prior) * risk_p_pos + torch.clamp(risk_u_neg - float(prior) * risk_p_neg, min=0.0)
            optimizer.zero_grad()
            risk.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(Xte).squeeze(1)).cpu().numpy()
    return np.clip(pred, 0, 1), {**prior_info, "prior": float(prior)}


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
    p_mask, n_mask = lab == "P", lab == "N"
    group_cache = {split: group_values(df, groups, split) for split in SPLITS}
    results = {
        "meta": {
            "script": "matched_backbone_pu.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "features": "Morgan radius=2, 2048 bits",
            "backbone": "2048 -> 128 -> dropout(0.2) -> 64 -> 1",
            "training": {"epochs": 40, "batch_size": 256, "learning_rate": 1e-3, "weight_decay": 1e-5},
            "seeds": SEEDS,
            "test_contract": "P/N test subset for every method; train roles only; no test-derived tuning",
            "prior_note": "nnPU prior uses an explicitly disclosed in-sample logistic SCAR heuristic q/c; not cross-fitted",
            "runtime_device": str(runtime_device()),
        },
        "splits": {},
    }
    prediction_rows = []
    for split in SPLITS:
        agg = defaultdict(list)
        per_seed = []
        for seed in SEEDS:
            roles = role_indices(df, manifest, "all_records_2way", split, seed)
            assert_group_disjoint(df, groups, roles, split)
            tr_all, te_all = roles["train"], roles["test"]
            p_train, n_train = tr_all[p_mask[tr_all]], tr_all[n_mask[tr_all]]
            u_pool = tr_all[~p_mask[tr_all]]
            p_test, n_test = te_all[p_mask[te_all]], te_all[n_mask[te_all]]
            if min(len(p_train), len(n_train), len(p_test), len(n_test)) == 0:
                continue
            te = np.concatenate([p_test, n_test])
            y_te = np.concatenate([np.ones(len(p_test)), np.zeros(len(n_test))]).astype(int)
            X_pn = np.vstack([X[p_train], X[n_train]])
            y_pn = np.concatenate([np.ones(len(p_train)), np.zeros(len(n_train))]).astype(int)
            X_pu = np.vstack([X[p_train], X[u_pool]])
            s_pu = np.concatenate([np.ones(len(p_train)), np.zeros(len(u_pool))]).astype(int)
            predictions = {}
            predictions["supervised_mlp"], _ = (mlp_fit_predict(X_pn, y_pn, X[te], seed=seed), None)
            predictions["u_as_n_mlp"] = mlp_fit_predict(X_pu, s_pu, X[te], seed=seed)
            predictions["elkan_noto_mlp"] = None
            try:
                from sklearn.linear_model import LogisticRegression

                obs = LogisticRegression(max_iter=500, solver="liblinear", random_state=seed)
                obs.fit(X_pu, s_pu)
                g_test = obs.predict_proba(X[te])[:, 1]
                c = float(np.mean(obs.predict_proba(X[p_train])[:, 1]))
                predictions["elkan_noto_mlp"] = np.clip(g_test / max(c, 1e-6), 0, 1)
            except Exception:
                predictions.pop("elkan_noto_mlp")
            predictions["nnpu_mlp"], prior_info = nnpu_fit_predict(X_pu, s_pu, X[te], seed=seed)
            seed_entry = {
                "seed": int(seed),
                "n_train_p": int(len(p_train)),
                "n_train_hidden_n": int(len(n_train)),
                "n_train_u": int(np.sum(lab[u_pool] == "U")),
                "n_test_p": int(len(p_test)),
                "n_test_n": int(len(n_test)),
                "nnpu_prior": prior_info,
                "metrics": {},
            }
            for method, pred in predictions.items():
                if pred is None:
                    continue
                metrics = classification_metrics(y_te, pred)
                seed_entry["metrics"][method] = metrics
                for metric, value in metrics.items():
                    agg[f"{method}::{metric}"].append(value)
                for rid, yt, yp, gid in zip(df.iloc[te]["record_id"], y_te, pred, group_cache[split][te]):
                    prediction_rows.append({
                        "split_regime": split,
                        "seed": int(seed),
                        "method": method,
                        "record_id": int(rid),
                        "group_id": str(gid),
                        "y_true": int(yt),
                        "y_pred": float(yp),
                    })
            per_seed.append(seed_entry)
        entry = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=0))} for k, v in agg.items()}
        entry["per_seed"] = per_seed
        entry["manifest_family"] = "all_records_2way"
        results["splits"][split] = entry
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {PREDICTIONS_PATH} ({len(prediction_rows):,} predictions)")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
