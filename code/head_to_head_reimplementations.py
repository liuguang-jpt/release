# -*- coding: utf-8 -*-
"""Frozen-manifest head-to-head reimplementations of representative models.

The official DeepPROTACs and DegradeMaster repositories require protein-pocket,
ligand and/or 3-D coordinates that are absent from the record-level release.
This script therefore implements two explicitly labelled, paper-aligned
surrogates using only available PROTAC features. It never reports official
weights or official-paper numbers as reproduced results.
"""
from __future__ import annotations

import hashlib
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

RESULTS_PATH = REPORTS_DIR / "head_to_head_reimplementations_v4.json"
PREDICTIONS_PATH = REPORTS_DIR / "head_to_head_reimplementations_v4.csv"

OFFICIAL = {
    "DeepPROTACs": {
        "paper": "Li et al., Nature Communications 2022",
        "doi": "10.1038/s41467-022-34807-3",
        "repo": "https://github.com/fenglei104/DeepPROTACs",
        "commit": "dfb62c4d137b7133d5ea4834a2615779630f2a52",
        "class": "paper-aligned reimplementation",
        "missing_inputs": ["target pocket", "E3-ligase pocket", "target ligand", "E3-ligase ligand", "linker segmentation"],
        "deviation": "2D PROTAC Morgan representation plus target/E3 identity and RDKit descriptors; no pocket or ligand graph inputs",
    },
    "DegradeMaster": {
        "paper": "Liu et al., Bioinformatics 2025",
        "doi": "10.1093/bioinformatics/btaf191",
        "repo": "https://github.com/ABILiLab/DegradeMaster",
        "commit": "aa149beaf067a051e070b3b281f7dba43c2f3e90",
        "class": "protocol-constrained reimplementation",
        "missing_inputs": ["PROTAC 3D coordinates", "target-pocket coordinates", "E3-pocket coordinates", "official PROTAC-8K preprocessing"],
        "deviation": "Morgan plus RDKit descriptors and target/E3 identity; confidence-threshold pseudo-label enrichment replaces 3D E(3)-equivariant graph encoding",
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def make_model(input_dim, hidden=256):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )


def fit_mlp(X_train, y_train, X_test, seed, epochs=40, hidden=256):
    import torch

    set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    device = runtime_device()
    model = make_model(X_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device)
    Xte = torch.tensor(X_test, device=device)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xt))
        for start in range(0, len(order), 256):
            idx = order[start : start + 256]
            loss = loss_fn(model(Xt[idx]).squeeze(1), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(1)).cpu().numpy()


def rdkit_descriptors(smiles):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        return np.zeros(9, dtype=np.float32)
    vals = [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol), rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
    ]
    return np.asarray(vals, dtype=np.float32)


def build_features(df, X_morgan):
    # These categorical descriptors are label-free and created before role filtering.
    blocks = [X_morgan.astype(np.float32)]
    for col in ["target", "e3_ligase"]:
        blocks.append(pd.get_dummies(df[col].fillna("UNKNOWN"), prefix=col, dtype=np.float32).to_numpy())
    desc = np.vstack([rdkit_descriptors(s) for s in df["smiles"].fillna("")]).astype(np.float32)
    # Scale continuous descriptors using fixed full-table statistics; no labels or test outcomes are used.
    mean, sd = desc.mean(axis=0), desc.std(axis=0)
    desc = (desc - mean) / np.where(sd > 0, sd, 1.0)
    blocks.append(desc.astype(np.float32))
    return np.hstack(blocks).astype(np.float32), {
        "morgan": "radius=2, 2048 bits",
        "categorical": ["target", "e3_ligase"],
        "descriptors": ["MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotB", "RingCount", "FractionCSP3", "AromaticRings"],
        "feature_dim": int(sum(b.shape[1] for b in blocks)),
    }


def xgb_fit_predict(X_train, y_train, X_test, seed):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        n_jobs=max(1, (os.cpu_count() or 2) - 1), random_state=seed, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def degrademaster_pseudolabel(X_pn, y_pn, X_u, X_test, seed):
    # Teacher/student analogue of memory-based label enrichment. The threshold
    # and per-class cap are fixed before seeing the test role.
    set_seed(seed)
    teacher_test = fit_mlp(X_pn, y_pn, X_u, seed=seed, epochs=40, hidden=256)
    candidates = np.where((teacher_test >= 0.9) | (teacher_test <= 0.1))[0]
    max_each = max(1, 2 * len(y_pn))
    pos = candidates[teacher_test[candidates] >= 0.9]
    neg = candidates[teacher_test[candidates] <= 0.1]
    pos = pos[np.argsort(-teacher_test[pos])[:max_each]]
    neg = neg[np.argsort(teacher_test[neg])[:max_each]]
    chosen = np.concatenate([pos, neg])
    if len(chosen):
        X_aug = np.vstack([X_pn, X_u[chosen]])
        y_aug = np.concatenate([y_pn, (teacher_test[chosen] >= 0.5).astype(int)])
    else:
        X_aug, y_aug = X_pn, y_pn
    pred = fit_mlp(X_aug, y_aug, X_test, seed=seed + 100000, epochs=40, hidden=256)
    return pred, {"n_candidates": int(len(candidates)), "n_pseudo": int(len(chosen)), "threshold": 0.9, "max_each": int(max_each)}


def main():
    t0 = time.time()
    contract = load_contract(validate_hashes=True)
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    X_morgan = np.load(FEATURE_PATH).astype(np.float32)
    X, feature_info = build_features(df, X_morgan)
    lab = df[LABEL_COL].to_numpy()
    p_mask, n_mask = lab == "P", lab == "N"
    group_cache = {split: group_values(df, groups, split) for split in SPLITS}
    results = {
        "meta": {
            "script": "head_to_head_reimplementations.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": contract["artifacts"]["manifest_sha256"],
            "split_schema": contract["split_schema"],
            "feature_info": feature_info,
            "task": "P/N classification on common test subset; frozen all_records_2way train/test roles",
            "models": OFFICIAL,
            "boundary": "Neither neural result is an official-weight reproduction; missing structural inputs are documented above.",
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
            predictions = {
                "morgan_xgboost": xgb_fit_predict(X_morgan[np.concatenate([p_train, n_train])], y_pn, X_morgan[te], seed),
                "deepprotacs_aligned": fit_mlp(X_pn, y_pn, X[te], seed=seed, epochs=40, hidden=256),
            }
            predictions["degrademaster_aligned"], pseudo_info = degrademaster_pseudolabel(X_pn, y_pn, X[u_pool], X[te], seed)
            seed_entry = {
                "seed": int(seed), "n_train_p": int(len(p_train)), "n_train_n": int(len(n_train)),
                "n_train_u": int(np.sum(lab[u_pool] == "U")), "n_test_p": int(len(p_test)), "n_test_n": int(len(n_test)),
                "degrademaster_pseudolabel": pseudo_info, "metrics": {},
            }
            for method, pred in predictions.items():
                metrics = classification_metrics(y_te, pred)
                seed_entry["metrics"][method] = metrics
                for metric, value in metrics.items():
                    agg[f"{method}::{metric}"].append(value)
                for rid, yt, yp, gid in zip(df.iloc[te]["record_id"], y_te, pred, group_cache[split][te]):
                    prediction_rows.append({
                        "split_regime": split, "seed": int(seed), "method": method,
                        "record_id": int(rid), "group_id": str(gid), "y_true": int(yt), "y_pred": float(yp),
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
