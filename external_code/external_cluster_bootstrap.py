# -*- coding: utf-8 -*-
"""Cluster-level bootstrap and T3 label-conflict sensitivity for external validation."""
from __future__ import annotations
import hashlib, json, warnings
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("PROTAC_EXTERNAL_ROOT", str(REPO_ROOT / "external_data")))
PROC = BASE / "processed"
RESULTS = BASE / "results"
COHORT = PROC / "external_validation_cohort.csv"
PRED = RESULTS / "external_predictions.csv"
STRUCT_OUT = RESULTS / "external_structure_cluster_bootstrap.csv"
PATENT_OUT = RESULTS / "external_patent_cluster_bootstrap.csv"
CONFLICT_OUT = RESULTS / "external_conflict_label_sensitivity.csv"
B = 1000


def ece_binary(y, p, bins=10):
    edges = np.linspace(0, 1, bins+1); out = 0.0
    for i in range(bins):
        m = ((p >= edges[i]) & (p <= edges[i+1])) if i == bins-1 else ((p >= edges[i]) & (p < edges[i+1]))
        if m.any(): out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def clf_metrics(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); pred = (p >= 0.5).astype(int)
    out = {"roc_auc": np.nan, "pr_auc": np.nan, "balanced_accuracy": np.nan, "mcc": np.nan, "brier": np.nan, "ece_10bin": np.nan, "sensitivity": np.nan, "specificity": np.nan}
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y,p)); out["pr_auc"] = float(average_precision_score(y,p))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y,pred)); out["mcc"] = float(matthews_corrcoef(y,pred)); out["brier"] = float(brier_score_loss(y,p)); out["ece_10bin"] = ece_binary(y,p)
    tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    out["sensitivity"] = float(tp/(tp+fn)) if tp+fn else np.nan
    out["specificity"] = float(tn/(tn+fp)) if tn+fp else np.nan
    return out


def reg_metrics(y, p):
    y = np.asarray(y,float); p = np.asarray(p,float)
    out = {"mae": float(mean_absolute_error(y,p)), "rmse": float(np.sqrt(mean_squared_error(y,p))), "r2": np.nan, "spearman": np.nan}
    if len(y)>1 and np.std(y)>0: out["r2"] = float(r2_score(y,p))
    if len(y)>1 and np.std(y)>0 and np.std(p)>0:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); out["spearman"] = float(spearmanr(y,p).statistic)
    return out


def stable_seed(*parts):
    raw = '|'.join(map(str,parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**32-1)


def one_cluster_bootstrap(y, p, groups, kind, task, unit, split, seed):
    groups = pd.Series(groups).fillna('UNKNOWN').astype(str).to_numpy()
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == u) for u in unique]
    point = clf_metrics(y,p) if kind == 'clf' else reg_metrics(y,p)
    metrics = list(point)
    vals = {m: [] for m in metrics}
    rng = np.random.default_rng(stable_seed(task, unit, split, seed, B))
    for _ in range(B):
        sampled = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[i] for i in sampled])
        bm = clf_metrics(y[idx], p[idx]) if kind == 'clf' else reg_metrics(y[idx], p[idx])
        for m in metrics: vals[m].append(bm[m])
    rows=[]
    for m in metrics:
        a=np.asarray(vals[m],float); a=a[np.isfinite(a)]
        rows.append({"task":task,"cluster_unit":unit,"split_regime":split,"seed":int(seed),"metric":m,
                     "point_estimate":float(point[m]) if np.isfinite(point[m]) else np.nan,
                     "bootstrap_mean":float(a.mean()) if len(a) else np.nan,
                     "bootstrap_sd":float(a.std(ddof=1)) if len(a)>1 else np.nan,
                     "q025":float(np.quantile(a,.025)) if len(a) else np.nan,
                     "q500":float(np.quantile(a,.5)) if len(a) else np.nan,
                     "q975":float(np.quantile(a,.975)) if len(a) else np.nan,
                     "n_records":int(len(y)),"n_clusters":int(len(unique)),"bootstrap_reps":B,
                     "finite_bootstrap_reps":int(len(a))})
    return rows


def main():
    pred = pd.read_csv(PRED, low_memory=False)
    cohort = pd.read_csv(COHORT, low_memory=False)
    ext = cohort[["external_cohort_row","inchikey","source_patent_id","activity_evidence_external_v1"]].copy()
    ext = ext.rename(columns={"external_cohort_row":"external_row"})
    ext["inchikey"] = ext.inchikey.fillna('UNKNOWN').astype(str)
    ext["source_patent_id"] = ext.source_patent_id.fillna('UNKNOWN').astype(str)
    pred = pred.merge(ext, on='external_row', how='left', suffixes=('','_cohort'))
    all_boot=[]
    for task, kind in [("T3_pn_clf","clf"),("T1_pdc50_reg","reg"),("T2_dmax_reg","reg")]:
        sub = pred[pred.task.eq(task)]
        for (split, seed), g in sub.groupby(['split_regime','seed'], sort=True):
            y=g.y_true.to_numpy(float); p=g.y_pred.to_numpy(float)
            all_boot.extend(one_cluster_bootstrap(y,p,g.inchikey.to_numpy(),kind,task,'inchikey',split,int(seed)))
            all_boot.extend(one_cluster_bootstrap(y,p,g.source_patent_id.to_numpy(),kind,task,'source_patent_id',split,int(seed)))
    boot_df=pd.DataFrame(all_boot)
    boot_df[boot_df.cluster_unit.eq('inchikey')].to_csv(STRUCT_OUT,index=False,encoding='utf-8-sig')
    boot_df[boot_df.cluster_unit.eq('source_patent_id')].to_csv(PATENT_OUT,index=False,encoding='utf-8-sig')

    # T3 label protocol sensitivity. The official record-level protocol excludes U/A before prediction.
    t3 = pred[pred.task.eq('T3_pn_clf')].copy()
    labels = t3.activity_evidence_external_v1.astype(str)
    conflict_structures = set(t3.groupby('inchikey').activity_evidence_external_v1.nunique().loc[lambda s: s > 1].index)
    protocols=[]
    for (split, seed), g in t3.groupby(['split_regime','seed'], sort=True):
        def add(protocol, y, p, n_structures, dropped_records=0, ties=0):
            m=clf_metrics(y,p)
            for metric,val in m.items():
                protocols.append({"task":"T3_pn_clf","protocol":protocol,"split_regime":split,"seed":int(seed),"metric":metric,"value":val,
                                  "n_records":int(len(y)),"n_structures":int(n_structures),"conflict_structures_total":int(len(conflict_structures)),
                                  "dropped_records":int(dropped_records),"majority_ties_dropped":int(ties)})
        add('record_level_all',g.y_true.to_numpy(float),g.y_pred.to_numpy(float),g.inchikey.nunique())
        no_conf=g[~g.inchikey.isin(conflict_structures)]
        add('exclude_conflict_structures',no_conf.y_true.to_numpy(float),no_conf.y_pred.to_numpy(float),no_conf.inchikey.nunique(),len(g)-len(no_conf))
        # Structure-level majority label with mean predicted probability. Ties are excluded.
        rows=[]; ties=0
        for ik, sg in g.groupby('inchikey', sort=False):
            counts=sg.activity_evidence_external_v1.value_counts()
            if len(counts)>1 and counts.iloc[0] == counts.iloc[1]:
                ties += 1; continue
            label=1 if counts.get('P',0) > counts.get('N',0) else 0
            rows.append((label, float(sg.y_pred.mean())))
        arr=np.asarray(rows,float)
        add('structure_majority_label',arr[:,0],arr[:,1],len(rows),ties=ties)
    pd.DataFrame(protocols).to_csv(CONFLICT_OUT,index=False,encoding='utf-8-sig')
    print(f'wrote {STRUCT_OUT} rows={sum(1 for x in all_boot if x["cluster_unit"]=="inchikey")}')
    print(f'wrote {PATENT_OUT} rows={sum(1 for x in all_boot if x["cluster_unit"]=="source_patent_id")}')
    print(f'wrote {CONFLICT_OUT} rows={len(protocols)} conflict_structures={len(conflict_structures)}')

if __name__ == '__main__':
    main()


