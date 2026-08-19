# -*- coding: utf-8 -*-
"""Frozen-role condition aggregation sensitivity analysis for PROTAC benchmark v3."""
from __future__ import annotations
import os
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT=BASE/'40_项目_Bias-Aware'
CODE=PROJECT/'code'
PROC=BASE/'data'/'derived'
REPORTS=BASE/'reports'
FIGDIR=PROJECT/'01_论文'/'04_盲审v3修订初稿'/'figures'
sys.path.insert(0,str(CODE))
from benchmark_contract import FEATURE_PATH, SEEDS, SPLITS, load_dataset, load_groups, load_manifest
OUT_DATA=PROC/'condition_aggregate_sensitivity.csv'
OUT_RESULTS=PROC/'condition_aggregate_sensitivity_results.csv'
OUT_JSON=PROC/'condition_aggregate_sensitivity_results.json'
OUT_REPORT=REPORTS/'CONDITION_AGGREGATION_SENSITIVITY_V3.md'
OUT_FIG=FIGDIR/'fig5_condition_aggregation_sensitivity.png'

def txt(x):
    return '__MISSING__' if pd.isna(x) or str(x).strip()=='' else str(x).strip()

def num(x):
    if pd.isna(x): return '__MISSING__'
    try:
        x=float(x)
        return str(int(x)) if x.is_integer() else f'{x:.8g}'
    except (TypeError, ValueError):
        return str(x).strip()

def key(parts):
    return '||'.join(str(x).replace('|','/') for x in parts)

def make_keys(df):
    co=df.compound_id.map(txt); tar=df.target.map(txt); e3=df.e3_ligase.map(txt)
    cell=df.cell_line.map(num); th=df.treatment_time_h.map(num); pub=df.article_doi.map(txt)
    ctx=[f'publication:{p}' if c=='__MISSING__' and t=='__MISSING__' else f'cell:{c};time_h:{t}' for c,t,p in zip(cell,th,pub)]
    return {
      'condition':pd.Series([key([a,b,c,d]) for a,b,c,d in zip(co,tar,e3,ctx)],index=df.index),
      'compound_poi_e3':pd.Series([key([a,b,c]) for a,b,c in zip(co,tar,e3)],index=df.index),
      'publication_aware':pd.Series([key([a,b,c,d,f'cell:{e};time_h:{f}']) for a,b,c,d,e,f in zip(co,tar,e3,pub,cell,th)],index=df.index)
    }

def agg_label(s):
    v=set(s.dropna().astype(str))
    if 'P' in v and 'N' not in v:return 'P'
    if 'N' in v and 'P' not in v:return 'N'
    if 'P' in v and 'N' in v:return 'A'
    return 'U'

def first(s):
    for x in s:
        if not pd.isna(x) and str(x).strip()!='': return x
    return np.nan

def build_agg(df, keys):
    rows=[]
    for level,ks in keys.items():
        w=df.copy(); w['aggregate_id']=ks.astype(str).values
        for aid,g in w.groupby('aggregate_id',sort=False):
            p=pd.to_numeric(g.loc[g.dc50_obs_type.eq('exact'),'pdc50_value'],errors='coerce').dropna()
            d=pd.to_numeric(g.loc[g.dmax_obs_type.eq('exact'),'dmax_value'],errors='coerce').dropna()
            rows.append({
              'aggregation_level':level,'aggregate_id':aid,
              'record_ids':';'.join(map(str,g.record_id.astype(int))),
              'n_raw':len(g),'n_exact_pdc50':len(p),
              'n_censored_pdc50':int(g.dc50_obs_type.isin(['left-censored','right-censored','interval-censored']).sum()),
              'n_exact_dmax':len(d),'n_publications':int(g.article_doi.dropna().astype(str).nunique()),
              'n_compounds':int(g.compound_id.nunique()),
              'pdc50_agg_exact_median':float(p.median()) if len(p) else np.nan,
              'dmax_agg_exact_median':float(d.median()) if len(d) else np.nan,
              'pdc50_exact_iqr':float(p.quantile(.75)-p.quantile(.25)) if len(p)>=2 else np.nan,
              'dmax_exact_iqr':float(d.quantile(.75)-d.quantile(.25)) if len(d)>=2 else np.nan,
              'activity_evidence_agg':agg_label(g.activity_evidence_v2),
              'activity_has_P':bool((g.activity_evidence_v2=='P').any()),
              'activity_has_N':bool((g.activity_evidence_v2=='N').any()),
              'activity_has_U':bool((g.activity_evidence_v2=='U').any()),
              'activity_has_A':bool((g.activity_evidence_v2=='A').any()),
              'record_id_first':int(g.record_id.iloc[0]),
              'compound_id':first(g.compound_id),'target':first(g.target),
              'e3_ligase':first(g.e3_ligase),'scaffold':first(g.scaffold)
            })
    return pd.DataFrame(rows)

def clf_metrics(y,p):
    from sklearn.metrics import average_precision_score,balanced_accuracy_score,brier_score_loss,matthews_corrcoef,roc_auc_score
    y=np.asarray(y,int); p=np.asarray(p,float)
    if len(y)<2 or len(np.unique(y))<2:return None
    return {'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'mcc':float(matthews_corrcoef(y,(p>=.5).astype(int))),
            'balanced_acc':float(balanced_accuracy_score(y,(p>=.5).astype(int)))}

def reg_metrics(y,p):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    from scipy.stats import spearmanr
    y=np.asarray(y,float); p=np.asarray(p,float)
    if len(y)<2:return None
    rho=spearmanr(y,p).statistic
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(math.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p)) if np.unique(y).size>1 else np.nan,
            'spearman':float(rho) if np.isfinite(rho) else np.nan}

def boot_delta(y,pa,pb,groups,metric,n=500,seed=42):
    rng=np.random.default_rng(seed); groups=np.asarray(groups); uniq=np.unique(groups)
    if len(uniq)<2:return (np.nan,np.nan,np.nan,len(uniq))
    vals=[]
    for _ in range(n):
        sel=rng.choice(uniq,len(uniq),replace=True)
        idx=np.concatenate([np.flatnonzero(groups==g) for g in sel])
        if metric=='roc_auc':
            a=clf_metrics(np.asarray(y)[idx],np.asarray(pa)[idx]); b=clf_metrics(np.asarray(y)[idx],np.asarray(pb)[idx])
        else:
            a=reg_metrics(np.asarray(y)[idx],np.asarray(pa)[idx]); b=reg_metrics(np.asarray(y)[idx],np.asarray(pb)[idx])
        if a and b and np.isfinite(a[metric]) and np.isfinite(b[metric]):vals.append(a[metric]-b[metric])
    return (float(np.mean(vals)),float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)),len(uniq)) if vals else (np.nan,np.nan,np.nan,len(uniq))

def run_models(df,agg,keys,manifest,X):
    raw_keys={level:dict(zip(df.record_id.astype(int),ks.astype(str))) for level,ks in keys.items()}
    all_rows=[]; preds=[]
    import xgboost as xgb
    nj=max(1,(os.cpu_count() or 2)-1)
    for level in keys:
        key_to_raw={}
        for rid,aid in raw_keys[level].items():key_to_raw.setdefault(aid,[]).append(rid)
        sub=agg[agg.aggregation_level.eq(level)].copy().reset_index(drop=True)
        for split in SPLITS:
          for seed in SEEDS:
            raw_role={(int(r.record_id)):r.role for r in manifest[(manifest.split_family=='all_records_2way')&(manifest.split_regime==split)&(manifest.seed==seed)].itertuples()}
            role={}; mixed=0
            for aid,rids in key_to_raw.items():
                rs={raw_role.get(int(r),'excluded') for r in rids}
                if len(rs)==1:role[aid]=next(iter(rs))
                else:role[aid]='mixed'; mixed+=1
            sub['frozen_role']=sub.aggregate_id.map(role).fillna('excluded')
            for task,eligible in [('T3_pn_clf',sub.activity_evidence_agg.isin(['P','N'])),('T4_pnu_clf',sub.activity_evidence_agg.isin(['P','N','U']))]:
                tr=sub.index[eligible&sub.frozen_role.eq('train')].to_numpy(); te=sub.index[eligible&sub.frozen_role.eq('test')].to_numpy()
                if len(tr)<4 or len(te)<4 or sub.loc[tr,'activity_evidence_agg'].nunique()<2 or sub.loc[te,'activity_evidence_agg'].nunique()<2:continue
                ytr=(sub.loc[tr,'activity_evidence_agg'].to_numpy()=='P').astype(int); yte=(sub.loc[te,'activity_evidence_agg'].to_numpy()=='P').astype(int)
                m=xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=nj,random_state=seed,eval_metric='logloss',verbosity=0)
                m.fit(X[sub.loc[tr,'feature_row'].to_numpy()],ytr)
                p=m.predict_proba(X[sub.loc[te,'feature_row'].to_numpy()])[:,1]; met=clf_metrics(yte,p)
                if met:
                    all_rows.append({'level':level,'split_regime':split,'seed':seed,'task':task,'metric_family':'classification','n_aggregate_total':len(sub),'n_mixed_excluded':mixed,'n_train':len(tr),'n_test':len(te),'n_test_P':int(yte.sum()),'n_test_nonP':int((1-yte).sum()),**met})
                    for i,yt,yp in zip(te,yte,p):preds.append({'level':level,'split_regime':split,'seed':seed,'task':task,'aggregate_id':sub.loc[i,'aggregate_id'],'y_true':int(yt),'y_pred':float(yp)})
            for task,col in [('T1_pdc50_reg','pdc50_agg_exact_median'),('T2_dmax_reg','dmax_agg_exact_median')]:
                eligible=sub[col].notna(); tr=sub.index[eligible&sub.frozen_role.eq('train')].to_numpy(); te=sub.index[eligible&sub.frozen_role.eq('test')].to_numpy()
                if len(tr)<8 or len(te)<4:continue
                ytr=sub.loc[tr,col].to_numpy(float); yte=sub.loc[te,col].to_numpy(float)
                m=xgb.XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=nj,random_state=seed,objective='reg:squarederror',eval_metric='rmse',verbosity=0)
                m.fit(X[sub.loc[tr,'feature_row'].to_numpy()],ytr)
                p=m.predict(X[sub.loc[te,'feature_row'].to_numpy()]); met=reg_metrics(yte,p)
                if met:
                    all_rows.append({'level':level,'split_regime':split,'seed':seed,'task':task,'metric_family':'regression','n_aggregate_total':len(sub),'n_mixed_excluded':mixed,'n_train':len(tr),'n_test':len(te),'n_test_P':np.nan,'n_test_nonP':np.nan,**met})
                    for i,yt,yp in zip(te,yte,p):preds.append({'level':level,'split_regime':split,'seed':seed,'task':task,'aggregate_id':sub.loc[i,'aggregate_id'],'y_true':float(yt),'y_pred':float(yp)})
    return pd.DataFrame(all_rows),preds,raw_keys

def main():
    t0=time.time(); df=load_dataset(); load_groups(df); manifest=load_manifest(validate=True)
    X=np.load(FEATURE_PATH).astype(np.float32); keys=make_keys(df); agg=build_agg(df,keys)
    agg['feature_row']=agg.record_id_first.astype(int); agg.to_csv(OUT_DATA,index=False,encoding='utf-8-sig')
    res,preds,raw_keys=run_models(df,agg,keys,manifest,X); res.to_csv(OUT_RESULTS,index=False,encoding='utf-8-sig')

    baseline=pd.read_csv(PROC/'baseline_predictions_v3.csv',encoding='utf-8-sig'); baseline=baseline[baseline.feature.eq('F1_morgan')]
    pair=[]
    for r in preds:
        bt={'T3_pn_clf':'T3_pn_clf','T4_pnu_clf':'T4_diag_uan_clf','T1_pdc50_reg':'T1_pdc50_reg','T2_dmax_reg':'T2_dmax_reg'}[r['task']]
        rids=[rid for rid,aid in raw_keys[r['level']].items() if aid==r['aggregate_id']]
        b=baseline[baseline.task.eq(bt)&baseline.split_regime.eq(r['split_regime'])&baseline.seed.eq(r['seed'])&baseline.record_id.isin(rids)]
        if b.empty:continue
        bp=float(b.y_pred.mean()) if r['task'] in ['T3_pn_clf','T4_pnu_clf'] else float(b.y_pred.median())
        pair.append({**r,'record_pred_collapsed':bp})
    pairdf=pd.DataFrame(pair); paired=[]
    if not pairdf.empty:
      from sklearn.metrics import roc_auc_score
      for (level,split,seed,task),g in pairdf.groupby(['level','split_regime','seed','task']):
        metric='roc_auc' if task in ['T3_pn_clf','T4_pnu_clf'] else 'mae'; y=g.y_true.to_numpy(); pa=g.y_pred.to_numpy(); pb=g.record_pred_collapsed.to_numpy()
        if metric=='roc_auc' and len(np.unique(y))<2:continue
        delta=float(roc_auc_score(y,pa)-roc_auc_score(y,pb)) if metric=='roc_auc' else float(np.mean(np.abs(y-pa))-np.mean(np.abs(y-pb)))
        d,lo,hi,ng=boot_delta(y,pa,pb,g.aggregate_id.to_numpy(),metric,500,int(seed))
        paired.append({'level':level,'split_regime':split,'seed':int(seed),'task':task,'metric':metric,'n_paired_aggregates':len(g),'delta_aggregate_minus_collapsed_record':delta,'paired_ci_lower':lo,'paired_ci_upper':hi,'bootstrap_groups':ng})
    pairout=pd.DataFrame(paired)
    summary={'meta':{
      'generated':time.strftime('%Y-%m-%d %H:%M:%S'),
      'analysis':'condition-level and alternative observation-unit aggregation sensitivity',
      'record_level_n':int(len(df)),'levels':list(keys),
      'aggregation_rules':{
        'condition':'compound_id x target x e3_ligase x (cell_line,treatment_time_h); both missing -> article_doi fallback',
        'compound_poi_e3':'compound_id x target x e3_ligase',
        'publication_aware':'compound_id x target x e3_ligase x article_doi x cell_line x treatment_time_h',
        'continuous':'exact values only; group median; raw/exact/censored counts retained',
        'classification':'P only -> P; N only -> N; P+N -> A; otherwise U'},
      'split_rule':'all_records_2way frozen roles; mixed-role aggregate units excluded and counted',
      'model':'Morgan radius=2, 2048 bits; XGBoost 300 trees depth 6 learning_rate 0.05 subsample 0.8 colsample_bytree 0.8; three frozen seeds',
      'bootstrap':'500 aggregate-group resamples; conditional on frozen predictions'},
      'aggregate_counts':agg.groupby('aggregation_level').size().to_dict(),
      'aggregate_raw_counts':agg.groupby('aggregation_level').n_raw.sum().to_dict(),
      'metrics':res.to_dict('records'),
      'paired_collapsed_record_comparisons':pairout.to_dict('records') if not pairout.empty else []}
    OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:None),encoding='utf-8')

    lines=['# Condition-level aggregation sensitivity analysis (v3)','',f"> Generated: {summary['meta']['generated']}",'> Scope: conditional sensitivity analysis using frozen all_records_2way roles; no new random split was created.','','## Protocol','','Three observation-unit definitions were compared: condition-level, compound–POI–E3 and publication-aware condition units. Exact pDC50 and Dmax values were summarized by the group median. P/N/U labels were combined by explicit conflict rules. Aggregate units crossing frozen train/test roles were excluded rather than reassigned.','','## Aggregate counts','','| Level | Aggregate units | Raw records represented |','|---|---:|---:|']
    for level in keys:lines.append(f"| {level} | {summary['aggregate_counts'][level]:,} | {summary['aggregate_raw_counts'][level]:,} |")
    lines += ['', '## Model metrics','', 'Mean ± SD across the three frozen seeds; values are descriptive and conditional on the existing split contract.','', '| Level | Split | Task | n train | n test | ROC-AUC / MAE | PR-AUC / RMSE | R² / Brier |','|---|---|---|---:|---:|---:|---:|---:|']
    for (level,split,task),g in res.groupby(['level','split_regime','task']):
        if g.metric_family.iloc[0]=='classification':
            a=f"{g.roc_auc.mean():.3f} ± {g.roc_auc.std(ddof=0):.3f}"; b=f"{g.pr_auc.mean():.3f} ± {g.pr_auc.std(ddof=0):.3f}"; c=f"{g.brier.mean():.3f} ± {g.brier.std(ddof=0):.3f}"
        else:
            a=f"{g.mae.mean():.3f} ± {g.mae.std(ddof=0):.3f}"; b=f"{g.rmse.mean():.3f} ± {g.rmse.std(ddof=0):.3f}"; c=f"{g.r2.mean():.3f} ± {g.r2.std(ddof=0):.3f}"
        lines.append(f"| {level} | {split} | {task} | {g.n_train.mean():.1f} | {g.n_test.mean():.1f} | {a} | {b} | {c} |")
    lines += ['', '## Paired comparison to collapsed record-level predictions','', 'Paired group-bootstrap deltas are aggregate-model minus collapsed record-level predictions. Positive ROC-AUC deltas favor aggregation; negative MAE deltas favor aggregation. Intervals are conditional on frozen predictions.','', '| Level | Split | Task | Metric | Paired aggregates | Delta | 95% CI |','|---|---|---|---|---:|---:|---:|']
    if pairout.empty:lines.append('| — | — | — | — | 0 | — | No paired rows available |')
    else:
      for r in pairout.to_dict('records'):lines.append(f"| {r['level']} | {r['split_regime']} | {r['task']} | {r['metric']} | {r['n_paired_aggregates']} | {r['delta_aggregate_minus_collapsed_record']:.4f} | [{r['paired_ci_lower']:.4f}, {r['paired_ci_upper']:.4f}] |")
    lines += ['', '## Interpretation','', '- This tests whether conclusions are sensitive to repeated assay records receiving less direct weight.', '- A changed metric is a changed estimand, not causal identification of a bias mechanism.', '- Aggregation can hide genuine heterogeneity across dose, time and cell line; the publication fallback prevents unknown conditions being silently merged.', '- The external dataset is pending and is not represented by this internal analysis.', '', '## Reproducibility artifacts', '', f'- Aggregate table: `{OUT_DATA.as_posix()}`', f'- Metrics CSV: `{OUT_RESULTS.as_posix()}`', f'- Results JSON: `{OUT_JSON.as_posix()}`', f'- Figure: `{OUT_FIG.as_posix()}`', f'- Script: `{(CODE/"condition_aggregation_sensitivity.py").as_posix()}`', '']
    OUT_REPORT.write_text('\n'.join(lines),encoding='utf-8')

    try:
      import matplotlib.pyplot as plt
      plot=res[res.task.eq('T3_pn_clf')]
      if not plot.empty:
        piv=plot.groupby(['split_regime','level']).roc_auc.mean().unstack('level').reindex(SPLITS)
        fig,ax=plt.subplots(1,2,figsize=(11,4.5)); piv.plot(kind='bar',ax=ax[0],ylim=(.45,1),rot=0)
        ax[0].set_ylabel('Mean ROC-AUC (T3 P/N)'); ax[0].set_xlabel('Frozen split regime'); ax[0].legend(title='Aggregation',fontsize=8)
        counts=agg.groupby('aggregation_level').size().reindex(list(keys)); ax[1].bar(counts.index,counts.values,color=['#4C72B0','#55A868','#C44E52'])
        ax[1].set_ylabel('Aggregate units'); ax[1].tick_params(axis='x',rotation=25); fig.tight_layout(); FIGDIR.mkdir(parents=True,exist_ok=True); fig.savefig(OUT_FIG,dpi=180); plt.close(fig)
    except Exception as e:
      summary['figure_error']=repr(e); OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:None),encoding='utf-8')
    print('wrote',OUT_DATA); print('wrote',OUT_RESULTS); print('wrote',OUT_JSON); print('wrote',OUT_REPORT); print('elapsed',round(time.time()-t0,1),'s')

if __name__=='__main__':main()

