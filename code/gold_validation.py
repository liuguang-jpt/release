# -*- coding: utf-8 -*-
"""Internal-consistency annotation-set checks for rules and a frozen P/N model."""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from benchmark_contract import FEATURE_PATH, LABEL_COL, PROCESSED_DIR, REPORTS_DIR, load_dataset

GOLD_PATH = PROCESSED_DIR / "gold_set_annotations" / "gold_final.csv"
RESULTS_PATH = PROCESSED_DIR / "gold_validation_v3.json"
FAILURES_PATH = PROCESSED_DIR / "gold_rule_failures_v3.csv"
MODEL_PREDICTIONS_PATH = PROCESSED_DIR / "gold_model_predictions_v3.csv"
REPORT_PATH = REPORTS_DIR / "GOLD_VALIDATION_V3.md"
LABELS = ["P", "N", "A", "U"]
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260817
MODEL_SEED = 20260815


def scalar_ci(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "lower": float(np.quantile(x, 0.025)),
        "upper": float(np.quantile(x, 0.975)),
        "n_valid": int(len(x)),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }


def bootstrap_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_true)
    store: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for label in LABELS:
        for metric in ["precision", "recall", "f1"]:
            store[f"{label}::{metric}"] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, size=n)
        metrics = multiclass_metrics(y_true[idx], y_pred[idx])
        store["accuracy"].append(metrics["accuracy"])
        store["macro_f1"].append(metrics["macro_f1"])
        for label in LABELS:
            for metric in ["precision", "recall", "f1"]:
                store[f"{label}::{metric}"].append(metrics["per_class"][label][metric])
    return {key: scalar_ci(value) for key, value in store.items()}


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "accuracy_at_0.5": float(accuracy_score(y_true, pred)),
        "macro_f1_at_0.5": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y_true, pred)),
        "mcc_at_0.5": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def bootstrap_binary(y_true: np.ndarray, prob: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    n = len(y_true)
    store: dict[str, list[float]] = Counter()
    store = {key: [] for key in binary_metrics(y_true, prob)}
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics = binary_metrics(y_true[idx], prob[idx])
        for key, value in metrics.items():
            store[key].append(value)
    return {key: scalar_ci(value) for key, value in store.items()}


def failure_taxonomy(row: pd.Series) -> str:
    if row["activity_evidence_v2"] == "U" and row["final_evidence"] in {"P", "N"} and pd.notna(row.get("pctdeg_raw")):
        return "pctdeg_signal_ignored"
    if row["activity_evidence_v2"] == "P" and row["final_evidence"] == "A" and row.get("dc50_obs_type") == "interval-censored":
        return "interval_overcalling"
    note = str(row.get("adjudication_note", ""))
    pct = str(row.get("pctdeg_raw", ""))
    if "矛盾" in note or (row["activity_evidence_v2"] == "P" and row["final_evidence"] == "A" and pct.strip() in {"0", "0.0"}):
        return "cross_endpoint_conflict"
    return "borderline_or_combined_evidence_overcalling"


def markdown_confusion(matrix: np.ndarray, row_name: str, col_name: str) -> str:
    lines = [f"| {row_name} \\ {col_name} | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|---:|"]
    for label, row in zip(LABELS, matrix):
        lines.append(f"| {label} | " + " | ".join(str(int(x)) for x in row) + " |")
    return "\n".join(lines)


def fmt_ci(point: float, ci: dict) -> str:
    return f"{point:.3f} (95% CI {ci['lower']:.3f}–{ci['upper']:.3f})"


def main() -> None:
    started = time.time()
    df = load_dataset()
    gold = pd.read_csv(GOLD_PATH, encoding="utf-8-sig")
    if len(gold) != 132 or not gold["record_id"].is_unique:
        raise AssertionError("expected the frozen 132-record internal-consistency annotation set")
    missing = set(gold["record_id"]) - set(df["record_id"])
    if missing:
        raise AssertionError(f"internal-consistency IDs missing from main dataset: {len(missing)}")

    merged = gold.merge(
        df[[
            "record_id", LABEL_COL, "dc50_obs_type", "pdc50_value", "pdc50_lower", "pdc50_upper",
            "dmax_obs_type", "dmax_value", "dmax_lower", "dmax_upper"
        ]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    y_a = merged["annotator_A_label"].astype(str).to_numpy()
    y_b = merged["annotator_B_label"].astype(str).to_numpy()
    y_gold = merged["final_evidence"].astype(str).to_numpy()
    y_rule = merged[LABEL_COL].astype(str).to_numpy()

    annotator_confusion = confusion_matrix(y_a, y_b, labels=LABELS)
    rule_confusion = confusion_matrix(y_gold, y_rule, labels=LABELS)
    rule_metrics = multiclass_metrics(y_gold, y_rule)
    rule_ci = bootstrap_multiclass(y_gold, y_rule)

    failures = merged.loc[y_gold != y_rule].copy()
    failures["failure_taxonomy"] = failures.apply(failure_taxonomy, axis=1)
    failure_cols = [
        "record_id", "target", "e3_ligase", "cell_line", "treatment_time_h",
        "dc50_raw", "dmax_raw", "pctdeg_raw", "annotator_A_label", "annotator_B_label",
        "final_evidence", LABEL_COL, "failure_taxonomy", "final_confidence", "adjudication_note",
        "dc50_obs_type", "pdc50_value", "pdc50_lower", "pdc50_upper", "dmax_obs_type",
        "dmax_value", "dmax_lower", "dmax_upper",
    ]
    # The raw endpoint columns came from the internal-consistency sample and retain their unsuffixed names.
    failures[[c for c in failure_cols if c in failures.columns]].to_csv(
        FAILURES_PATH, index=False, encoding="utf-8-sig"
    )

    # Frozen Morgan + XGBoost P/N classifier. Internal-consistency record IDs are excluded from training.
    X = np.load(FEATURE_PATH, mmap_mode="r")
    if X.shape[0] != len(df):
        raise AssertionError("feature rows do not match record-level dataset")
    gold_ids = set(gold["record_id"].astype(int))
    train_mask = df[LABEL_COL].isin(["P", "N"]) & ~df["record_id"].isin(gold_ids)
    test_mask_gold = merged["final_evidence"].isin(["P", "N"]).to_numpy()
    train_idx = np.flatnonzero(train_mask.to_numpy())
    id_to_pos = pd.Series(np.arange(len(df)), index=df["record_id"])
    test_idx = id_to_pos.loc[merged.loc[test_mask_gold, "record_id"]].to_numpy(dtype=int)
    y_train = (df.iloc[train_idx][LABEL_COL].to_numpy() == "P").astype(int)
    y_model_gold = (merged.loc[test_mask_gold, "final_evidence"].to_numpy() == "P").astype(int)

    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=MODEL_SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X[train_idx], y_train)
    model_prob = model.predict_proba(X[test_idx])[:, 1]
    model_metrics = binary_metrics(y_model_gold, model_prob)
    model_ci = bootstrap_binary(y_model_gold, model_prob)
    model_rows = merged.loc[test_mask_gold, [
        "record_id", "target", "e3_ligase", "final_evidence", LABEL_COL
    ]].copy()
    model_rows["y_true"] = y_model_gold
    model_rows["p_positive"] = model_prob
    model_rows["y_pred_at_0.5"] = (model_prob >= 0.5).astype(int)
    model_rows.to_csv(MODEL_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    note_nonempty = merged["adjudication_note"].notna()
    disagreement = y_a != y_b
    result = {
        "meta": {
            "script": "gold_validation.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gold_path": str(GOLD_PATH),
            "n_gold": int(len(merged)),
            "label_order": LABELS,
            "bootstrap": {
                "unit": "record",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile 95%",
            },
        },
        "annotator_agreement": {
            "n": int(len(merged)),
            "n_agree": int((y_a == y_b).sum()),
            "agreement": float((y_a == y_b).mean()),
            "cohen_kappa": float(cohen_kappa_score(y_a, y_b, labels=LABELS)),
            "confusion_A_rows_B_columns": annotator_confusion.tolist(),
            "n_A_B_disagreements": int(disagreement.sum()),
            "n_final_equals_A": int((y_gold == y_a).sum()),
            "n_final_equals_B": int((y_gold == y_b).sum()),
            "n_nonempty_adjudication_notes": int(note_nonempty.sum()),
            "note_reconciliation": "5 A/B disagreements plus 2 data-correction notes on A/B-agreed records",
            "personnel_documentation_status": "author input required; identities, qualifications, independence, blinding, and adjudicator role are not documented in the frozen files",
        },
        "rule_vs_final": {
            "confusion_gold_rows_rule_columns": rule_confusion.tolist(),
            "metrics": rule_metrics,
            "bootstrap_ci": rule_ci,
            "n_match": int((y_gold == y_rule).sum()),
            "n_mismatch": int((y_gold != y_rule).sum()),
            "failure_taxonomy_counts": {k: int(v) for k, v in failures["failure_taxonomy"].value_counts().items()},
        },
        "frozen_model_gold_pn": {
            "description": "fixed Morgan-2048 + XGBoost classifier; all internal-consistency record IDs excluded from training; threshold fixed at 0.5",
            "model_seed": MODEL_SEED,
            "n_train": int(len(train_idx)),
            "n_train_p": int(y_train.sum()),
            "n_train_n": int((1 - y_train).sum()),
            "n_test_gold_pn": int(len(y_model_gold)),
            "n_test_p": int(y_model_gold.sum()),
            "n_test_n": int((1 - y_model_gold).sum()),
            "metrics": model_metrics,
            "bootstrap_ci": model_ci,
            "scope": "record-held-out annotation-set transfer check, not external validation and not a population accuracy estimate",
        },
        "reporting_context": [
            "Annotator A was the first author and rule developer; annotator B was AI-assisted; adjudication was performed by the first author and was not independent third-party review.",
            "The five disagreements were resolved by literal application of the frozen rule and all five final labels equal annotator B's labels.",
            "The nominal target of 200 was unreachable under the capped quota formula: 121 quota records plus 11 rare-censor safeguards yielded 132.",
            "The stratified internal-consistency aggregate is not automatically population-prevalence weighted.",
        ],
    }
    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Adjudicated internal-consistency annotation-set validation (v3)",
        "",
        f"> Generated: {result['meta']['generated']}  ",
        f"> Frozen file: `{GOLD_PATH}`  ",
        f"> Record bootstrap: {BOOTSTRAP_REPLICATES:,} replicates, percentile 95% CI.",
        "",
        "## 1. Annotator agreement",
        "",
        f"- A/B agreement: **{result['annotator_agreement']['n_agree']}/{len(merged)} = {result['annotator_agreement']['agreement']*100:.1f}%**.",
        f"- Cohen's κ: **{result['annotator_agreement']['cohen_kappa']:.3f}**.",
        f"- A/B disagreements: **{result['annotator_agreement']['n_A_B_disagreements']}**.",
        f"- Final equals annotator A: {result['annotator_agreement']['n_final_equals_A']}/{len(merged)}; final equals annotator B: **{result['annotator_agreement']['n_final_equals_B']}/{len(merged)}**.",
        "- Seven non-empty notes do not mean seven annotation disagreements: five notes correspond to A/B disagreements and two document source-data corrections on records already labeled P/P.",
        "",
        markdown_confusion(annotator_confusion, "Annotator A", "Annotator B"),
        "",
        "## 2. Automatic rule versus final four-class label",
        "",
        f"- Agreement/accuracy: **{int((y_gold == y_rule).sum())}/{len(merged)} = {fmt_ci(rule_metrics['accuracy'], rule_ci['accuracy'])}**.",
        f"- Macro-F1: **{fmt_ci(rule_metrics['macro_f1'], rule_ci['macro_f1'])}**.",
        "",
        markdown_confusion(rule_confusion, "Final label", "Automatic rule"),
        "",
        "### Per-class performance",
        "",
        "| Class | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        m = rule_metrics["per_class"][label]
        lines.append(
            f"| {label} | {fmt_ci(m['precision'], rule_ci[f'{label}::precision'])} | "
            f"{fmt_ci(m['recall'], rule_ci[f'{label}::recall'])} | "
            f"{fmt_ci(m['f1'], rule_ci[f'{label}::f1'])} | {m['support']} |"
        )
    lines += [
        "",
        "### Failure taxonomy (14 mismatches)",
        "",
        "| Failure type | Count | Interpretation |",
        "|---|---:|---|",
    ]
    descriptions = {
        "pctdeg_signal_ignored": "DC50/Dmax were absent, but percent-degradation evidence supported P or N while the rule returned U.",
        "interval_overcalling": "An interval endpoint was treated as sufficiently positive although the expert label remained ambiguous.",
        "cross_endpoint_conflict": "Strong DC50/Dmax and percent-degradation signals conflicted, but the OR rule returned P.",
        "borderline_or_combined_evidence_overcalling": "Moderate/borderline combined evidence was overcalled as P.",
    }
    for key, count in failures["failure_taxonomy"].value_counts().items():
        lines.append(f"| `{key}` | {int(count)} | {descriptions[key]} |")
    lines += [
        "",
        f"The complete record-level table is saved as `{FAILURES_PATH}`.",
        "",
        "## 3. Frozen-model transfer check on final P/N records",
        "",
        f"- Training records: {len(train_idx):,} ({int(y_train.sum()):,} P; {int((1-y_train).sum()):,} N), with all 132 internal-consistency record IDs excluded.",
        f"- Test records: {len(y_model_gold)} ({int(y_model_gold.sum())} P; {int((1-y_model_gold).sum())} N).",
        "- Model and threshold were fixed: Morgan-2048 + XGBoost; decision threshold = 0.5; no tuning on the internal-consistency sample.",
        "",
        "| Metric | Estimate (95% record-bootstrap CI) |",
        "|---|---:|",
    ]
    for key, value in model_metrics.items():
        lines.append(f"| {key} | {fmt_ci(value, model_ci[key])} |")
    lines += [
        "",
        "This is a **record-held-out annotation-set transfer check**, not external validation: the annotations were sampled from the same underlying database, and only record IDs—not all related scaffolds/publications/targets—were excluded.",
        "",
        "## 4. Reporting context",
        "",
        "1. Annotator A was the first author and rule developer; annotator B was AI-assisted; adjudication was performed by the first author and was not independent third-party review.",
        "2. All five disagreements were resolved by literal application of the frozen rule and all five final labels equal annotator B's labels.",
        "3. The nominal target of 200 was unreachable under the capped quota formula: 121 quota records plus 11 rare-censor safeguards yielded 132.",
        "4. The stratified internal-consistency aggregate is not automatically a prevalence-weighted estimate for the full database.",
        "",
        "## 5. Claim boundary",
        "",
        "The 96.2% agreement and κ=0.948 validate **annotator concordance**. The 89.4% rule accuracy validates the **automatic rule against final labels**. The frozen P/N experiment evaluates **model transfer on the final P/N subset**. These are separate questions and must not be combined into one 'independent validation' claim.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"annotator agreement={result['annotator_agreement']['agreement']:.4f}, kappa={result['annotator_agreement']['cohen_kappa']:.4f}")
    print(f"rule accuracy={rule_metrics['accuracy']:.4f}, macro-F1={rule_metrics['macro_f1']:.4f}, mismatches={len(failures)}")
    print(f"internal-consistency P/N frozen-model AUC={model_metrics['roc_auc']:.4f}, macro-F1@0.5={model_metrics['macro_f1_at_0.5']:.4f}")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {FAILURES_PATH}")
    print(f"wrote {MODEL_PREDICTIONS_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()


