# -*- coding: utf-8 -*-
"""Paired hierarchical group bootstrap for pre-defined primary comparisons.

Each replicate samples split seeds with replacement, then samples test groups
with replacement inside each selected seed.  Both methods use the identical
sampled records, and the metric difference is computed directly within the
replicate.  Marginal confidence intervals are never averaged or compared for
overlap.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmark_contract import PROCESSED_DIR, REPORTS_DIR, SEEDS

PU_PREDICTIONS = PROCESSED_DIR / "pu_predictions_v3.csv"
CENSOR_PREDICTIONS = PROCESSED_DIR / "censored_predictions_v3.csv"
RESULTS_PATH = PROCESSED_DIR / "paired_hierarchical_bootstrap_v3.json"
REPORT_PATH = REPORTS_DIR / "PAIRED_HIERARCHICAL_BOOTSTRAP_V3.md"
N_BOOT = 5000
BOOTSTRAP_SEED = 20260817


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    source: str
    split_regime: str
    evaluation_set: str | None
    method_a: str
    method_b: str
    metric: str
    favorable_direction: str


COMPARISONS = [
    *[
        Comparison(
            comparison_id=f"pu_nnpu_minus_supervised_auc__{split}",
            source="pu",
            split_regime=split,
            evaluation_set=None,
            method_a="nnpu",
            method_b="supervised",
            metric="roc_auc",
            favorable_direction="positive",
        )
        for split in ["random", "scaffold", "pub", "poi"]
    ],
    *[
        Comparison(
            comparison_id=f"pu_elkan_noto_minus_u_as_n_auc__{split}",
            source="pu",
            split_regime=split,
            evaluation_set=None,
            method_a="elkan_noto",
            method_b="u_as_n",
            metric="roc_auc",
            favorable_direction="positive",
        )
        for split in ["random", "scaffold", "pub", "poi"]
    ],
    *[
        Comparison(
            comparison_id=f"censor_mlp_censored_minus_mlp_drop_violation__{split}",
            source="censor",
            split_regime=split,
            evaluation_set="censored",
            method_a="mlp_censored",
            method_b="mlp_drop",
            metric="violation_rate",
            favorable_direction="negative",
        )
        for split in ["random", "scaffold"]
    ],
    *[
        Comparison(
            comparison_id=f"censor_mlp_censored_minus_mlp_drop_exact_mae__{split}",
            source="censor",
            split_regime=split,
            evaluation_set="exact",
            method_a="mlp_censored",
            method_b="mlp_drop",
            metric="mae",
            favorable_direction="negative",
        )
        for split in ["random", "scaffold"]
    ],
]


def auc_metric(frame: pd.DataFrame, pred_col: str) -> float:
    y = frame["y_true"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, frame[pred_col].to_numpy(dtype=float)))


def mae_metric(frame: pd.DataFrame, pred_col: str) -> float:
    return float(np.mean(np.abs(frame["y_true"].to_numpy(float) - frame[pred_col].to_numpy(float))))


def violation_metric(frame: pd.DataFrame, pred_col: str) -> float:
    pred = frame[pred_col].to_numpy(float)
    bound = frame["bound"].to_numpy(float)
    side = frame["censor_side"].astype(str).to_numpy()
    violation = np.where(side == "right", pred > bound, pred < bound)
    return float(violation.mean())


def metric_function(metric: str) -> Callable[[pd.DataFrame, str], float]:
    return {"roc_auc": auc_metric, "mae": mae_metric, "violation_rate": violation_metric}[metric]


def prepare_paired_frame(predictions: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    sub = predictions[predictions["split_regime"].eq(comparison.split_regime)].copy()
    if comparison.evaluation_set is not None:
        sub = sub[sub["evaluation_set"].eq(comparison.evaluation_set)].copy()
    sub = sub[sub["method"].isin([comparison.method_a, comparison.method_b])].copy()
    key = ["seed", "record_id"]
    if sub.duplicated(key + ["method"]).any():
        raise AssertionError(f"duplicate method prediction rows: {comparison.comparison_id}")

    metadata_cols = ["seed", "record_id", "group_id", "y_true"]
    if comparison.source == "censor":
        metadata_cols += ["bound", "censor_side", "obs_type", "evaluation_set"]
    meta = sub[metadata_cols].drop_duplicates(key)
    if meta.duplicated(key).any():
        raise AssertionError(f"inconsistent paired metadata: {comparison.comparison_id}")
    wide = sub.pivot(index=key, columns="method", values="y_pred").reset_index()
    wide.columns.name = None
    frame = meta.merge(wide, on=key, how="inner", validate="one_to_one")
    if frame[[comparison.method_a, comparison.method_b]].isna().any().any():
        raise AssertionError(f"unpaired predictions: {comparison.comparison_id}")
    expected_methods = sub.groupby(key)["method"].nunique()
    if not expected_methods.eq(2).all() or len(frame) * 2 != len(sub):
        raise AssertionError(f"method record sets differ: {comparison.comparison_id}")
    if sorted(frame["seed"].unique().tolist()) != sorted(SEEDS):
        raise AssertionError(f"unexpected seeds: {comparison.comparison_id}")
    return frame


def resample_groups(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    unique_groups = frame["group_id"].astype(str).unique()
    selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    by_group = {group: part for group, part in frame.groupby(frame["group_id"].astype(str), sort=False)}
    return pd.concat([by_group[group] for group in selected], ignore_index=True)


def paired_hierarchical_bootstrap(frame: pd.DataFrame, comparison: Comparison) -> dict:
    metric = metric_function(comparison.metric)
    seeds = np.asarray(sorted(frame["seed"].unique()), dtype=int)
    by_seed = {int(seed): frame[frame["seed"].eq(seed)].copy() for seed in seeds}

    per_seed = []
    for seed in seeds:
        seed_frame = by_seed[int(seed)]
        value_a = metric(seed_frame, comparison.method_a)
        value_b = metric(seed_frame, comparison.method_b)
        per_seed.append(
            {
                "seed": int(seed),
                "n_records": int(len(seed_frame)),
                "n_groups": int(seed_frame["group_id"].nunique()),
                "method_a": float(value_a),
                "method_b": float(value_b),
                "difference_a_minus_b": float(value_a - value_b),
            }
        )
    point_a = float(np.mean([x["method_a"] for x in per_seed]))
    point_b = float(np.mean([x["method_b"] for x in per_seed]))
    point_delta = float(np.mean([x["difference_a_minus_b"] for x in per_seed]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_delta: list[float] = []
    attempts = 0
    max_attempts = N_BOOT * 3
    while len(bootstrap_delta) < N_BOOT and attempts < max_attempts:
        attempts += 1
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_differences = []
        valid = True
        for sampled_seed in sampled_seeds:
            sampled_frame = resample_groups(by_seed[int(sampled_seed)], rng)
            value_a = metric(sampled_frame, comparison.method_a)
            value_b = metric(sampled_frame, comparison.method_b)
            if not np.isfinite(value_a) or not np.isfinite(value_b):
                valid = False
                break
            seed_differences.append(value_a - value_b)
        if valid:
            bootstrap_delta.append(float(np.mean(seed_differences)))
    if len(bootstrap_delta) < N_BOOT:
        raise RuntimeError(
            f"only {len(bootstrap_delta)} valid replicates for {comparison.comparison_id} after {attempts} attempts"
        )

    values = np.asarray(bootstrap_delta, dtype=float)
    lower, upper = np.quantile(values, [0.025, 0.975])
    p_le_zero = float(np.mean(values <= 0))
    p_ge_zero = float(np.mean(values >= 0))
    if comparison.favorable_direction == "positive":
        probability_favorable = float(np.mean(values > 0))
    else:
        probability_favorable = float(np.mean(values < 0))
    return {
        "comparison_id": comparison.comparison_id,
        "source": comparison.source,
        "split_regime": comparison.split_regime,
        "evaluation_set": comparison.evaluation_set,
        "metric": comparison.metric,
        "method_a": comparison.method_a,
        "method_b": comparison.method_b,
        "difference_definition": f"{comparison.method_a} - {comparison.method_b}",
        "favorable_direction": comparison.favorable_direction,
        "point_estimate": {
            "method_a_mean_across_seeds": point_a,
            "method_b_mean_across_seeds": point_b,
            "difference_mean_across_seeds": point_delta,
        },
        "difference_ci_95": {"lower": float(lower), "upper": float(upper)},
        "bootstrap_probability_favorable": probability_favorable,
        "bootstrap_two_sided_sign_tail": float(min(1.0, 2 * min(p_le_zero, p_ge_zero))),
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
        "per_seed": per_seed,
        "n_bootstrap_valid": int(len(values)),
        "n_bootstrap_attempts": int(attempts),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    started = time.time()
    pu = pd.read_csv(PU_PREDICTIONS, encoding="utf-8-sig")
    censor = pd.read_csv(CENSOR_PREDICTIONS, encoding="utf-8-sig")
    outputs = []
    for comparison in COMPARISONS:
        source = pu if comparison.source == "pu" else censor
        paired = prepare_paired_frame(source, comparison)
        result = paired_hierarchical_bootstrap(paired, comparison)
        outputs.append(result)
        point = result["point_estimate"]["difference_mean_across_seeds"]
        ci = result["difference_ci_95"]
        print(
            f"{comparison.comparison_id}: delta={point:.4f} "
            f"95% CI [{ci['lower']:.4f}, {ci['upper']:.4f}]"
        )

    payload = {
        "meta": {
            "script": "paired_hierarchical_bootstrap.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_bootstrap": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "seeds": SEEDS,
            "protocol": (
                "sample split seeds with replacement; within each sampled seed, sample test groups with replacement; "
                "apply the same sampled records to both methods; compute the metric difference inside the replicate; "
                "average seed-level differences within the replicate"
            ),
            "inference_scope": (
                "conditional on the frozen fitted predictions; group composition and the empirical three-seed distribution are resampled. "
                "This does not reproduce full model-training uncertainty."
            ),
            "prohibited_old_procedure": "no averaging of per-seed confidence-interval endpoints and no significance inference from marginal CI overlap",
        },
        "comparisons": outputs,
    }
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Paired hierarchical group bootstrap (v3)",
        "",
        f"> Generated: {payload['meta']['generated']}  ",
        f"> Replicates: {N_BOOT:,}; seeds: {', '.join(str(x) for x in SEEDS)}.",
        "",
        "## Protocol",
        "",
        "For every replicate, split seeds were sampled with replacement. Within each selected seed, test groups were sampled with replacement, and the identical sampled records were used for both methods. The method difference was calculated directly inside that replicate and then averaged across the sampled seed instances.",
        "",
        "These intervals are conditional on the frozen predictions. They represent uncertainty from test-group composition plus the empirical three-seed distribution; they do not reproduce full nested retraining uncertainty.",
        "",
        "The previous procedure of averaging lower and upper CI endpoints across seeds has been retired. Marginal CI overlap is not used as a significance test.",
        "",
        "## Primary paired differences",
        "",
        "| Comparison | Split | Metric | Method A | Method B | A mean | B mean | Delta A-B (95% CI) | Favorable direction |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for result in outputs:
        point = result["point_estimate"]
        ci = result["difference_ci_95"]
        lines.append(
            f"| `{result['comparison_id']}` | {result['split_regime']} | {result['metric']} | "
            f"{result['method_a']} | {result['method_b']} | "
            f"{fmt(point['method_a_mean_across_seeds'])} | {fmt(point['method_b_mean_across_seeds'])} | "
            f"{fmt(point['difference_mean_across_seeds'])} [{fmt(ci['lower'])}, {fmt(ci['upper'])}] | "
            f"{result['favorable_direction']} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "- For ROC-AUC, a positive Delta favors method A.",
        "- For violation rate and MAE, a negative Delta favors method A.",
        "- Whether the paired interval includes zero is reported descriptively. Claims should focus on effect size and uncertainty, not on overlap between separate marginal intervals.",
        "- Publication and cold-POI results may have few heterogeneous groups; their intervals remain conditional stress-test summaries rather than proof of deployment generalization.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
