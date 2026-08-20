#!/usr/bin/env python
"""Sensitivity of nnPU metrics to 0.8x, 1.0x and 1.2x SCAR priors."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TMP", str(args.output.parent / "tmp"))
    os.environ.setdefault("TEMP", os.environ["TMP"])
    Path(os.environ["TMP"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code_root))
    os.environ["PROTAC_ROOT"] = str(args.data_root)
    from benchmark_contract import SEEDS, SPLITS, assert_group_disjoint, group_values, load_dataset, load_groups, load_manifest, role_indices
    from pu_pipeline import classification_metrics, estimate_scar_prior, nnpu_fit_predict

    frame = load_dataset()
    groups = load_groups(frame)
    manifest = load_manifest(validate=True)
    features = np.load(args.feature_path).astype(np.float32)
    if len(frame) != len(features):
        raise AssertionError("feature rows are not aligned with the frozen table")
    labels = frame["activity_evidence_v2"].to_numpy()
    p_mask, n_mask = labels == "P", labels == "N"
    result = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "protocol": "Frozen all_records_2way roles; per-seed in-sample SCAR q/c prior multiplied by 0.8, 1.0 or 1.2 and clipped to [0.001, 0.9]. Morgan features, architecture, epochs and P/N tests are unchanged.",
            "multipliers": [0.8, 1.0, 1.2],
            "seeds": SEEDS,
        },
        "splits": {},
    }
    for split in SPLITS:
        aggregate: dict[str, list[float]] = defaultdict(list)
        seed_rows = []
        for seed in SEEDS:
            roles = role_indices(frame, manifest, "all_records_2way", split, seed)
            assert_group_disjoint(frame, groups, roles, split)
            train, test = roles["train"], roles["test"]
            p_train, n_train = train[p_mask[train]], train[n_mask[train]]
            p_test, n_test = test[p_mask[test]], test[n_mask[test]]
            pool = train[~p_mask[train]]
            if min(len(p_train), len(n_train), len(p_test), len(n_test)) == 0:
                raise AssertionError(f"empty P/N role for {split}/{seed}")
            x_train = np.vstack([features[p_train], features[pool]])
            observed = np.concatenate([np.ones(len(p_train)), np.zeros(len(pool))]).astype(int)
            base_prior, prior_info = estimate_scar_prior(x_train, observed)
            test_idx = np.concatenate([p_test, n_test])
            y_test = np.concatenate([np.ones(len(p_test)), np.zeros(len(n_test))]).astype(int)
            row = {"seed": seed, "base_prior": base_prior, "prior_info": prior_info, "metrics": {}}
            for multiplier in (0.8, 1.0, 1.2):
                prior = float(np.clip(base_prior * multiplier, 0.001, 0.9))
                predictions = nnpu_fit_predict(x_train, observed, features[test_idx], seed=seed, prior=prior, epochs=40)
                metrics = classification_metrics(y_test, predictions)
                key = f"pi_x{multiplier:.1f}"
                row["metrics"][key] = {"prior": prior, **metrics}
                for metric, value in metrics.items():
                    aggregate[f"{key}::{metric}"].append(float(value))
            seed_rows.append(row)
            print(f"{split}/{seed}: complete")
        result["splits"][split] = {
            "summary": {key: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=0))} for key, values in aggregate.items()},
            "per_seed": seed_rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
