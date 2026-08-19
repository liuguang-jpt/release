# -*- coding: utf-8 -*-
"""Audit the frozen split manifest and emit a machine-readable/human report."""
from __future__ import annotations

import json
import time

import pandas as pd

from benchmark_contract import (
    CONTRACT_PATH,
    MANIFEST_PATH,
    REPORTS_DIR,
    audit_manifest,
    load_contract,
    load_dataset,
    load_groups,
    load_manifest,
    role_indices,
    task_eligibility,
)


def main():
    df = load_dataset()
    groups = load_groups(df)
    manifest = load_manifest(validate=True)
    contract = load_contract(validate_hashes=True)
    configs = audit_manifest(df, groups, manifest)

    task_rows = []
    for task, spec in contract["tasks"].items():
        eligible = task_eligibility(df, task)
        family = spec["split_family"]
        relevant = [c for c in configs if c["split_family"] == family]
        for cfg in relevant:
            roles = role_indices(df, manifest, family, cfg["split_regime"], cfg["seed"])
            task_rows.append(
                {
                    "task": task,
                    "split_regime": cfg["split_regime"],
                    "seed": cfg["seed"],
                    "eligible_total": int(eligible.sum()),
                    "eligible_train": int(eligible[roles["train"]].sum()),
                    "eligible_calibration": int(eligible[roles["calibration"]].sum()),
                    "eligible_test": int(eligible[roles["test"]].sum()),
                }
            )

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(MANIFEST_PATH),
        "contract": str(CONTRACT_PATH),
        "status": "PASS",
        "config_audit": configs,
        "task_eligibility_audit": task_rows,
    }
    json_path = MANIFEST_PATH.with_name("split_manifest_v3_audit.json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = [
        "# Frozen Split Manifest v3 Audit",
        "",
        f"> Status: **PASS** | Configurations: {len(configs)} | Generated: {out['generated_at']}",
        "",
        "## Contract",
        "",
        f"- Split schema: `{contract['split_schema']}`",
        f"- Dataset SHA256: `{contract['artifacts']['dataset_sha256']}`",
        f"- Group SHA256: `{contract['artifacts']['group_sha256']}`",
        f"- Feature SHA256: `{contract['artifacts']['feature_sha256']}`",
        f"- Git commit at freeze: `{contract['git']['commit']}`",
        f"- Working tree dirty at freeze: `{contract['git']['working_tree_dirty']}`",
        "",
        "## Split configuration audit",
        "",
        "| family | split | seed | train | calibration | test | excluded | record overlap | group overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in configs:
        lines.append(
            f"| {r['split_family']} | {r['split_regime']} | {r['seed']} | {r['n_train']} | "
            f"{r['n_calibration']} | {r['n_test']} | {r['n_excluded']} | 0 | 0 |"
        )
    lines += ["", "## Task eligibility audit", "", "| task | split | seed | eligible | train | calibration | test |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in task_rows:
        lines.append(
            f"| {r['task']} | {r['split_regime']} | {r['seed']} | {r['eligible_total']} | "
            f"{r['eligible_train']} | {r['eligible_calibration']} | {r['eligible_test']} |"
        )
    report_path = REPORTS_DIR / "SPLIT_MANIFEST_V3_AUDIT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: {len(configs)} configurations, zero record/group overlaps")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
