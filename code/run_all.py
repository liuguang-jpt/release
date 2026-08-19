# -*- coding: utf-8 -*-
"""Reproduce the frozen v3 benchmark workflow without touching raw data.

Default behavior
----------------
1. Rebuild processed data and labels (unless --skip-etl).
2. Validate/rebuild Morgan features and their record_id sidecar.
3. Freeze and audit the split manifest and temporal matched controls.
4. Run all v3 experiments, including the paired hierarchical bootstrap.
5. Write a machine-readable run receipt for the scientific workflow.

Human annotation templates are never regenerated unless the explicit
--prepare-human-annotation flag is supplied.
"""
from __future__ import annotations

import argparse
import os
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
ROOT_DIR = Path(os.environ.get("PROTAC_ROOT", str(PROJECT_DIR)))
PROCESSED_DIR = ROOT_DIR / "data" / "derived"
RECEIPT_PATH = PROCESSED_DIR / "run_all_v3_receipt.json"
GOLD_PATH = PROCESSED_DIR / "gold_set_annotations" / "gold_final.csv"

ETL_STEPS = [
    ("etl_protac.py", "Build the record-level processed table"),
    ("build_split_groups.py", "Build scaffold/publication/POI grouping keys"),
    ("relabel_v2.py", "Apply the frozen v2 activity-evidence rules"),
    ("relabel_semantics.py", "Assign U/N semantic subtypes and evidence tiers"),
]
CONTRACT_STEPS = [
    ("build_morgan_features.py", "Validate or rebuild Morgan features plus record_id sidecar"),
    ("make_split_manifest.py", "Freeze the benchmark split manifest and contract"),
    ("audit_split_manifest.py", "Audit record and group separation"),
    ("make_temporal_matched_controls.py", "Freeze matched temporal-control assignments"),
]
EXPERIMENT_STEPS = [
    ("baseline_pipeline.py", "Run supervised baseline tasks"),
    ("pu_pipeline.py", "Run PU comparisons"),
    ("calib_sensitivity_pipeline.py", "Run calibration and prior-sensitivity analyses"),
    ("censored_eval_pipeline.py", "Run same-backbone censoring comparisons"),
    ("shortcut_controls.py", "Run manifest-driven shortcut controls"),
    ("gold_validation.py", "Run internal-consistency rule and frozen-model checks"),
    ("external_validation.py", "Run the internal post-cutoff temporal stress test"),
    ("paired_hierarchical_bootstrap.py", "Run paired hierarchical group bootstrap"),
]
HUMAN_TEMPLATE_STEPS = [
    ("sample_annotation_150.py", "Prepare the rule-audit annotation template"),
    ("sample_gold_set.py", "Prepare the internal-consistency annotation template"),
]


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "pandas", "sklearn", "scipy", "xgboost", "torch", "rdkit"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def run_step(script: str, description: str, dry_run: bool) -> dict:
    path = CODE_DIR / script
    if not path.exists():
        raise FileNotFoundError(path)
    command = [sys.executable, str(path)]
    print(f"\n=== {script}: {description} ===", flush=True)
    if dry_run:
        print("DRY RUN: " + " ".join(command), flush=True)
        return {"script": script, "description": description, "status": "dry-run", "seconds": 0.0}
    started = time.time()
    completed = subprocess.run(command, cwd=CODE_DIR, check=False)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {completed.returncode}")
    print(f"completed {script} in {elapsed:.1f}s", flush=True)
    return {"script": script, "description": description, "status": "completed", "seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-etl", action="store_true", help="reuse the current processed table, groups, and labels")
    parser.add_argument(
        "--prepare-human-annotation",
        action="store_true",
        help="explicitly regenerate human-annotation templates; off by default to protect completed annotations",
    )
    parser.add_argument("--skip-shortcut-controls", action="store_true", help="skip shortcut-control model fitting")
    parser.add_argument("--skip-slow-bootstrap", action="store_true", help="skip the approximately 30-minute paired bootstrap")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved workflow without executing scripts")
    args = parser.parse_args()

    steps: list[tuple[str, str]] = []
    if not args.skip_etl:
        steps.extend(ETL_STEPS)
    if args.prepare_human_annotation:
        steps.extend(HUMAN_TEMPLATE_STEPS)
    steps.extend(CONTRACT_STEPS)
    for script, description in EXPERIMENT_STEPS:
        if script == "shortcut_controls.py" and args.skip_shortcut_controls:
            continue
        if script == "paired_hierarchical_bootstrap.py" and args.skip_slow_bootstrap:
            continue
        if script == "gold_validation.py" and not GOLD_PATH.exists():
            print(f"skip {script}: frozen gold file not found at {GOLD_PATH}")
            continue
        steps.append((script, description))

    receipt = {
        "schema": "protac-run-all-v3.0.0",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "arguments": vars(args),
        "steps": [],
        "status": "running",
    }
    started = time.time()
    try:
        for script, description in steps:
            receipt["steps"].append(run_step(script, description, args.dry_run))
        receipt["status"] = "dry-run" if args.dry_run else "completed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = str(exc)
        raise
    finally:
        receipt["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        receipt["total_seconds"] = time.time() - started
        if not args.dry_run:
            RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nrun receipt: {RECEIPT_PATH}", flush=True)

    print(f"\nworkflow {receipt['status']} in {receipt['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()



