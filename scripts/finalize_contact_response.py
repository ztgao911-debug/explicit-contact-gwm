"""Finalize saved completed computations without recollecting or retraining."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from contact_response_evaluation import summarize_response_rows
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def finalize_existing(output):
    output = output.resolve()
    config = json.loads((output / "resolved_config.json").read_text())
    reports = json.loads((output / "training_summaries.json").read_text())
    checks = json.loads((output / "checks.json").read_text())
    evaluation = json.loads((output / "ordinary_evaluation/evaluation.json").read_text())
    history = json.loads((output / "historical_hashes.json").read_text())
    assert len(reports) == 4 and all(r["updates"] == 2000 for r in reports)
    assert history == {name: file_sha256(ROOT / name) for name in history}
    rows = []
    with (output / "response_evaluation/per_case_response_metrics.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    k: v
                    if k in ("method", "variant", "region")
                    else int(v)
                    if k in ("frame", "horizon", "node_count")
                    else float(v)
                    if v
                    else None
                    for k, v in row.items()
                }
            )
    response_checks = json.loads((output / "response_evaluation/checks.json").read_text())
    failures = evaluation["checks"]["rollout_failures"] + json.loads(
        (output / "response_evaluation/failures.json").read_text()
    )
    env_path = output / "environment.json"
    if env_path.exists():
        environment = json.loads(env_path.read_text())
    else:
        environment = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "elapsed_seconds_estimate_from_file_times": (output / "training_summaries.json")
            .stat()
            .st_mtime
            - (output / "config.yaml").stat().st_mtime,
            "training_elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in reports),
            "output_bytes_before_summary": sum(
                p.stat().st_size for p in output.rglob("*") if p.is_file()
            ),
        }
        write_json(env_path, environment)
    previous_failure = json.loads((output / "failures.json").read_text())
    if previous_failure and not (output / "execution_failures.json").exists():
        write_json(output / "execution_failures.json", previous_failure)
    snapshots = [p for p in (output / "source_snapshot").rglob("*") if p.is_file()]
    mainline = yaml.safe_load((ROOT / config["mainline_config"]).read_text())
    response = {
        "summary": summarize_response_rows(rows),
        "failures": json.loads((output / "response_evaluation/failures.json").read_text()),
        "all_20_step_cases_complete": all(c["completed_steps"] == 20 for c in response_checks),
        "inverted_states": sum(c["inverted_states"] for c in response_checks),
    }
    write_json(
        output / "summary.json",
        {
            "environment": environment,
            "training": reports,
            "checks": checks,
            "derived_caps": json.loads((output / "central_derived_caps.json").read_text()),
            "central_bound_audit": json.loads(
                (output / "central_output_bound_audit.json").read_text()
            ),
            "ordinary_one_step": evaluation["one_step"],
            "ordinary_rollout": evaluation["rollout"],
            "response_evaluation": response,
            "executed_source_snapshot_hashes": {
                str(p.relative_to(output / "source_snapshot")): file_sha256(p) for p in snapshots
            },
            "postprocessing_source_hashes": hashes(
                [
                    ROOT / "scripts/finalize_contact_response.py",
                    ROOT / "scripts/contact_response_evaluation.py",
                ]
            ),
            "frozen_force_parameter_hashes": hashes(
                [
                    ROOT / mainline[k]
                    for k in ("point_calibration", "area_material", "integration_config")
                ]
            ),
            "scope": (
                "one_step_teacher_forcing_and_initial_pair_response_training_then_"
                "autoregressive_multistep_evaluation"
            ),
            "export_recovery_only_no_recollection_or_retraining": True,
        },
    )
    write_json(output / "failures.json", failures)
    write_json(
        output / "export_recovery.json",
        {
            "cause": "NumPy integer negative-direction count rejected by JSON serialization",
            "fix": "Convert direction count to Python int; recompute summaries from saved CSV",
            "recollection": False,
            "retraining": False,
            "model_or_prediction_changes": False,
            "original_exception_preserved": str(output / "execution_failures.json"),
        },
    )
    write_json(output / "run_status.json", {"complete": True, "export_recovered": True})
    print(
        json.dumps(
            {"output": str(output), "complete": True, "runtime_exceptions": failures}, indent=2
        )
    )
