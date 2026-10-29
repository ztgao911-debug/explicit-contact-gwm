"""Frozen 12-configuration conditional multi-step search with a six-hour deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from bounded_search_data import load_search_data, validation_windows
from bounded_search_evaluation import evaluate_trial
from bounded_search_training import (
    AlgorithmFailure,
    SearchDeadline,
    check_deadline,
    gradient_preflight,
    make_schedules,
    train_candidate,
)
from train_contact_response import model_for
from train_joint_state import load_data
from validate_contact_pipeline import hashes, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256
from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.joint_state_dynamics import JointStatePipeline
from contact_gwm.pinned_contact_pipeline import PinnedContactPipeline
from contact_gwm.repaired_contact_pipeline import RepairedContactConditions, RepairedJointPipeline
from contact_gwm.search_protocol import frozen_protocol, protocol_digest

ROOT = Path(__file__).resolve().parents[1]


def latest_results(results):
    """One latest attempt per configuration/seed; raw ledger remains immutable."""
    latest = {}
    for result in results:
        key = result["candidate_id"], result["seed"]
        previous = latest.get(key)
        if previous is None or result.get("attempt_number", 1) >= previous.get("attempt_number", 1):
            latest[key] = result
    return list(latest.values())


def read_ledger(report, output):
    path = report / "trial_ledger.json"
    results = json.loads(path.read_text()) if path.exists() else []
    known = {(r["candidate_id"], r["seed"], r.get("attempt_number", 1)) for r in results}
    # A process can finish an attempt just before the ledger write is interrupted.
    for file in sorted((output / "trials").glob("*/attempt_*/result.json")):
        result = json.loads(file.read_text())
        key = result["candidate_id"], result["seed"], result["attempt_number"]
        if key not in known:
            results.append(result)
            known.add(key)
    return results


def failure_status(error):
    if isinstance(error, SearchDeadline):
        return "timeout"
    if isinstance(
        error, (AlgorithmFailure, FloatingPointError, np.linalg.LinAlgError, mujoco.FatalError)
    ):
        return "algorithm_failure"
    state_failure_terms = (
        "non-finite",
        "nonfinite",
        "infinite",
        "infinity",
        "degenerate",
        "singular",
        "exterior sphere",
        "sphere center lies",
        "sphere center on",
        "area resultant",
        "linalg",
        "cholesky",
        "svd did not converge",
        "native intervention warning",
    )
    if isinstance(error, (ValueError, RuntimeError)) and any(
        term in str(error).lower() for term in state_failure_terms
    ):
        return "algorithm_failure"
    return "technical_failure"


def new_attempt_directory(candidate, seed, config, output, technical_fix_reason):
    parent = output / "trials" / f"{candidate['id']}_seed_{seed}"
    attempts = sorted(parent.glob("attempt_*")) if parent.exists() else []
    maximum = 1 + config["technical_fault_policy"]["maximum_retries_per_configuration_seed"]
    if maximum > 2:
        raise RuntimeError("Frozen policy permits at most one technical retry")
    if attempts:
        last = attempts[-1] / "result.json"
        if last.exists() and json.loads(last.read_text())["status"] != "technical_failure":
            raise RuntimeError(
                "Algorithm failures, invalid candidates and timeouts cannot be retried"
            )
        if not technical_fix_reason:
            raise RuntimeError("Technical retry requires --technical-fix-reason")
    number = max((int(p.name.split("_")[-1]) for p in attempts), default=0) + 1
    if number > maximum:
        raise RuntimeError(f"Technical retry budget exhausted for {candidate['id']}/{seed}")
    directory = parent / f"attempt_{number:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory, number


def _snapshot_sources(output, directory_name):
    for file in source_files():
        target = output / directory_name / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)


def _source_changes(expected, actual):
    return {
        name: {"old_sha256": expected.get(name), "new_sha256": actual.get(name)}
        for name in sorted(expected.keys() | actual.keys())
        if expected.get(name) != actual.get(name)
    }


def _record_frozen_source_change(config, report, output, protocol, technical_fix_reason):
    if protocol.get("configuration") != config:
        raise RuntimeError("Frozen configuration/math/data protocol cannot change on resume")
    for name, expected in protocol["data_hashes"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    versions_path = report / "technical_source_versions.json"
    versions = json.loads(versions_path.read_text()) if versions_path.exists() else []
    expected = versions[-1]["source_hashes"] if versions else protocol["source_hashes"]
    actual = hashes(source_files())
    changes = _source_changes(expected, actual)
    if not changes:
        return False
    ledger = read_ledger(report, output)
    formal_attempt_started = bool(ledger) or any(
        (output / "trials").glob("*/attempt_*/configuration.json")
    )
    if not formal_attempt_started:
        # Preparation probes have no optimizer updates. Preserve each technical
        # preparation version, then regenerate the SAME candidates and scoring.
        number = len(list(report.glob("protocol_preparation_v*.json"))) + 1
        write_json(report / f"protocol_preparation_v{number:02d}.json", protocol)
        archive = output / f"source_snapshot_preparation_v{number:02d}"
        if (output / "source_snapshot").exists():
            shutil.move(str(output / "source_snapshot"), str(archive))
        history_path = report / "preparation_changes.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(
            {
                "version": number,
                "changed_files": changes,
                "reason": technical_fix_reason
                or "Technical preparation before any optimizer updates",
                "formal_training_attempts": 0,
            }
        )
        write_json(history_path, history)
        return True
    if not technical_fix_reason:
        raise RuntimeError("Frozen source changed; explicit --technical-fix-reason is required")
    version = len(versions) + 1
    versions.append(
        {
            "version": version,
            "reason": technical_fix_reason,
            "changed_files": changes,
            "source_hashes": actual,
            "affected_technical_configuration_seeds": [
                {"candidate_id": r["candidate_id"], "seed": r["seed"]}
                for r in latest_results(ledger)
                if r["status"] == "technical_failure"
            ],
            "completed_configuration_seeds_at_change": [
                {"candidate_id": r["candidate_id"], "seed": r["seed"]}
                for r in latest_results(ledger)
                if r["status"].startswith("complete")
            ],
            "original_protocol_immutable": True,
            "review_scope": (
                "Technical correction only; candidate list, fixed data and scoring unchanged"
            ),
        }
    )
    write_json(versions_path, versions)
    _snapshot_sources(output, f"source_snapshot_technical_v{version:02d}")
    return False


def foundation(config):
    base = yaml.safe_load((ROOT / config["base_protocol_config"]).read_text())
    mainline = yaml.safe_load((ROOT / base["mainline_config"]).read_text())
    base["_variants"] = mainline["interventions"]["variants"]
    base["rollout"]["start_frames"] = config["evaluation"]["ordinary_starts"]
    base["rollout"]["horizons"] = config["evaluation"]["horizons"]
    mesh, trajectories, splits = load_data(ROOT / config["data_directory"], base)
    mesh["tetra_tensor"] = torch.tensor(mesh["tetrahedra"], dtype=torch.long)
    mesh["rest_volume_tensor"] = torch.tensor(mesh["rest_signed_tetra_volumes"]).double()
    legacy_config = yaml.safe_load((ROOT / mainline["legacy_pipeline_config"]).read_text())
    legacy = PinnedContactPipeline(
        mesh,
        ROOT / config["data_directory"] / "scene.xml",
        ROOT / legacy_config["checkpoint"],
        legacy_config,
    )
    point = json.loads((ROOT / mainline["point_calibration"]).read_text())
    settings = yaml.safe_load((ROOT / mainline["integration_config"]).read_text())["integration"]
    conditions = RepairedContactConditions(
        legacy,
        mesh,
        point["stiffness_N_m"],
        ElasticAreaMaterial.load(ROOT / mainline["area_material"]),
        settings,
    )
    reference_artifact = torch.load(
        ROOT / config["reference_checkpoint"], map_location="cpu", weights_only=False
    )
    reference = model_for(mesh, config["model"], {}, config["seed"])
    reference.load_state_dict(reference_artifact["model_state"], strict=True)
    reference.eval().requires_grad_(False)
    encoder = RepairedJointPipeline(legacy, reference, conditions, "mujoco_force").encoder
    return base, mesh, trajectories, splits, legacy, conditions, encoder, reference


def source_files():
    return sorted(
        [
            *ROOT.glob("src/contact_gwm/**/*.py"),
            *ROOT.glob("scripts/*.py"),
            ROOT / "configs/bounded_multistep_search.yaml",
        ]
    )


def prepare(config, report, output, objects, deadline, technical_fix_reason=None):
    base, mesh, trajectories, _, legacy, conditions, encoder, _ = objects
    data = load_search_data(ROOT, trajectories, mesh)
    windows = validation_windows(trajectories, mesh, ROOT)
    if (report / "protocol.json").exists():
        protocol = json.loads((report / "protocol.json").read_text())
        assert protocol["candidates"] == frozen_protocol(7)["candidates"]
        refreeze = _record_frozen_source_change(
            config, report, output, protocol, technical_fix_reason
        )
        if not refreeze:
            return data, windows, protocol
    if len(windows) != 30:
        raise RuntimeError("The frozen common five-step validation must contain 30 windows")
    # Reuse a successful, already executed probe when the preparation itself has
    # exhausted the total wall budget. This only permits evidence finalization;
    # the trial loop below still refuses to start any optimizer after deadline.
    gradient_path = report / "gradient_preflight.json"
    reuse_gradient = time.time() >= deadline and gradient_path.exists()
    if reuse_gradient:
        gradient_check = json.loads(gradient_path.read_text())
        if not (
            gradient_check.get("no_optimizer_updates")
            and gradient_check.get("terminal_to_first_x_gradient_norm", 0) > 0
            and gradient_check.get("terminal_to_first_v_gradient_norm", 0) > 0
            and gradient_check.get("state_features_have_grad_fn")
        ):
            raise RuntimeError("Expired budget cannot replace a missing successful gradient probe")
        write_json(
            report / "gradient_probe_reuse.json",
            {
                "reason": "Budget expired; reuse existing successful no-optimizer probe",
                "original_probe_sha256": file_sha256(gradient_path),
                "new_probe_executed": False,
            },
        )
    else:
        check_deadline(deadline)
    # Independent gradient check; no candidate is fitted and no optimizer runs.
    probe = model_for(mesh, config["model"], {}, 7)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(7)
        probe.internal_network[4].weight.copy_(
            torch.randn(probe.internal_network[4].weight.shape, generator=generator) * 1e-5
        )
        probe.contact_network[4].weight.copy_(
            torch.randn(probe.contact_network[4].weight.shape, generator=generator) * 1e-5
        )
    if not reuse_gradient:
        gradient_check = gradient_preflight(
            probe, legacy, conditions, encoder, data, mesh, deadline
        )
        write_json(gradient_path, gradient_check)
    write_json(report / "data_loader_checks.json", data["checks"])
    protocol = frozen_protocol(7)
    protocol["configuration"] = config
    protocol["reference_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    protocol["branch"] = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    protocol["source_hashes"] = hashes(source_files())
    protocol["data_hashes"] = json.loads((report / "preflight_native_data.json").read_text())[
        "file_sha256"
    ]
    protocol["validation_windows"] = [
        {
            "sourcecase": w["sourcecase"],
            "phase": w["phase"],
            "endpoints": 5,
            "roi_sha256": hashlib.sha256(w["roi"].numpy().tobytes()).hexdigest(),
        }
        for w in windows
    ]
    protocol["validation_definition"] = (
        "30 fixed M autoregressive windows; pool all five endpoints within initial phase/region; "
        "four phases, two regions and position/instantaneous velocity equal; no mesh term"
    )
    protocol["training_definition"] = (
        "ordinary one-step + central branch window endpoint mean + initial pair response + "
        "candidate mesh_weight times endpoint-mean dimensionless mesh penalty; "
        "weights of first three terms=1"
    )
    protocol["physical_gate_definition"] = (
        "all five ordinary trajectories including no_contact and approach windows, "
        "plus all 18 left branch windows, all 20 predicted endpoints; inversion<=0"
    )
    protocol["checkpoint_selection"]["interval_updates"] = config["training"]["validation_interval"]
    protocol["independent_test"] = "not_verified_right_arrays_never_loaded"
    write_json(report / "protocol.json", protocol)
    write_json(report / "protocol_digest.json", {"sha256": protocol_digest(protocol)})
    write_json(report / "candidates.json", protocol["candidates"])
    write_csv(report / "candidates.csv", protocol["candidates"])
    _snapshot_sources(output, "source_snapshot")
    write_json(
        output / "historical_hashes.json",
        hashes(
            [
                *(ROOT / config["data_directory"]).rglob("*"),
                *(ROOT / config["pilot_directory"]).rglob("*"),
                *(ROOT / base["validation_native_directory"]).rglob("*"),
            ]
        ),
    )
    write_json(
        report / "environment.json",
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "cpu_threads": config["cpu_threads"],
            "parameter_count": sum(p.numel() for p in probe.parameters()),
        },
    )
    print(
        f"Frozen {len(protocol['candidates'])} candidates, 30 validation windows, "
        f"protocol {protocol_digest(protocol)}",
        flush=True,
    )
    return data, windows, protocol


def guarded(predictor, deadline):
    def call(state, robot):
        check_deadline(deadline)
        return predictor(state, robot)

    return call


def evaluate_model(
    model, legacy, conditions, mesh, trajectories, splits, base, directory, deadline, routes=False
):
    methods = ("M", "F", "G") if routes else ("M",)
    mapping = {"M": "mujoco_force", "F": "analytic_force", "G": "gaussian_geometry"}
    predictors = {
        name: guarded(
            RepairedJointPipeline(legacy, model, conditions, mapping[name]).step, deadline
        )
        for name in methods
    }
    if routes:
        predictors["I"] = guarded(
            JointStatePipeline(legacy, model, contact_method="none").step, deadline
        )
        predictors["CV"] = guarded(
            lambda state, robot: legacy.step(state, robot, "constant_velocity"), deadline
        )
    return evaluate_trial(predictors, legacy, trajectories, splits, mesh, base, directory)


def publish_metrics(directory, report_directory):
    report_directory.mkdir(parents=True, exist_ok=True)
    allowed = {
        "training_curve.csv",
        "training_updates.csv",
        "checkpoint_selection.json",
        "training_summary.json",
        "ordinary/one_step_metrics.csv",
        "ordinary/one_step_per_trajectory.csv",
        "ordinary/rollout_metrics.csv",
        "ordinary/rollout_per_trajectory.csv",
        "ordinary/rollout_failures.json",
        "ordinary_physics.json",
        "response_physics.json",
        "ordinary_physics_windows.csv",
        "response_physics_windows.csv",
        "static_stability.csv",
        "response_evaluation/summary.csv",
        "response_evaluation/per_case_response_metrics.csv",
        "response_evaluation/failures.json",
        "response_evaluation/checks.json",
    }
    for relative in allowed:
        source = directory / relative
        if source.exists():
            target = report_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for file in (directory / "recovery").glob("*.csv") if (directory / "recovery").exists() else []:
        target = report_directory / "recovery" / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)


def run_trial(
    candidate,
    seed,
    role,
    config,
    protocol,
    data,
    windows,
    objects,
    report,
    output,
    deadline,
    technical_fix_reason=None,
):
    base, mesh, trajectories, splits, legacy, conditions, encoder, _ = objects
    key = f"{candidate['id']}_seed_{seed}"
    directory, attempt_number = new_attempt_directory(
        candidate, seed, config, output, technical_fix_reason
    )
    published = report / "trials" / key / f"attempt_{attempt_number:02d}"
    published.mkdir(parents=True, exist_ok=True)
    result = {
        "candidate_id": candidate["id"],
        "seed": seed,
        "role": role,
        "candidate": candidate,
        "status": "running",
        "eligible": False,
        "protocol_sha256": protocol_digest(protocol),
        "attempt_number": attempt_number,
        "artifact_directory": str(directory.relative_to(ROOT)),
        "published_directory": str(published.relative_to(ROOT)),
        "actual_source_hashes": hashes(source_files()),
        "technical_fix_reason": technical_fix_reason if attempt_number > 1 else None,
        "attempt_started_epoch": time.time(),
    }
    write_json(directory / "configuration.json", {**result, "model": config["model"]})
    start = time.perf_counter()
    try:
        check_deadline(deadline)
        model = model_for(mesh, config["model"], {}, seed)
        initial_hash = tensor_digest(model.state_dict())
        schedules = make_schedules(
            data, seed, config["training"]["updates"], config["training"]["batch_size"]
        )
        schedule_path = output / f"batch_schedules_seed_{seed}.npz"
        if not schedule_path.exists():
            np.savez_compressed(schedule_path, **{k: v.numpy() for k, v in schedules.items()})
        settings = {**config["training"], "seed": seed}
        trained = train_candidate(
            model,
            legacy,
            conditions,
            encoder,
            data,
            windows,
            mesh,
            candidate,
            settings,
            schedules,
            directory,
            deadline,
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "architecture": config["model"],
            "candidate": candidate,
            "training_summary": trained,
            "initial_weight_sha256": initial_hash,
            "protocol_sha256": protocol_digest(protocol),
        }
        torch.save(checkpoint, directory / "checkpoint.pt")
        evaluation = evaluate_model(
            model,
            legacy,
            conditions,
            mesh,
            trajectories,
            splits,
            base,
            directory / "evaluation",
            deadline,
        )
        outcome = evaluation["per_method"]["M"]
        result.update(
            status="complete_valid" if outcome["eligible"] else "complete_invalid",
            eligible=outcome["eligible"],
            training=trained,
            metrics=outcome,
            initial_weight_sha256=initial_hash,
            checkpoint_sha256=file_sha256(directory / "checkpoint.pt"),
        )
        publish_metrics(directory / "evaluation", published / "evaluation")
    except BaseException as error:
        result.update(
            status=failure_status(error),
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = time.perf_counter() - start
    result["attempt_completed_epoch"] = time.time()
    previous_results = [
        json.loads(p.read_text()) for p in directory.parent.glob("attempt_*/result.json")
    ]
    result["all_attempt_elapsed_seconds"] = result["elapsed_seconds"] + sum(
        r["elapsed_seconds"] for r in previous_results
    )
    write_json(directory / "result.json", result)
    write_json(published / "result.json", result)
    publish_metrics(directory, published)
    print(
        f"Finished {key}: {result['status']}, elapsed {result['elapsed_seconds']:.1f}s", flush=True
    )
    return result


def rankings(results, report):
    results = latest_results(results)
    rows = []
    for result in results:
        if result["role"] != "search":
            continue
        metrics = result.get("metrics", {})
        scores = metrics.get("scores", {})
        rows.append(
            {
                "candidate_id": result["candidate_id"],
                "status": result["status"],
                "eligible": result["eligible"],
                **result["candidate"],
                "joint_score": scores.get("joint_score"),
                "position_score": scores.get("position_score"),
                "velocity_score": scores.get("velocity_score"),
                "inverted_states": metrics.get("inverted_states"),
                "inverted_windows": metrics.get("inverted_windows"),
                "minimum_volume_ratio": metrics.get("minimum_volume_ratio"),
                "selected_step": result.get("training", {}).get("selected_step"),
                "elapsed_seconds": result["elapsed_seconds"],
                "all_attempt_elapsed_seconds": result.get(
                    "all_attempt_elapsed_seconds", result["elapsed_seconds"]
                ),
                "attempt_number": result.get("attempt_number", 1),
                "artifact_directory": result.get("artifact_directory"),
            }
        )
    for field, rank in (
        ("joint_score", "joint_rank"),
        ("position_score", "position_rank"),
        ("velocity_score", "velocity_rank"),
    ):
        valid = sorted(
            [row for row in rows if row["eligible"] and row[field] is not None],
            key=lambda row: (row[field], row["candidate_id"]),
        )
        for row in rows:
            row[rank] = next((i + 1 for i, value in enumerate(valid) if value is row), None)
    if rows:
        write_csv(report / "candidate_rankings.csv", rows)
    else:
        (report / "candidate_rankings.csv").write_text(
            "candidate_id,status,eligible,joint_score,position_score,velocity_score,"
            "joint_rank,position_rank,velocity_rank\n"
        )
    return sorted(
        [r for r in results if r["role"] == "search" and r["eligible"]],
        key=lambda r: (r["metrics"]["scores"]["joint_score"], r["candidate_id"]),
    )


def seed_summary(results, selected, report):
    results = latest_results(results)
    rows = []
    for candidate_id in selected:
        group = [r for r in results if r["candidate_id"] == candidate_id]
        complete = (
            len(group) == 3
            and {r["seed"] for r in group} == {7, 17, 27}
            and all("metrics" in r for r in group)
        )
        for metric in ("joint_score", "position_score", "velocity_score"):
            values = [
                r["metrics"]["scores"][metric]
                for r in group
                if r.get("metrics", {}).get("scores", {}).get(metric) is not None
            ]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "seeds_expected": "7,17,27",
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "all_three_completed": complete,
                    "all_three_physically_valid": complete and all(r["eligible"] for r in group),
                }
            )
    if rows:
        write_csv(report / "multiseed_summary.csv", rows)
    else:
        (report / "multiseed_summary.csv").write_text(
            "candidate_id,metric,seeds_expected,n,mean,sample_std,all_three_completed,"
            "all_three_physically_valid\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bounded_multistep_search.yaml"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--technical-fix-reason")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    budget = json.loads((report / "budget.json").read_text())
    deadline = budget["deadline_epoch"]
    objects = foundation(config)
    data, windows, protocol = prepare(
        config, report, output, objects, deadline, args.technical_fix_reason
    )
    if args.prepare_only:
        return
    results = read_ledger(report, output)
    write_json(report / "trial_ledger.json", results)
    if config["maximum_configurations"] > 12 or config["maximum_replication_candidates"] > 2:
        raise RuntimeError("Search/replication configuration budget exceeds the frozen user limit")
    if config["replication_seeds"] != [17, 27]:
        raise RuntimeError("Replication seeds must remain 17 and 27")
    timed_out = False
    for candidate in protocol["candidates"]:
        previous = next(
            (
                r
                for r in latest_results(results)
                if r["candidate_id"] == candidate["id"] and r["seed"] == 7
            ),
            None,
        )
        if previous and previous["status"] != "technical_failure":
            continue
        if previous and not args.technical_fix_reason:
            raise RuntimeError(
                "Technical retry requires --technical-fix-reason; no new configuration drawn"
            )
        distinct_search = {r["candidate_id"] for r in results if r["role"] == "search"}
        if (
            candidate["id"] not in distinct_search
            and len(distinct_search) >= config["maximum_configurations"]
        ):
            break
        if time.time() >= deadline:
            timed_out = True
            break
        result = run_trial(
            candidate,
            7,
            "search",
            config,
            protocol,
            data,
            windows,
            objects,
            report,
            output,
            deadline,
            args.technical_fix_reason,
        )
        results.append(result)
        write_json(report / "trial_ledger.json", results)
        rankings(results, report)
        if result["status"] == "technical_failure":
            write_json(
                report / "status.json",
                {"status": "technical_failure_requires_recorded_fix", "trial": candidate["id"]},
            )
            raise RuntimeError(f"Technical failure in {candidate['id']}: {result['error']}")
        if result["status"] == "timeout":
            timed_out = True
            break
    valid = rankings(results, report)
    selected = [r["candidate_id"] for r in valid[: config["maximum_replication_candidates"]]]
    write_json(
        report / "selected_for_replication.json",
        {
            "candidate_ids": selected,
            "rule": "top_two_physically_valid_seed7_joint_score",
            "selected_before_replication": True,
        },
    )
    for candidate_id in selected:
        candidate = next(c for c in protocol["candidates"] if c["id"] == candidate_id)
        for seed in config["replication_seeds"]:
            previous = next(
                (
                    r
                    for r in latest_results(results)
                    if r["candidate_id"] == candidate_id and r["seed"] == seed
                ),
                None,
            )
            if previous and previous["status"] != "technical_failure":
                continue
            if previous and not args.technical_fix_reason:
                raise RuntimeError("Technical replication retry requires --technical-fix-reason")
            distinct_replications = {
                (r["candidate_id"], r["seed"]) for r in results if r["role"] == "replication"
            }
            if (candidate_id, seed) not in distinct_replications and len(
                distinct_replications
            ) >= 4:
                raise RuntimeError("Four distinct replication trainings have already been started")
            if time.time() >= deadline:
                timed_out = True
                break
            result = run_trial(
                candidate,
                seed,
                "replication",
                config,
                protocol,
                data,
                windows,
                objects,
                report,
                output,
                deadline,
                args.technical_fix_reason,
            )
            results.append(result)
            write_json(report / "trial_ledger.json", results)
            seed_summary(results, selected, report)
            if result["status"] == "technical_failure":
                raise RuntimeError(
                    f"Technical failure replicating {candidate_id}/{seed}: {result['error']}"
                )
            if result["status"] == "timeout":
                timed_out = True
                break
    seed_summary(results, selected, report)
    base, mesh, trajectories, splits, legacy, conditions, _, reference = objects
    final_results = []
    for result in latest_results(results):
        if result["candidate_id"] not in selected or "training" not in result:
            continue
        if time.time() >= deadline:
            timed_out = True
            break
        route_key = (
            f"{result['candidate_id']}_seed_{result['seed']}_"
            f"attempt_{result.get('attempt_number', 1):02d}"
        )
        directory = output / "final_routes" / route_key
        if (directory / "evaluation_summary.json").exists():
            evaluation = json.loads((directory / "evaluation_summary.json").read_text())
        else:
            try:
                saved = torch.load(
                    ROOT / result["artifact_directory"] / "checkpoint.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model = model_for(mesh, config["model"], {}, result["seed"])
                model.load_state_dict(saved["model_state"], strict=True)
                model.eval().requires_grad_(False)
                evaluation = evaluate_model(
                    model,
                    legacy,
                    conditions,
                    mesh,
                    trajectories,
                    splits,
                    base,
                    directory,
                    deadline,
                    routes=True,
                )
            except Exception as error:
                failed = {
                    "candidate_id": result["candidate_id"],
                    "seed": result["seed"],
                    "status": failure_status(error),
                    "stage": "final_routes",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "artifact_directory": str(directory.relative_to(ROOT)),
                }
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "failure.json", failed)
                published = report / "final_routes" / directory.name
                published.mkdir(parents=True, exist_ok=True)
                write_json(published / "failure.json", failed)
                publish_metrics(directory, published)
                final_results.append(failed)
                write_json(report / "final_routes_summary.json", final_results)
                if isinstance(error, SearchDeadline):
                    timed_out = True
                    break
                continue
        final_results.append(
            {
                "candidate_id": result["candidate_id"],
                "seed": result["seed"],
                "status": "complete",
                "per_method": evaluation["per_method"],
            }
        )
        publish_metrics(directory, report / "final_routes" / directory.name)
        write_json(report / "final_routes_summary.json", final_results)
    # Fixed-reference baselines do not consume a configuration or a training.
    if time.time() < deadline:
        directory = output / "baselines"
        if not (directory / "evaluation_summary.json").exists():
            predictors = {
                "E0": guarded(
                    RepairedJointPipeline(legacy, reference, conditions, "mujoco_force").step,
                    deadline,
                ),
                "CV": guarded(
                    lambda state, robot: legacy.step(state, robot, "constant_velocity"), deadline
                ),
            }
            try:
                evaluate_trial(predictors, legacy, trajectories, splits, mesh, base, directory)
            except Exception as error:
                failed = {
                    "status": failure_status(error),
                    "stage": "baselines",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "failure.json", failed)
                write_json(report / "baseline_failure.json", failed)
                publish_metrics(directory, report / "baselines")
                if isinstance(error, SearchDeadline):
                    timed_out = True
        if (directory / "evaluation_summary.json").exists():
            evaluation = json.loads((directory / "evaluation_summary.json").read_text())
            write_json(report / "baseline_summary.json", evaluation["per_method"])
            publish_metrics(directory, report / "baselines")
    before = json.loads((output / "historical_hashes.json").read_text())
    unchanged = before == {name: file_sha256(ROOT / name) for name in before}
    write_json(
        report / "preservation_checks.json",
        {
            "historical_files_sha256_unchanged": len(before) if unchanged else False,
            "right_arrays_loaded": False,
            "independent_test_verified": False,
        },
    )
    write_json(
        report / "status.json",
        {
            "status": "budget_stop" if timed_out else "completed",
            "search_configurations_started": len(
                {r["candidate_id"] for r in results if r["role"] == "search"}
            ),
            "replication_trainings_started": len(
                {(r["candidate_id"], r["seed"]) for r in results if r["role"] == "replication"}
            ),
            "training_attempts_including_technical_retries": len(results),
            "technical_retry_attempts": sum(r.get("attempt_number", 1) > 1 for r in results),
            "technical_failures_recorded": sum(r["status"] == "technical_failure" for r in results),
            "selected_candidate_ids": selected,
            "final_route_evaluations_completed": sum(
                r.get("status") == "complete" for r in final_results
            ),
            "final_route_evaluation_failures": [
                r for r in final_results if r.get("status") != "complete"
            ],
            "wall_elapsed_seconds": time.time() - budget["wall_start_epoch"],
            "completed_epoch": time.time(),
            "independent_test_verified": False,
        },
    )
    print(json.dumps(json.loads((report / "status.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
