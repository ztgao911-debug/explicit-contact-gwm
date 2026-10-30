"""Replicate the frozen fair RAW/G pilot at seeds 17 and 27, four trainings only."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from bounded_search_data import _tensor_cache, load_search_data, validation_windows
from bounded_search_evaluation import evaluate_trial
from bounded_search_training import check_deadline, make_schedules, train_candidate
from run_bounded_search import failure_status, foundation, guarded, publish_metrics, source_files
from run_fair_mainline import native_forbidden, pipeline_for, reload_check
from train_contact_response import model_for
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256
from contact_gwm.fair_conditioning import training_path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 27)
VARIANTS = ("RAW", "G")


def verify_configuration(config, parent):
    """Replication may vary seeds/paths, never mathematical settings or masks."""
    for key in ("model", "training", "validation", "evaluation", "comparison"):
        if config[key] != parent["configuration"][key]:
            raise ValueError(f"Frozen scientific configuration changed: {key}")
    if config["replication"]["new_seeds"] != list(SEEDS):
        raise ValueError("Only seeds 17 and 27 are authorized")
    if config["replication"]["maximum_new_trainings"] != 4:
        raise ValueError("Four new RAW/G trainings maximum")


def gradient_check(model, variant, objects, data, seed):
    _, _, _, _, legacy, conditions, encoder, _ = objects
    probe = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for network in (probe.internal_network, probe.contact_network):
            network[4].weight.copy_(
                torch.randn(network[4].weight.shape, generator=generator) * 1e-5
            )
    path = training_path(variant, legacy, probe, conditions, encoder)
    x = data["branch_positions"][6, 0][None].clone().requires_grad_(True)
    v = data["branch_velocities"][6, 0][None].clone().requires_grad_(True)
    states = path.rollout(x, v, [[data["branch_robots"][6][t]] for t in range(3)])
    loss = ((states[-1][0] - data["branch_positions"][6, 3][None]) / 0.001).square().mean()
    loss += ((states[-1][1] - data["branch_velocities"][6, 3][None]) / 0.1).square().mean()
    gx, gv = torch.autograd.grad(loss, states[1])
    assert torch.isfinite(gx).all() and torch.isfinite(gv).all()
    assert gx.norm() > 0 and gv.norm() > 0
    return {
        "position_gradient_norm": float(gx.norm()),
        "velocity_gradient_norm": float(gv.norm()),
        "probe_only_optimizer_updates": 0,
        "seed": seed,
    }


def cached_data(source, parent_output, parent_manifest, variant, mesh):
    caches = {}
    for name, count in (("ordinary_train", 600), ("branch_cache", 2160)):
        path = parent_output / variant / f"{name}.npz"
        if file_sha256(path) != parent_manifest[str(path.relative_to(ROOT))]["sha256"]:
            raise ValueError(f"Original deterministic current-input cache changed: {path}")
        cache = _tensor_cache(path, mesh, count)
        for key in ("features", "velocity", "target_dx", "target_dv", "neighborhood", "pinned"):
            if not torch.equal(cache[key], source[name][key]):
                raise ValueError(f"Replica state/label/mask differs: {variant}/{name}/{key}")
        caches[name] = cache
    return {**source, **caches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/fair_mainline_multiseed.yaml"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    if (report / "protocol.json").exists():
        raise RuntimeError("Immutable replication already prepared; preserve prior attempts")
    budget = json.loads((report / "budget.json").read_text())
    deadline = budget["deadline_epoch"]
    check_deadline(deadline)
    torch.set_num_threads(1)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    parent_report, parent_output = (
        ROOT / config["replication"]["parent_report"],
        ROOT / config["replication"]["parent_output"],
    )
    parent = json.loads((parent_report / "protocol.json").read_text())
    verify_configuration(config, parent)
    for name, expected in {**parent["source_hashes"], **parent["data_hashes"]}.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Original frozen scientific source/data changed: {name}")
    before = hashes([p for p in parent_report.rglob("*") if p.is_file()])
    old_artifacts = json.loads((parent_report / "large_artifact_manifest.json").read_text())[
        "ignored_large_artifacts"
    ]
    for name, record in old_artifacts.items():
        if file_sha256(ROOT / name) != record["sha256"]:
            raise ValueError(f"Original model/cache/prediction changed: {name}")
    objects = foundation(config)
    base, mesh, trajectories, splits, legacy, conditions, encoder, _ = objects
    source, windows = (
        load_search_data(ROOT, trajectories, mesh),
        validation_windows(trajectories, mesh, ROOT),
    )
    data = {v: cached_data(source, parent_output, old_artifacts, v, mesh) for v in VARIANTS}
    models = {s: {v: model_for(mesh, config["model"], {}, s) for v in VARIANTS} for s in SEEDS}
    initial = {
        s: {v: tensor_digest(m.state_dict()) for v, m in group.items()}
        for s, group in models.items()
    }
    counts = {v: sum(p.numel() for p in models[17][v].parameters()) for v in VARIANTS}
    assert counts == parent["parameter_counts"]
    assert all(len(set(d.values())) == 1 for d in initial.values())
    assert (
        len({initial[17]["RAW"], initial[27]["RAW"], parent["initial_weight_sha256"]["RAW"]}) == 3
    )
    schedules = {s: make_schedules(source, s, 200, 16) for s in SEEDS}
    for seed, schedule in schedules.items():
        np.savez_compressed(
            output / f"shared_schedule_seed_{seed}.npz",
            **{k: v.numpy() for k, v in schedule.items()},
        )
    with patch.object(legacy.native, "query", side_effect=native_forbidden) as native_guard:
        probes = {
            s: {v: gradient_check(models[s][v], v, objects, source, s) for v in VARIANTS}
            for s in SEEDS
        }
        write_json(report / "gradient_preflight.json", probes)
        protocol = {
            "experiment_kind": "frozen_fair_mainline_seed_replication_no_search",
            "configuration": config,
            "new_seeds": list(SEEDS),
            "all_reported_seeds": [7, 17, 27],
            "maximum_new_trainings": 4,
            "original_seed7_retrained": False,
            "parent_protocol_sha256": file_sha256(parent_report / "protocol.json"),
            "scientific_configuration_verified_identical": True,
            "data_hashes": parent["data_hashes"],
            "source_hashes": hashes([*source_files(), args.config.resolve()]),
            "parameter_counts": counts,
            "initial_weight_sha256": initial,
            "reuse_caches": {
                str(p.relative_to(ROOT)): file_sha256(p)
                for v in VARIANTS
                for p in (parent_output / v).glob("*.npz")
            },
            "seed7_report_hashes_before": before,
            "seed7_large_artifact_hashes": old_artifacts,
            "shared_schedule_hashes": {
                str(s): file_sha256(output / f"shared_schedule_seed_{s}.npz") for s in SEEDS
            },
            "checkpoint_selection": parent["checkpoint_selection"],
            "ranking": parent["ranking"],
            "validation_windows": parent["validation_windows"],
            "RAW_condition": parent["RAW_condition"],
            "G_condition": parent["G_condition"],
            "condition_gradient": (
                "RAW current kinematics attached; G current geometry/force stop-gradient"
            ),
            "native_forward_queries_allowed": False,
            "statistics": "Report every seed, mean and sample std ddof=1; no failed seed deletion",
            "independent_test_verified": False,
            "reference_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        write_json(report / "protocol.json", protocol)
        for file in [*source_files(), args.config.resolve()]:
            target = output / "source_snapshot" / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        print(f"Frozen four replications; parameters={counts}; seeds={SEEDS}", flush=True)
        ledger, evaluations = [], {}
        for seed in SEEDS:
            seed_output, seed_report = output / f"seed_{seed}", report / f"seed_{seed}"
            seed_output.mkdir()
            seed_report.mkdir()
            predictors = {}
            for variant in VARIANTS:
                check_deadline(deadline)
                started = time.perf_counter()
                directory = seed_output / variant
                directory.mkdir()
                result = {
                    "seed": seed,
                    "variant": variant,
                    "status": "running",
                    "initial_weight_sha256": initial[seed][variant],
                    "started_epoch": time.time(),
                    "artifact_directory": str(directory.relative_to(ROOT)),
                }
                write_json(directory / "attempt_configuration.json", result)
                try:
                    model = models[seed][variant]
                    candidate = {
                        "id": variant,
                        "mode": "autoregressive",
                        **{
                            k: config["comparison"][k]
                            for k in ("horizon", "learning_rate", "velocity_weight", "mesh_weight")
                        },
                    }
                    trained = train_candidate(
                        model,
                        legacy,
                        conditions,
                        encoder,
                        data[variant],
                        windows,
                        mesh,
                        candidate,
                        {**config["training"], "seed": seed},
                        schedules[seed],
                        directory,
                        deadline,
                        path=training_path(variant, legacy, model, conditions, encoder),
                    )
                    trained["conditional_gradient"] = (
                        "RAW current kinematics attached; no contact query"
                        if variant == "RAW"
                        else "G geometry/analytic force stop-gradient; state/features attached"
                    )
                    trained["conditioner_calls_training_and_selection"] = trained[
                        "native_queries_training_and_selection"
                    ]
                    trained["actual_native_queries"] = 0
                    write_json(directory / "training_summary.json", trained)
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "architecture": config["model"],
                            "variant": variant,
                            "seed": seed,
                            "training_summary": trained,
                        },
                        directory / "checkpoint.pt",
                    )
                    loaded, reload = reload_check(
                        variant, model, directory / "checkpoint.pt", objects, config["model"]
                    )
                    predictors[variant] = guarded(
                        pipeline_for(variant, legacy, loaded, conditions).step, deadline
                    )
                    result.update(
                        status="trained",
                        training=trained,
                        reload=reload,
                        checkpoint_sha256=file_sha256(directory / "checkpoint.pt"),
                    )
                except Exception as error:
                    result.update(
                        status=failure_status(error),
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                result.update(
                    elapsed_seconds=time.perf_counter() - started, completed_epoch=time.time()
                )
                ledger.append(result)
                write_json(directory / "result.json", result)
                write_json(report / "training_ledger.json", ledger)
                publish_metrics(directory, seed_report / variant)
                write_json(seed_report / variant / "result.json", result)
                print(f"Finished {variant} seed{seed}: {result['status']}", flush=True)
            predictors["CV"] = guarded(
                lambda state, robot: legacy.step(state, robot, "constant_velocity"), deadline
            )
            try:
                evaluation = evaluate_trial(
                    predictors, legacy, trajectories, splits, mesh, base, seed_output / "evaluation"
                )
                evaluations[str(seed)] = {
                    "status": "completed",
                    "per_method": evaluation["per_method"],
                }
                write_json(seed_report / "evaluation_summary.json", evaluation["per_method"])
            except Exception as error:
                evaluations[str(seed)] = {
                    "status": failure_status(error),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(seed_report / "evaluation_failure.json", evaluations[str(seed)])
            publish_metrics(seed_output / "evaluation", seed_report / "evaluation")
            write_json(report / "evaluation_ledger.json", evaluations)
        write_json(
            report / "input_permission_checks.json",
            {
                "native_contact_or_force_query_calls": native_guard.call_count,
                "original_caches_reused_read_only_and_hash_verified": True,
                "same_model_initialization_within_each_seed": True,
                "same_batch_schedule_within_each_seed": True,
                "different_initialization_and_batch_order_across_seeds": True,
                "G_recomputed_from_predicted_current_state_during_rollout": True,
                "future_soft_state_only_supervision_or_scoring": True,
                "independent_test_verified": False,
            },
        )
    after = {name: file_sha256(ROOT / name) for name in before}
    write_json(
        report / "preservation_checks.json",
        {
            "original_seed7_report_unchanged": before == after,
            "data_files_unchanged": all(
                file_sha256(ROOT / n) == h for n, h in parent["data_hashes"].items()
            ),
            "right_arrays_loaded": False,
            "independent_test_verified": False,
        },
    )
    status = {
        "status": "completed",
        "new_trainings_started": len(ledger),
        "new_trainings_completed": sum(r["status"] == "trained" for r in ledger),
        "new_training_failures": [r for r in ledger if r["status"] != "trained"],
        "wall_elapsed_seconds": time.time() - budget["wall_start_epoch"],
        "completed_epoch": time.time(),
        "independent_test_verified": False,
    }
    write_json(report / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
