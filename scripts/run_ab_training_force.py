"""A: fixed long training prefixes. B: fresh geometry-only G0 force ablation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
from bounded_search_training import (
    branch_window_loss,
    check_deadline,
    make_schedules,
    train_candidate,
)
from replicate_fair_mainline import cached_data
from run_bounded_search import failure_status, foundation, guarded, publish_metrics, source_files
from run_fair_mainline import native_forbidden, pipeline_for
from train_contact_response import losses, model_for
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256
from contact_gwm.fair_conditioning import training_path
from contact_gwm.force_input_ablation import (
    geometry_only_pipeline,
    geometry_only_training_path,
    without_force_cache,
)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("RAW", "G", "G0")
SEEDS = (7, 17, 27)
STAGES = ("prefix_200", "prefix_500", "prefix_1000", "prefix_2000", "exact_2000")


def pipeline(variant, legacy, model, conditions):
    return (
        geometry_only_pipeline(legacy, model, conditions)
        if variant == "G0"
        else pipeline_for(variant, legacy, model, conditions)
    )


def rollout_path(variant, legacy, model, conditions, encoder):
    return (
        geometry_only_training_path(legacy, model, conditions, encoder)
        if variant == "G0"
        else training_path(variant, legacy, model, conditions, encoder)
    )


def candidate_for(config, variant):
    return {
        "id": variant,
        "mode": "autoregressive",
        **{
            k: config["comparison"][k]
            for k in ("horizon", "learning_rate", "velocity_weight", "mesh_weight")
        },
    }


def load_case(config):
    objects = foundation(config)
    _, mesh, trajectories, _, _, _, _, _ = objects
    source = load_search_data(ROOT, trajectories, mesh)
    windows = validation_windows(trajectories, mesh, ROOT)
    return objects, source, windows


def verify_frozen(protocol):
    for name, digest in {**protocol["source_hashes"], **protocol["data_hashes"]}.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen source/data changed: {name}")


def diagnostic_callback(variant, objects, data, config, directory):
    _, mesh, _, _, legacy, conditions, encoder, _ = objects
    rows = []

    @torch.no_grad()
    def callback(step, model):
        before, rng = tensor_digest(model.state_dict()), torch.get_rng_state().clone()
        settings = {
            **config["training"],
            "velocity_weight": config["comparison"]["velocity_weight"],
        }
        started = time.perf_counter()
        row = {
            "update": step,
            "teacher_forced_central": losses(
                model, data["ordinary_train"], data["branch_cache"], data["pairs"], settings
            ),
        }
        try:
            path = rollout_path(variant, legacy, model, conditions, encoder)
            probe, grid, details = branch_window_loss(
                model,
                path,
                data,
                torch.linspace(0, 107, 16).long(),
                torch.linspace(0, 0.999999, 16),
                candidate_for(config, variant),
                settings,
                mesh,
                json.loads((ROOT / config["report_directory"] / "budget.json").read_text())[
                    "deadline_epoch"
                ],
            )
            row.update(
                central_fixed_AR_probe_loss=float(probe), mesh_penalty=float(grid), **details
            )
        except Exception as error:
            row.update(central_fixed_AR_probe_loss=None, diagnostic_failure=str(error))
        row.update(
            parameters_bitwise_unchanged=before == tensor_digest(model.state_dict()),
            RNG_unchanged=torch.equal(rng, torch.get_rng_state()),
            elapsed_seconds=time.perf_counter() - started,
        )
        assert row["parameters_bitwise_unchanged"] and row["RNG_unchanged"]
        rows.append(row)
        write_json(directory / "milestone_diagnostics.json", rows)

    return callback


def prepare(config, report, output, deadline):
    parent_report, parent_output = (
        ROOT / config["experiments"]["parent_report"],
        ROOT / config["experiments"]["parent_output"],
    )
    parent = json.loads((parent_report / "protocol.json").read_text())
    for key in ("model", "validation", "evaluation"):
        assert config[key] == parent["configuration"][key]
    original_train = parent["configuration"]["training"]
    assert {
        k: v for k, v in config["training"].items() if k not in ("updates", "milestone_updates")
    } == {k: v for k, v in original_train.items() if k != "updates"}
    assert config["training"]["updates"] == 2000
    assert config["experiments"]["seeds"] == list(SEEDS)
    assert config["experiments"]["maximum_trainings"] == 9
    for name, digest in parent["data_hashes"].items():
        assert file_sha256(ROOT / name) == digest, name
    dependencies = json.loads((parent_report / "conditioning_dependencies.json").read_text())[
        "read_only_dependencies"
    ]
    for dep in dependencies.values():
        assert file_sha256(ROOT / dep["path"]) == dep["sha256"]
    objects, source, windows = load_case(config)
    _, mesh, _, _, legacy, conditions, encoder, _ = objects
    old_artifacts = json.loads((parent_report / "large_artifact_manifest.json").read_text())[
        "ignored_large_artifacts"
    ]
    caches = {v: cached_data(source, parent_output, old_artifacts, v, mesh) for v in ("RAW", "G")}
    specs = {
        v: {
            name: {
                "path": str((parent_output / v / f"{name}.npz").relative_to(ROOT)),
                "sha256": file_sha256(parent_output / v / f"{name}.npz"),
            }
            for name in ("ordinary_train", "branch_cache")
        }
        for v in ("RAW", "G")
    }
    (output / "caches").mkdir()
    for name in ("ordinary_train", "branch_cache"):
        cache = without_force_cache(caches["G"][name])
        file = output / "caches" / f"G0_{name}.npz"
        np.savez_compressed(file, **{k: v.numpy() for k, v in cache.items()})
        specs.setdefault("G0", {})[name] = {
            "path": str(file.relative_to(ROOT)),
            "sha256": file_sha256(file),
        }
    initial = {
        s: {
            v: tensor_digest(model_for(mesh, config["model"], {}, s).state_dict()) for v in VARIANTS
        }
        for s in SEEDS
    }
    assert all(len(set(v.values())) == 1 for v in initial.values())
    for seed in SEEDS:
        schedule = make_schedules(source, seed, 2000, 16)
        np.savez_compressed(
            output / f"schedule_{seed}.npz", **{k: v.numpy() for k, v in schedule.items()}
        )
    probes = {}
    with patch.object(legacy.native, "query", side_effect=native_forbidden):
        for variant in VARIANTS:
            model = model_for(mesh, config["model"], {}, 7)
            assert sum(p.numel() for p in model.parameters()) == 45324
            gen = torch.Generator().manual_seed(7)
            with torch.no_grad():
                for net in (model.internal_network, model.contact_network):
                    net[4].weight.copy_(torch.randn(net[4].weight.shape, generator=gen) * 1e-5)
            path = rollout_path(variant, legacy, model, conditions, encoder)
            x = source["branch_positions"][6, 0][None].clone().requires_grad_(True)
            v = source["branch_velocities"][6, 0][None].clone().requires_grad_(True)
            states = path.rollout(x, v, [[source["branch_robots"][6][t]] for t in range(3)])
            loss = (
                ((states[-1][0] - source["branch_positions"][6, 3][None]) / 0.001).square().mean()
            )
            loss += (
                ((states[-1][1] - source["branch_velocities"][6, 3][None]) / 0.1).square().mean()
            )
            gx, gv = torch.autograd.grad(loss, states[1])
            assert (
                torch.isfinite(gx).all()
                and torch.isfinite(gv).all()
                and gx.norm() > 0
                and gv.norm() > 0
            )
            probes[variant] = {
                "position_gradient_norm": float(gx.norm()),
                "velocity_gradient_norm": float(gv.norm()),
                "optimizer_updates": 0,
            }
    historical = hashes(
        [
            p
            for folder in (parent_report, ROOT / "experiments/fair_mainline_multiseed/run_001")
            for p in folder.rglob("*")
            if p.is_file()
        ]
    )
    protocol = {
        "experiment_kind": "A_training_budget_B_force_input_ablation",
        "configuration": config,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "milestones": [0, 200, 500, 1000, 2000],
        "evaluation_stages": list(STAGES),
        "initial_weight_sha256": initial,
        "parameter_count_each": 45324,
        "G0_force_channels": [11, 12, 13],
        "G0_unexcited_first_layer_force_weights": 384,
        "G0_same_nominal_not_same_effective_input_capacity": True,
        "cache_specs": specs,
        "data_hashes": parent["data_hashes"],
        "conditioning_dependencies": dependencies,
        "historical_report_hashes": historical,
        "source_hashes": hashes([*source_files(), ROOT / "configs/ab_training_force.yaml"]),
        "reference_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_extension": (
            "Optional milestone serialization/callback; losses, Adam, LR formula "
            "and default behavior unchanged"
        ),
        "checkpoint_selection": parent["checkpoint_selection"],
        "ranking": parent["ranking"],
        "validation_windows": parent["validation_windows"],
        "native_queries_allowed": False,
        "short_200_run_not_a_matched_prefix": True,
        "independent_test_verified": False,
    }
    write_json(report / "protocol.json", protocol)
    write_json(report / "gradient_preflight.json", probes)
    write_json(report / "data_loader_checks.json", source["checks"])
    for file in [*source_files(), ROOT / "configs/ab_training_force.yaml"]:
        target = output / "source_snapshot" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    check_deadline(deadline)
    print("Frozen nine trainings, milestones and evaluation protocol", flush=True)


def worker(config, seed, variant, attempt):
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    deadline = json.loads((report / "budget.json").read_text())["deadline_epoch"]
    protocol = json.loads((report / "protocol.json").read_text())
    directory = output / "train" / f"seed_{seed}" / variant / f"attempt_{attempt:02d}"
    started = time.perf_counter()
    result = {
        "seed": seed,
        "variant": variant,
        "attempt": attempt,
        "status": "running",
        "artifact_directory": str(directory.relative_to(ROOT)),
        "started_epoch": time.time(),
    }
    write_json(directory / "configuration.json", result)
    try:
        verify_frozen(protocol)
        objects, source, windows = load_case(config)
        _, mesh, _, _, legacy, conditions, encoder, _ = objects
        cache = {}
        for name, count in (("ordinary_train", 600), ("branch_cache", 2160)):
            spec = protocol["cache_specs"][variant][name]
            assert file_sha256(ROOT / spec["path"]) == spec["sha256"]
            cache[name] = _tensor_cache(ROOT / spec["path"], mesh, count)
            for key in ("features", "velocity", "target_dx", "target_dv", "neighborhood", "pinned"):
                assert torch.equal(cache[name][key], source[name][key])
        data = {**source, **cache}
        with np.load(output / f"schedule_{seed}.npz") as saved:
            schedules = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
        model = model_for(mesh, config["model"], {}, seed)
        assert (
            tensor_digest(model.state_dict())
            == protocol["initial_weight_sha256"][str(seed)][variant]
        )
        with patch.object(legacy.native, "query", side_effect=native_forbidden) as native:
            trained = train_candidate(
                model,
                legacy,
                conditions,
                encoder,
                data,
                windows,
                mesh,
                candidate_for(config, variant),
                {**config["training"], "seed": seed},
                schedules,
                directory,
                deadline,
                path=rollout_path(variant, legacy, model, conditions, encoder),
                milestone_callback=diagnostic_callback(variant, objects, data, config, directory),
            )
            trained["actual_native_queries"] = native.call_count
            trained["conditional_gradient"] = (
                "Current state attached; G/G0 geometry/force stop-gradient; RAW kinematics attached"
            )
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
        result.update(
            status="trained",
            training=trained,
            actual_native_queries=trained["actual_native_queries"],
        )
    except Exception as error:
        result.update(
            status=failure_status(error), error=str(error), traceback=traceback.format_exc()
        )
    result.update(elapsed_seconds=time.perf_counter() - started, completed_epoch=time.time())
    write_json(directory / "result.json", result)
    print(json.dumps(result), flush=True)


def evaluate_all(config, report, output, ledger, deadline):
    objects, _, _ = load_case(config)
    base, mesh, trajectories, splits, legacy, conditions, _, _ = objects
    records = []
    with patch.object(legacy.native, "query", side_effect=native_forbidden) as native:
        for seed in SEEDS:
            for stage in STAGES:
                check_deadline(deadline)
                directory = output / "evaluation" / f"seed_{seed}" / stage
                if (directory / "evaluation_summary.json").exists():
                    evaluation = json.loads((directory / "evaluation_summary.json").read_text())
                else:
                    predictors = {}
                    for variant in VARIANTS:
                        result = next(
                            r for r in ledger if r["seed"] == seed and r["variant"] == variant
                        )
                        if result["status"] != "trained":
                            continue
                        kind, limit = stage.split("_")
                        artifact = (
                            ROOT / result["artifact_directory"] / f"{kind}_{int(limit):04d}.pt"
                        )
                        saved = torch.load(artifact, map_location="cpu", weights_only=False)
                        if saved["model_state"] is None:
                            continue
                        model = model_for(mesh, config["model"], {}, seed)
                        model.load_state_dict(saved["model_state"], strict=True)
                        model.eval().requires_grad_(False)
                        predictors[variant] = guarded(
                            pipeline(variant, legacy, model, conditions).step, deadline
                        )
                    predictors["CV"] = guarded(
                        lambda state, robot: legacy.step(state, robot, "constant_velocity"),
                        deadline,
                    )
                    try:
                        evaluation = evaluate_trial(
                            predictors, legacy, trajectories, splits, mesh, base, directory
                        )
                    except Exception as error:
                        records.append(
                            {
                                "seed": seed,
                                "stage": stage,
                                "status": failure_status(error),
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        write_json(report / "evaluation_ledger.json", records)
                        continue
                records.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "status": "completed",
                        "per_method": evaluation["per_method"],
                    }
                )
                published = report / "evaluation" / f"seed_{seed}" / stage
                write_json(published / "summary.json", evaluation["per_method"])
                publish_metrics(directory, published)
                write_json(report / "evaluation_ledger.json", records)
                print(f"Evaluated seed{seed} {stage}", flush=True)
    write_json(
        report / "input_permission_checks.json",
        {
            "evaluation_native_queries": native.call_count,
            "training_native_queries": sum(r.get("actual_native_queries", 0) for r in ledger),
            "right_arrays_loaded": False,
            "independent_test_verified": False,
        },
    )


def master(config, resume):
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    deadline = json.loads((report / "budget.json").read_text())["deadline_epoch"]
    if not (report / "protocol.json").exists():
        prepare(config, report, output, deadline)
    elif not resume:
        raise RuntimeError("Existing run requires explicit --resume; never overwrite")
    protocol = json.loads((report / "protocol.json").read_text())
    verify_frozen(protocol)
    jobs = [(s, v) for s in SEEDS for v in VARIANTS]
    ledger = []
    active = {}
    pending = []
    for seed, variant in jobs:
        directory = output / "train" / f"seed_{seed}" / variant / "attempt_01"
        if (directory / "result.json").exists():
            ledger.append(json.loads((directory / "result.json").read_text()))
        elif directory.exists():
            raise RuntimeError(f"Interrupted attempt requires investigation: {directory}")
        else:
            pending.append((seed, variant))
    while pending or active:
        check_deadline(deadline)
        while pending and len(active) < config["experiments"]["maximum_parallel_workers"]:
            seed, variant = pending.pop(0)
            directory = output / "train" / f"seed_{seed}" / variant / "attempt_01"
            directory.mkdir(parents=True)
            log = (directory / "execution.log").open("w")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--seed",
                    str(seed),
                    "--variant",
                    variant,
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[proc.pid] = (proc, log, seed, variant, directory)
            print(f"Started seed{seed} {variant}, pid={proc.pid}", flush=True)
        for pid, (proc, log, seed, variant, directory) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            result = (
                json.loads((directory / "result.json").read_text())
                if (directory / "result.json").exists()
                else {
                    "seed": seed,
                    "variant": variant,
                    "status": "technical_failure",
                    "error": f"Worker exit {code}",
                    "artifact_directory": str(directory.relative_to(ROOT)),
                }
            )
            ledger.append(result)
            del active[pid]
            write_json(report / "training_ledger.json", ledger)
            publish_metrics(directory, report / "training" / f"seed_{seed}" / variant)
            for name in ("result.json", "milestone_diagnostics.json"):
                if (directory / name).exists():
                    shutil.copy2(
                        directory / name, report / "training" / f"seed_{seed}" / variant / name
                    )
            print(f"Finished seed{seed} {variant}: {result['status']}", flush=True)
        if active:
            time.sleep(1)
    evaluate_all(config, report, output, ledger, deadline)
    changed = [
        n for n, h in protocol["historical_report_hashes"].items() if file_sha256(ROOT / n) != h
    ]
    write_json(
        report / "preservation_checks.json",
        {
            "historical_report_changed": changed,
            "data_changed": [
                n for n, h in protocol["data_hashes"].items() if file_sha256(ROOT / n) != h
            ],
            "right_arrays_loaded": False,
        },
    )
    write_json(
        report / "status.json",
        {
            "status": "completed",
            "trainings_started": len(ledger),
            "trainings_completed": sum(r["status"] == "trained" for r in ledger),
            "failures": [r for r in ledger if r["status"] != "trained"],
            "wall_elapsed_seconds": time.time()
            - json.loads((report / "budget.json").read_text())["wall_start_epoch"],
            "independent_test_verified": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/ab_training_force.yaml").read_text())
    torch.set_num_threads(1)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / config["output_directory"] / "matplotlib"))
    if args.worker:
        worker(config, args.seed, args.variant, 1)
    else:
        master(config, args.resume)


if __name__ == "__main__":
    main()
