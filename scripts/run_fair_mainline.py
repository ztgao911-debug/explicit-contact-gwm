"""Two fresh same-network trainings: raw current kinematics versus explicit G."""

from __future__ import annotations

import argparse
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
from bounded_search_data import load_search_data, validation_windows
from bounded_search_evaluation import evaluate_trial
from bounded_search_training import check_deadline, make_schedules, train_candidate
from fair_mainline_data import reencode_training_data
from run_bounded_search import failure_status, foundation, guarded, publish_metrics, source_files
from train_contact_response import model_for
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256
from contact_gwm.fair_conditioning import RawJointPipeline, training_path
from contact_gwm.pinned_contact_pipeline import IndenterInput
from contact_gwm.repaired_contact_pipeline import RepairedJointPipeline
from contact_gwm.search_protocol import frozen_protocol, protocol_digest

ROOT = Path(__file__).resolve().parents[1]


def pipeline_for(variant, legacy, model, conditions):
    if variant == "RAW":
        return RawJointPipeline(legacy, model)
    if variant == "G":
        return RepairedJointPipeline(legacy, model, conditions, "gaussian_geometry")
    raise ValueError("Train only RAW and G")


def native_forbidden(*args, **kwargs):
    raise AssertionError("Fair model forwards must not query native contact or force")


def preflight(variant, objects, data, architecture, deadline):
    _, mesh, _, _, legacy, conditions, encoder, _ = objects
    model = model_for(mesh, architecture, {}, 7)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for network in (model.internal_network, model.contact_network):
            network[4].weight.copy_(
                torch.randn(network[4].weight.shape, generator=generator) * 1e-5
            )
    path = training_path(variant, legacy, model, conditions, encoder)
    x = data["branch_positions"][6, 0][None].clone().requires_grad_(True)
    v = data["branch_velocities"][6, 0][None].clone().requires_grad_(True)
    robots = [[data["branch_robots"][6][t]] for t in range(3)]
    states = path.rollout(x, v, robots)
    loss = ((states[-1][0] - data["branch_positions"][6, 3][None]) / 0.001).square().mean()
    loss += ((states[-1][1] - data["branch_velocities"][6, 3][None]) / 0.1).square().mean()
    gx, gv = torch.autograd.grad(loss, states[1])
    finite_nonzero = bool(
        torch.isfinite(gx).all() and torch.isfinite(gv).all() and gx.norm() > 0 and gv.norm() > 0
    )
    assert finite_nonzero and path.last_features.grad_fn is not None
    forward = pipeline_for(variant, legacy, model, conditions).step(
        legacy.gaussian_state(x[0].detach().numpy(), v[0].detach().numpy()), robots[0][0]
    )
    equal_forward = torch.equal(states[1][0][0].detach(), forward.state.centers) and torch.equal(
        states[1][1][0].detach(), forward.state.velocities
    )
    assert equal_forward
    check_deadline(deadline)
    return {
        "terminal_to_first_position_gradient_norm": float(gx.norm()),
        "terminal_to_first_velocity_gradient_norm": float(gv.norm()),
        "finite_nonzero_gradients": finite_nonzero,
        "train_inference_one_step_bitwise_equal": equal_forward,
        "optimizer_updates": 0,
        "native_queries_allowed": False,
    }


def reload_check(variant, model, checkpoint, objects, architecture):
    _, mesh, trajectories, _, legacy, conditions, _, _ = objects
    loaded = model_for(mesh, architecture, {}, 7)
    loaded.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state"], strict=True)
    loaded.eval().requires_grad_(False)
    arrays, begin = trajectories["center_deep"], 75
    x, v = arrays["node_position_world"][begin], arrays["node_velocity_world"][begin]
    states = [legacy.gaussian_state(x, v) for _ in range(2)]
    pipelines = [pipeline_for(variant, legacy, m, conditions) for m in (model, loaded)]
    for t in range(begin, begin + 20):
        robot = IndenterInput.from_arrays(arrays, t)
        states = [p.step(s, robot).state for p, s in zip(pipelines, states, strict=True)]
        assert torch.equal(states[0].centers, states[1].centers)
        assert torch.equal(states[0].velocities, states[1].velocities)
    return loaded, {"twenty_step_position_velocity_bitwise_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fair_mainline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    if (report / "protocol.json").exists():
        raise RuntimeError("Immutable run already prepared; do not overwrite or silently retry")
    budget = json.loads((report / "budget.json").read_text())
    deadline = budget["deadline_epoch"]
    check_deadline(deadline)
    torch.set_num_threads(1)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    parent = json.loads((ROOT / config["comparison"]["parent_protocol"]).read_text())
    for name, expected in parent["data_hashes"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Historical input changed: {name}")
    objects = foundation(config)
    base, mesh, trajectories, splits, legacy, conditions, encoder, _ = objects
    data, windows = (
        load_search_data(ROOT, trajectories, mesh),
        validation_windows(trajectories, mesh, ROOT),
    )
    variants = config["comparison"]["variants"]
    models = {v: model_for(mesh, config["model"], {}, 7) for v in variants}
    initial = {v: tensor_digest(m.state_dict()) for v, m in models.items()}
    counts = {v: sum(p.numel() for p in m.parameters()) for v, m in models.items()}
    assert len(set(initial.values())) == len(set(counts.values())) == 1
    assert counts["G"] == 45324
    schedules = make_schedules(data, 7, 200, 16)
    np.savez_compressed(
        output / "shared_batch_schedule.npz", **{k: v.numpy() for k, v in schedules.items()}
    )
    with patch.object(legacy.native, "query", side_effect=native_forbidden) as native_guard:
        probes = {v: preflight(v, objects, data, config["model"], deadline) for v in variants}
        write_json(report / "gradient_preflight.json", probes)
        protocol = frozen_protocol(7)
        protocol.update(
            experiment_kind="fair_RAW_vs_G_from_scratch_short_mainline",
            configuration=config,
            candidates=[dict(config["comparison"], id=v) for v in variants],
            max_search_configurations=2,
            max_updates_per_training=200,
            max_replication_candidates=0,
            replication_seeds=[],
            max_replication_trainings=0,
            wall_clock_budget_seconds=1800,
            data_hashes=parent["data_hashes"],
            source_hashes=hashes([*source_files(), args.config.resolve()]),
            initial_weight_sha256=initial,
            parameter_counts=counts,
            shared_batch_schedule_sha256=file_sha256(output / "shared_batch_schedule.npz"),
            reference_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            validation_windows=parent["validation_windows"],
            fair_input_permissions=(
                "Current x/v/rest, actual tool x/v and command x/v; no future soft input"
            ),
            RAW_condition="36 raw state/kinematic channels, gate=1; no contact queries",
            G_condition=(
                "Existing topology-assisted surface + calibrated point force + contact gate; "
                "no native query"
            ),
            differentiation=(
                "Current contact geometry/force stop-gradient; state/features/kernels attached"
            ),
            scope=(
                "Whole conditioning inductive bias comparison; not force-feature-only attribution"
            ),
            independent_test="not_verified_right_arrays_never_loaded",
        )
        protocol["checkpoint_selection"]["interval_updates"] = 50
        write_json(report / "protocol.json", protocol)
        write_json(report / "protocol_digest.json", {"sha256": protocol_digest(protocol)})
        write_json(report / "data_loader_checks.json", data["checks"])
        for file in [*source_files(), args.config.resolve()]:
            target = output / "source_snapshot" / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        print(f"Frozen fair protocol; counts={counts}, same initial weights=True", flush=True)
        results, predictors = [], {}
        for variant in variants:
            started = time.perf_counter()
            directory = output / variant
            directory.mkdir()
            result = {
                "variant": variant,
                "status": "running",
                "initial_weight_sha256": initial[variant],
            }
            write_json(
                directory / "attempt_configuration.json",
                {**result, "protocol_sha256": protocol_digest(protocol)},
            )
            try:
                model = models[variant]
                pipeline = pipeline_for(variant, legacy, model, conditions)
                cached = reencode_training_data(
                    ROOT, data, trajectories, pipeline, deadline, directory
                )
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
                    cached,
                    windows,
                    mesh,
                    candidate,
                    {**config["training"], "seed": 7},
                    schedules,
                    directory,
                    deadline,
                    path=training_path(variant, legacy, model, conditions, encoder),
                )
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "architecture": config["model"],
                        "variant": variant,
                        "training_summary": trained,
                    },
                    directory / "checkpoint.pt",
                )
                model, reload = reload_check(
                    variant, model, directory / "checkpoint.pt", objects, config["model"]
                )
                predictors[variant] = guarded(
                    pipeline_for(variant, legacy, model, conditions).step, deadline
                )
                result.update(
                    status="trained",
                    training=trained,
                    reload=reload,
                    checkpoint_sha256=file_sha256(directory / "checkpoint.pt"),
                )
            except Exception as error:
                result.update(
                    status=failure_status(error), error=str(error), traceback=traceback.format_exc()
                )
            result["elapsed_seconds"] = time.perf_counter() - started
            results.append(result)
            publish_metrics(directory, report / variant)
            write_json(report / "training_ledger.json", results)
            print(f"{variant}: {result['status']}, {result['elapsed_seconds']:.2f}s", flush=True)
        predictors["CV"] = guarded(
            lambda state, robot: legacy.step(state, robot, "constant_velocity"), deadline
        )
        evaluation = evaluate_trial(
            predictors, legacy, trajectories, splits, mesh, base, output / "evaluation"
        )
        write_json(report / "evaluation_summary.json", evaluation["per_method"])
        publish_metrics(output / "evaluation", report / "evaluation")
        write_json(
            report / "input_permission_checks.json",
            {
                "native_contact_or_force_query_calls": native_guard.call_count,
                "native_queries_forbidden_during_caching_training_and_evaluation": True,
                "same_initial_weights": len(set(initial.values())) == 1,
                "same_parameter_count": len(set(counts.values())) == 1,
                "raw_receives_tool_state_and_commands": True,
                "G_reencoded_for_training_and_recomputed_during_rollout": True,
                "historical_M_contexts_and_gates_replaced": True,
                "query_counter_note": (
                    "Legacy native_queries fields count conditioner calls here; "
                    "actual native queries=0"
                ),
            },
        )
    changed = [n for n, h in parent["data_hashes"].items() if file_sha256(ROOT / n) != h]
    write_json(
        report / "preservation_checks.json",
        {
            "frozen_data_files": len(parent["data_hashes"]),
            "changed": changed,
            "right_arrays_loaded": False,
            "independent_test_verified": False,
        },
    )
    write_json(
        report / "status.json",
        {
            "status": "completed",
            "trainings_started": len(results),
            "successful_trainings": sum(r["status"] == "trained" for r in results),
            "wall_elapsed_seconds": time.time() - budget["wall_start_epoch"],
            "completed_epoch": time.time(),
            "independent_test_verified": False,
        },
    )
    print(json.dumps(json.loads((report / "status.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
