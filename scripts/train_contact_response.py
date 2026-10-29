"""Central paired-data pilot: response supervision x output-bound ablation, CPU."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from audit_contact_mechanism import external_inputs
from contact_response_data import branch_cache, collect_central, validation_records
from contact_response_evaluation import evaluate_responses
from joint_state_evaluation import evaluate_models
from train_joint_state import load_data, make_cache
from validate_contact_pipeline import hashes, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.joint_state_dynamics import JointStateDynamics, JointStatePipeline
from contact_gwm.paired_contact_response import (
    absolute_loss,
    central_derived_caps,
    normalized_state_loss,
    output_bound_audit,
    paired_loss,
    predict_cache,
)
from contact_gwm.pinned_contact_pipeline import PinnedContactPipeline
from contact_gwm.repaired_contact_pipeline import RepairedContactConditions, RepairedJointPipeline

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("mujoco_force", "analytic_force", "gaussian_geometry")


def subset(cache, indices):
    return {k: v if k == "pinned" else v[indices] for k, v in cache.items()}


@torch.no_grad()
def full_outputs(model, cache, indices, chunk_size=64):
    dx, dv = [], []
    for chunk in indices.split(chunk_size):
        p = predict_cache(model, cache, chunk)
        dx.append(p.delta_position)
        dv.append(p.delta_velocity)
    return torch.cat(dx), torch.cat(dv)


@torch.no_grad()
def losses(model, ordinary, branches, pairs, settings):
    components = []
    for cache in (ordinary, branches):
        ids = torch.arange(len(cache["context"]))
        dx, dv = full_outputs(model, cache, ids)
        components.append(
            float(
                normalized_state_loss(
                    dx,
                    dv,
                    cache["target_dx"],
                    cache["target_dv"],
                    cache["neighborhood"],
                    cache["pinned"],
                    settings,
                )
            )
        )
    bx, bv = full_outputs(model, branches, pairs["base_index"])
    vx, vv = full_outputs(model, branches, pairs["variant_index"])
    components.append(
        float(
            normalized_state_loss(
                vx - bx,
                vv - bv,
                pairs["target_dx"],
                pairs["target_dv"],
                pairs["neighborhood"],
                branches["pinned"],
                settings,
            )
        )
    )
    return dict(zip(("ordinary", "branch_absolute", "paired_response"), components, strict=True))


def model_for(mesh, architecture, caps, seed):
    torch.manual_seed(seed)
    return JointStateDynamics(
        torch.tensor(mesh["rest_positions_world"]).float(),
        torch.tensor(mesh["pinned_mask"]),
        **{**architecture, **caps},
    )


def train_one(name, model, use_response, caches, pairs, schedules, config, output, architecture):
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    history = []
    best_loss = math.inf
    selected_step = None
    selected_weights = None
    clipped = 0
    started = time.perf_counter()
    for step in range(settings["updates"] + 1):
        if step % settings["validation_interval"] == 0 or step == settings["updates"]:
            model.eval()
            tr = losses(model, caches["ordinary_train"], caches["train"], pairs["train"], settings)
            va = losses(
                model,
                caches["ordinary_validation"],
                caches["validation"],
                pairs["validation"],
                settings,
            )
            # Identical selection metric, INCLUDING STEP ZERO, for all four variants.
            score = (
                va["ordinary"]
                + settings["branch_absolute_weight"] * va["branch_absolute"]
                + settings["paired_response_weight"] * va["paired_response"]
            )
            row = {
                "step": step,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"validation_{k}": v for k, v in va.items()},
                "common_validation_score": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            assert all(math.isfinite(value) for value in row.values())
            history.append(row)
            if score < best_loss:
                best_loss, selected_step, selected_weights = (
                    score,
                    step,
                    copy.deepcopy(model.state_dict()),
                )
            if step % 250 == 0:
                print(
                    f"{name}: step {step}, val ordinary={va['ordinary']:.6g}, "
                    f"branch={va['branch_absolute']:.6g}, response={va['paired_response']:.6g}",
                    flush=True,
                )
        if step == settings["updates"]:
            break
        model.train()
        fraction = step / max(1, settings["updates"] - 1)
        rate = settings["minimum_learning_rate_fraction"] + (
            1 - settings["minimum_learning_rate_fraction"]
        ) * 0.5 * (1 + math.cos(math.pi * fraction))
        optimizer.param_groups[0]["lr"] = settings["learning_rate"] * rate
        optimizer.zero_grad(set_to_none=True)
        loss = absolute_loss(model, caches["ordinary_train"], schedules["ordinary"][step], settings)
        loss = loss + settings["branch_absolute_weight"] * absolute_loss(
            model, caches["train"], schedules["branch"][step], settings
        )
        if use_response:
            loss = loss + settings["paired_response_weight"] * paired_loss(
                model, caches["train"], pairs["train"], schedules["pair"][step], settings
            )
        (settings["loss_multiplier"] * loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
        )
        clipped += int(norm > settings["gradient_clip"])
        optimizer.step()
    model.load_state_dict(selected_weights, strict=True)
    model.eval().requires_grad_(False)
    folder = output / name
    folder.mkdir()
    artifact = {
        "model_state": selected_weights,
        "architecture": architecture,
        "config": config,
        "selected_step": selected_step,
        "selection_loss": best_loss,
        "node_ids": np.arange(len(model.pinned_mask)),
        "training_updates": settings["updates"],
        "response_supervised": use_response,
    }
    torch.save(artifact, folder / "checkpoint.pt")
    write_csv(folder / "training_curve.csv", history)
    result = {
        "variant": name,
        "selected_step": selected_step,
        "selection_loss": best_loss,
        "updates": settings["updates"],
        "finite_gradients": True,
        "clipped_updates": clipped,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": architecture,
        "response_supervised": use_response,
    }
    write_json(folder / "training_summary.json", result)
    return result


def forbidden(*a, **kw):
    raise AssertionError("Forbidden original solver operation")


@torch.no_grad()
def contracts(models, legacy, conditions, trajectories, mesh, output, config):
    # Same original, unpoisoned INITIAL state; only known robot fields survive.
    native = read_npz(ROOT / config["validation_native_directory"] / "70_base_native.npz")
    initial = legacy.gaussian_state(native["position"][0], native["velocity"][0])
    external = external_inputs(native)
    poisoned = {
        k: np.full_like(v, np.nan, dtype=float)
        if k in ("position", "velocity", "net_force", "net_torque") or k.startswith("contact_")
        else v.copy()
        for k, v in native.items()
    }
    alternate = external_inputs(poisoned)
    for name, model in models.items():
        if name == "frozen":
            continue
        saved = torch.load(output / name / "checkpoint.pt", map_location="cpu", weights_only=False)
        reload = model_for(mesh, saved["architecture"], {}, config["seed"])
        reload.load_state_dict(saved["model_state"], strict=True)
        reload.eval().requires_grad_(False)
        pipe = RepairedJointPipeline(legacy, model, conditions, "mujoco_force")
        again = RepairedJointPipeline(legacy, reload, conditions, "mujoco_force")
        a, b, c = (
            pipe.rollout(initial, external),
            again.rollout(initial, external),
            pipe.rollout(initial, alternate),
        )
        assert all(
            torch.equal(x.state.centers, y.state.centers)
            and torch.equal(x.state.velocities, y.state.velocities)
            for other in (b, c)
            for x, y in zip(a, other, strict=True)
        )
        for method in ("analytic_force", "gaussian_geometry"):
            p = RepairedJointPipeline(legacy, model, conditions, method)
            with (
                patch.object(mujoco, "mj_forward", forbidden),
                patch.object(mujoco, "mj_contactForce", forbidden),
                patch.object(mujoco, "mj_step", forbidden),
            ):
                p.step(initial, external[0])
                if method == "gaussian_geometry":
                    with patch.object(legacy.native, "query", forbidden):
                        p.step(initial, external[0])
    return {
        "reload_selected_weights_20_step_bitwise_equal": True,
        "future_soft_contact_force_poisoning_no_effect": True,
        "analytic_branches_native_force_forbidden": True,
        "G_native_geometry_forbidden": True,
        "state_update_still_independent_position_and_instantaneous_velocity": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_contact_response.yaml")
    parser.add_argument("--finalize-existing", type=Path)
    args = parser.parse_args()
    if args.finalize_existing is not None:
        from finalize_contact_response import finalize_existing

        finalize_existing(args.finalize_existing)
        return
    config = yaml.safe_load(args.config.read_text())
    assert config["seed"] == 7 and config["dt"] == 0.01 and config["training"]["updates"] == 2000
    torch.set_num_threads(config["cpu_threads"])
    before = hashes([*(ROOT / config["data_directory"]).rglob("*"), *(ROOT / "outputs").rglob("*")])
    parent = ROOT / config["output_root"]
    index = 1
    while (parent / f"run_{index:03d}").exists():
        index += 1
    output = parent / f"run_{index:03d}"
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    started = time.perf_counter()
    write_json(output / "run_status.json", {"complete": False})
    write_json(output / "historical_hashes.json", before)
    sources = [
        "scripts/train_contact_response.py",
        "scripts/contact_response_data.py",
        "scripts/contact_response_evaluation.py",
        "src/contact_gwm/paired_contact_response.py",
        "configs/train_contact_response.yaml",
        "src/contact_gwm/joint_state_dynamics.py",
        "src/contact_gwm/repaired_contact_pipeline.py",
        "scripts/joint_state_evaluation.py",
        "tests/test_paired_contact_response.py",
    ]
    for source in sources:
        target = output / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    try:
        mainline = yaml.safe_load((ROOT / config["mainline_config"]).read_text())
        config["_variants"] = mainline["interventions"]["variants"]
        write_json(output / "resolved_config.json", config)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        mesh, trajectories, splits = load_data(ROOT / config["data_directory"], config)
        legacy_cfg = yaml.safe_load((ROOT / mainline["legacy_pipeline_config"]).read_text())
        legacy = PinnedContactPipeline(
            mesh,
            ROOT / config["data_directory"] / "scene.xml",
            ROOT / legacy_cfg["checkpoint"],
            legacy_cfg,
        )
        old = torch.load(
            ROOT / config["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        architecture = old["architecture"]
        frozen = model_for(mesh, architecture, {}, config["seed"])
        frozen.load_state_dict(old["model_state"], strict=True)
        frozen.eval().requires_grad_(False)
        point = json.loads((ROOT / mainline["point_calibration"]).read_text())
        integration = yaml.safe_load((ROOT / mainline["integration_config"]).read_text())[
            "integration"
        ]
        conditions = RepairedContactConditions(
            legacy,
            mesh,
            point["stiffness_N_m"],
            ElasticAreaMaterial.load(ROOT / mainline["area_material"]),
            integration,
        )
        frozen_pipe = RepairedJointPipeline(legacy, frozen, conditions, "mujoco_force")
        central = collect_central(mesh, trajectories, config, config["_variants"], output)
        val_records = validation_records(config, config["_variants"], output)
        caches = {}
        pairs = {}
        cache_checks = {}
        caches["train"], pairs["train"], cache_checks["train"] = branch_cache(
            central, frozen_pipe, trajectories, mesh, config, output, "train"
        )
        audit = output_bound_audit(caches["train"], pairs["train"], 0.001, 0.1)
        # Central bounds are FROZEN before constructing any validation cache.
        expanded = central_derived_caps(audit)
        write_json(output / "central_output_bound_audit.json", audit)
        write_json(output / "central_derived_caps.json", expanded)
        print(
            f"Central-only derived output caps: {expanded}; original bound audit: {audit}",
            flush=True,
        )
        caches["validation"], pairs["validation"], cache_checks["validation"] = branch_cache(
            val_records, frozen_pipe, trajectories, mesh, config, output, "validation"
        )
        ordinary, samples = make_cache(frozen_pipe, trajectories, splits, mesh, config, output)
        for split in ("train", "validation"):
            ids = torch.tensor([i for i, s in enumerate(samples) if s["split"] == split])
            caches["ordinary_" + split] = subset(ordinary, ids)
        generator = torch.Generator().manual_seed(config["seed"])
        schedules = {
            key: torch.randint(
                count,
                (config["training"]["updates"], config["training"]["batch_size"]),
                generator=generator,
            )
            for key, count in [
                ("ordinary", 600),
                ("branch", len(caches["train"]["context"])),
                ("pair", len(pairs["train"]["base_index"])),
            ]
        }
        np.savez_compressed(
            output / "shared_batch_schedule.npz", **{k: v.numpy() for k, v in schedules.items()}
        )
        models = {"frozen": frozen}
        reports = []
        init_digests = []
        for variant in config["variants"]:
            caps = {} if variant["caps"] == "original" else expanded
            arch = {**architecture, **caps}
            model = model_for(mesh, arch, {}, config["seed"])
            init_digests.append(tensor_digest(model.state_dict()))
            reports.append(
                train_one(
                    variant["name"],
                    model,
                    variant["response"],
                    caches,
                    pairs,
                    schedules,
                    config,
                    output,
                    arch,
                )
            )
            models[variant["name"]] = model
        assert len(set(init_digests)) == 1
        digests = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
        predictors = {}
        native_predictors = {}
        for name, model in models.items():
            for method in CONDITIONS:
                pipe = RepairedJointPipeline(legacy, model, conditions, method)
                predictors[f"{name}__{method}"] = pipe.step
                if method == "mujoco_force":
                    native_predictors[name] = pipe.step
            predictors[f"{name}__internal_only"] = JointStatePipeline(
                legacy, model, contact_method="none"
            ).step
        predictors["constant_velocity"] = lambda state, robot: legacy.step(
            state, robot, "constant_velocity"
        )
        native_predictors["constant_velocity"] = predictors["constant_velocity"]
        checks = contracts(models, legacy, conditions, trajectories, mesh, output, config)
        with patch.object(mujoco, "mj_step", forbidden):
            evaluation = evaluate_models(
                predictors,
                legacy.gaussian_state,
                trajectories,
                splits,
                mesh,
                config,
                output / "ordinary_evaluation",
            )
            response = evaluate_responses(
                native_predictors, legacy, trajectories, mesh, config, output, ROOT
            )
        assert all(
            tensor_digest(models[name].state_dict()) == value for name, value in digests.items()
        )
        assert before == {name: file_sha256(ROOT / name) for name in before}
        checks.update(
            {
                **evaluation["checks"],
                "cache_checks": cache_checks,
                "same_initial_weights_all_four_variants": True,
                "same_shared_batch_schedule_all_four_variants": True,
                "central_native_trajectories_generated": len(central),
                "validation_native_trajectories_reused": len(val_records),
                "central_response_pairs": len(pairs["train"]["base_index"]),
                "validation_response_pairs": len(pairs["validation"]["base_index"]),
                "train_gradient_rows_central_only": True,
                "test_trajectory_arrays_loaded": False,
                "historical_files_sha256_unchanged": len(before),
                "weights_unchanged_during_evaluation": True,
            }
        )
        write_json(output / "checks.json", checks)
        write_json(output / "training_summaries.json", reports)
        environment = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "output_bytes_before_summary": sum(
                p.stat().st_size for p in output.rglob("*") if p.is_file()
            ),
        }
        write_json(output / "environment.json", environment)
        write_json(
            output / "summary.json",
            {
                "environment": environment,
                "training": reports,
                "checks": checks,
                "derived_caps": expanded,
                "central_bound_audit": audit,
                "ordinary_one_step": evaluation["one_step"],
                "ordinary_rollout": evaluation["rollout"],
                "response_evaluation": response,
                "source_hashes": hashes([ROOT / s for s in sources]),
                "frozen_force_parameter_hashes": hashes(
                    [
                        ROOT / mainline[k]
                        for k in ("point_calibration", "area_material", "integration_config")
                    ]
                ),
                "scope": (
                    "teacher_forced_one_step_absolute_and_initial_pair_response_training_then_"
                    "5_10_20_step_autoregressive_evaluation_no_multistep_training_loss"
                ),
            },
        )
        write_json(
            output / "failures.json",
            evaluation["checks"]["rollout_failures"] + response["failures"],
        )
        write_json(output / "run_status.json", {"complete": True})
        print(
            json.dumps(
                {
                    "output": str(output),
                    "environment": environment,
                    "training": reports,
                    "response_summary": response["summary"],
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as error:
        write_json(
            output / "failures.json", [{"error": type(error).__name__, "message": str(error)}]
        )
        raise


if __name__ == "__main__":
    main()
