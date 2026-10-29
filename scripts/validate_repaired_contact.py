"""Frozen M/F/G mainline, separate area adapter, short rollouts and interventions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from audit_contact_mechanism import force_geometry_audit, run_interventions
from joint_state_evaluation import evaluate_models
from train_joint_state import load_data
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.joint_state_dynamics import JointStateDynamics
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline
from contact_gwm.repaired_contact_pipeline import (
    CONTACT_METHODS,
    METHODS,
    RepairedContactConditions,
    repaired_pipelines,
)

ROOT = Path(__file__).resolve().parents[1]


def predictors_for(legacy, pipes):
    return {
        **{name: pipe.step for name, pipe in pipes.items()},
        "constant_velocity": lambda state, robot: legacy.step(state, robot, "constant_velocity"),
    }


def forbidden(*a, **kw):
    raise AssertionError("Forbidden native force, geometry or integration call")


def load_model(artifact, mesh):
    model = JointStateDynamics(
        torch.tensor(mesh["rest_positions_world"]).float(),
        torch.tensor(mesh["pinned_mask"]),
        **artifact["architecture"],
    )
    model.load_state_dict(artifact["model_state"], strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def runtime_contracts(legacy, model, conditions, pipes, mesh, trajectories, output):
    arrays = trajectories["left_deep"]
    initial = legacy.gaussian_state(
        arrays["node_position_world"][70], arrays["node_velocity_world"][70]
    )
    external = [IndenterInput.from_arrays(arrays, f) for f in range(70, 90)]
    artifact = torch.load(output / "frozen_model.pt", map_location="cpu", weights_only=False)
    reload = repaired_pipelines(legacy, load_model(artifact, mesh), conditions)
    poisoned = {
        key: np.full_like(value, np.nan, dtype=float)
        if key.startswith(("node_", "contact_", "net_contact_")) or key == "integration_state"
        else value.copy()
        for key, value in arrays.items()
    }
    alternate = [IndenterInput.from_arrays(poisoned, f) for f in range(70, 90)]
    for name, pipe in pipes.items():
        a, b, c = (
            pipe.rollout(initial, external),
            reload[name].rollout(initial, external),
            pipe.rollout(initial, alternate),
        )
        assert all(
            torch.equal(x.state.centers, y.state.centers)
            and torch.equal(x.state.velocities, y.state.velocities)
            for other in (b, c)
            for x, y in zip(a, other, strict=True)
        )
    for name in ("analytic_force", "gaussian_geometry", "area_geometry", "sparse_geometry"):
        with (
            patch.object(mujoco, "mj_forward", forbidden),
            patch.object(mujoco, "mj_contactForce", forbidden),
            patch.object(mujoco, "mj_step", forbidden),
        ):
            pipes[name].step(initial, external[0])
            if name != "analytic_force":
                with (
                    patch.object(legacy.native, "query", forbidden),
                    patch.object(mujoco, "mj_fwdPosition", forbidden),
                ):
                    pipes[name].step(initial, external[0])
    # The extended per-face exports must not change the previously calibrated
    # TOTAL elastic basis. Compare the same current stored frame, not a label.
    cached = read_npz(ROOT / "outputs/elastic_area_calibration/run_002/force_predictions.npz")
    # Calibration order is central shallow, central deep, left shallow, left deep, control.
    reference = 3 * 300 + 70
    _, force = conditions.condition(initial, external[0], "area_geometry")
    np.testing.assert_allclose(
        force.forces.sum(0).numpy(), cached["area_force_N"][reference], rtol=1e-12, atol=1e-15
    )
    return {
        "reload_all_network_conditions_20_step_bitwise_equal": True,
        "future_soft_contact_force_label_poisoning_no_effect": True,
        "analytic_branches_native_force_calls_forbidden": True,
        "surface_area_sparse_native_geometry_calls_forbidden": True,
        "area_total_force_matches_frozen_calibration_frame": True,
        "primary_F_G_identical_point_force_implementation_and_parameters": True,
        "area_adapter": "one_resultant_per_original_face_unchanged_normalized_mean_encoder",
        "area_adapter_invariant_to_mesh_repartition": False,
        "neural_architecture_or_state_update_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/validate_repaired_contact.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["seed"] != 7 or config["dt"] != 0.01 or tuple(config["methods"]) != METHODS:
        raise ValueError("Keep the frozen seven-method protocol")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    data = ROOT / config["data_directory"]
    before = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
    output = args.output
    if output is None:
        parent, index = ROOT / config["output_root"], 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        output = parent / f"run_{index:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve existing outputs: {output}")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    started = time.perf_counter()
    write_json(output / "run_status.json", {"complete": False})
    write_json(output / "historical_hashes.json", before)
    sources = [
        "configs/validate_repaired_contact.yaml",
        "scripts/validate_repaired_contact.py",
        "src/contact_gwm/repaired_contact_pipeline.py",
        "src/contact_gwm/contact_integration.py",
        "src/contact_gwm/joint_state_dynamics.py",
        "src/contact_gwm/deformation.py",
        "src/contact_gwm/continuous_contact_surface.py",
        "scripts/joint_state_evaluation.py",
        "scripts/audit_contact_mechanism.py",
        "tests/test_repaired_contact_pipeline.py",
    ]
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        mesh, trajectories, splits = load_data(data, config)
        legacy_config = yaml.safe_load((ROOT / config["legacy_pipeline_config"]).read_text())
        legacy = PinnedContactPipeline(
            mesh, data / "scene.xml", ROOT / legacy_config["checkpoint"], legacy_config
        )
        checkpoint = ROOT / config["checkpoint"]
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert artifact["training_pairs"] == 600 and np.array_equal(
            artifact["node_ids"], mesh["node_ids"]
        )
        model = load_model(artifact, mesh)
        digest = tensor_digest(model.state_dict())
        shutil.copy2(checkpoint, output / "frozen_model.pt")
        point_fit = json.loads((ROOT / config["point_calibration"]).read_text())
        assert point_fit["training_trajectories"] == config["splits"]["train"]
        assert point_fit["unit"] == "N/m" and point_fit["fit_frames"] == 200
        area = ElasticAreaMaterial.load(ROOT / config["area_material"])
        integration = yaml.safe_load((ROOT / config["integration_config"]).read_text())[
            "integration"
        ]
        for key in ("point_calibration", "area_material", "integration_config"):
            shutil.copy2(ROOT / config[key], output / Path(config[key]).name)
        conditions = RepairedContactConditions(
            legacy, mesh, point_fit["stiffness_N_m"], area, integration
        )
        pipes = repaired_pipelines(legacy, model, conditions)
        predictors = predictors_for(legacy, pipes)
        provenance = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "selected_step": artifact["selected_step"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_updates_this_run": 0,
            "point_stiffness_N_m": point_fit["stiffness_N_m"],
            "area_stiffness_N_m3": area.stiffness_N_m3,
            "calibrated_this_run": False,
            "damping_and_friction": "zero_for_all_analytic_branches",
            "topology_assisted_surface": True,
            "test_trajectory_arrays_loaded": False,
            "normalization_network_and_independent_velocity_heads": "frozen_unchanged",
            "sources": hashes([ROOT / name for name in sources]),
            "calibration_hashes": hashes(
                [
                    ROOT / config[key]
                    for key in ("point_calibration", "area_material", "integration_config")
                ]
            ),
        }
        write_json(output / "provenance.json", provenance)
        contracts = runtime_contracts(legacy, model, conditions, pipes, mesh, trajectories, output)
        print(
            "Runtime reload, current-only, force/geometry permission contracts passed.", flush=True
        )
        with patch.object(mujoco, "mj_step", forbidden):
            evaluation = evaluate_models(
                predictors, legacy.gaussian_state, trajectories, splits, mesh, config, output
            )
            facade = SimpleNamespace(
                gaussian_state=legacy.gaussian_state,
                center_of_mass=legacy.center_of_mass,
                condition=conditions.condition,
            )
            force_metrics, force_checks = force_geometry_audit(
                facade, trajectories, splits, output, contact_methods=CONTACT_METHODS
            )
        old = read_npz(ROOT / config["historical_audit"] / "one_step_predictions.npz")
        new = read_npz(output / "one_step_predictions.npz")
        for name in trajectories:
            for field in ("next_position", "next_velocity"):
                assert np.array_equal(
                    new[f"{name}__mujoco_force__{field}"], old[f"{name}__mujoco_force__{field}"]
                )
        print(
            "Full native-route one-step historical reproduction and force audit passed.", flush=True
        )
        interventions = run_interventions(legacy, predictors, mesh, trajectories, config, output)
        assert tensor_digest(model.state_dict()) == digest
        assert before == {name: file_sha256(ROOT / name) for name in before}
        checks = {
            **evaluation["checks"],
            **contracts,
            **force_checks,
            "native_route_one_step_bitwise_equal_historical_audit": True,
            "frozen_weights_unchanged": True,
            "historical_files_sha256_unchanged": len(before),
            "test_trajectory_arrays_loaded": False,
        }
        write_json(output / "checks.json", checks)
        write_json(output / "area_integration_calls.json", conditions.integration_diagnostics)
        environment = {
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "cpu_threads": config["cpu_threads"],
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
                "provenance": provenance,
                "checks": checks,
                "one_step": evaluation["one_step"],
                "rollout": evaluation["rollout"],
                "force_geometry": force_metrics,
                "interventions": interventions,
                "scope": (
                    "frozen_network_diagnostic_prescribed_indenter_soft_state_autoregression_"
                    "not_fair_retraining_or_heldout_test"
                ),
            },
        )
        write_json(
            output / "failures.json",
            evaluation["checks"]["rollout_failures"] + interventions["failed_cases"],
        )
        write_json(output / "run_status.json", {"complete": True})
        print(
            json.dumps(
                {"output": str(output), "environment": environment, "checks": checks}, indent=2
            ),
            flush=True,
        )
    except BaseException as exc:
        write_json(output / "failures.json", [{"error": type(exc).__name__, "message": str(exc)}])
        raise


if __name__ == "__main__":
    main()
