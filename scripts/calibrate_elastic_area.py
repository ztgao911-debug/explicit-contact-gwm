"""Central-only elastic area-stiffness calibration and frozen left validation.

One coefficient, normal-force labels only, zero intercept/damping/friction.
Cached geometric integrals are frozen. This is force-model validation using
true current states, not dynamics training or a free rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from validate_adaptive_contact_integration import origin_for
from validate_contact_pipeline import hashes, write_csv, write_json
from validate_contact_surface import CURRENT_FIELDS, PHASES, current_inputs

from contact_gwm.contact_integration import ClippedSurfaceQuadrature, IntegrationSettings
from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.elastic_area import ElasticAreaMaterial, fit_elastic_area_stiffness

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("zero_force", "historical_point_elastic", "matched_point_elastic", "area_elastic")


def cached_bases(directory, names):
    # Only basis columns are extracted; previous score/reference columns are
    # never passed to either calibration or prediction.
    archive = read_npz(directory / "trajectory_integrals.npz")["values"]
    result = {}
    with (directory / "trajectory_per_case.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["trajectory"] not in names:
                continue
            case = int(row["case"])
            record = archive[case]
            assert int(record[0]) == case
            result[row["trajectory"], int(row["frame"])] = {
                "case": case,
                "split": row["split"],
                "phase": row["phase"],
                "elastic_vector_m3": record[3:6].copy(),
                "elastic_torque_m4": record[9:12].copy(),
                "area_m2": float(record[1]),
            }
    return result


def force_targets(arrays, frame, origin):
    first, last = arrays["contact_offsets"][frame : frame + 2]
    all_ids = np.arange(first, last)
    selected = all_ids[arrays["contact_signed_distance"][all_ids, 0] <= 0]
    n = len(selected)
    sign = np.where(arrays["contact_flex_ids"][selected, 1] == 0, 1.0, -1.0)
    normals = arrays["contact_frame_world"][selected, :3] * sign[:, None]
    total = arrays["contact_force_on_block_world"][selected]
    magnitude = (total * normals).sum(-1)
    if n and (
        np.max(np.abs(np.linalg.norm(normals, axis=1) - 1)) > 1e-10 or magnitude.min() < -1e-10
    ):
        raise ValueError("Invalid world normal or native normal-force sign")
    normal = np.maximum(magnitude, 0)[:, None] * normals
    lever = arrays["contact_position_world"][selected] - origin
    local_moment = arrays["contact_wrench_local"][selected, 3:]
    rotation = arrays["contact_frame_world"][selected].reshape(-1, 3, 3)
    intrinsic = sign[:, None] * np.einsum("nki,nk->ni", rotation, local_moment)
    total_force = total.sum(0)
    stored_error = float(
        np.max(np.abs(total_force - arrays["net_contact_force_on_block_world"][frame]))
    )
    if stored_error > 1e-10:
        raise ValueError("Filtered contact force does not reproduce recorded net force")
    # These point baselines consume geometry only, not normal-force magnitudes.
    penetration = (-arrays["contact_signed_distance"][selected, 0]).clip(min=0)
    point_basis = penetration[:, None] * normals
    return {
        "normal_force_N": normal.sum(0),
        "total_force_N": total_force,
        "normal_torque_N_m": np.cross(lever, normal).sum(0),
        "total_torque_N_m": np.cross(lever, total).sum(0) + intrinsic.sum(0),
        "point_basis_m": point_basis.sum(0),
        "point_torque_basis_m2": np.cross(lever, point_basis).sum(0),
        "geometric_contacts": n,
        "stored_net_force_max_abs_error_N": stored_error,
        "max_intrinsic_contact_torque_N_m": float(np.abs(intrinsic).max()) if n else 0.0,
    }


def load_records(names, split, config, mesh, bases):
    records = []
    for name in names:
        fields = (
            "node_ids",
            "time",
            "phase_id",
            "node_position_world",
            "indenter_position_world",
            "contact_offsets",
            "contact_signed_distance",
            "contact_force_on_block_world",
            "contact_frame_world",
            "contact_flex_ids",
            "contact_position_world",
            "contact_wrench_local",
            "net_contact_force_on_block_world",
        )
        with np.load(ROOT / config["data_directory"] / f"{name}.npz", allow_pickle=False) as f:
            arrays = {k: f[k] for k in fields}
        if not np.array_equal(arrays["node_ids"], mesh["node_ids"]):
            raise ValueError("Node order changed")
        for frame in range(config["evaluation"]["frame_stop_exclusive"]):
            basis = bases[name, frame]
            phase = int(arrays["phase_id"][frame])
            assert basis["phase"] == PHASES[phase] and basis["split"] == split
            origin = origin_for(torch.tensor(arrays["node_position_world"][frame]), mesh).numpy()
            target = force_targets(arrays, frame, origin)
            records.append(
                {
                    "trajectory": name,
                    "split": split,
                    "frame": frame,
                    "time_s": float(arrays["time"][frame]),
                    "phase": PHASES[phase],
                    "phase_id": phase,
                    "origin_world_m": origin,
                    "robot_position_m": arrays["indenter_position_world"][frame],
                    **basis,
                    **target,
                }
            )
    return records


def predict(material, record):
    # This boundary accepts only current geometric bases, never targets.
    force, torque = material.predict(
        torch.tensor(record["elastic_vector_m3"]), torch.tensor(record["elastic_torque_m4"])
    )
    return force.numpy(), torque.numpy()


def fit_matched_point_reference(records):
    """Same central frames/objective, one coefficient with the correct N/m unit."""
    basis = np.stack([r["point_basis_m"] for r in records])
    target = np.stack([r["normal_force_N"] for r in records])
    scale = 1e-4
    design = basis / scale
    denominator = float(np.square(design).sum())
    if denominator <= 0:
        raise ValueError("No point-contact basis for matched comparison")
    coefficient = max(0.0, float((design * target).sum() / denominator)) / scale
    return coefficient, {
        "stiffness_N_m": coefficient,
        "unit": "N/m",
        "fit_frames": len(records),
        "objective": "equal_frame_weight_vector_squared_error",
        "intercept_N": 0.0,
        "normal_force_only": True,
        "torque_used_for_fitting": False,
        "vector_rmse_N": float(np.sqrt(np.square(coefficient * basis - target).sum(-1).mean())),
    }


def evaluate(records, material, old_k, matched_k, threshold):
    rows = []
    for record in records:
        predictions = {
            "zero_force": (np.zeros(3), np.zeros(3)),
            "historical_point_elastic": (
                old_k * record["point_basis_m"],
                old_k * record["point_torque_basis_m2"],
            ),
            "area_elastic": predict(material, record),
            "matched_point_elastic": (
                matched_k * record["point_basis_m"],
                matched_k * record["point_torque_basis_m2"],
            ),
        }
        for name, (force, torque) in predictions.items():
            if not np.isfinite(np.r_[force, torque]).all():
                raise FloatingPointError("Non-finite force/torque")
            rows.append(
                {
                    **{
                        k: record[k]
                        for k in ("trajectory", "split", "frame", "time_s", "phase", "case")
                    },
                    "method": name,
                    "geometric_contacts": record["geometric_contacts"],
                    "force_contact": bool(np.linalg.norm(record["normal_force_N"]) > threshold),
                    "predicted_force_contact": bool(np.linalg.norm(force) > threshold),
                    **{f"force_{a}_N": float(v) for a, v in zip("xyz", force, strict=True)},
                    **{f"torque_{a}_N_m": float(v) for a, v in zip("xyz", torque, strict=True)},
                    **{
                        f"true_normal_force_{a}_N": float(v)
                        for a, v in zip("xyz", record["normal_force_N"], strict=True)
                    },
                    **{
                        f"true_total_force_{a}_N": float(v)
                        for a, v in zip("xyz", record["total_force_N"], strict=True)
                    },
                    **{
                        f"true_normal_torque_{a}_N_m": float(v)
                        for a, v in zip("xyz", record["normal_torque_N_m"], strict=True)
                    },
                    **{
                        f"true_total_torque_{a}_N_m": float(v)
                        for a, v in zip("xyz", record["total_torque_N_m"], strict=True)
                    },
                    "normal_force_error_squared_N2": float(
                        np.square(force - record["normal_force_N"]).sum()
                    ),
                    "total_force_error_squared_N2": float(
                        np.square(force - record["total_force_N"]).sum()
                    ),
                    "normal_torque_error_squared_N2_m2": float(
                        np.square(torque - record["normal_torque_N_m"]).sum()
                    ),
                    "total_torque_error_squared_N2_m2": float(
                        np.square(torque - record["total_torque_N_m"]).sum()
                    ),
                    "true_normal_force_squared_N2": float(
                        np.square(record["normal_force_N"]).sum()
                    ),
                    "elastic_basis_norm_m3": float(np.linalg.norm(record["elastic_vector_m3"])),
                    "contact_area_m2": record["area_m2"],
                }
            )
    return rows


def summarize(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in fields), []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        contacts = [r for r in values if r["force_contact"]]
        result_row = {
            **dict(zip(fields, key, strict=True)),
            "frames": len(values),
            "force_contact_frames": len(contacts),
        }
        for label, selected in [("all", values), ("contact", contacts)]:
            for metric, source in [
                ("normal_force_rmse_N", "normal_force_error_squared_N2"),
                ("total_force_rmse_N", "total_force_error_squared_N2"),
                ("normal_torque_rmse_N_m", "normal_torque_error_squared_N2_m2"),
                ("total_torque_rmse_N_m", "total_torque_error_squared_N2_m2"),
            ]:
                result_row[f"{label}_{metric}"] = (
                    float(np.sqrt(np.mean([r[source] for r in selected]))) if selected else None
                )
            scale = sum(r["true_normal_force_squared_N2"] for r in selected)
            result_row[f"{label}_normal_force_relative_l2"] = (
                float(np.sqrt(sum(r["normal_force_error_squared_N2"] for r in selected) / scale))
                if scale > 0
                else None
            )
        result_row["force_contact_false_positive_frames"] = sum(
            not r["force_contact"] and r["predicted_force_contact"] for r in values
        )
        result_row["force_contact_false_negative_frames"] = sum(
            r["force_contact"] and not r["predicted_force_contact"] for r in values
        )
        result.append(result_row)
    return result


def replay_and_contracts(records, material, config, mesh, output, basis_config):
    integration = ClippedSurfaceQuadrature(
        mesh["rest_positions_world"],
        mesh["surface_triangles"],
        mesh["pinned_mask"],
        IntegrationSettings(**basis_config["integration"]),
    )
    reload = ElasticAreaMaterial.load(output / "material.json")
    errors = []
    for record in records:
        if record["frame"] not in config["replay_frames"]:
            continue
        with np.load(ROOT / config["data_directory"] / f"{record['trajectory']}.npz") as f:
            arrays = {k: f[k] for k in CURRENT_FIELDS}
        x, v, p, rv, _, _ = current_inputs(arrays, record["frame"])
        basis = integration.normal_bases(
            x,
            v,
            p,
            rv,
            basis_config["_base"]["sphere_radius_m"],
            basis_config["_base"]["skin_radius_m"],
            torch.tensor(record["origin_world_m"]),
        )
        assert np.array_equal(basis["elastic_vector_m3"].numpy(), record["elastic_vector_m3"])
        assert np.array_equal(basis["elastic_torque_m4"].numpy(), record["elastic_torque_m4"])
        assert basis["unresolved_area_intervals"] == basis["unresolved_elastic_intervals"] == 0
        a, b = predict(material, record), predict(reload, record)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        poisoned = {
            **record,
            "normal_force_N": np.full(3, np.nan),
            "total_force_N": np.full(3, np.nan),
            "future_position": np.full((len(mesh["node_ids"]), 3), np.nan),
        }
        c = predict(material, poisoned)
        assert all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))
        independent = np.cross(
            record["robot_position_m"] - record["origin_world_m"], record["elastic_vector_m3"]
        )
        errors.append(float(np.max(np.abs(independent - record["elastic_torque_m4"]))))
    return {
        "cached_geometric_basis_replay_bitwise_equal": True,
        "replayed_current_states": len(errors),
        "material_reload_predictions_bitwise_equal": True,
        "force_and_future_label_poisoning_invariant": True,
        "torque_basis_radial_identity_max_abs_m4": max(errors),
        "state_updated": False,
        "neural_weights_trained": False,
        "damping_or_friction_fitted": False,
        "test_trajectory_arrays_loaded": False,
    }


def plots(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, name in zip(
        axes.flat, ("center_shallow", "center_deep", "left_shallow", "left_deep"), strict=True
    ):
        selected = [r for r in rows if r["trajectory"] == name]
        truth = [r for r in selected if r["method"] == "area_elastic"]
        axis.plot(
            [r["time_s"] for r in truth],
            [r["true_normal_force_z_N"] for r in truth],
            "k-",
            label="native normal target",
        )
        for method in ("historical_point_elastic", "matched_point_elastic", "area_elastic"):
            records = [r for r in selected if r["method"] == method]
            axis.plot(
                [r["time_s"] for r in records], [r["force_z_N"] for r in records], label=method
            )
        axis.set(title=name, xlabel="Time (s)", ylabel="Force on block Z (N)")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "normal_force_curves.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, name in zip(axes, ("left_shallow", "left_deep"), strict=True):
        truth = [r for r in rows if r["trajectory"] == name and r["method"] == "area_elastic"]
        axis.plot(
            [r["time_s"] for r in truth],
            [r["true_normal_torque_y_N_m"] * 1e3 for r in truth],
            "k-",
            label="native normal torque",
        )
        axis.plot(
            [r["time_s"] for r in truth],
            [r["true_total_torque_y_N_m"] * 1e3 for r in truth],
            "k:",
            label="native total torque",
        )
        for method in ("historical_point_elastic", "matched_point_elastic", "area_elastic"):
            records = [r for r in rows if r["trajectory"] == name and r["method"] == method]
            axis.plot(
                [r["time_s"] for r in records],
                [r["torque_y_N_m"] * 1e3 for r in records],
                label=method,
            )
        axis.set(title=name, xlabel="Time (s)", ylabel="Torque Y (N mm)")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "validation_torque_curves.png", dpi=160)
    plt.close(figure)


def forbidden(*a, **kw):
    raise AssertionError("Calibration used a native solver/integrator instead of recorded labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/calibrate_elastic_area.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (
        config["seed"] != 7
        or config["fit"]["phase_ids"] != [1, 2]
        or config["fit"]["intercept"]
        or not config["fit"]["nonnegative"]
        or config["fit"]["parameters"] != ["elastic_area_stiffness_N_m3"]
        or tuple(config["evaluation"]["methods"]) != METHODS
    ):
        raise ValueError("Keep the fixed single-parameter calibration protocol")
    if config["splits"]["train"] != ["center_shallow", "center_deep"] or config["splits"][
        "validation"
    ] != ["left_shallow", "left_deep"]:
        raise ValueError("Keep whole-trajectory splits")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    data = ROOT / config["data_directory"]
    basis_directory = ROOT / config["basis_run"]
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
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    sources = [
        "scripts/calibrate_elastic_area.py",
        "src/contact_gwm/elastic_area.py",
        "configs/calibrate_elastic_area.yaml",
        "tests/test_elastic_area.py",
        "src/contact_gwm/contact_integration.py",
    ]
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    try:
        mesh = read_npz(data / "mesh.npz")
        basis_config = yaml.safe_load((basis_directory / "config.yaml").read_text())
        basis_config["_base"] = yaml.safe_load((basis_directory / "base_config.yaml").read_text())
        train_names = config["splits"]["train"]
        train = load_records(
            train_names, "train", config, mesh, cached_bases(basis_directory, train_names)
        )
        fit_records = [r for r in train if r["phase_id"] in config["fit"]["phase_ids"]]
        model, fit = fit_elastic_area_stiffness(
            np.stack([r["elastic_vector_m3"] for r in fit_records]),
            np.stack([r["normal_force_N"] for r in fit_records]),
            numerical_basis_scale_m3=config["fit"]["numerical_basis_scale_m3"],
        )
        model.save(output / "material.json")
        frozen_hash = file_sha256(output / "material.json")
        sample_rows = [
            {
                **{k: r[k] for k in ("trajectory", "frame", "time_s", "phase", "case")},
                "basis_norm_m3": float(np.linalg.norm(r["elastic_vector_m3"])),
                "normal_force_norm_N": float(np.linalg.norm(r["normal_force_N"])),
            }
            for r in fit_records
        ]
        write_csv(output / "fit_samples.csv", sample_rows)
        fit.update(
            {
                "stiffness_N_m3": model.stiffness_N_m3,
                "unit": "N/m^3",
                "training_trajectories": train_names,
                "fit_phase_ids": config["fit"]["phase_ids"],
                "objective": config["fit"]["objective"],
                "target": config["fit"]["target"],
                "torque_used_for_fitting": False,
                "validation_targets_loaded_before_freezing": False,
            }
        )
        write_json(output / "calibration.json", fit)
        matched_k, matched_fit = fit_matched_point_reference(fit_records)
        matched_fit["training_trajectories"] = train_names
        matched_fit["fit_phase_ids"] = config["fit"]["phase_ids"]
        matched_fit["validation_targets_loaded_before_freezing"] = False
        write_json(output / "matched_point_calibration.json", matched_fit)
        print(
            f"Central fit frozen: k_A={model.stiffness_N_m3:.9g} N/m^3; loading left validation.",
            flush=True,
        )
        records = list(train)
        for split in ("validation", "control"):
            names = config["splits"][split]
            records.extend(
                load_records(names, split, config, mesh, cached_bases(basis_directory, names))
            )
        frozen = ElasticAreaMaterial.load(output / "material.json")
        old = json.loads((ROOT / config["old_point_calibration"]).read_text())
        old_k = float(old["material"]["normal_stiffness"])
        with (
            patch.object(mujoco, "mj_fwdPosition", forbidden),
            patch.object(mujoco, "mj_forward", forbidden),
            patch.object(mujoco, "mj_contactForce", forbidden),
            patch.object(mujoco, "mj_step", forbidden),
        ):
            rows = evaluate(
                records, frozen, old_k, matched_k, config["evaluation"]["force_contact_threshold_N"]
            )
            contracts = replay_and_contracts(records, frozen, config, mesh, output, basis_config)
        write_csv(output / "per_frame_metrics.csv", rows)
        split = summarize(rows, ("method", "split"))
        write_csv(output / "split_metrics.csv", split)
        write_csv(output / "phase_metrics.csv", summarize(rows, ("method", "split", "phase")))
        write_csv(
            output / "trajectory_metrics.csv", summarize(rows, ("method", "trajectory", "split"))
        )
        fit_keys = {(r["trajectory"], r["frame"]) for r in fit_records}
        write_csv(
            output / "calibration_frame_metrics.csv",
            summarize(
                [r for r in rows if (r["trajectory"], r["frame"]) in fit_keys], ("method", "split")
            ),
        )
        np.savez_compressed(
            output / "force_predictions.npz",
            case=np.array([r["case"] for r in records]),
            elastic_basis_m3=np.stack([r["elastic_vector_m3"] for r in records]),
            torque_basis_m4=np.stack([r["elastic_torque_m4"] for r in records]),
            origin_world_m=np.stack([r["origin_world_m"] for r in records]),
            target_normal_force_N=np.stack([r["normal_force_N"] for r in records]),
            target_total_force_N=np.stack([r["total_force_N"] for r in records]),
            target_normal_torque_N_m=np.stack([r["normal_torque_N_m"] for r in records]),
            target_total_torque_N_m=np.stack([r["total_torque_N_m"] for r in records]),
            area_force_N=np.stack([predict(frozen, r)[0] for r in records]),
            area_torque_N_m=np.stack([predict(frozen, r)[1] for r in records]),
            point_basis_m=np.stack([r["point_basis_m"] for r in records]),
            point_torque_basis_m2=np.stack([r["point_torque_basis_m2"] for r in records]),
            historical_point_force_N=np.stack([old_k * r["point_basis_m"] for r in records]),
            historical_point_torque_N_m=np.stack(
                [old_k * r["point_torque_basis_m2"] for r in records]
            ),
            matched_point_force_N=np.stack([matched_k * r["point_basis_m"] for r in records]),
            matched_point_torque_N_m=np.stack(
                [matched_k * r["point_torque_basis_m2"] for r in records]
            ),
        )
        plots(rows, output)
        assert file_sha256(output / "material.json") == frozen_hash
        assert before == {name: file_sha256(ROOT / name) for name in before}
        with np.load(output / "force_predictions.npz") as f:
            assert all(np.isfinite(f[k]).all() for k in f.files)
        contracts.update(
            {
                "validation_targets_loaded_after_model_frozen": True,
                "stiffness_unchanged_after_validation": True,
                "historical_files_sha256_unchanged": len(before),
                "stored_total_force_max_abs_error_N": max(
                    r["stored_net_force_max_abs_error_N"] for r in records
                ),
                "max_intrinsic_contact_torque_N_m": max(
                    r["max_intrinsic_contact_torque_N_m"] for r in records
                ),
            }
        )
        environment = {
            "experiment_date": config["experiment_date"],
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cpu_threads": config["cpu_threads"],
            "elapsed_seconds": time.perf_counter() - started,
            "output_bytes_before_summary": sum(
                f.stat().st_size for f in output.rglob("*") if f.is_file()
            ),
        }
        summary = {
            "environment": environment,
            "calibration": fit,
            "split_metrics": split,
            "checks": contracts,
            "old_point_stiffness_N_m": old_k,
            "matched_point_calibration": matched_fit,
            "source_hashes": hashes([ROOT / name for name in sources]),
            "basis_source_hashes": hashes(
                [
                    basis_directory / "trajectory_integrals.npz",
                    basis_directory / "config.yaml",
                    data / "mesh.npz",
                ]
            ),
            "material_sha256": frozen_hash,
            "scope": "one_scalar_effective_normal_force_model_true_current_states_no_rollout",
        }
        write_json(output / "summary.json", summary)
        write_json(output / "checks.json", contracts)
        write_json(output / "environment.json", environment)
        write_json(output / "failures.json", [])
        write_json(output / "run_status.json", {"complete": True})
        print(
            json.dumps(
                {
                    "output": str(output),
                    "calibration": fit,
                    "validation": [r for r in split if r["split"] == "validation"],
                    "elapsed_seconds": environment["elapsed_seconds"],
                },
                indent=2,
            )
        )
    except BaseException as error:
        write_json(
            output / "failures.json", [{"error": type(error).__name__, "message": str(error)}]
        )
        raise


if __name__ == "__main__":
    main()
