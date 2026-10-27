"""Frozen full-trajectory joint model: contact replacements and native interventions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from joint_state_evaluation import PHASES, evaluate_models
from train_joint_state import load_data
from validate_contact_pipeline import hashes, reference_masks, write_csv, write_json

from contact_gwm.contact_audit import (
    NativeContactIntervention,
    intervention_robot,
    signed_response_metrics,
)
from contact_gwm.contact_force import ContactMaterial
from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.data.mujoco_soft import tetra_volumes
from contact_gwm.joint_state_dynamics import JointStateDynamics, JointStatePipeline
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline

ROOT = Path(__file__).resolve().parents[1]
CONTACT_METHODS = ("mujoco_force", "analytic_force", "gaussian_geometry")
METHODS = (*CONTACT_METHODS, "internal_only", "constant_velocity")


def make_pipelines(legacy, model):
    return {
        method: JointStatePipeline(legacy, model, contact_method=condition)
        for method, condition in (
            *((name, name) for name in CONTACT_METHODS),
            ("internal_only", "none"),
        )
    }


def make_predictors(legacy, pipelines):
    return {
        **{name: pipeline.step for name, pipeline in pipelines.items()},
        "constant_velocity": lambda state, robot: legacy.step(state, robot, "constant_velocity"),
    }


def summarize_force_geometry(rows):
    groups = defaultdict(list)
    for row in rows:
        phases = [row["phase"], "all"]
        if row["phase"] in ("press", "hold"):
            phases.append("press_hold")
        for phase in phases:
            groups[row["method"], row["split"], phase].append(row)
    result = []
    for (method, split, phase), values in sorted(groups.items()):
        contact = [r for r in values if r["true_force_contact"]]
        matched = [r for r in values if r["contact_position_error_mm"] is not None]
        summary = {
            "method": method,
            "split": split,
            "phase": phase,
            "frames": len(values),
            "zero_force_baseline_rmse_N": float(
                np.sqrt(
                    np.mean([sum(r[f"true_force_{axis}_N"] ** 2 for axis in "xyz") for r in values])
                )
            ),
            "zero_force_contact_frames_rmse_N": float(
                np.sqrt(
                    np.mean(
                        [sum(r[f"true_force_{axis}_N"] ** 2 for axis in "xyz") for r in contact]
                    )
                )
            )
            if contact
            else None,
            "net_force_rmse_N": float(
                np.sqrt(np.mean([r["force_error_squared_N2"] for r in values]))
            ),
            "net_force_contact_frames_rmse_N": float(
                np.sqrt(np.mean([r["force_error_squared_N2"] for r in contact]))
            )
            if contact
            else None,
            "net_torque_rmse_N_m": float(
                np.sqrt(np.mean([r["torque_error_squared_N2_m2"] for r in values]))
            ),
            "matched_geometry_frames": len(matched),
            "matched_only_mean_contact_position_error_mm": float(
                np.mean([r["contact_position_error_mm"] for r in matched])
            )
            if matched
            else None,
            "matched_only_mean_normal_angle_deg": float(
                np.mean([r["normal_angle_error_deg"] for r in matched])
            )
            if matched
            else None,
        }
        for label, reference, prediction in (
            ("geometry", "true_geometric_contact", "predicted_contact"),
            ("force_frame", "true_force_contact", "predicted_contact"),
        ):
            tp = sum(r[reference] and r[prediction] for r in values)
            fp = sum(not r[reference] and r[prediction] for r in values)
            fn = sum(r[reference] and not r[prediction] for r in values)
            summary.update(
                {
                    f"{label}_tp": tp,
                    f"{label}_fp": fp,
                    f"{label}_fn": fn,
                    f"{label}_recall": tp / (tp + fn) if tp + fn else None,
                    f"{label}_precision": tp / (tp + fp) if tp + fp else None,
                }
            )
        result.append(summary)
    return result


@torch.no_grad()
def force_geometry_audit(legacy, trajectories, splits, output):
    rows, events = [], []
    min_normal, torque_error = 0.0, 0.0
    for name, arrays in trajectories.items():
        per_method = defaultdict(list)
        for frame in range(len(arrays["time"]) - 1):
            state = legacy.gaussian_state(
                arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
            )
            robot = IndenterInput.from_arrays(arrays, frame)
            a, b = arrays["contact_offsets"][frame : frame + 2]
            ids = np.arange(a, b)
            ids = ids[arrays["contact_signed_distance"][ids, 0] <= 0]
            force_ids = ids[
                np.linalg.norm(arrays["contact_force_on_block_world"][ids], axis=1) > 1e-8
            ]
            true_points = arrays["contact_position_world"][ids]
            sign = np.where(arrays["contact_flex_ids"][ids, 1] == 0, 1.0, -1.0)
            true_normals = arrays["contact_frame_world"][ids, :3] * sign[:, None]
            true_force = arrays["contact_force_on_block_world"][force_ids].sum(0)
            com = legacy.center_of_mass(state).numpy()
            true_torque = np.cross(
                arrays["contact_position_world"][force_ids] - com,
                arrays["contact_force_on_block_world"][force_ids],
            ).sum(0)
            native_geometry = None
            for method in CONTACT_METHODS:
                contacts, force = legacy.condition(state, robot, method)
                if method == "mujoco_force":
                    native_geometry = contacts
                elif method == "analytic_force":
                    assert all(
                        torch.equal(getattr(contacts, key), getattr(native_geometry, key))
                        for key in (
                            "points",
                            "normals",
                            "signed_gaps",
                            "relative_velocities",
                            "object_indices",
                        )
                    )
                net, torque = force.forces.sum(0).numpy(), force.torques.sum(0).numpy()
                point_error, normal_error = None, None
                if contacts.num_contacts and len(ids):
                    distance = np.linalg.norm(
                        contacts.points.numpy()[:, None] - true_points, axis=-1
                    )
                    point_error = float(
                        0.5 * (distance.min(0).mean() + distance.min(1).mean()) * 1e3
                    )
                    nearest = distance.argmin(1)
                    cosine = (contacts.normals.numpy() * true_normals[nearest]).sum(-1)
                    normal_error = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())
                if contacts.num_contacts:
                    min_normal = min(
                        min_normal, float((force.forces * contacts.normals).sum(-1).min())
                    )
                    torque_error = max(
                        torque_error,
                        float(
                            (
                                torch.cross(
                                    contacts.points - torch.tensor(com), force.forces, dim=-1
                                )
                                - force.torques
                            )
                            .abs()
                            .max()
                        ),
                    )
                row = {
                    "method": method,
                    "split": splits[name],
                    "trajectory": name,
                    "frame": frame,
                    "phase": PHASES[int(arrays["phase_id"][frame])],
                    "true_geometric_contact": bool(len(ids)),
                    "true_force_contact": bool(len(force_ids)),
                    "predicted_contact": bool(contacts.num_contacts),
                    "predicted_contact_count": contacts.num_contacts,
                    "predicted_nonzero_force_contact_count": int(
                        (torch.linalg.vector_norm(force.forces, dim=-1) > 1e-8).sum()
                    ),
                    "force_error_squared_N2": float(np.square(net - true_force).sum()),
                    "torque_error_squared_N2_m2": float(np.square(torque - true_torque).sum()),
                    "contact_position_error_mm": point_error,
                    "normal_angle_error_deg": normal_error,
                    **{f"force_{axis}_N": float(v) for axis, v in zip("xyz", net, strict=True)},
                    **{
                        f"true_force_{axis}_N": float(v)
                        for axis, v in zip("xyz", true_force, strict=True)
                    },
                }
                rows.append(row)
                per_method[method].append(row)
        for method, values in per_method.items():
            truth = [r["frame"] for r in values if r["true_geometric_contact"]]
            predicted = [r["frame"] for r in values if r["predicted_contact"]]
            events.append(
                {
                    "trajectory": name,
                    "split": splits[name],
                    "method": method,
                    "true_first_frame": truth[0] if truth else None,
                    "predicted_first_frame": predicted[0] if predicted else None,
                    "true_last_frame": truth[-1] if truth else None,
                    "predicted_last_frame": predicted[-1] if predicted else None,
                    "first_contact_error_s": (predicted[0] - truth[0]) * 0.01
                    if truth and predicted
                    else None,
                    "last_contact_error_s": (predicted[-1] - truth[-1]) * 0.01
                    if truth and predicted
                    else None,
                }
            )
    summaries = summarize_force_geometry(rows)
    write_csv(output / "force_geometry_per_frame.csv", rows)
    write_csv(output / "force_geometry_metrics.csv", summaries)
    write_csv(output / "contact_event_times.csv", events)
    return summaries, {
        "native_and_analytic_geometry_bitwise_equal": True,
        "minimum_force_dot_normal_N": min_normal,
        "torque_reference_identity_max_error_N_m": torque_error,
    }


@torch.no_grad()
def intervention_predictions(initial, external, predictors, mesh):
    """Only initial soft state + known external robot inputs cross this boundary."""
    saved, checks = {}, {"max_fixed_position_or_velocity_error": 0.0, "inverted_recorded_states": 0}
    for method, predict in predictors.items():
        state = initial
        x, v, force, contact = [state.centers.numpy()], [state.velocities.numpy()], [], []
        for robot in external:
            result = predict(state, robot)
            state = result.state
            x.append(state.centers.numpy())
            v.append(state.velocities.numpy())
            force.append(result.forces.forces.sum(0).numpy())
            contact.append(result.contacts.num_contacts)
        x, v = np.asarray(x), np.asarray(v)
        if not np.isfinite([x, v]).all() or not np.isfinite(force).all():
            raise FloatingPointError("Non-finite intervention prediction")
        pin = mesh["pinned_mask"]
        checks["max_fixed_position_or_velocity_error"] = max(
            checks["max_fixed_position_or_velocity_error"],
            float(np.abs(x[:, pin] - mesh["rest_positions_world"][pin]).max()),
            float(np.abs(v[:, pin]).max()),
        )
        checks["inverted_recorded_states"] += int(
            np.any(
                tetra_volumes(x, mesh["tetrahedra"]) / mesh["rest_signed_tetra_volumes"] <= 0,
                axis=-1,
            ).sum()
        )
        saved.update(
            {
                f"{method}__position": x,
                f"{method}__velocity": v,
                f"{method}__force": np.asarray(force),
                f"{method}__contact_count": np.asarray(contact),
            }
        )
    return saved, checks


def external_inputs(native):
    return [
        IndenterInput(p.copy(), v.copy(), native["command_position"].copy(), np.zeros(3))
        for p, v in zip(native["robot_position"][:-1], native["robot_velocity"][:-1], strict=True)
    ]


def run_interventions(legacy, predictors, mesh, trajectories, config, output):
    settings = config["interventions"]
    arrays = trajectories[settings["trajectory"]]
    folder = output / "interventions"
    folder.mkdir()
    solver = NativeContactIntervention(ROOT / config["data_directory"] / "scene.xml", mesh)
    reference_rois = reference_masks(arrays, mesh, config["contact_neighborhood_radius_m"])
    all_checks, response_rows, absolute_rows, initial_rows = [], [], [], []
    failures, total_steps, representative = [], 0, {}
    base_repeated = False
    for frame in settings["start_frames"]:
        native_cases, predictions = {}, {}
        for variant in settings["variants"]:
            name = variant["name"]
            robot = intervention_robot(IndenterInput.from_arrays(arrays, frame), variant)
            # No trajectory is provided to this solver after initialization.
            try:
                native, native_checks = solver.rollout(
                    arrays["integration_state"][frame].copy(),
                    robot,
                    settings["steps"],
                    config["dt"],
                )
                total_steps += native_checks["native_internal_steps"]
                assert np.array_equal(native["position"][0], arrays["node_position_world"][frame])
                assert np.array_equal(native["velocity"][0], arrays["node_velocity_world"][frame])
                native_cases[name] = native
                initial = legacy.gaussian_state(native["position"][0], native["velocity"][0])
                predicted, prediction_checks = intervention_predictions(
                    initial, external_inputs(native), predictors, mesh
                )
                predictions[name] = predicted
                np.savez_compressed(folder / f"{frame}_{name}_native.npz", **native)
                np.savez_compressed(
                    folder / f"{frame}_{name}_predictions.npz",
                    node_ids=mesh["node_ids"],
                    **predicted,
                )
                reloaded_native = read_npz(folder / f"{frame}_{name}_native.npz")
                reloaded_predictions = read_npz(folder / f"{frame}_{name}_predictions.npz")
                assert all(
                    np.array_equal(value, reloaded_native[key]) for key, value in native.items()
                )
                assert all(
                    np.array_equal(value, reloaded_predictions[key])
                    for key, value in predicted.items()
                )
                all_checks.append(
                    {
                        "frame": frame,
                        "variant": name,
                        "native": native_checks,
                        "predicted": prediction_checks,
                    }
                )
                a, b = native["contact_offsets"][:2]
                ids = np.arange(a, b)
                ids = ids[native["contact_signed_distance"][ids, 0] <= 0]
                initial_rows.append(
                    {
                        "frame": frame,
                        "source_phase": PHASES[int(arrays["phase_id"][frame])],
                        "variant": name,
                        "initial_native_geometric_contacts": len(ids),
                        "initial_native_force_N": float(np.linalg.norm(native["net_force"][0])),
                        **{
                            f"robot_{axis}_m": float(value)
                            for axis, value in zip("xyz", robot.position, strict=True)
                        },
                    }
                )
                if frame == settings["start_frames"][0] and name == "base":
                    repeated, _ = solver.rollout(
                        arrays["integration_state"][frame].copy(),
                        robot,
                        settings["steps"],
                        config["dt"],
                    )
                    total_steps += settings["steps"] * round(
                        config["dt"] / solver.model.opt.timestep
                    )
                    assert all(np.array_equal(native[key], repeated[key]) for key in native)
                    base_repeated = True
            except (RuntimeError, ValueError, AssertionError, FloatingPointError) as exc:
                failures.append({"frame": frame, "variant": name, "error": str(exc)})
        # Failures are explicit; do not silently change which variants are compared.
        if len(native_cases) != len(settings["variants"]) or len(predictions) != len(
            settings["variants"]
        ):
            continue
        roi = reference_rois[frame].copy()
        for native in native_cases.values():
            a, b = native["contact_offsets"][:2]
            points = native["contact_position_world"][a:b]
            if len(points):
                roi |= (
                    np.linalg.norm(
                        arrays["node_position_world"][frame, :, None] - points, axis=-1
                    ).min(-1)
                    <= config["contact_neighborhood_radius_m"]
                )
        roi &= ~mesh["pinned_mask"]
        np.savez_compressed(
            folder / f"{frame}_evaluation_masks.npz",
            node_ids=mesh["node_ids"],
            active=~mesh["pinned_mask"],
            contact_neighborhood=roi,
        )
        baseline = native_cases["base"]
        for variant, native in native_cases.items():
            for method in METHODS:
                x, v = (
                    predictions[variant][f"{method}__position"],
                    predictions[variant][f"{method}__velocity"],
                )
                bx, bv = (
                    predictions["base"][f"{method}__position"],
                    predictions["base"][f"{method}__velocity"],
                )
                for h in settings["horizons"]:
                    for region, mask in (
                        ("active", ~mesh["pinned_mask"]),
                        ("contact_neighborhood", roi),
                    ):
                        metadata = {
                            "frame": frame,
                            "source_phase": PHASES[int(arrays["phase_id"][frame])],
                            "variant": variant,
                            "method": method,
                            "horizon": h,
                            "region": region,
                            "node_count": int(mask.sum()),
                        }
                        absolute_rows.append(
                            {
                                **metadata,
                                "position_rmse_um": float(
                                    np.sqrt(
                                        np.square(x[h, mask] - native["position"][h, mask])
                                        .sum(-1)
                                        .mean()
                                    )
                                    * 1e6
                                ),
                                "velocity_rmse_mm_s": float(
                                    np.sqrt(
                                        np.square(v[h, mask] - native["velocity"][h, mask])
                                        .sum(-1)
                                        .mean()
                                    )
                                    * 1e3
                                ),
                            }
                        )
                        if variant != "base":
                            row = dict(metadata)
                            for prefix, pred, truth, scale in (
                                (
                                    "position",
                                    x[h] - bx[h],
                                    native["position"][h] - baseline["position"][h],
                                    1e6,
                                ),
                                (
                                    "velocity",
                                    v[h] - bv[h],
                                    native["velocity"][h] - baseline["velocity"][h],
                                    1e3,
                                ),
                            ):
                                row.update(
                                    {
                                        f"{prefix}_{key}": value
                                        for key, value in signed_response_metrics(
                                            pred, truth, mask, scale
                                        ).items()
                                    }
                                )
                            response_rows.append(row)
                if method in ("internal_only", "constant_velocity"):
                    assert np.array_equal(x, bx) and np.array_equal(v, bv)
        if frame == settings["start_frames"][1]:
            representative = {
                "native": native_cases,
                "predictions": predictions,
                "roi": roi,
                "frame": frame,
                "rest": mesh["rest_positions_world"],
            }
        print(f"native interventions done: frame {frame}, {len(native_cases)} variants", flush=True)
    write_json(folder / "failures.json", failures)
    write_json(folder / "checks.json", all_checks)
    if not response_rows:
        raise RuntimeError("No complete physical-intervention comparison; see failures.json")
    write_csv(folder / "initial_conditions.csv", initial_rows)
    write_csv(folder / "absolute_metrics.csv", absolute_rows)
    write_csv(folder / "response_metrics.csv", response_rows)
    grouped = defaultdict(list)
    for row in response_rows:
        grouped[row["method"], row["horizon"], row["region"]].append(row)
    summary = []
    for (method, h, region), rows in sorted(grouped.items()):
        n = sum(r["node_count"] for r in rows)
        summary.append(
            {
                "method": method,
                "horizon": h,
                "region": region,
                "comparisons": len(rows),
                "position_response_rmse_um": float(
                    np.sqrt(
                        sum(r["node_count"] * r["position_response_rmse"] ** 2 for r in rows) / n
                    )
                ),
                "velocity_response_rmse_mm_s": float(
                    np.sqrt(
                        sum(r["node_count"] * r["velocity_response_rmse"] ** 2 for r in rows) / n
                    )
                ),
                "zero_response_position_rmse_um": float(
                    np.sqrt(
                        sum(
                            r["node_count"] * r["position_reference_response_rms"] ** 2
                            for r in rows
                        )
                        / n
                    )
                ),
                "zero_response_velocity_rmse_mm_s": float(
                    np.sqrt(
                        sum(
                            r["node_count"] * r["velocity_reference_response_rms"] ** 2
                            for r in rows
                        )
                        / n
                    )
                ),
            }
        )
    write_csv(folder / "response_summary.csv", summary)
    intervention_plots(response_rows, representative, settings, folder)
    return {
        "native_internal_steps_including_repeat": total_steps,
        "native_cases": len(all_checks),
        "failed_cases": failures,
        "base_repeat_bitwise_equal": base_repeated,
        "native_and_predicted_npz_reload_bitwise_equal": True,
        "same_initial_soft_states_all_cases": True,
        "internal_only_and_cv_response_bitwise_zero": True,
        "response_summary": summary,
        "max_fixed_position_or_velocity_error": max(
            max(
                c["native"]["max_fixed_position_or_velocity_error"],
                c["predicted"]["max_fixed_position_or_velocity_error"],
            )
            for c in all_checks
        ),
    }


def intervention_plots(rows, representative, settings, folder):
    os.environ.setdefault("MPLCONFIGDIR", str(folder.parent / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    variants = [v["name"] for v in settings["variants"] if v["name"] != "base"]
    for ax, key in zip(axes, ("position_response_rmse", "velocity_response_rmse"), strict=True):
        for method in METHODS:
            values = []
            for variant in variants:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["horizon"] == 1
                    and r["region"] == "active"
                    and r["variant"] == variant
                ]
                values.append(np.sqrt(np.mean([r[key] ** 2 for r in selected])))
            ax.plot(variants, values, marker="o", label=method)
        ax.set(title="One-step signed intervention response", ylabel=key)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.savefig(folder / "one_step_response_errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, prefix in zip(axes, ("position", "velocity"), strict=True):
        for method in CONTACT_METHODS:
            gains = []
            for variant in variants:
                selected = [
                    r[f"{prefix}_projection_gain"]
                    for r in rows
                    if r["method"] == method
                    and r["horizon"] == 1
                    and r["region"] == "active"
                    and r["variant"] == variant
                    and r[f"{prefix}_projection_gain"] is not None
                ]
                gains.append(float(np.mean(selected)) if selected else np.nan)
            ax.plot(variants, gains, marker="o", label=method)
        ax.axhline(1, color="k", linestyle="--", label="Native exact response")
        ax.axhline(0, color="grey", linestyle=":", label="Zero-response baselines")
        ax.set(
            title=f"Signed {prefix} response projection", ylabel="Projection onto native response"
        )
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.savefig(folder / "one_step_projection_gains.png", dpi=150)
    plt.close(fig)
    if representative:
        native, predicted = representative["native"], representative["predictions"]
        # Spatial first-step response to left/right relocation, independently scaled
        # only by a single common color limit for ALL methods and native labels.
        fields = {}
        for variant in ("shift_left", "shift_right"):
            fields[variant, "native"] = (
                native[variant]["position"][1] - native["base"]["position"][1]
            )[:, 2] * 1e6
            for method in CONTACT_METHODS:
                fields[variant, method] = (
                    predicted[variant][f"{method}__position"][1]
                    - predicted["base"][f"{method}__position"][1]
                )[:, 2] * 1e6
        limit = max(abs(value).max() for value in fields.values())
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
        xy = native["base"]["position"][0, :, :2]
        rest = representative["rest"]
        top = np.isclose(rest[:, 2], rest[:, 2].max())
        for row, variant in enumerate(("shift_left", "shift_right")):
            for column, method in enumerate(("native", *CONTACT_METHODS)):
                ax = axes[row, column]
                art = ax.scatter(
                    xy[top, 0] * 1e3,
                    xy[top, 1] * 1e3,
                    c=fields[variant, method][top],
                    vmin=-limit,
                    vmax=limit,
                    cmap="coolwarm",
                )
                ax.set(
                    title=f"{variant}: {method}", xlabel="x (mm)", ylabel="y (mm)", aspect="equal"
                )
        fig.colorbar(art, ax=axes, label="Signed z response vs base (um)")
        fig.savefig(folder / "relocation_response_maps.png", dpi=150)
        plt.close(fig)


def runtime_contracts(legacy, model, pipelines, mesh, trajectories, output):
    arrays = trajectories["left_deep"]
    initial = legacy.gaussian_state(
        arrays["node_position_world"][70], arrays["node_velocity_world"][70]
    )
    external = [IndenterInput.from_arrays(arrays, f) for f in range(70, 90)]
    copied = torch.load(output / "frozen_model.pt", map_location="cpu", weights_only=False)
    reloaded = JointStateDynamics(
        torch.tensor(mesh["rest_positions_world"]).float(),
        torch.tensor(mesh["pinned_mask"]),
        **copied["architecture"],
    )
    reloaded.load_state_dict(copied["model_state"], strict=True)
    reloaded.eval()
    reload_pipes = make_pipelines(legacy, reloaded)
    with torch.no_grad():
        for method in pipelines:
            a = pipelines[method].rollout(initial, external)
            b = reload_pipes[method].rollout(initial, external)
            assert all(
                torch.equal(x.state.centers, y.state.centers)
                and torch.equal(x.state.velocities, y.state.velocities)
                for x, y in zip(a, b, strict=True)
            )

        # Analytic branches may not consult the native FORCE solver.
        def forbidden(*args, **kwargs):
            raise AssertionError("Forbidden native force/geometry query")

        for method in ("analytic_force", "gaussian_geometry"):
            with (
                patch.object(mujoco, "mj_contactForce", forbidden),
                patch.object(mujoco, "mj_forward", forbidden),
            ):
                pipelines[method].step(initial, external[0])
        with patch.object(legacy.native, "query", forbidden):
            pipelines["gaussian_geometry"].step(initial, external[0])
            pipelines["internal_only"].step(initial, external[0])
        # All future soft/contact labels poisoned; only known robot arrays are read.
        poisoned = {
            key: np.full_like(value, np.nan, dtype=float)
            if key.startswith(("node_", "contact_")) or key == "integration_state"
            else value.copy()
            for key, value in arrays.items()
        }
        alternate = [IndenterInput.from_arrays(poisoned, f) for f in range(70, 90)]
        for pipeline in pipelines.values():
            a, b = pipeline.rollout(initial, external), pipeline.rollout(initial, alternate)
            assert all(
                torch.equal(x.state.centers, y.state.centers)
                and torch.equal(x.state.velocities, y.state.velocities)
                for x, y in zip(a, b, strict=True)
            )
    return {
        "reload_all_conditions_20_step_bitwise_equal": True,
        "analytic_and_gaussian_native_force_calls_forbidden": True,
        "gaussian_and_internal_only_native_geometry_calls_forbidden": True,
        "future_soft_label_poisoning_no_effect": True,
        "frozen_parameter_count": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/audit_contact_mechanism.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["seed"] != 7 or config["dt"] != 0.01 or tuple(config["methods"]) != METHODS:
        raise ValueError("Keep the fixed audit protocol")
    torch.set_num_threads(config["cpu_threads"])
    data = ROOT / config["data_directory"]
    before = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
    output = args.output
    if output is None:
        parent, index = ROOT / config["output_root"], 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        output = parent / f"run_{index:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve previous results: {output}")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    started = time.perf_counter()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for name in (
        "scripts/audit_contact_mechanism.py",
        "scripts/joint_state_evaluation.py",
        "src/contact_gwm/contact_audit.py",
        "src/contact_gwm/joint_state_dynamics.py",
        "src/contact_gwm/pinned_contact_pipeline.py",
        "configs/audit_contact_mechanism.yaml",
    ):
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    mesh, trajectories, splits = load_data(data, config)
    legacy_config = yaml.safe_load((ROOT / config["legacy_pipeline_config"]).read_text())
    legacy = PinnedContactPipeline(
        mesh, data / "scene.xml", ROOT / legacy_config["checkpoint"], legacy_config
    )
    calibration = json.loads((ROOT / config["calibration"]).read_text())
    assert set(s["trajectory"] for s in calibration["samples"]) == set(config["splits"]["train"])
    legacy.material = ContactMaterial(**calibration["material"])
    shutil.copy2(ROOT / config["calibration"], output / "calibration.json")
    checkpoint = ROOT / config["checkpoint"]
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert artifact["training_pairs"] == 600 and np.array_equal(
        artifact["node_ids"], mesh["node_ids"]
    )
    model = JointStateDynamics(
        torch.tensor(mesh["rest_positions_world"]).float(),
        torch.tensor(mesh["pinned_mask"]),
        **artifact["architecture"],
    )
    model.load_state_dict(artifact["model_state"], strict=True)
    model.eval().requires_grad_(False)
    initial_digest = tensor_digest(model.state_dict())
    shutil.copy2(checkpoint, output / "frozen_model.pt")
    pipelines = make_pipelines(legacy, model)
    predictors = make_predictors(legacy, pipelines)
    provenance = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "selected_step": artifact["selected_step"],
        "training_pairs": artifact["training_pairs"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_updates_this_run": 0,
        "calibration_source_sha256": file_sha256(ROOT / config["calibration"]),
    }
    write_json(output / "provenance.json", provenance)
    contracts = runtime_contracts(legacy, model, pipelines, mesh, trajectories, output)
    evaluation = evaluate_models(
        predictors, legacy.gaussian_state, trajectories, splits, mesh, config, output
    )
    # Reproduce historical full-model M, not the fixed recovery-window networks.
    old = read_npz(checkpoint.parent.parent / "one_step_predictions.npz")
    new = read_npz(output / "one_step_predictions.npz")
    for name in trajectories:
        for field in ("next_position", "next_velocity"):
            assert np.array_equal(
                new[f"{name}__mujoco_force__{field}"], old[f"{name}__autonomous_joint__{field}"]
            )
    contracts["native_condition_one_step_bitwise_equal_historical_full_model"] = True
    force_metrics, force_checks = force_geometry_audit(legacy, trajectories, splits, output)
    interventions = run_interventions(legacy, predictors, mesh, trajectories, config, output)
    assert tensor_digest(model.state_dict()) == initial_digest
    assert before == {name: file_sha256(ROOT / name) for name in before}
    checks = {
        **evaluation["checks"],
        **contracts,
        **force_checks,
        "historical_file_count_unchanged_sha256": len(before),
        "frozen_weights_unchanged": True,
        "test_trajectory_arrays_loaded": False,
    }
    write_json(output / "checks.json", checks)
    write_json(output / "historical_hashes.json", before)
    environment = {
        "experiment_date": config["experiment_date"],
        "host_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": sum(f.stat().st_size for f in output.rglob("*") if f.is_file()),
        "cpu_threads": config["cpu_threads"],
        "training_updates": 0,
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
            "validation_mechanism_audit_not_held_out_test": True,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "environment": environment,
                "intervention_cases": interventions["native_cases"],
                "intervention_failures": interventions["failed_cases"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
