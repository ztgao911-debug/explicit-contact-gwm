"""CPU contact-pipeline validation, frozen local network, prescribed indenter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from contact_gwm.contact_force import ContactMaterial
from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256, read_npz
from contact_gwm.data.mujoco_soft import tetra_volumes
from contact_gwm.local_training import ConstantVelocityResidualDeformation
from contact_gwm.pinned_contact_pipeline import (
    METHODS,
    IndenterInput,
    PinnedContactPipeline,
    nonnegative_two_column_fit,
)

ROOT = Path(__file__).resolve().parents[1]
PHASES = {0: "approach", 1: "press", 2: "hold", 3: "release", 4: "recover"}
REGIONS = ("active", "contact_neighborhood")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hashes(paths):
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths if path.is_file()}


def source_labels(arrays, frame):
    a, b = arrays["contact_offsets"][frame : frame + 2]
    ids = np.arange(a, b)
    if len(ids):
        ids = ids[
            (arrays["contact_signed_distance"][ids, 0] <= 0)
            & (np.linalg.norm(arrays["contact_force_on_block_world"][ids], axis=1) > 1e-8)
        ]
    return ids


def reference_masks(arrays, mesh, radius):
    """Scoring ONLY: current GT-contact ROI; last GT ROI during separation.

    Before the first contact use a rest-shape ROI within radius of the point
    immediately below the prescribed indenter on the rest top. No method sees
    these masks; they include every frame, including the no-contact control.
    """
    active = ~mesh["pinned_mask"]
    rest = mesh["rest_positions_world"]
    initial_point = arrays["indenter_position_world"][0].copy()
    initial_point[2] = rest[:, 2].max()
    previous = (np.linalg.norm(rest - initial_point, axis=1) <= radius) & active
    masks = []
    for frame in range(len(arrays["time"])):
        indices = source_labels(arrays, frame)
        if len(indices):
            points = arrays["contact_position_world"][indices]
            distance = np.linalg.norm(
                arrays["node_position_world"][frame, :, None] - points, axis=2
            )
            previous = (distance.min(axis=1) <= radius) & active
        masks.append(previous.copy())
    return np.asarray(masks)


def calibrate(pipeline, trajectories, config):
    design, target, samples = [], [], []
    for name in config["calibration"]["trajectories"]:
        arrays = trajectories[name]
        for frame in range(len(arrays["time"]) - 1):
            if int(arrays["phase_id"][frame]) not in config["calibration"]["phase_ids"]:
                continue
            state = pipeline.gaussian_state(
                arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
            )
            contacts, force = pipeline.condition(
                state, IndenterInput.from_arrays(arrays, frame), "mujoco_force"
            )
            if not contacts.num_contacts:
                continue
            closing = (contacts.relative_velocities * contacts.normals).sum(-1).clamp_min(0)
            design.extend(torch.stack((contacts.penetrations, closing), dim=1).numpy())
            target.extend((force.forces * contacts.normals).sum(-1).clamp_min(0).numpy())
            samples.append({"trajectory": name, "frame": frame, "contacts": contacts.num_contacts})
    design, target = np.asarray(design), np.asarray(target)
    fit = nonnegative_two_column_fit(design, target)
    pipeline.material = ContactMaterial(
        **{
            **config["initial_material"],
            "normal_stiffness": float(fit[0]),
            "normal_damping": float(fit[1]),
        }
    )
    return {
        "material": asdict(pipeline.material),
        "units": {"stiffness": "N/m", "damping": "N s/m"},
        "source": "current-state native solver forces on central train press/hold frames only",
        "samples": samples,
        "contact_count": len(target),
        "normal_force_contact_rmse_N": float(np.sqrt(np.square(design @ fit - target).mean())),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "young_modulus_Pa": 5000.0,
        "young_modulus_used_as_stiffness": False,
    }


def aggregate_errors(records, key_fields):
    groups = defaultdict(lambda: [0.0, 0.0, 0])
    for row in records:
        # All groups use the same reference masks, not predicted contacts.
        keys = [tuple(row[field] for field in key_fields)]
        if "phase" in key_fields:
            key = list(keys[0])
            key[key_fields.index("phase")] = "all"
            keys.append(tuple(key))
            if row["phase"] in ("press", "hold"):
                key = list(key)
                key[key_fields.index("phase")] = "press_hold"
                keys.append(tuple(key))
        for key in keys:
            value = groups[key]
            value[0] += row["position_sse_m2"]
            value[1] += row["velocity_sse_m2_s2"]
            value[2] += row["node_count"]
    return [
        dict(
            zip(key_fields, key, strict=True),
            node_count=value[2],
            position_or_increment_rmse_um=float(np.sqrt(value[0] / value[2]) * 1e6),
            endpoint_velocity_rmse_mm_s=float(np.sqrt(value[1] / value[2]) * 1e3),
        )
        for key, value in sorted(groups.items())
        if value[2]
    ]


def error_record(error, velocity_error, mask, **metadata):
    return {
        **metadata,
        "node_count": int(mask.sum()),
        "position_sse_m2": float(np.square(error[mask]).sum()),
        "velocity_sse_m2_s2": float(np.square(velocity_error[mask]).sum()),
    }


def evaluate_one_step(pipeline, trajectories, splits, masks, output):
    errors, force_rows = [], []
    saved = {"node_ids": np.arange(len(pipeline.rest)), "rest_positions": pipeline.rest.numpy()}
    max_pin, min_normal, torque_error = 0.0, 1.0, 0.0
    for name, arrays in trajectories.items():
        split = splits[name]
        nframes = len(arrays["time"]) - 1
        trajectory_predictions = {method: [] for method in METHODS}
        for frame in range(nframes):
            state = pipeline.gaussian_state(
                arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
            )
            robot = IndenterInput.from_arrays(arrays, frame)
            ids = source_labels(arrays, frame)
            true_force = arrays["contact_force_on_block_world"][ids].sum(axis=0)
            true_points = arrays["contact_position_world"][ids]
            com = pipeline.center_of_mass(state).numpy()
            true_torque = np.cross(
                true_points - com, arrays["contact_force_on_block_world"][ids]
            ).sum(axis=0)
            target = arrays["node_position_world"][frame + 1] - state.centers.numpy()
            for method in METHODS:
                result = pipeline.step(state, robot, method)
                prediction = result.delta.numpy()
                trajectory_predictions[method].append(prediction)
                error = prediction - target
                verror = result.state.velocities.numpy() - arrays["node_velocity_world"][frame + 1]
                for region in REGIONS:
                    mask = ~pipeline.pinned.numpy() if region == "active" else masks[name][frame]
                    errors.append(
                        error_record(
                            error,
                            verror,
                            mask,
                            method=method,
                            split=split,
                            trajectory=name,
                            frame=frame,
                            phase=PHASES[int(arrays["phase_id"][frame])],
                            region=region,
                        )
                    )
                if method != "constant_velocity":
                    force = result.forces.forces.sum(0).numpy()
                    torque = result.forces.torques.sum(0).numpy()
                    force_rows.append(
                        {
                            "method": method,
                            "split": split,
                            "trajectory": name,
                            "frame": frame,
                            "phase": PHASES[int(arrays["phase_id"][frame])],
                            "true_contact": bool(len(ids)),
                            "predicted_contact": bool(result.contacts.num_contacts),
                            "true_contact_count": len(ids),
                            "predicted_contact_count": result.contacts.num_contacts,
                            "force_error_squared_N2": float(np.square(force - true_force).sum()),
                            "torque_error_squared_N2_m2": float(
                                np.square(torque - true_torque).sum()
                            ),
                            **{
                                f"force_{axis}_N": float(value)
                                for axis, value in zip("xyz", force, strict=True)
                            },
                            **{
                                f"true_force_{axis}_N": float(value)
                                for axis, value in zip("xyz", true_force, strict=True)
                            },
                        }
                    )
                    if result.contacts.num_contacts:
                        min_normal = min(
                            min_normal,
                            float((result.forces.forces * result.contacts.normals).sum(-1).min()),
                        )
                        computed = torch.cross(
                            result.contacts.points - torch.tensor(com), result.forces.forces, dim=-1
                        )
                        torque_error = max(
                            torque_error, float((computed - result.forces.torques).abs().max())
                        )
                max_pin = max(
                    max_pin,
                    float(result.delta[pipeline.pinned].abs().max()),
                    float(result.state.velocities[pipeline.pinned].abs().max()),
                )
        for method, predictions in trajectory_predictions.items():
            saved[f"{name}__{method}__delta"] = np.asarray(predictions)
        saved[f"{name}__contact_neighborhood_mask"] = masks[name]
        print(f"one-step done: {name} ({nframes} pairs x 4 methods)", flush=True)
    np.savez_compressed(output / "one_step_predictions.npz", **saved)
    write_csv(output / "one_step_per_frame.csv", errors)
    write_csv(output / "forces_per_frame.csv", force_rows)
    pooled = aggregate_errors(errors, ["method", "split", "phase", "region"])
    write_csv(output / "one_step_metrics.csv", pooled)
    per_trajectory = aggregate_errors(errors, ["method", "trajectory", "phase", "region"])
    write_csv(output / "one_step_per_trajectory.csv", per_trajectory)
    force_metrics = summarize_forces(force_rows)
    write_csv(output / "force_detection_metrics.csv", force_metrics)
    return (
        pooled,
        force_metrics,
        {
            "max_fixed_node_increment_or_velocity": max_pin,
            "minimum_force_dot_normal_N": min_normal,
            "torque_cross_product_max_error_N_m": torque_error,
            "one_step_frame_pairs_per_method": sum(
                len(a["time"]) - 1 for a in trajectories.values()
            ),
        },
    )


def summarize_forces(rows):
    groups = defaultdict(list)
    for row in rows:
        for phase in (row["phase"], "all"):
            groups[row["method"], row["split"], phase].append(row)
    result = []
    for (method, split, phase), values in sorted(groups.items()):
        tp = sum(row["true_contact"] and row["predicted_contact"] for row in values)
        fp = sum(not row["true_contact"] and row["predicted_contact"] for row in values)
        fn = sum(row["true_contact"] and not row["predicted_contact"] for row in values)
        true_contact = [row for row in values if row["true_contact"]]
        result.append(
            {
                "method": method,
                "split": split,
                "phase": phase,
                "frames": len(values),
                "net_force_rmse_N": float(
                    np.sqrt(np.mean([r["force_error_squared_N2"] for r in values]))
                ),
                "net_force_contact_frames_rmse_N": float(
                    np.sqrt(np.mean([r["force_error_squared_N2"] for r in true_contact]))
                )
                if true_contact
                else None,
                "net_torque_rmse_N_m": float(
                    np.sqrt(np.mean([r["torque_error_squared_N2_m2"] for r in values]))
                ),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "mean_predicted_contact_count": float(
                    np.mean([r["predicted_contact_count"] for r in values])
                ),
                "mean_true_contact_count": float(
                    np.mean([r["true_contact_count"] for r in values])
                ),
            }
        )
    return result


def evaluate_rollout(pipeline, trajectories, splits, masks, mesh, config, output):
    errors, windows, failures = [], [], []
    saved = {"node_ids": mesh["node_ids"], "rest_positions": mesh["rest_positions_world"]}
    invalid_count, max_pin = 0, 0.0
    detached_checks = []
    max_horizon = max(config["rollout"]["horizons"])
    for name, arrays in trajectories.items():
        for start in config["rollout"]["start_frames"]:
            initial = pipeline.gaussian_state(
                arrays["node_position_world"][start], arrays["node_velocity_world"][start]
            )
            # After this boundary, rollout accepts no soft trajectory/contact/force labels.
            external = [
                IndenterInput.from_arrays(arrays, frame)
                for frame in range(start, start + max_horizon)
            ]
            windows.append(
                {
                    "trajectory": name,
                    "start_frame": start,
                    "initial_time_s": float(arrays["time"][start]),
                    "initial_phase": PHASES[int(arrays["phase_id"][start])],
                    "horizons": config["rollout"]["horizons"],
                }
            )
            for method in METHODS:
                try:
                    predictions = pipeline.rollout(initial, external, method)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    failures.append(
                        {"trajectory": name, "start": start, "method": method, "error": str(exc)}
                    )
                    continue
                positions = np.asarray(
                    [initial.centers.numpy(), *[p.state.centers.numpy() for p in predictions]]
                )
                velocities = np.asarray(
                    [initial.velocities.numpy(), *[p.state.velocities.numpy() for p in predictions]]
                )
                contact_counts = np.asarray([p.contacts.num_contacts for p in predictions])
                gate_max = np.asarray([float(p.gates.max()) for p in predictions])
                key = f"{name}__{start}__{method}"
                saved[f"{key}__position"] = positions
                saved[f"{key}__velocity"] = velocities
                saved[f"{key}__contact_count"] = contact_counts
                saved[f"{key}__gate_max"] = gate_max
                saved[f"{key}__force"] = np.asarray(
                    [p.forces.forces.sum(0).numpy() for p in predictions]
                )
                invalid = (
                    tetra_volumes(positions, mesh["tetrahedra"]) / mesh["rest_signed_tetra_volumes"]
                )
                invalid_count += int(np.any(invalid <= 0, axis=-1).sum())
                max_pin = max(
                    max_pin,
                    float(
                        np.abs(
                            positions[:, mesh["pinned_mask"]]
                            - mesh["rest_positions_world"][mesh["pinned_mask"]]
                        ).max()
                    ),
                    float(np.abs(velocities[:, mesh["pinned_mask"]]).max()),
                )
                for horizon in range(1, max_horizon + 1):
                    error = positions[horizon] - arrays["node_position_world"][start + horizon]
                    verror = velocities[horizon] - arrays["node_velocity_world"][start + horizon]
                    for region in REGIONS:
                        mask = ~mesh["pinned_mask"] if region == "active" else masks[name][start]
                        errors.append(
                            error_record(
                                error,
                                verror,
                                mask,
                                method=method,
                                split=splits[name],
                                trajectory=name,
                                start=start,
                                phase=PHASES[int(arrays["phase_id"][start])],
                                horizon=horizon,
                                region=region,
                            )
                        )
                # A detached window is determined by current true labels ONLY in evaluator.
                if start in (170, 210) and name != "no_contact":
                    detached_checks.append(
                        {
                            "trajectory": name,
                            "start": start,
                            "method": method,
                            "true_contact_frames": sum(
                                bool(len(source_labels(arrays, f)))
                                for f in range(start, start + max_horizon)
                            ),
                            "predicted_contact_frames": int((contact_counts > 0).sum()),
                            "max_gate": float(gate_max.max()),
                            "true_rms_deformation_start_um": float(
                                np.sqrt(
                                    np.square(
                                        arrays["node_position_world"][start, ~mesh["pinned_mask"]]
                                        - mesh["rest_positions_world"][~mesh["pinned_mask"]]
                                    )
                                    .sum(axis=-1)
                                    .mean()
                                )
                                * 1e6
                            ),
                            "true_rms_deformation_end_um": float(
                                np.sqrt(
                                    np.square(
                                        arrays["node_position_world"][
                                            start + max_horizon, ~mesh["pinned_mask"]
                                        ]
                                        - mesh["rest_positions_world"][~mesh["pinned_mask"]]
                                    )
                                    .sum(axis=-1)
                                    .mean()
                                )
                                * 1e6
                            ),
                            "predicted_rms_deformation_end_um": float(
                                np.sqrt(
                                    np.square(
                                        positions[-1, ~mesh["pinned_mask"]]
                                        - mesh["rest_positions_world"][~mesh["pinned_mask"]]
                                    )
                                    .sum(axis=-1)
                                    .mean()
                                )
                                * 1e6
                            ),
                        }
                    )
        print(f"rollout done: {name} (6 windows x 4 methods x 20 steps)", flush=True)
    np.savez_compressed(output / "rollout_predictions.npz", **saved)
    write_json(output / "rollout_windows.json", windows)
    write_json(output / "rollout_failures.json", failures)
    write_csv(output / "rollout_per_window.csv", errors)
    pooled = aggregate_errors(errors, ["method", "split", "phase", "horizon", "region"])
    write_csv(output / "rollout_all_horizons.csv", pooled)
    write_csv(
        output / "rollout_metrics.csv",
        [row for row in pooled if row["horizon"] in config["rollout"]["horizons"]],
    )
    write_csv(output / "detached_recovery.csv", detached_checks)
    return pooled, {
        "rollout_exception_count": len(failures),
        "rollout_failures": failures,
        "rollout_states_with_inverted_tetrahedra": invalid_count,
        "rollout_max_fixed_node_position_or_velocity_error": max_pin,
        "rollout_windows": len(windows),
        "rollout_steps_per_method": len(windows) * max_horizon,
    }


def consistency_checks(pipeline, mesh, checkpoint, config, trajectories, output):
    arrays = trajectories["center_deep"]
    frame = 71
    state = pipeline.gaussian_state(
        arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
    )
    robot = IndenterInput.from_arrays(arrays, frame)
    reload = PinnedContactPipeline(
        mesh, ROOT / config["data_directory"] / "scene.xml", checkpoint, config
    )
    reload.material = pipeline.material
    reload_errors = {}
    for method in METHODS:
        a, b = pipeline.step(state, robot, method), reload.step(state, robot, method)
        reload_errors[method] = float((a.state.centers - b.state.centers).abs().max())
        assert torch.equal(a.state.centers, b.state.centers)
        assert torch.equal(a.state.velocities, b.state.velocities)
    # Finite gradients through the frozen decoder, diagnostic only; no optimizer/update.
    model = pipeline.model
    context = torch.zeros((len(mesh["node_ids"]), 36), requires_grad=True)
    prediction, _ = model.deformation.decode_context(model.augment_context(context))
    prediction.square().sum().backward()
    finite_grad = (
        all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        and torch.isfinite(context.grad).all().item()
    )
    assert finite_grad
    model.zero_grad(set_to_none=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": pipeline.training_config,
            "source_checkpoint_sha256": file_sha256(checkpoint),
            "pipeline_config": config,
            "analytic_material": asdict(pipeline.material),
        },
        output / "frozen_model.pt",
    )
    saved = torch.load(output / "frozen_model.pt", map_location="cpu", weights_only=False)
    assert all(
        torch.equal(value, saved["model_state"][key]) for key, value in model.state_dict().items()
    )
    roundtrip = PinnedContactPipeline(
        mesh, ROOT / config["data_directory"] / "scene.xml", output / "frozen_model.pt", config
    )
    roundtrip.material = ContactMaterial(**saved["analytic_material"])
    for method in METHODS:
        original = pipeline.rollout(state, [robot] * 20, method)
        restored = roundtrip.rollout(state, [robot] * 20, method)
        assert all(
            torch.equal(a.state.centers, b.state.centers)
            and torch.equal(a.state.velocities, b.state.velocities)
            for a, b in zip(original, restored, strict=True)
        )
    for filename in ("one_step_predictions.npz", "rollout_predictions.npz"):
        loaded = read_npz(output / filename)
        assert np.array_equal(loaded["node_ids"], mesh["node_ids"])
        assert all(np.isfinite(value).all() for value in loaded.values())
    return {
        "reload_max_position_error_m": reload_errors,
        "finite_decoder_gradients": finite_grad,
        "saved_checkpoint_tensor_equality": True,
        "saved_checkpoint_20_step_rollout_bitwise_equal": True,
        "prediction_archives_reloaded_finite_stable_ids": True,
        "network_training_steps_this_run": 0,
    }


def adapter_compatibility_probe(pipeline, trajectories, mesh, config, output):
    """Concrete diagnostic: old training adapter discarded zero-force candidates.

    Reproduce the ORIGINAL eight-pair oracle-only fit, then compare against the
    common geometry-only candidate selection used by BOTH main native branches.
    The force-filtered variant is never used by an analytic prediction/rollout.
    """
    source = ROOT / config["checkpoint"]
    samples = json.loads((source.parent.parent / "fixed_samples.json").read_text())
    dataset = GaussianPairDataset(ROOT / config["data_directory"], pipeline.training_config)
    encoder = ConstantVelocityResidualDeformation(**pipeline.training_config["model"])
    records, predictions, counts = [], {}, []
    with torch.no_grad():
        for sample in samples:
            name, frame = sample["trajectory"], sample["frame"]
            arrays = trajectories[name]
            inputs = dataset.frame_inputs(name, frame)
            context, gate = encoder.encode(inputs)
            original = (
                pipeline.model.predict_encoded(
                    context, gate, inputs.pinned_mask, inputs.gaussians.velocities
                )
                .double()
                .numpy()
            )
            current = pipeline.gaussian_state(
                arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
            )
            native = pipeline.step(
                current, IndenterInput.from_arrays(arrays, frame), "mujoco_force"
            )
            target = arrays["node_position_world"][frame + 1] - arrays["node_position_world"][frame]
            predictions["original_force_filtered"] = original
            predictions["common_geometric_candidates"] = native.delta.numpy()
            predictions["constant_velocity"] = current.velocities.numpy() * config["dt"]
            counts.append(
                {
                    "trajectory": name,
                    "frame": frame,
                    "original_force_filtered_contacts": inputs.contacts.num_contacts,
                    "common_geometric_contacts": native.contacts.num_contacts,
                }
            )
            for method, delta in predictions.items():
                for region in REGIONS:
                    mask = ~mesh["pinned_mask"] if region == "active" else inputs.neighborhood_mask
                    records.append(
                        error_record(
                            delta - target,
                            np.zeros_like(delta),
                            np.asarray(mask),
                            method=method,
                            phase=sample["phase"],
                            region=region,
                        )
                    )
    write_json(output / "adapter_probe_fixed_samples.json", samples)
    write_csv(output / "adapter_probe_contact_counts.csv", counts)
    metrics = aggregate_errors(records, ["method", "phase", "region"])
    write_csv(output / "adapter_probe_metrics.csv", metrics)
    with (source.parent.parent / "comparison.csv").open() as stream:
        prior = next(row for row in csv.DictReader(stream) if row["group"] == "B")
    for region, column in (
        ("active", "fit_rmse_um"),
        ("contact_neighborhood", "fit_neighborhood_rmse_um"),
    ):
        value = next(
            row["position_or_increment_rmse_um"]
            for row in metrics
            if row["method"] == "original_force_filtered"
            and row["phase"] == "all"
            and row["region"] == region
        )
        assert abs(value - float(prior[column])) < 1e-4, "Original frozen fit did not reproduce"
    return {
        "original_eight_pair_oracle_adapter_reproduced": True,
        "probe_is_not_an_additional_main_method": True,
        "common_geometry_does_not_filter_by_true_force": True,
    }


def make_plots(output, one_step, rollout, trajectories, mesh):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, region in zip(axes, REGIONS, strict=True):
        for method in METHODS:
            rows = sorted(
                [
                    r
                    for r in rollout
                    if r["method"] == method
                    and r["split"] == "validation"
                    and r["phase"] == "all"
                    and r["region"] == region
                ],
                key=lambda row: row["horizon"],
            )
            axis.plot(
                [r["horizon"] for r in rows],
                [r["position_or_increment_rmse_um"] for r in rows],
                label=method,
            )
        axis.set(
            title=f"Left validation: {region}",
            xlabel="Prediction steps",
            ylabel="Position vector RMSE (um)",
            yscale="log",
        )
        axis.grid(alpha=0.3)
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "rollout_errors.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, phase in enumerate(("press", "hold")):
        for j, region in enumerate(REGIONS):
            values = [
                next(
                    r["position_or_increment_rmse_um"]
                    for r in one_step
                    if r["method"] == method
                    and r["split"] == "validation"
                    and r["phase"] == phase
                    and r["region"] == region
                )
                for method in METHODS
            ]
            axes[i, j].bar(np.arange(len(METHODS)), values)
            axes[i, j].set(
                xticks=np.arange(len(METHODS)),
                xticklabels=METHODS,
                title=f"Left: {phase}, {region}",
                ylabel="Increment vector RMSE (um)",
            )
            axes[i, j].tick_params(axis="x", labelsize=7)
    figure.tight_layout()
    figure.savefig(output / "one_step_comparison.png", dpi=160)
    plt.close(figure)

    def rms(x):
        active = ~mesh["pinned_mask"]
        rest = mesh["rest_positions_world"]
        return np.sqrt(np.square(x[:, active] - rest[active]).sum(-1).mean(-1)) * 1e6

    archive = read_npz(output / "rollout_predictions.npz")
    figure, axes = plt.subplots(2, 3, figsize=(14, 7))
    name = "left_deep"
    for axis, start in zip(axes.flat, (70, 110, 140, 150, 170, 210), strict=True):
        true = trajectories[name]["node_position_world"][start : start + 21]
        axis.plot(np.arange(21), rms(true), color="black", label="MuJoCo reference")
        for method in METHODS:
            key = f"{name}__{start}__{method}__position"
            if key in archive:
                axis.plot(np.arange(21), rms(archive[key]), label=method)
        axis.set(
            title=f"{name}, start={start} ({PHASES[int(trajectories[name]['phase_id'][start])]})",
            xlabel="Prediction steps",
            ylabel="RMS deformation (um)",
        )
        axis.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "phase_deformation_curves.png", dpi=160)
    plt.close(figure)
    with (output / "forces_per_frame.csv").open() as stream:
        force_rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for method in METHODS[:-1]:
        rows = [r for r in force_rows if r["trajectory"] == "left_deep" and r["method"] == method]
        axes[0].plot(
            [int(r["frame"]) * 0.01 for r in rows],
            [float(r["force_z_N"]) for r in rows],
            label=method,
        )
        axes[1].step(
            [int(r["frame"]) * 0.01 for r in rows],
            [int(r["predicted_contact_count"]) for r in rows],
            label=method,
        )
    rows = [
        r for r in force_rows if r["trajectory"] == "left_deep" and r["method"] == "mujoco_force"
    ]
    axes[0].plot(
        [int(r["frame"]) * 0.01 for r in rows],
        [float(r["true_force_z_N"]) for r in rows],
        "k--",
        label="recorded reference",
    )
    axes[0].set(ylabel="Net force z (N)")
    axes[1].set(ylabel="Detected contacts", xlabel="Time (s)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "force_and_detection.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/validate_contact_pipeline.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data = ROOT / config["data_directory"]
    checkpoint = ROOT / config["checkpoint"]
    splits = {name: split for split, names in config["splits"].items() for name in names}
    if set(config["calibration"]["trajectories"]) != set(config["splits"]["train"]):
        raise ValueError("Calibration must use exactly the central training trajectories")
    previous = list((ROOT / "outputs").rglob("*"))
    before = hashes([*data.rglob("*"), *previous])
    output = args.output
    if output is None:
        parent = ROOT / config["output_root"]
        index = 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        output = parent / f"run_{index:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve existing results: {output}")
    output.mkdir(parents=True)
    config["resolved_output"] = (
        str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)
    )
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(Path(__file__), output / "experiment_entrypoint.py")
    import contact_gwm.pinned_contact_pipeline as pipeline_module

    shutil.copy2(pipeline_module.__file__, output / "pipeline_implementation.py")
    sources = {"checkpoint": file_sha256(checkpoint), "scene.xml": file_sha256(data / "scene.xml")}
    mesh = read_npz(data / "mesh.npz")
    manifest = json.loads((data / "manifest.json").read_text())
    if file_sha256(data / "mesh.npz") != manifest["shared_sha256"]["mesh.npz"]:
        raise ValueError("Mesh checksum mismatch")
    trajectories = {name: read_npz(data / f"{name}.npz") for name in splits}
    for name, arrays in trajectories.items():
        expected = next(
            item["sha256"] for item in manifest["trajectories"] if item["file"] == f"{name}.npz"
        )
        if file_sha256(data / f"{name}.npz") != expected:
            raise ValueError("Trajectory checksum mismatch")
        if not np.array_equal(arrays["node_ids"], mesh["node_ids"]):
            raise ValueError("Stable node IDs do not match")
        if not np.allclose(np.diff(arrays["time"]), config["dt"], atol=1e-9, rtol=0):
            raise ValueError("Recorded dt mismatch")
    start_time = time.perf_counter()
    pipeline = PinnedContactPipeline(mesh, data / "scene.xml", checkpoint, config)
    calibration = calibrate(pipeline, trajectories, config)
    write_json(output / "calibration.json", calibration)
    print(f"calibrated: {calibration['material']}", flush=True)
    masks = {
        name: reference_masks(arrays, mesh, config["contact_neighborhood_radius_m"])
        for name, arrays in trajectories.items()
    }
    one_step, force_metrics, checks = evaluate_one_step(
        pipeline, trajectories, splits, masks, output
    )
    rollout, rollout_checks = evaluate_rollout(
        pipeline, trajectories, splits, masks, mesh, config, output
    )
    checks.update(rollout_checks)
    checks.update(consistency_checks(pipeline, mesh, checkpoint, config, trajectories, output))
    checks.update(adapter_compatibility_probe(pipeline, trajectories, mesh, config, output))
    write_json(
        output / "conventions.json",
        {
            "frame": "All positions, velocities, normals, forces and torques are in world axes",
            "normal": "robot -> soft; local contact-frame row 0, flipped if flex is side 0",
            "force_transform": "F_soft = side_sign * contact_frame.T @ local_wrench[:3]",
            "units": {"position_gap": "m", "velocity": "m/s", "force": "N", "torque": "N m"},
            "relative_velocity": "current robot v minus soft contact-point v",
            "interpolation_approximation": (
                "tetrahedral barycentrics at contact midpoint, clipped and normalized"
            ),
            "closing_speed": "positive dot(relative_velocity, robot_to_soft_normal)",
            "normal_law": "k * max(-gap,0) + c * max(closing_speed,0); gap <= 0 only",
            "friction_law": "along relative tangential v, magnitude min(4*speed, 0.3*normal_force)",
            "torque_origin": (
                "current dynamic-node COM, using stored node masses; diagnostic only"
            ),
            "fixed_boundary": "pin x to rest, v=0, delta=0 each step; bypass rigid integration",
            "velocity_update": "x_next=x+v*dt+r; v_next=v+2*r/dt (constant acceleration)",
            "geometry_selection": "all gap<=0 native candidates; never filter by true force",
            "network": "unchanged spatial B, 24326 parameters, trained on central 8 pairs",
            "rollout": "prescribed indenter; no soft x/v/contact/force after initialization",
            "mask": (
                "scoring only; current GT ROI or last ROI when detached, rest-top ROI "
                "before contact; rollout holds initial ROI fixed"
            ),
        },
    )
    # Endpoint-velocity error under this assumption when displacement is exact;
    # this is a diagnostic, not a lower bound on all velocity prediction models.
    endpoint_oracle = []
    for name, arrays in trajectories.items():
        x, v = arrays["node_position_world"], arrays["node_velocity_world"]
        vhat = 2 * (x[1:] - x[:-1]) / config["dt"] - v[:-1]
        for phase in (1, 2, 3, 4):
            selection = arrays["phase_id"][:-1] == phase
            error = (vhat - v[1:])[selection][:, ~mesh["pinned_mask"]]
            endpoint_oracle.append(
                {
                    "trajectory": name,
                    "phase": PHASES[phase],
                    "endpoint_velocity_rmse_mm_s": float(
                        np.sqrt(np.square(error).sum(-1).mean()) * 1e3
                    ),
                }
            )
    write_csv(output / "constant_acceleration_oracle_velocity.csv", endpoint_oracle)
    make_plots(output, one_step, rollout, trajectories, mesh)
    # Every loaded and historical data/result file must remain byte-identical.
    after = {path: file_sha256(ROOT / path) for path in before}
    assert before == after
    checks["old_data_and_results_sha256_unchanged"] = True
    checks["preserved_file_count"] = len(before)
    assert checks["max_fixed_node_increment_or_velocity"] == 0
    assert checks["rollout_max_fixed_node_position_or_velocity_error"] == 0
    write_json(output / "checks.json", checks)
    write_json(output / "source_hashes.json", {"historical_files": before, **sources})
    import mujoco

    metadata = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "cpu_threads": config["cpu_threads"],
        "seed": config["seed"],
        "elapsed_seconds": time.perf_counter() - start_time,
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "parameter_count": sum(p.numel() for p in pipeline.model.parameters()),
        "no_training_this_run": True,
        "network_source_training_pairs": 8,
        "loaded_trajectories": list(trajectories),
        "right_test_trajectories_loaded": False,
        "actual_completed": [
            "calibration",
            "one_step_all_phases",
            "rollout_5_10_20",
            "reload_checks",
            "finite_gradient_check",
            "plots",
            "original_adapter_compatibility_probe",
        ],
    }
    write_json(output / "environment.json", metadata)
    print(json.dumps({"output": str(output), **metadata, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
