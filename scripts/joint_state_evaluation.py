"""Common-mask evaluation for pinned joint position/instantaneous-velocity models.

Predictors receive only a current GaussianSet and current IndenterInput. During
rollout all subsequent soft states come from their own preceding prediction;
the trajectory arrays below are used only outside that prediction boundary.
RMSE is vector RMSE: sum of XYZ squared error divided by the number of nodes.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from contact_gwm.data.mujoco_soft import tetra_volumes
from contact_gwm.pinned_contact_pipeline import IndenterInput, StepResult
from contact_gwm.structures import GaussianSet

PHASES = {0: "approach", 1: "press", 2: "hold", 3: "release", 4: "recover"}
REGIONS = ("active", "contact_neighborhood")
Predictor = Callable[[GaussianSet, IndenterInput], StepResult]


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _label_contacts(arrays, frame):
    """Scoring-only contact label; never passed into a predictor."""
    a, b = arrays["contact_offsets"][frame : frame + 2]
    indices = np.arange(a, b)
    return indices[
        (arrays["contact_signed_distance"][indices, 0] <= 0)
        & (np.linalg.norm(arrays["contact_force_on_block_world"][indices], axis=1) > 1e-8)
    ]


def _masks(arrays, mesh, radius):
    """Current reference contact ROI, carried forward after contact loss.

    Before first contact, use the rest-shape region below the known initial
    indenter. This is identical for all models and includes no-contact frames.
    """
    active, rest = ~mesh["pinned_mask"], mesh["rest_positions_world"]
    point = arrays["indenter_position_world"][0].copy()
    point[2] = rest[:, 2].max()
    previous = (np.linalg.norm(rest - point, axis=1) <= radius) & active
    result = []
    for frame in range(len(arrays["time"])):
        indices = _label_contacts(arrays, frame)
        if len(indices):
            distance = np.linalg.norm(
                arrays["node_position_world"][frame, :, None]
                - arrays["contact_position_world"][indices],
                axis=2,
            )
            previous = (distance.min(axis=1) <= radius) & active
        result.append(previous.copy())
    return np.asarray(result)


def _record(error, velocity_error, mask, quantity, **metadata):
    count = int(mask.sum())
    psse = float(np.square(error[mask]).sum())
    vsse = float(np.square(velocity_error[mask]).sum())
    return {
        **metadata,
        "node_count": count,
        "position_sse_m2": psse,
        "velocity_sse_m2_s2": vsse,
        f"{quantity}_rmse_um": float(np.sqrt(psse / count) * 1e6) if count else None,
        "velocity_rmse_mm_s": float(np.sqrt(vsse / count) * 1e3) if count else None,
    }


def aggregate_errors(records, fields, quantity):
    """Pool node SSE, not a mean of per-frame RMSE; retain every phase/group."""
    groups = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for row in records:
        phases = [row["phase"], "all"] if "phase" in fields else [None]
        if "phase" in fields and row["phase"] in ("press", "hold"):
            phases.append("press_hold")
        for phase in phases:
            contact_groups = (row["contact_group"], "all") if "contact_group" in fields else (None,)
            for group in contact_groups:
                key = tuple(
                    phase if field == "phase" else group if field == "contact_group" else row[field]
                    for field in fields
                )
                value = groups[key]
                value[0] += row["position_sse_m2"]
                value[1] += row["velocity_sse_m2_s2"]
                value[2] += row["node_count"]
                value[3] += 1
    return [
        {
            **dict(zip(fields, key, strict=True)),
            "node_count": count,
            "frame_count": frames,
            f"{quantity}_rmse_um": float(np.sqrt(psse / count) * 1e6),
            "velocity_rmse_mm_s": float(np.sqrt(vsse / count) * 1e3),
        }
        for key, (psse, vsse, count, frames) in sorted(groups.items())
        if count
    ]


def _check(result, mesh, checks):
    x, v, delta = (
        _numpy(value) for value in (result.state.centers, result.state.velocities, result.delta)
    )
    values = (
        x,
        v,
        delta,
        _numpy(result.residual),
        _numpy(result.gates),
        _numpy(result.forces.forces),
        _numpy(result.forces.torques),
        _numpy(result.contacts.points),
        _numpy(result.contacts.normals),
        _numpy(result.contacts.signed_gaps),
        _numpy(result.contacts.relative_velocities),
    )
    if not all(np.isfinite(value).all() for value in values):
        checks["nonfinite_result_count"] += 1
        raise FloatingPointError("Non-finite state, increment, gate, or force")
    pin = mesh["pinned_mask"]
    for key, error in (
        ("max_fixed_position_error_m", np.abs(x[pin] - mesh["rest_positions_world"][pin])),
        ("max_fixed_velocity_m_s", np.abs(v[pin])),
        ("max_fixed_increment_m", np.abs(delta[pin])),
    ):
        checks[key] = max(checks[key], float(error.max(initial=0)))
    return x.copy(), v.copy()


def _rms(value, mask, multiplier):
    return float(np.sqrt(np.square(value[mask]).sum(axis=-1).mean()) * multiplier)


@torch.no_grad()
def evaluate_models(
    predictors: dict[str, Predictor],
    state_factory: Callable[[np.ndarray, np.ndarray], GaussianSet],
    trajectories: dict[str, dict],
    splits: dict[str, str],
    mesh: dict,
    config: dict,
    output: Path,
) -> dict:
    """Evaluate every adjacent pair and prescribed-indenter autoregressive window.

    ``config`` accepts contact_neighborhood_radius_m and rollout.{start_frames,
    horizons}; defaults are .02 m, [70,110,140,150,170,210], [5,10,20].
    ``splits`` maps trajectory name to train/validation/control. The returned
    dictionary contains all pooled metrics and numerical checks, and is JSON
    serializable. All arrays and positions/velocities use SI units on disk.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    active, rest = ~mesh["pinned_mask"], mesh["rest_positions_world"]
    masks = {
        name: _masks(arrays, mesh, config.get("contact_neighborhood_radius_m", 0.02))
        for name, arrays in trajectories.items()
    }
    checks = {
        "nonfinite_result_count": 0,
        "max_fixed_position_error_m": 0.0,
        "max_fixed_velocity_m_s": 0.0,
        "max_fixed_increment_m": 0.0,
        "rollout_states_with_inverted_tetrahedra": 0,
    }
    one_rows, contact_rows = [], []
    one_saved = {"node_ids": mesh["node_ids"], "rest_positions": rest}
    for name, arrays in trajectories.items():
        nframes = len(arrays["time"]) - 1
        positions = {method: [] for method in predictors}
        velocities = {method: [] for method in predictors}
        for frame in range(nframes):
            robot = IndenterInput.from_arrays(arrays, frame)
            group = "contact" if len(_label_contacts(arrays, frame)) else "no_contact"
            current = arrays["node_position_world"][frame]
            target_delta = arrays["node_position_world"][frame + 1] - current
            for method, predictor in predictors.items():
                # Each model gets a fresh state: no in-place mutation can leak across models.
                state = state_factory(current, arrays["node_velocity_world"][frame])
                result = predictor(state, robot)
                x, v = _check(result, mesh, checks)
                positions[method].append(x)
                velocities[method].append(v)
                error = _numpy(result.delta) - target_delta
                verror = v - arrays["node_velocity_world"][frame + 1]
                for region in REGIONS:
                    mask = active if region == "active" else masks[name][frame]
                    one_rows.append(
                        _record(
                            error,
                            verror,
                            mask,
                            "increment",
                            method=method,
                            split=splits[name],
                            trajectory=name,
                            frame=frame,
                            time_s=float(arrays["time"][frame]),
                            phase=PHASES[int(arrays["phase_id"][frame])],
                            region=region,
                            contact_group=group,
                        )
                    )
                contact_rows.append(
                    {
                        "method": method,
                        "trajectory": name,
                        "frame": frame,
                        "reference_contact": group == "contact",
                        "predicted_contact_count": result.contacts.num_contacts,
                        "max_gate": float(_numpy(result.gates).max(initial=0)),
                    }
                )
        for method in predictors:
            one_saved[f"{name}__{method}__next_position"] = np.asarray(positions[method])
            one_saved[f"{name}__{method}__next_velocity"] = np.asarray(velocities[method])
        one_saved[f"{name}__contact_neighborhood_mask"] = masks[name]
        one_saved[f"{name}__time"] = arrays["time"]
        print(f"joint-state one-step done: {name} ({nframes} pairs)", flush=True)
    _csv(output / "one_step_per_frame.csv", one_rows)
    _csv(output / "one_step_contacts.csv", contact_rows)
    np.savez_compressed(output / "one_step_predictions.npz", **one_saved)
    fields = ["method", "split", "phase", "region", "contact_group"]
    one_metrics = aggregate_errors(one_rows, fields, "increment")
    _csv(output / "one_step_metrics.csv", one_metrics)
    _csv(
        output / "one_step_per_trajectory.csv",
        aggregate_errors(
            one_rows,
            ["method", "split", "trajectory", "phase", "region", "contact_group"],
            "increment",
        ),
    )

    rollout_config = config.get("rollout", {})
    starts = rollout_config.get("start_frames", [70, 110, 140, 150, 170, 210])
    horizons = rollout_config.get("horizons", [5, 10, 20])
    steps = max(horizons)
    roll_rows, profiles, detached, failures, windows = [], [], [], [], []
    completed_windows = {method: set() for method in predictors}
    roll_saved = {"node_ids": mesh["node_ids"], "rest_positions": rest}
    for name, arrays in trajectories.items():
        for start in starts:
            if start + steps >= len(arrays["time"]):
                raise ValueError(f"Window {name}/{start} exceeds trajectory length")
            external = [IndenterInput.from_arrays(arrays, f) for f in range(start, start + steps)]
            windows.append({"trajectory": name, "start": start, "steps": steps})
            for method, predictor in predictors.items():
                state = state_factory(
                    arrays["node_position_world"][start], arrays["node_velocity_world"][start]
                )
                # The prediction loop sees only this initial state and external robot inputs.
                predicted, results = (
                    [(_numpy(state.centers).copy(), _numpy(state.velocities).copy())],
                    [],
                )
                try:
                    for robot in external:
                        result = predictor(state, robot)
                        predicted.append(_check(result, mesh, checks))
                        results.append(result)
                        state = result.state
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    failures.append(
                        {"trajectory": name, "start": start, "method": method, "error": str(exc)}
                    )
                    continue
                x, v = (np.asarray([pair[i] for pair in predicted]) for i in (0, 1))
                completed_windows[method].add((name, start))
                counts = np.asarray([result.contacts.num_contacts for result in results])
                key = f"{name}__{start}__{method}"
                for suffix, value in (("position", x), ("velocity", v), ("contact_count", counts)):
                    roll_saved[f"{key}__{suffix}"] = value
                volumes = tetra_volumes(x, mesh["tetrahedra"]) / mesh["rest_signed_tetra_volumes"]
                checks["rollout_states_with_inverted_tetrahedra"] += int(
                    np.any(volumes <= 0, axis=-1).sum()
                )
                # All label reads below are scoring, after the whole prediction window.
                for horizon in range(1, steps + 1):
                    frame = start + horizon
                    for region in REGIONS:
                        mask = active if region == "active" else masks[name][start]
                        roll_rows.append(
                            _record(
                                x[horizon] - arrays["node_position_world"][frame],
                                v[horizon] - arrays["node_velocity_world"][frame],
                                mask,
                                "position",
                                method=method,
                                split=splits[name],
                                trajectory=name,
                                start=start,
                                phase=PHASES[int(arrays["phase_id"][start])],
                                horizon=horizon,
                                region=region,
                                contact_group="contact"
                                if len(_label_contacts(arrays, start))
                                else "no_contact",
                            )
                        )
                window_profiles = []
                for horizon in range(steps + 1):
                    frame = start + horizon
                    row = {
                        "method": method,
                        "split": splits[name],
                        "trajectory": name,
                        "start": start,
                        "horizon": horizon,
                        "time_s": float(arrays["time"][frame]),
                        "true_rms_deformation_um": _rms(
                            arrays["node_position_world"][frame] - rest, active, 1e6
                        ),
                        "predicted_rms_deformation_um": _rms(x[horizon] - rest, active, 1e6),
                        "true_rms_speed_mm_s": _rms(
                            arrays["node_velocity_world"][frame], active, 1e3
                        ),
                        "predicted_rms_speed_mm_s": _rms(v[horizon], active, 1e3),
                        "true_contact_count": len(_label_contacts(arrays, frame)),
                        # Contact queries belong to current states, not predicted endpoints.
                        "predicted_contact_count": int(counts[horizon])
                        if horizon < steps
                        else None,
                    }
                    profiles.append(row)
                    window_profiles.append(row)
                if start in (170, 210) and name != "no_contact":
                    detached.append(
                        {
                            "method": method,
                            "split": splits[name],
                            "trajectory": name,
                            "start": start,
                            "true_contact_frames": sum(
                                row["true_contact_count"] > 0 for row in window_profiles[:-1]
                            ),
                            "predicted_contact_frames": int((counts > 0).sum()),
                            **{
                                f"{label}_{edge}": window_profiles[index][label]
                                for label in (
                                    "true_rms_deformation_um",
                                    "predicted_rms_deformation_um",
                                    "true_rms_speed_mm_s",
                                    "predicted_rms_speed_mm_s",
                                )
                                for edge, index in (("start", 0), ("end", -1))
                            },
                        }
                    )
        print(f"joint-state rollout done: {name} ({len(starts)} windows)", flush=True)
    _csv(output / "rollout_per_window.csv", roll_rows)
    _csv(output / "window_profiles.csv", profiles)
    _csv(output / "detached_recovery.csv", detached)
    _json(output / "rollout_failures.json", failures)
    _json(output / "rollout_windows.json", windows)
    np.savez_compressed(output / "rollout_predictions.npz", **roll_saved)
    roll_metrics = aggregate_errors(roll_rows, [*fields, "horizon"], "position")
    _csv(output / "rollout_all_horizons.csv", roll_metrics)
    _csv(
        output / "rollout_metrics.csv", [row for row in roll_metrics if row["horizon"] in horizons]
    )
    _csv(
        output / "rollout_per_trajectory.csv",
        aggregate_errors(
            roll_rows,
            ["method", "split", "trajectory", "phase", "region", "contact_group", "horizon"],
            "position",
        ),
    )
    checks.update(
        {
            "one_step_pairs_per_method": sum(len(a["time"]) - 1 for a in trajectories.values()),
            "rollout_windows_per_method": len(windows),
            "rollout_steps_per_method": len(windows) * steps,
            "rollout_exception_count": len(failures),
            "rollout_failures": failures,
            "common_evaluation_masks": all(
                len(completed_windows[method]) == len(windows) for method in predictors
            ),
            "completed_rollout_windows_per_method": {
                method: len(value) for method, value in completed_windows.items()
            },
            "rollout_contact_group_definition": "reference contact at initialization, scoring only",
            "rollout_region_definition": "reference contact neighborhood frozen at initialization",
        }
    )
    reload_checks = {}
    for filename, expected in (
        ("one_step_predictions.npz", one_saved),
        ("rollout_predictions.npz", roll_saved),
    ):
        with np.load(output / filename) as saved:
            reload_checks[filename] = bool(
                np.array_equal(saved["node_ids"], mesh["node_ids"])
                and all(np.isfinite(saved[key]).all() for key in saved.files)
                and all(np.array_equal(saved[key], expected[key]) for key in saved.files)
            )
    checks["prediction_reload_finite_and_node_ids"] = reload_checks
    _plots(roll_metrics, profiles, predictors, starts, output)
    metrics = {
        "one_step": one_metrics,
        "rollout": roll_metrics,
        "detached": detached,
        "checks": checks,
    }
    _json(output / "evaluation.json", metrics)
    return metrics


def _plots(metrics, profiles, predictors, starts, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, region in enumerate(REGIONS):
        for row_index, metric in enumerate(("position_rmse_um", "velocity_rmse_mm_s")):
            ax = axes[row_index, column]
            for method in predictors:
                values = sorted(
                    (
                        row
                        for row in metrics
                        if row["method"] == method
                        and row["split"] == "validation"
                        and row["phase"] == "all"
                        and row["region"] == region
                        and row["contact_group"] == "all"
                    ),
                    key=lambda row: row["horizon"],
                )
                ax.plot(
                    [row["horizon"] for row in values],
                    [row[metric] for row in values],
                    label=method,
                )
            ax.set(title=f"Validation / {region}", xlabel="Prediction steps", ylabel=metric)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.savefig(output / "rollout_position_velocity.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(starts), 2, figsize=(13, 3 * len(starts)), squeeze=False, layout="constrained"
    )
    for index, start in enumerate(starts):
        for column, quantity in enumerate(("rms_deformation_um", "rms_speed_mm_s")):
            ax = axes[index, column]
            for method_index, method in enumerate(predictors):
                rows = [
                    row
                    for row in profiles
                    if row["trajectory"] == "left_deep"
                    and row["start"] == start
                    and row["method"] == method
                ]
                if not rows:
                    continue
                times = [row["time_s"] for row in rows]
                if method_index == 0:
                    ax.plot(
                        times,
                        [row[f"true_{quantity}"] for row in rows],
                        "k--",
                        label="MuJoCo label",
                    )
                ax.plot(times, [row[f"predicted_{quantity}"] for row in rows], label=method)
            ax.set(title=f"left_deep / start frame {start}", xlabel="Time (s)", ylabel=quantity)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
    fig.savefig(output / "left_deep_deformation_speed_windows.png", dpi=150)
    plt.close(fig)
