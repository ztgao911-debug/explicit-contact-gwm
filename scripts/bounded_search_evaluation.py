"""Frozen search evaluation, physical gate, and explicit failed-window evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from contact_response_evaluation import evaluate_responses
from joint_state_evaluation import _check, evaluate_models

from contact_gwm.budget_evaluation import (
    physical_archive_checks,
    physical_state_checks,
    recovery_waveform_metrics,
)
from contact_gwm.pinned_contact_pipeline import IndenterInput
from contact_gwm.search_protocol import fixed_metric_scores

ROOT = Path(__file__).resolve().parents[1]


def _array(value):
    return (
        value.detach().cpu().numpy().copy()
        if isinstance(value, torch.Tensor)
        else np.array(value, copy=True)
    )


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _input_snapshot(state, robot):
    return {
        "input_position": _array(state.centers),
        "input_velocity": _array(state.velocities),
        "input_features": _array(state.features),
        "robot_position": _array(robot.position),
        "robot_velocity": _array(robot.velocity),
        "command_position": _array(robot.target_position),
        "command_velocity": _array(robot.target_velocity),
    }


def _result_snapshot(result):
    return {
        "output_position": _array(result.state.centers),
        "output_velocity": _array(result.state.velocities),
        "output_increment": _array(result.delta),
        "output_force": _array(result.forces.forces),
        "output_contact_points": _array(result.contacts.points),
        "output_contact_normals": _array(result.contacts.normals),
        "output_contact_signed_gaps": _array(result.contacts.signed_gaps),
        "output_contact_relative_velocities": _array(result.contacts.relative_velocities),
        "output_torque": _array(result.forces.torques),
        "output_gate": _array(result.gates),
        "output_residual": _array(result.residual),
    }


def _replay_failed_windows(predictors, legacy, trajectories, mesh, config, failures, folder):
    """Replay only actual failures to retain partial arrays; never rescue a score."""
    folder.mkdir(parents=True, exist_ok=True)
    steps = max(config["rollout"]["horizons"])
    reports = []
    for index, failure in enumerate(failures):
        started = time.perf_counter()
        name, start, method = failure["trajectory"], int(failure["start"]), failure["method"]
        arrays = trajectories[name]
        state = legacy.gaussian_state(
            arrays["node_position_world"][start], arrays["node_velocity_world"][start]
        )
        external = [IndenterInput.from_arrays(arrays, t) for t in range(start, start + steps)]
        x, v = [_array(state.centers)], [_array(state.velocities)]
        checks = {
            "nonfinite_result_count": 0,
            "max_fixed_position_error_m": 0.0,
            "max_fixed_velocity_m_s": 0.0,
            "max_fixed_increment_m": 0.0,
        }
        bad_output, last_input, error = {}, {}, None
        for robot in external:
            last_input = _input_snapshot(state, robot)
            try:
                result = predictors[method](state, robot)
                bad_output = _result_snapshot(result)
                px, pv = _check(result, mesh, checks)
                x.append(px)
                v.append(pv)
                state = result.state
                bad_output = {}
            except (ValueError, RuntimeError, FloatingPointError) as exception:
                error = {"type": type(exception).__name__, "message": str(exception)}
                break
        saved = {
            "node_ids": mesh["node_ids"],
            "position": np.asarray(x),
            "velocity": np.asarray(v),
            "external_position": np.asarray([robot.position for robot in external]),
            "external_velocity": np.asarray([robot.velocity for robot in external]),
            "external_command_position": np.asarray([robot.target_position for robot in external]),
            "external_command_velocity": np.asarray([robot.target_velocity for robot in external]),
            # Reads after prediction; scoring labels are never arguments to the predictor.
            "reference_position_scoring_only": arrays["node_position_world"][
                start : start + steps + 1
            ],
            "reference_velocity_scoring_only": arrays["node_velocity_world"][
                start : start + steps + 1
            ],
            **last_input,
            **bad_output,
        }
        path = folder / f"failure_{index:03d}.npz"
        np.savez_compressed(path, **saved)
        reports.append(
            {
                **failure,
                "replay_completed_steps": len(x) - 1,
                "replay_error": error,
                "replay_seconds": time.perf_counter() - started,
                "evidence_file": path.name,
                "original_failure_remains_disqualifying": True,
            }
        )
    _json(folder / "replay_records.json", reports)
    return reports


def _response_physics(folder, predictors, mesh, config):
    rows = []
    variants = [row["name"] for row in config["_variants"]]
    for frame in config["collection"]["validation_frames"]:
        for variant in variants:
            path = folder / f"{frame}_{variant}_predictions.npz"
            for method in predictors:
                metadata = {"frame": frame, "variant": variant, "method": method}
                if not path.exists():
                    rows.append({**metadata, "eligible": False, "error": "missing_case"})
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    xkey, vkey = f"{method}__position", f"{method}__velocity"
                    if xkey not in saved or vkey not in saved:
                        rows.append({**metadata, "eligible": False, "error": "missing_method"})
                    else:
                        rows.append(
                            {
                                **metadata,
                                **physical_state_checks(
                                    saved[xkey], saved[vkey], mesh, expected_steps=20
                                ),
                            }
                        )
    result = {}
    for method in predictors:
        values = [row for row in rows if row["method"] == method]
        minima = [
            row["minimum_volume_ratio"]
            for row in values
            if row.get("minimum_volume_ratio") is not None
        ]
        result[method] = {
            "eligible": bool(values and all(row["eligible"] for row in values)),
            "expected_windows": len(config["collection"]["validation_frames"]) * len(variants),
            "incomplete_or_invalid_windows": sum(not row["eligible"] for row in values),
            "inverted_windows": sum(row.get("inverted_states", 0) > 0 for row in values),
            "inverted_states": sum(row.get("inverted_states", 0) for row in values),
            "nonfinite_states": sum(row.get("nonfinite_states", 0) for row in values),
            "fixed_boundary_violating_states": sum(
                row.get("fixed_boundary_violating_states", 0) for row in values
            ),
            "minimum_volume_ratio": min(minima) if minima else None,
        }
    return {"methods": result, "windows": rows}


def _static_stability(archive, predictors, mesh, rollout_archive=None):
    rows = []
    rest = mesh["rest_positions_world"]
    with np.load(archive, allow_pickle=False) as saved:
        for method in predictors:
            xkey, vkey = (
                f"no_contact__{method}__next_position",
                f"no_contact__{method}__next_velocity",
            )
            if xkey not in saved or vkey not in saved:
                continue
            rows.append(
                {
                    "method": method,
                    "source": "one_step_no_contact",
                    "max_absolute_displacement_m": float(np.max(np.abs(saved[xkey] - rest))),
                    "max_absolute_instantaneous_velocity_m_s": float(np.max(np.abs(saved[vkey]))),
                }
            )
    if rollout_archive is not None:
        with np.load(rollout_archive, allow_pickle=False) as saved:
            for key in saved.files:
                if not key.startswith("no_contact__") or not key.endswith("__position"):
                    continue
                parts = key.split("__")
                method = "__".join(parts[2:-1])
                velocity_key = key.removesuffix("__position") + "__velocity"
                if method not in predictors or velocity_key not in saved:
                    continue
                rows.append(
                    {
                        "method": method,
                        "source": "rollout_no_contact",
                        "start": int(parts[1]),
                        "max_absolute_displacement_m": float(np.max(np.abs(saved[key] - rest))),
                        "max_absolute_instantaneous_velocity_m_s": float(
                            np.max(np.abs(saved[velocity_key]))
                        ),
                    }
                )
    return rows


def evaluate_trial(predictors, legacy, trajectories, splits, mesh, config, output):
    """One immutable evaluator for search candidates, seed checks, and M/F/G.

    Scores cover four phases x two regions x three horizons, with fixed equal
    position/velocity evaluation weights. Search hyperparameters never alter
    this function's scoring. Physical screening covers every ordinary window,
    no-contact control, and all existing 18 intervention branches.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last_calls = {}
    last_started = [None]
    forward_contracts = {
        method: {
            "nonfinite_outputs": 0,
            "max_fixed_increment_m": 0.0,
            "max_fixed_position_error_m": 0.0,
            "max_fixed_velocity_m_s": 0.0,
        }
        for method in predictors
    }

    def record(method, predictor):
        def wrapped(state, robot):
            last_started[0] = method
            last_calls[method] = _input_snapshot(state, robot)
            result = predictor(state, robot)
            snapshot = _result_snapshot(result)
            last_calls[method].update(snapshot)
            contract = forward_contracts[method]
            contract["nonfinite_outputs"] += int(
                any(not np.isfinite(value).all() for value in snapshot.values())
            )
            for field, error in (
                (
                    "max_fixed_increment_m",
                    np.abs(snapshot["output_increment"][mesh["pinned_mask"]]),
                ),
                (
                    "max_fixed_position_error_m",
                    np.abs(
                        snapshot["output_position"][mesh["pinned_mask"]]
                        - mesh["rest_positions_world"][mesh["pinned_mask"]]
                    ),
                ),
                (
                    "max_fixed_velocity_m_s",
                    np.abs(snapshot["output_velocity"][mesh["pinned_mask"]]),
                ),
            ):
                contract[field] = max(contract[field], float(error.max(initial=0)))
            return result

        return wrapped

    recorded = {method: record(method, predictor) for method, predictor in predictors.items()}
    try:
        ordinary = evaluate_models(
            recorded, legacy.gaussian_state, trajectories, splits, mesh, config, output / "ordinary"
        )
    except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as exception:
        folder = output / "failed_forward"
        folder.mkdir(exist_ok=True)
        metadata = []
        for method, snapshot in last_calls.items():
            path = folder / f"{method}.npz"
            np.savez_compressed(path, **snapshot)
            matches = []
            for name, arrays in trajectories.items():
                for frame, (x, v) in enumerate(
                    zip(arrays["node_position_world"], arrays["node_velocity_world"], strict=True)
                ):
                    if np.array_equal(x, snapshot["input_position"]) and np.array_equal(
                        v, snapshot["input_velocity"]
                    ):
                        matches.append({"trajectory": name, "frame": frame})
            metadata.append(
                {
                    "method": method,
                    "matching_true_current_frames": matches,
                    "input_state_sha256": hashlib.sha256(
                        snapshot["input_position"].tobytes() + snapshot["input_velocity"].tobytes()
                    ).hexdigest(),
                    "file": path.name,
                }
            )
        _json(
            folder / "failure.json",
            {
                "type": type(exception).__name__,
                "message": str(exception),
                "last_forward_evidence": metadata,
                "last_started_method": last_started[0],
            },
        )
        raise
    replay = _replay_failed_windows(
        predictors,
        legacy,
        trajectories,
        mesh,
        config,
        ordinary["checks"]["rollout_failures"],
        output / "failed_windows",
    )
    response = evaluate_responses(recorded, legacy, trajectories, mesh, config, output, ROOT)
    expected = [
        (name, int(start)) for name in trajectories for start in config["rollout"]["start_frames"]
    ]
    physics = physical_archive_checks(
        output / "ordinary/rollout_predictions.npz",
        mesh,
        expected_windows=expected,
        expected_steps=max(config["rollout"]["horizons"]),
    )
    response_physics = _response_physics(output / "response_evaluation", predictors, mesh, config)
    _json(output / "ordinary_physics.json", physics)
    _json(output / "response_physics.json", response_physics)
    _csv(output / "ordinary_physics_windows.csv", physics["windows"])
    _csv(output / "response_physics_windows.csv", response_physics["windows"])
    recovery = recovery_waveform_metrics(
        output / "ordinary/rollout_predictions.npz", mesh, trajectories, output / "recovery"
    )
    static = _static_stability(
        output / "ordinary/one_step_predictions.npz",
        predictors,
        mesh,
        output / "ordinary/rollout_predictions.npz",
    )
    _csv(output / "static_stability.csv", static)
    per_method = {}
    for method in predictors:
        rows = [row for row in ordinary["rollout"] if row["method"] == method]
        scores = fixed_metric_scores(rows)
        op = physics["methods"].get(method, {})
        rp = response_physics["methods"].get(method, {})
        failures = [
            row for row in ordinary["checks"]["rollout_failures"] if row["method"] == method
        ]
        response_failures = [row for row in response["failures"] if row["method"] == method]
        eligible = (
            scores["coverage_complete"]
            and op.get("eligible", False)
            and rp.get("eligible", False)
            and not failures
            and not response_failures
            and not any(forward_contracts[method].values())
        )
        minima = [
            p["minimum_volume_ratio"] for p in (op, rp) if p.get("minimum_volume_ratio") is not None
        ]
        per_method[method] = {
            "eligible": bool(eligible),
            "scores": scores,
            "ordinary_physics": op,
            "intervention_physics": rp,
            "inverted_states": op.get("inverted_states", 0) + rp.get("inverted_states", 0),
            "inverted_windows": op.get("inverted_windows", 0) + rp.get("inverted_windows", 0),
            "minimum_volume_ratio": min(minima) if minima else None,
            "ordinary_failures": failures,
            "intervention_failures": response_failures,
            "forward_contracts": forward_contracts[method],
        }
    result = {
        "per_method": per_method,
        "ordinary": ordinary,
        "response": response,
        "recovery": recovery,
        "static_stability": static,
        "failed_window_replays": replay,
        "independent_test_verified": False,
    }
    _json(output / "evaluation_summary.json", result)
    return result
