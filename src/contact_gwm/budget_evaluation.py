"""Fixed short-rollout scoring, physical validity, and signed recovery evidence.

These routines are scoring only. No reference state, reference mask, or future
quantity crosses a predictor boundary. Incomplete or invalid windows remain
explicit and cannot enter the eligible ranking by disappearing from an average.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .data.mujoco_soft import tetra_volumes


def physical_state_checks(
    positions: np.ndarray,
    velocities: np.ndarray,
    mesh: Mapping,
    *,
    expected_steps: int | None = None,
    pin_tolerance: float = 0.0,
) -> dict:
    """Check all saved states, including initialization; no invalid state is cut.

    Counts refer to recorded states, not individual inverted tetrahedra. A
    negative or zero signed volume ratio is a mesh failure. Pin tolerance is
    zero by default because the model enforces the boundary by construction.
    """
    x, v = np.asarray(positions), np.asarray(velocities)
    rest = np.asarray(mesh["rest_positions_world"])
    pinned = np.asarray(mesh["pinned_mask"], dtype=bool)
    if x.ndim != 3 or x.shape[1:] != rest.shape or v.shape != x.shape:
        return {"eligible": False, "error": "state_shape_mismatch", "completed_steps": 0}
    finite = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(v).all(axis=(1, 2))
    pin_x = np.abs(x[:, pinned] - rest[pinned])
    pin_v = np.abs(v[:, pinned])
    pin_bad = np.any(pin_x > pin_tolerance, axis=(1, 2)) | np.any(
        pin_v > pin_tolerance, axis=(1, 2)
    )
    increment_bad = (
        np.any(np.abs(np.diff(x[:, pinned], axis=0)) > pin_tolerance, axis=(1, 2))
        if len(x) > 1
        else np.zeros(0, dtype=bool)
    )
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        ratios = tetra_volumes(x, np.asarray(mesh["tetrahedra"])) / np.asarray(
            mesh["rest_signed_tetra_volumes"]
        )
    finite_volume = np.isfinite(ratios).all(axis=-1)
    inverted = np.any(ratios <= 0, axis=-1)
    finite_values = ratios[np.isfinite(ratios)]
    finite_pin_x, finite_pin_v = pin_x[np.isfinite(pin_x)], pin_v[np.isfinite(pin_v)]
    complete = expected_steps is None or len(x) == expected_steps + 1
    return {
        "eligible": bool(
            finite.all()
            and finite_volume.all()
            and not pin_bad.any()
            and not increment_bad.any()
            and not inverted.any()
            and complete
        ),
        "recorded_states": len(x),
        "completed_steps": max(0, len(x) - 1),
        "expected_steps": expected_steps,
        "complete": bool(complete),
        "nonfinite_states": int((~finite).sum()),
        "nonfinite_volume_states": int((~finite_volume).sum()),
        "fixed_boundary_violating_states": int(pin_bad.sum()),
        "fixed_increment_violating_states": int(increment_bad.sum()),
        "max_fixed_position_error_m": float(finite_pin_x.max(initial=0)),
        "max_fixed_velocity_m_s": float(finite_pin_v.max(initial=0)),
        "inverted_states": int(inverted.sum()),
        "inverted_horizons": np.flatnonzero(inverted).astype(int).tolist(),
        "minimum_volume_ratio": float(finite_values.min()) if finite_values.size else None,
    }


def _window_key(key):
    if not key.endswith("__position"):
        return None
    parts = key.split("__")
    if len(parts) < 4:
        return None
    try:
        start = int(parts[1])
    except ValueError:
        return None
    return parts[0], start, "__".join(parts[2:-1])


def physical_archive_checks(
    archive_path: str | Path,
    mesh: Mapping,
    *,
    method: str | None = None,
    expected_windows: Iterable[tuple[str, int]] | None = None,
    expected_steps: int | None = 20,
) -> dict:
    """Disaggregate methods/windows of a common evaluator rollout NPZ.

    Callers should pass the protocol's expected windows. Inferring from saved
    keys alone cannot detect a window missing for every method.
    """
    rows, found = [], defaultdict(set)
    with np.load(archive_path, allow_pickle=False) as archive:
        if "node_ids" not in archive or not np.array_equal(archive["node_ids"], mesh["node_ids"]):
            return {"eligible": False, "error": "node_ids_mismatch", "methods": {}, "windows": []}
        discovered = []
        for key in archive.files:
            parsed = _window_key(key)
            if parsed is not None and (method is None or parsed[2] == method):
                discovered.append((key, parsed))
        names = {parsed[2] for _, parsed in discovered}
        if method is not None:
            names.add(method)
        expected = (
            {(str(name), int(start)) for name, start in expected_windows}
            if expected_windows is not None
            else {(parsed[0], parsed[1]) for _, parsed in discovered}
        )
        for key, (name, start, candidate) in discovered:
            velocity_key = key.removesuffix("__position") + "__velocity"
            if velocity_key not in archive:
                check = {"eligible": False, "error": "missing_velocity", "completed_steps": 0}
            else:
                check = physical_state_checks(
                    archive[key], archive[velocity_key], mesh, expected_steps=expected_steps
                )
            rows.append({"trajectory": name, "start": start, "method": candidate, **check})
            found[candidate].add((name, start))
        for candidate in sorted(names):
            for name, start in sorted(expected - found[candidate]):
                rows.append(
                    {
                        "trajectory": name,
                        "start": start,
                        "method": candidate,
                        "eligible": False,
                        "error": "missing_window",
                        "completed_steps": 0,
                    }
                )
    summaries = {}
    for candidate in sorted(names):
        selected = [row for row in rows if row["method"] == candidate]
        minima = [
            row["minimum_volume_ratio"]
            for row in selected
            if row.get("minimum_volume_ratio") is not None
        ]
        summaries[candidate] = {
            "eligible": bool(selected and all(row["eligible"] for row in selected)),
            "expected_windows": len(expected),
            "saved_windows": len(found[candidate]),
            "missing_windows": len(expected - found[candidate]),
            "incomplete_or_invalid_windows": sum(not row["eligible"] for row in selected),
            "nonfinite_states": sum(row.get("nonfinite_states", 0) for row in selected),
            "fixed_boundary_violating_states": sum(
                row.get("fixed_boundary_violating_states", 0) for row in selected
            ),
            "inverted_states": sum(row.get("inverted_states", 0) for row in selected),
            "inverted_windows": sum(row.get("inverted_states", 0) > 0 for row in selected),
            "minimum_volume_ratio": min(minima) if minima else None,
            "max_fixed_position_error_m": max(
                (row.get("max_fixed_position_error_m", 0.0) for row in selected), default=0.0
            ),
            "max_fixed_velocity_m_s": max(
                (row.get("max_fixed_velocity_m_s", 0.0) for row in selected), default=0.0
            ),
        }
    return {
        "eligible": bool(summaries and all(row["eligible"] for row in summaries.values())),
        "methods": summaries,
        "windows": rows,
        "expected_window_list_provided": expected_windows is not None,
    }


def assess_existing_evaluation(evaluation: Mapping, *, method: str, physical_checks=None) -> dict:
    """Use the one frozen ranking on existing evaluator metrics and checks."""
    from .search_protocol import fixed_metric_scores

    rows = [row for row in evaluation["rollout"] if row["method"] == method]
    scores = fixed_metric_scores(rows)
    checks = evaluation["checks"]
    reasons = []
    for key in (
        "nonfinite_result_count",
        "max_fixed_position_error_m",
        "max_fixed_velocity_m_s",
        "max_fixed_increment_m",
        "rollout_exception_count",
    ):
        if checks.get(key, 0):
            reasons.append(key)
    if checks.get("common_evaluation_masks") is False:
        reasons.append("incomplete_evaluation_windows")
    if physical_checks is None:
        if checks.get("rollout_states_with_inverted_tetrahedra", 0):
            reasons.append("inverted_tetrahedra")
    else:
        selected = physical_checks.get("methods", {}).get(method, {})
        if not selected.get("eligible", False):
            reasons.append("physical_archive_check_failed")
    complete = scores.get("coverage_complete", scores.get("complete", False))
    if not complete:
        reasons.append("ranking_cells_incomplete")
    return {
        "method": method,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "scores": scores,
    }


def _crossings(signal):
    centered = np.asarray(signal) - np.mean(signal)
    ids = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0))
    return ids + (-centered[ids]) / (centered[ids + 1] - centered[ids])


def signed_waveform_summary(prediction, reference, dt_s: float) -> dict:
    """Amplitude and a conservative lag/phase proxy; no invented cycle phase.

    Both signals must contain three upward crossings (at least two periods) for
    a phase proxy. Short monotone recovery is marked unverified, not phase zero.
    Positive lag means prediction occurs later than the reference.
    """
    p, r = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    if p.ndim != 1 or p.shape != r.shape or len(p) < 2 or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Aligned 1D waveforms and positive dt required")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise FloatingPointError("Non-finite signed recovery waveform")
    pa, ra = float(np.ptp(p) / 2), float(np.ptp(r) / 2)
    pc, rc = _crossings(p), _crossings(r)
    result = {
        "predicted_mean": float(p.mean()),
        "reference_mean": float(r.mean()),
        "bias": float((p - r).mean()),
        "rmse": float(np.sqrt(np.square(p - r).mean())),
        "predicted_half_peak_to_peak_amplitude": pa,
        "reference_half_peak_to_peak_amplitude": ra,
        "amplitude_error": pa - ra,
        "amplitude_ratio": pa / ra if ra > 1e-12 else None,
        "predicted_upward_crossings": len(pc),
        "reference_upward_crossings": len(rc),
        "phase_status": "insufficient_cycles",
        "lag_s": None,
        "phase_error_degrees": None,
        "reference_period_s": None,
        "predicted_period_s": None,
        "maximum_lag_correlation": None,
    }
    if min(len(pc), len(rc)) < 3 or min(pa, ra) <= 1e-12:
        return result
    ref_period = float(np.median(np.diff(rc)) * dt_s)
    pred_period = float(np.median(np.diff(pc)) * dt_s)
    maximum_lag = min(len(p) // 2, int(np.floor(ref_period / (2 * dt_s))))
    candidates = []
    p, r = p - p.mean(), r - r.mean()
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag > 0:
            a, b = p[lag:], r[:-lag]
        elif lag < 0:
            a, b = p[:lag], r[-lag:]
        else:
            a, b = p, r
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        correlation = float(a @ b / norm) if norm > 1e-30 else -1.0
        candidates.append((correlation, -abs(lag), -lag, lag))
    correlation, _, _, lag = max(candidates)
    result.update(
        {
            "phase_status": "lag_proxy_at_least_two_cycles",
            "lag_s": lag * dt_s,
            "phase_error_degrees": 360 * lag * dt_s / ref_period,
            "reference_period_s": ref_period,
            "predicted_period_s": pred_period,
            "maximum_lag_correlation": correlation,
        }
    )
    return result


def _reference_rois(arrays, mesh, radius=0.02):
    rest = mesh["rest_positions_world"]
    active = ~mesh["pinned_mask"]
    point = arrays["indenter_position_world"][0].copy()
    point[2] = rest[:, 2].max()
    previous = (np.linalg.norm(rest - point, axis=1) <= radius) & active
    masks = []
    for frame in range(len(arrays["time"])):
        a, b = arrays["contact_offsets"][frame : frame + 2]
        ids = np.arange(a, b)
        ids = ids[
            (arrays["contact_signed_distance"][ids, 0] <= 0)
            & (np.linalg.norm(arrays["contact_force_on_block_world"][ids], axis=1) > 1e-8)
        ]
        if len(ids):
            distance = np.linalg.norm(
                arrays["node_position_world"][frame, :, None]
                - arrays["contact_position_world"][ids],
                axis=-1,
            )
            previous = (distance.min(axis=1) <= radius) & active
        masks.append(previous.copy())
    return np.asarray(masks)


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recovery_waveform_metrics(
    archive_path: str | Path,
    mesh: Mapping,
    trajectories: Mapping,
    output: str | Path,
    *,
    method: str | None = None,
    recover_starts=(170, 210),
    roi_masks: Mapping | None = None,
) -> dict:
    """Save signed z/vz curves at reproducible material node and mean regions.

    Full nodal x/v are already in the input NPZ. A representative material node
    is the REST top active node nearest initial indenter XY, chosen independently
    of predictions and held fixed across methods/windows. ROI is frozen at the
    initialization just as in the common evaluator. Mean curves can cancel, so
    the representative-node channel is always reported alongside them.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rest, active = np.asarray(mesh["rest_positions_world"]), ~np.asarray(mesh["pinned_mask"])
    top = np.flatnonzero(active & np.isclose(rest[:, 2], rest[:, 2].max()))
    if not len(top):
        raise ValueError("No active top material node for a fixed representative")
    rois = (
        roi_masks
        if roi_masks is not None
        else {name: _reference_rois(arrays, mesh) for name, arrays in trajectories.items()}
    )
    waveform_rows, summaries, missing = [], [], []
    seen, methods = set(), {method} if method is not None else set()
    with np.load(archive_path, allow_pickle=False) as archive:
        for key in archive.files:
            parsed = _window_key(key)
            if parsed is None:
                continue
            name, start, candidate = parsed
            if start not in recover_starts or (method is not None and candidate != method):
                continue
            if name not in trajectories:
                continue
            methods.add(candidate)
            seen.add((name, start, candidate))
            velocity_key = key.removesuffix("__position") + "__velocity"
            if velocity_key not in archive:
                missing.append(
                    {
                        "trajectory": name,
                        "start": start,
                        "method": candidate,
                        "error": "missing_velocity",
                    }
                )
                continue
            arrays = trajectories[name]
            x, v = archive[key], archive[velocity_key]
            steps = len(x)
            if steps < 2:
                missing.append(
                    {
                        "trajectory": name,
                        "start": start,
                        "method": candidate,
                        "error": "insufficient_samples",
                    }
                )
                continue
            if start + steps > len(arrays["time"]):
                raise ValueError("Recovery window exceeds reference trajectory")
            t = arrays["time"][start : start + steps]
            dt = float(np.median(np.diff(t)))
            representative = int(
                top[
                    np.argmin(
                        np.linalg.norm(
                            rest[top, :2] - arrays["indenter_position_world"][0, :2], axis=-1
                        )
                    )
                ]
            )
            channels = {
                "active_mean": active,
                "contact_neighborhood_mean": rois[name][start],
                "representative_node": np.arange(len(rest)) == representative,
            }
            for channel, mask in channels.items():
                if not np.any(mask):
                    missing.append(
                        {
                            "trajectory": name,
                            "start": start,
                            "method": candidate,
                            "channel": channel,
                            "error": "empty_mask",
                        }
                    )
                    continue
                true_x = arrays["node_position_world"][start : start + steps]
                true_v = arrays["node_velocity_world"][start : start + steps]
                pz = (x[:, mask, 2] - rest[mask, 2]).mean(axis=1) * 1e6
                rz = (true_x[:, mask, 2] - rest[mask, 2]).mean(axis=1) * 1e6
                pv = v[:, mask, 2].mean(axis=1) * 1e3
                rv = true_v[:, mask, 2].mean(axis=1) * 1e3
                metadata = {
                    "trajectory": name,
                    "start": start,
                    "method": candidate,
                    "channel": channel,
                    "representative_node": representative,
                    "node_count": int(np.sum(mask)),
                    "recorded_states": steps,
                }
                for h in range(steps):
                    waveform_rows.append(
                        {
                            **metadata,
                            "horizon": h,
                            "time_s": float(t[h]),
                            "predicted_z_displacement_um": float(pz[h]),
                            "reference_z_displacement_um": float(rz[h]),
                            "predicted_vz_mm_s": float(pv[h]),
                            "reference_vz_mm_s": float(rv[h]),
                        }
                    )
                metrics = dict(metadata)
                for label, prediction, reference in (("position", pz, rz), ("velocity", pv, rv)):
                    metrics.update(
                        {
                            f"{label}_{k}": value
                            for k, value in signed_waveform_summary(
                                prediction, reference, dt
                            ).items()
                        }
                    )
                summaries.append(metrics)
    for name, arrays in trajectories.items():
        for start in recover_starts:
            if start >= len(arrays["time"]):
                continue
            for candidate in methods:
                if (name, start, candidate) not in seen:
                    missing.append(
                        {
                            "trajectory": name,
                            "start": start,
                            "method": candidate,
                            "error": "missing_window",
                        }
                    )
    _write_csv(output / "recovery_signed_waveforms.csv", waveform_rows)
    _write_csv(output / "recovery_waveform_metrics.csv", summaries)
    result = {
        "waveforms": len(summaries),
        "samples": len(waveform_rows),
        "missing": missing,
        "representative_rule": "rest_top_active_node_nearest_initial_indenter_xy",
        "phase_proxy_minimum_upward_crossings_each_signal": 3,
        "phase_unverified_waveforms": sum(
            row["position_phase_status"] == "insufficient_cycles" for row in summaries
        ),
    }
    (output / "recovery_checks.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result
