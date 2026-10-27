"""Geometry-only CPU comparison of sparse and topology-bound dense Gaussians."""

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
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from validate_contact_pipeline import hashes, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.pinned_contact_pipeline import IndenterInput, NativeCurrentContact
from contact_gwm.structures import GaussianSet
from contact_gwm.surface_proxies import BoundSurfaceGaussians, closest_triangle_surface

ROOT = Path(__file__).resolve().parents[1]
PHASES = {0: "approach", 1: "press", 2: "hold", 3: "release", 4: "recover"}
CURRENT_FIELDS = (
    "node_position_world",
    "node_velocity_world",
    "indenter_position_world",
    "indenter_velocity_world",
    "action_target_position",
    "action_target_velocity",
)
METHODS = ("sparse", "dense")


def current_inputs(arrays, frame):
    """No future index, contact, force or integration-state field crosses here."""
    return tuple(torch.tensor(arrays[key][frame], dtype=torch.float64) for key in CURRENT_FIELDS)


def dynamics_state(x, v, mesh):
    rest, pinned = torch.tensor(mesh["rest_positions_world"]), torch.tensor(mesh["pinned_mask"])
    x, v = torch.where(pinned[:, None], rest, x), torch.where(pinned[:, None], 0.0, v)
    rotations = x.new_zeros((len(x), 4))
    rotations[:, 0] = 1
    return GaussianSet(
        x,
        x.new_full(x.shape, 0.005).log(),
        rotations,
        x.new_zeros((len(x), 1)),
        x.new_zeros((len(x), 0)),
        v,
    )


def native_query(native, state, robot):
    # Force oracle is explicitly disabled; this experiment never calls mj_step.
    contacts, force = native.query(state, robot, with_force=False)
    assert force is None
    return contacts


def rest_features(mesh, offset_grid=None):
    """All vertices/edges/faces, with an outward normal in the rest normal cone."""
    rest, triangles = mesh["rest_positions_world"], mesh["surface_triangles"]
    center = rest.mean(0)
    features = {}
    for triangle in triangles:
        vertices = rest[triangle]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, vertices.mean(0) - center) < 0:
            normal *= -1
        for ids in (
            *((int(node),) for node in triangle),
            tuple(sorted(map(int, triangle[:2]))),
            tuple(sorted(map(int, triangle[1:]))),
            tuple(sorted(map(int, triangle[[0, 2]]))),
            tuple(sorted(map(int, triangle))),
        ):
            if ids not in features:
                features[ids] = [rest[list(ids)].mean(0), normal.copy()]
            else:
                features[ids][1] += normal
    result = []
    for index, (ids, (point, normal)) in enumerate(features.items()):
        normal /= np.linalg.norm(normal)
        boundaries = sum(
            np.isclose(point[axis], rest[:, axis].min(), atol=1e-12, rtol=0)
            or np.isclose(point[axis], rest[:, axis].max(), atol=1e-12, rtol=0)
            for axis in range(3)
        )
        region = (
            "corner"
            if boundaries == 3
            else "edge"
            if boundaries == 2
            else ("top" if np.isclose(point[2], rest[:, 2].max()) else "side_or_bottom")
        )
        result.append(
            {
                "feature": index,
                "kind": {1: "vertex", 2: "edge_midpoint", 3: "face_centroid"}[len(ids)],
                "region": region,
                "point": point,
                "normal": normal,
            }
        )
    if offset_grid is not None:
        count = offset_grid["count_per_axis"]
        phase = offset_grid["fractional_offset"]
        if count < 2 or not 0 < phase < 1:
            raise ValueError("Invalid fixed offset grid")
        fractions = (np.arange(count) + phase) / count
        steps = [np.diff(np.unique(rest[:, axis])).min() for axis in (0, 1)]
        for u in fractions:
            for v in fractions:
                point = np.array(
                    [center[0] + u * steps[0], center[1] + v * steps[1], rest[:, 2].max()]
                )
                result.append(
                    {
                        "feature": len(result),
                        "kind": "offset_grid",
                        "region": "top",
                        "point": point,
                        "normal": np.array([0.0, 0.0, 1.0]),
                    }
                )
    return result


def geometry_row(query, native, reference, **metadata):
    contacts = query.contacts
    matched = bool(contacts.num_contacts and native.num_contacts)
    point_error = normal_error = native_gap_error = None
    if matched:
        distance = torch.cdist(contacts.points, native.points).numpy()
        point_error = float(0.5 * (distance.min(0).mean() + distance.min(1).mean()) * 1e3)
        nearest = distance.argmin(1)
        cosine = (contacts.normals.numpy() * native.normals.numpy()[nearest]).sum(-1)
        normal_error = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())
        native_gap_error = float((query.closest_gap - native.signed_gaps.min()) * 1e3)
    gap = float(query.closest_gap)
    reference_cosine = np.clip(np.dot(query.closest_normal.numpy(), reference[2]), -1, 1)
    return {
        **metadata,
        "native_contact": bool(native.num_contacts),
        "native_count": native.num_contacts,
        "mesh_contact": reference[0] <= 0,
        "predicted_contact": bool(contacts.num_contacts),
        "predicted_count": contacts.num_contacts,
        "closest_gap_mm": gap * 1e3,
        "mesh_gap_mm": reference[0] * 1e3,
        "gap_bias_vs_mesh_mm": (gap - reference[0]) * 1e3,
        "closest_normal_error_vs_mesh_deg": float(np.degrees(np.arccos(reference_cosine))),
        "closest_point_error_vs_mesh_mm": float(
            np.linalg.norm(query.closest_point.numpy() - reference[1]) * 1e3
        ),
        "matched_point_error_mm": point_error,
        "matched_normal_error_deg": normal_error,
        "matched_gap_error_vs_native_mm": native_gap_error,
    }


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        row = dict(zip(fields, key, strict=True))
        row["queries"] = len(values)
        for label, reference in (("native", "native_contact"), ("mesh", "mesh_contact")):
            tp = sum(r[reference] and r["predicted_contact"] for r in values)
            fp = sum(not r[reference] and r["predicted_contact"] for r in values)
            fn = sum(r[reference] and not r["predicted_contact"] for r in values)
            row.update(
                {
                    f"{label}_tp": tp,
                    f"{label}_fp": fp,
                    f"{label}_fn": fn,
                    f"{label}_recall": tp / (tp + fn) if tp + fn else None,
                    f"{label}_precision": tp / (tp + fp) if tp + fp else None,
                }
            )
        row["native_mesh_disagreement_frames"] = sum(
            r["native_contact"] != r["mesh_contact"] for r in values
        )
        row["mean_contact_count"] = float(np.mean([r["predicted_count"] for r in values]))
        row["max_contact_count"] = max(r["predicted_count"] for r in values)
        gaps = [r["gap_bias_vs_mesh_mm"] for r in values]
        row["gap_bias_vs_mesh_mean_mm"] = float(np.mean(gaps))
        row["gap_rmse_vs_mesh_mm"] = float(np.sqrt(np.mean(np.square(gaps))))
        row["gap_bias_vs_mesh_max_mm"] = max(gaps)
        contact = [r for r in values if r["native_contact"]]
        row["native_contact_frames_gap_rmse_vs_mesh_mm"] = (
            float(np.sqrt(np.mean([r["gap_bias_vs_mesh_mm"] ** 2 for r in contact])))
            if contact
            else None
        )
        for field in ("closest_normal_error_vs_mesh_deg", "closest_point_error_vs_mesh_mm"):
            row[f"native_contact_frames_mean_{field}"] = (
                float(np.mean([r[field] for r in contact])) if contact else None
            )
        row["matched_frames"] = sum(r["matched_point_error_mm"] is not None for r in values)
        for name in (
            "matched_point_error_mm",
            "matched_normal_error_deg",
            "matched_gap_error_vs_native_mm",
        ):
            samples = [r[name] for r in values if r[name] is not None]
            row[f"mean_{name}"] = float(np.mean(samples)) if samples else None
        summaries.append(row)
    return summaries


def pack_contacts(records):
    offsets = np.cumsum([0, *(r.num_contacts for r in records)])
    return {
        "offsets": offsets,
        **{
            key: np.concatenate([getattr(r, key).numpy() for r in records], axis=0)
            for key in ("object_indices", "points", "normals", "signed_gaps", "relative_velocities")
        },
    }


@torch.no_grad()
def static_audit(mesh, native, surfaces, config, output):
    x = torch.tensor(mesh["rest_positions_world"])
    v = torch.zeros_like(x)
    state = dynamics_state(x, v, mesh)
    features = rest_features(mesh, config["static"]["offset_grid"])
    rows, samples, onset = [], [], []
    records = defaultdict(list)
    radius, skin = config["sphere_radius_m"], config["skin_radius_m"]
    for feature in features:
        q, normal = feature["point"], feature["normal"]
        meta = {k: feature[k] for k in ("feature", "kind", "region")}
        for depth in config["static"]["penetration_depths_m"]:
            position = q + (radius + skin - depth) * normal
            robot = IndenterInput(position, np.zeros(3), position.copy(), np.zeros(3))
            truth = native_query(native, state, robot)
            records["native"].append(truth)
            reference = closest_triangle_surface(
                x.numpy(), mesh["surface_triangles"], position, radius, skin
            )
            assert abs(reference[0] + depth) < 1e-12
            samples.append([*q, *normal, depth, *position])
            for method, surface in surfaces.items():
                query = surface.query(x, v, torch.tensor(position), torch.zeros(3).double(), radius)
                records[method].append(query.contacts)
                rows.append(
                    geometry_row(
                        query, truth, reference, method=method, **meta, depth_mm=depth * 1e3
                    )
                )
        # Native onset is measured by bisection. Proxy onset is the exact first
        # intersection of a moving sphere center with the union of proxy balls.
        lo, hi = config["static"]["onset_bracket_m"]

        def detected(depth, q=q, normal=normal):
            position = q + (radius + skin - depth) * normal
            return bool(
                native_query(
                    native,
                    state,
                    IndenterInput(position, np.zeros(3), position.copy(), np.zeros(3)),
                ).num_contacts
            )

        if detected(lo) or not detected(hi):
            raise AssertionError("Native onset is not bracketed")
        for _ in range(config["static"]["onset_bisection_iterations"]):
            mid = 0.5 * (lo + hi)
            if detected(mid):
                hi = mid
            else:
                lo = mid
        for method, surface in surfaces.items():
            delta = surface.cloud(x, v).centers.numpy() - q
            along = delta @ normal
            tangent2 = np.maximum(0.0, np.square(delta).sum(-1) - along**2)
            valid = tangent2 <= (radius + skin) ** 2
            outermost = np.max(along[valid] + np.sqrt((radius + skin) ** 2 - tangent2[valid]))
            delay = radius + skin - outermost
            # Also execute the Gaussian query on either side of its analytic onset.
            epsilon = 1e-7
            checks = []
            for offset in (-epsilon, epsilon):
                position = q + (radius + skin - delay - offset) * normal
                checks.append(
                    bool(
                        surface.query(
                            x, v, torch.tensor(position), torch.zeros(3).double(), radius
                        ).contacts.num_contacts
                    )
                )
            assert checks == [False, True]
            onset.append(
                {
                    **meta,
                    "method": method,
                    "native_onset_low_mm": lo * 1e3,
                    "native_onset_high_mm": hi * 1e3,
                    "proxy_onset_mm": delay * 1e3,
                    "onset_delay_vs_native_mid_mm": (delay - 0.5 * (lo + hi)) * 1e3,
                }
            )
    write_csv(output / "static_per_query.csv", rows)
    write_csv(output / "static_onset.csv", onset)
    summaries = summarize(rows, ("method", "depth_mm", "region"))
    write_csv(output / "static_metrics.csv", summaries)
    write_csv(output / "static_overall.csv", summarize(rows, ("method", "depth_mm")))
    np.savez_compressed(output / "static_inputs.npz", samples=np.asarray(samples), rest=x.numpy())
    for method, values in records.items():
        np.savez_compressed(output / f"static_{method}_contacts.npz", **pack_contacts(values))
    return rows, onset, len(features)


@torch.no_grad()
def trajectory_audit(mesh, native, surfaces, config, data, output):
    rows, events, samples = [], [], []
    radius, skin = config["sphere_radius_m"], config["skin_radius_m"]
    loaded = {}
    for split, names in config["splits"].items():
        for name in names:
            with np.load(data / f"{name}.npz", allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in (*CURRENT_FIELDS, "time", "phase_id", "node_ids")}
            assert np.array_equal(arrays["node_ids"], mesh["node_ids"])
            loaded[name] = arrays
            records = defaultdict(list)
            for frame in range(len(arrays["time"]) - 1):
                x, v, p, rv, target, target_v = current_inputs(arrays, frame)
                state = dynamics_state(x, v, mesh)
                robot = IndenterInput(p.numpy(), rv.numpy(), target.numpy(), target_v.numpy())
                truth = native_query(native, state, robot)
                records["native"].append(truth)
                reference = closest_triangle_surface(
                    state.centers.numpy(), mesh["surface_triangles"], p.numpy(), radius, skin
                )
                samples.append(
                    {
                        "trajectory": name,
                        "split": split,
                        "frame": frame,
                        "time_s": float(arrays["time"][frame]),
                        "phase": PHASES[int(arrays["phase_id"][frame])],
                    }
                )
                for method, surface in surfaces.items():
                    query = surface.query(x, v, p, rv, radius)
                    records[method].append(query.contacts)
                    rows.append(geometry_row(query, truth, reference, method=method, **samples[-1]))
            for method, values in records.items():
                np.savez_compressed(
                    output / f"{name}_{method}_contacts.npz", **pack_contacts(values)
                )
                detected = [i for i, r in enumerate(values) if r.num_contacts]
                events.append(
                    {
                        "trajectory": name,
                        "split": split,
                        "method": method,
                        "first_frame": detected[0] if detected else None,
                        "last_frame": detected[-1] if detected else None,
                        "contact_frames": len(detected),
                    }
                )
    write_csv(output / "sample_manifest.csv", samples)
    write_csv(output / "trajectory_per_frame.csv", rows)
    missed = [r for r in rows if r["native_contact"] and not r["predicted_contact"]]
    if missed:
        write_csv(output / "missed_contact_frames.csv", missed)
    write_csv(output / "trajectory_metrics.csv", summarize(rows, ("method", "trajectory", "split")))
    write_csv(output / "phase_metrics.csv", summarize(rows, ("method", "split", "phase")))
    write_csv(output / "split_metrics.csv", summarize(rows, ("method", "split")))
    common = None
    for method in METHODS:
        keys = {
            (r["trajectory"], r["frame"])
            for r in rows
            if r["method"] == method and r["native_contact"] and r["predicted_contact"]
        }
        common = keys if common is None else common & keys
    common_rows = [r for r in rows if (r["trajectory"], r["frame"]) in common]
    write_csv(
        output / "common_detected_frame_metrics.csv", summarize(common_rows, ("method", "split"))
    )
    write_csv(output / "contact_events.csv", events)
    return rows, loaded


def runtime_checks(mesh, surfaces, arrays, radius, output):
    x, v, p, rv, _, _ = current_inputs(arrays, 110)
    dense = surfaces["dense"]
    before = (x.clone(), v.clone())
    a = dense.query(x, v, p, rv, radius)
    # Future states and ALL force/contact labels may be NaN without changing
    # current geometry. These labels are not accepted by BoundSurfaceGaussians.
    poisoned = {key: value.copy() for key, value in arrays.items()}
    for key in CURRENT_FIELDS:
        poisoned[key][111:] = np.nan
    poisoned["contact_force_on_block_world"] = np.full((1, 3), np.nan)
    poisoned["contact_position_world"] = np.full((1, 3), np.nan)
    px, pv, pp, prv, _, _ = current_inputs(poisoned, 110)
    b = dense.query(px, pv, pp, prv, radius)
    assert torch.equal(a.closest_gap, b.closest_gap)
    assert torch.equal(a.contacts.points, b.contacts.points)
    assert torch.equal(x, before[0]) and torch.equal(v, before[1])
    restored = BoundSurfaceGaussians.load(output / "dense_binding.npz")
    c = restored.query(x, v, p, rv, radius)
    for field in ("points", "normals", "signed_gaps", "relative_velocities", "object_indices"):
        assert torch.equal(getattr(a.contacts, field), getattr(c.contacts, field))
    assert torch.equal(a.closest_gap, c.closest_gap)
    bad_x, bad_v = x.clone(), v.clone()
    pin = torch.tensor(mesh["pinned_mask"])
    bad_x[pin] += 0.01
    bad_v[pin] += 1.0
    cloud = dense.cloud(bad_x, bad_v)
    rest_cloud = dense.cloud(torch.tensor(mesh["rest_positions_world"]), torch.zeros_like(v))
    fixed = np.all(dense.pinned[dense.parents] | (dense.weights == 0), axis=1)
    fixed = torch.tensor(fixed)
    fixed_x = float((cloud.centers[fixed] - rest_cloud.centers[fixed]).abs().max())
    fixed_v = float(cloud.velocities[fixed].abs().max())
    assert fixed_x == fixed_v == 0
    gx, gv, gp = x.clone().requires_grad_(), v.clone().requires_grad_(), p.clone().requires_grad_()
    query = dense.query(gx, gv, gp, rv, radius)
    objective = query.closest_gap + dense.cloud(gx, gv).velocities.square().sum()
    gradients = torch.autograd.grad(objective, (gx, gv, gp))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert not torch.any(gradients[0][pin]) and not torch.any(gradients[1][pin])
    # Every raw numeric archive is reopened; every stored value must be finite.
    reloaded = 0
    for path in output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                assert np.isfinite(archive[key]).all(), (path, key)
        reloaded += 1
    return {
        "fixed_proxy_position_error_m": fixed_x,
        "fixed_proxy_velocity_error_m_s": fixed_v,
        "fixed_proxy_count": int(fixed.sum()),
        "finite_gradients": True,
        "fixed_node_gradients_zero": True,
        "input_x_v_not_mutated": True,
        "future_and_force_label_poisoning_invariant": True,
        "binding_reload_bitwise_equal": True,
        "numeric_archives_reloaded_and_finite": reloaded,
        "predictor_calls_native_geometry_or_force": False,
        "native_reference_queries_with_force": False,
        "training_updates": 0,
        "test_trajectory_arrays_loaded": False,
    }


@torch.no_grad()
def benchmark(mesh, native, surfaces, arrays, settings, radius):
    x, v, p, rv, target, target_v = current_inputs(arrays, settings["frame"])
    state = dynamics_state(x, v, mesh)
    robot = IndenterInput(p.numpy(), rv.numpy(), target.numpy(), target_v.numpy())
    functions = {
        "native_geometry": lambda: native_query(native, state, robot),
        **{
            name: lambda surface=surface: surface.query(x, v, p, rv, radius)
            for name, surface in surfaces.items()
        },
    }
    rows = []
    for name, function in functions.items():
        for _ in range(settings["warmup"]):
            function()
        elapsed = []
        for _ in range(settings["repetitions"]):
            start = time.perf_counter()
            function()
            elapsed.append((time.perf_counter() - start) * 1e3)
        rows.append(
            {
                "method": name,
                "trajectory": settings["trajectory"],
                "frame": settings["frame"],
                "repetitions": len(elapsed),
                "median_ms": float(np.median(elapsed)),
                "p95_ms": float(np.percentile(elapsed, 95)),
                "mean_ms": float(np.mean(elapsed)),
            }
        )
    return rows


def plots(static, onset, trajectories, surfaces, mesh, loaded, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    depths = sorted({r["depth_mm"] for r in static if r["depth_mm"] > 0})
    for method in METHODS:
        values = [r for r in static if r["method"] == method]
        recall = [
            np.mean([r["predicted_contact"] for r in values if r["depth_mm"] == d]) for d in depths
        ]
        axes[0].plot(depths, recall, "o-", label=method)
        axes[1].hist(
            [r["onset_delay_vs_native_mid_mm"] for r in onset if r["method"] == method],
            bins=30,
            alpha=0.6,
            label=method,
        )
    axes[0].set(xlabel="Nominal penetration (mm)", ylabel="Recall vs native", ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Contact-onset delay vs native (mm)", ylabel="Surface features")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "static_detection_and_onset.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, name in zip(axes.flat, config_names(loaded), strict=True):
        values = [r for r in trajectories if r["trajectory"] == name]
        for method in METHODS:
            rows = [r for r in values if r["method"] == method]
            axis.plot(
                [r["time_s"] for r in rows], [r["closest_gap_mm"] for r in rows], label=method
            )
        rows = [r for r in values if r["method"] == "sparse"]
        axis.plot(
            [r["time_s"] for r in rows],
            [r["mesh_gap_mm"] for r in rows],
            "k--",
            label="continuous mesh (scoring only)",
        )
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(title=name, ylabel="Minimum gap (mm)", xlabel="Time (s)", ylim=(-1, 4))
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "trajectory_gaps.png", dpi=160)
    plt.close(figure)
    # One predeclared representative current state, with shared XY axes.
    x, v, _, _, _, _ = current_inputs(loaded["center_deep"], 110)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    rest = mesh["rest_positions_world"]
    for axis, method in zip(axes, METHODS, strict=True):
        surface = surfaces[method]
        cloud = surface.cloud(x, v)
        initial = surface.cloud(torch.tensor(rest), torch.zeros_like(v))
        top = np.isclose(initial.centers[:, 2].numpy(), rest[:, 2].max(), atol=1e-12)
        selected = cloud.centers.detach().numpy()[top]
        colors = (initial.centers.detach().numpy()[top, 2] - selected[:, 2]) * 1e3
        scatter = axis.scatter(
            selected[:, 0] * 1e3,
            selected[:, 1] * 1e3,
            c=colors,
            s=7 if method == "dense" else 35,
            vmin=0,
            vmax=6,
            cmap="viridis",
        )
        axis.set(
            title=f"{method}: top collision centers",
            xlabel="X (mm)",
            ylabel="Y (mm)",
            aspect="equal",
        )
    figure.colorbar(scatter, ax=axes.tolist(), label="Downward displacement (mm)")
    figure.savefig(output / "bound_deformed_surface.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    release_plot(trajectories, output)


def release_plot(rows, output):
    """Diagnostic zoom; all frames still participate in the metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, name in zip(axes, ("left_shallow", "left_deep"), strict=True):
        for method in METHODS:
            selected = [
                r
                for r in rows
                if r["trajectory"] == name and r["method"] == method and 140 <= r["frame"] <= 160
            ]
            axis.plot(
                [r["time_s"] for r in selected],
                [r["closest_gap_mm"] * 1e3 for r in selected],
                "o-",
                label=method,
                markersize=3,
            )
            if method == "dense":
                axis.plot(
                    [r["time_s"] for r in selected],
                    [r["mesh_gap_mm"] * 1e3 for r in selected],
                    "k--",
                    label="mesh reference",
                )
                missed = [r for r in selected if r["native_contact"] and not r["predicted_contact"]]
                axis.scatter(
                    [r["time_s"] for r in missed],
                    [r["closest_gap_mm"] * 1e3 for r in missed],
                    marker="x",
                    color="red",
                    label="missed native contact",
                    zorder=5,
                )
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(title=name, xlabel="Release time (s)", ylabel="Minimum gap (um)", ylim=(-50, 50))
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "release_contact_misses.png", dpi=160)
    plt.close(figure)


def config_names(loaded):
    return [name for name in loaded if name != "no_contact"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/validate_contact_surface.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["seed"] != 7:
        raise ValueError("Keep the fixed pilot seed")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    data = ROOT / config["data_directory"]
    before = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
    output = args.output
    if output is None:
        parent, number = ROOT / config["output_root"], 1
        while (parent / f"run_{number:03d}").exists():
            number += 1
        output = parent / f"run_{number:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve existing outputs: {output}")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    started = time.perf_counter()
    write_json(output / "historical_hashes.json", before)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for name in (
        "scripts/validate_contact_surface.py",
        "src/contact_gwm/surface_proxies.py",
        "src/contact_gwm/contact_geometry.py",
        "src/contact_gwm/pinned_contact_pipeline.py",
        "configs/validate_contact_surface.yaml",
        "tests/test_surface_proxies.py",
    ):
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    mesh = read_npz(data / "mesh.npz")
    native = NativeCurrentContact(data / "scene.xml", mesh)
    if not np.isclose(native.model.geom_size[native.sphere, 0], config["sphere_radius_m"]):
        raise ValueError("Sphere radius differs from the native reference")
    if not np.isclose(native.model.flex_radius[0], config["skin_radius_m"]):
        raise ValueError("Collision skin differs from the native reference")
    surfaces = {
        name: BoundSurfaceGaussians(
            mesh["rest_positions_world"],
            mesh["surface_triangles"],
            mesh["pinned_mask"],
            max_rest_edge_m=step,
            skin_radius_m=config["skin_radius_m"],
            surface_level=config["gaussian_surface_level"],
        )
        for name, step in (("sparse", None), ("dense", config["max_rest_edge_m"]))
    }
    assert surfaces["dense"].actual_max_rest_edge_m <= config["max_rest_edge_m"]
    # Audit that the n=1 control is exactly the original set of surface nodes.
    sparse = surfaces["sparse"]
    assert np.all(np.count_nonzero(sparse.weights, axis=1) == 1)
    ids = sparse.parents[np.arange(len(sparse.parents)), sparse.weights.argmax(1)]
    assert np.array_equal(np.sort(ids), np.unique(mesh["surface_triangles"]))
    for name, surface in surfaces.items():
        surface.save(output / f"{name}_binding.npz")
    print("Bindings saved; scanning static surface features.", flush=True)
    static, onset, count = static_audit(mesh, native, surfaces, config, output)
    print("Static scans complete; checking all five current-state trajectories.", flush=True)
    trajectories, loaded = trajectory_audit(mesh, native, surfaces, config, data, output)
    checks = runtime_checks(
        mesh, surfaces, loaded["center_deep"], config["sphere_radius_m"], output
    )
    checks["sparse_control_equals_original_surface_node_set"] = True
    timing = benchmark(
        mesh,
        native,
        surfaces,
        loaded[config["benchmark"]["trajectory"]],
        config["benchmark"],
        config["sphere_radius_m"],
    )
    write_csv(output / "query_timing.csv", timing)
    plots(static, onset, trajectories, surfaces, mesh, loaded, output)
    assert before == {name: file_sha256(ROOT / name) for name in before}
    checks["historical_files_unchanged_sha256"] = len(before)
    write_json(output / "checks.json", checks)
    write_json(output / "failures.json", [])
    validation = next(
        row
        for row in summarize(trajectories, ("method", "split"))
        if row["method"] == "dense" and row["split"] == "validation"
    )
    max_delay = float(
        max(r["onset_delay_vs_native_mid_mm"] for r in onset if r["method"] == "dense")
    )
    criteria = config["acceptance"]
    acceptance = {
        "static_onset_delay_pass": max_delay <= criteria["max_static_onset_delay_mm"],
        "validation_recall_pass": validation["native_recall"] >= criteria["min_validation_recall"],
        "validation_false_positive_pass": validation["native_fp"]
        <= criteria["max_validation_false_positive_frames"],
        "fixed_and_finite_and_reload_pass": True,
    }
    environment = {
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "experiment_date": config["experiment_date"],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "cpu_threads": config["cpu_threads"],
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes_before_summary": sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file()
        ),
    }
    summary = {
        "environment": environment,
        "static_features": count,
        "static_queries_per_method": len(static) // 2,
        "trajectory_queries_per_method": len(trajectories) // 2,
        "surfaces": {
            name: {
                "proxy_count": len(s.parents),
                "subdivisions": s.subdivisions,
                "max_rest_edge_m": s.actual_max_rest_edge_m,
                "skin_radius_m": s.skin_radius_m,
            }
            for name, s in surfaces.items()
        },
        "static_overall": summarize(static, ("method", "depth_mm")),
        "split_metrics": summarize(trajectories, ("method", "split")),
        "max_dense_static_onset_delay_vs_native_mm": max_delay,
        "acceptance": acceptance,
        "checks": checks,
        "timing": timing,
        "source_hashes": hashes(
            [
                data / "mesh.npz",
                data / "scene.xml",
                args.config,
                *(data / f"{name}.npz" for name in loaded),
            ]
        ),
        "scope": "geometry_only_known_rest_topology_current_state_queries_no_training",
    }
    write_json(output / "environment.json", environment)
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "surfaces": summary["surfaces"],
                "validation": validation,
                "acceptance": acceptance,
                "elapsed_seconds": environment["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
