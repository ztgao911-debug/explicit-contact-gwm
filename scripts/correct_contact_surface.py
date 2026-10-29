"""CPU critical-contact correction: dense balls vs topology-assisted patches.

No forces, learned model, physical contact margin, or new data are introduced.
This is explicitly a known-topology geometry model, not pure Gaussian CCD.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from validate_contact_pipeline import hashes, write_csv, write_json
from validate_contact_surface import (
    CURRENT_FIELDS,
    PHASES,
    current_inputs,
    dynamics_state,
    geometry_row,
    native_query,
    pack_contacts,
    rest_features,
    summarize,
)

from contact_gwm.continuous_contact_surface import ContinuousContactSurface, ContinuousSurfaceQuery
from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.pinned_contact_pipeline import IndenterInput, NativeCurrentContact
from contact_gwm.surface_proxies import BoundSurfaceGaussians, closest_triangle_surface

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dense_balls", "continuous_patch")


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def scored_row(query, native, reference, x, mesh, radius, skin, robot_position, **metadata):
    row = geometry_row(query, native, reference, **metadata)
    point_error = normal_error = None
    if native.num_contacts:
        point = query.closest_point.numpy()
        index = np.linalg.norm(native.points.numpy() - point, axis=-1).argmin()
        point_error = float(np.linalg.norm(native.points[index].numpy() - point) * 1e3)
        normal_error = angle(query.closest_normal.numpy(), native.normals[index].numpy())
    row.update(
        {
            # Both methods use ONE closest witness in these metrics, including
            # frames that the dense-ball detector missed. This avoids count bias.
            "closest_witness_native_point_error_mm": point_error,
            "closest_witness_native_normal_error_deg": normal_error,
            "raw_face_candidate_count": None,
            "selected_face_point_consistency_mm": None,
            "selected_face_normal_consistency_deg": None,
            "selected_face_gap_consistency_mm": None,
            "multiple_distinct_closest_points": None,
        }
    )
    if isinstance(query, ContinuousSurfaceQuery):
        local = closest_triangle_surface(
            x,
            mesh["surface_triangles"][[int(query.triangle_index)]],
            robot_position,
            radius,
            skin,
        )
        candidates = query.face_candidates
        tied = (candidates.signed_gaps - query.closest_gap).abs() < 1e-12
        distinct = bool(
            torch.any(
                torch.linalg.vector_norm(candidates.points[tied] - query.closest_point, dim=-1)
                > 1e-9
            )
        )
        row.update(
            {
                "raw_face_candidate_count": candidates.num_contacts,
                "selected_face_point_consistency_mm": float(
                    np.linalg.norm(query.closest_point.numpy() - local[1]) * 1e3
                ),
                "selected_face_normal_consistency_deg": angle(
                    query.closest_normal.numpy(), local[2]
                ),
                "selected_face_gap_consistency_mm": (float(query.closest_gap) - local[0]) * 1e3,
                "multiple_distinct_closest_points": distinct,
            }
        )
    return row


def extra_summary(rows, fields):
    summary = summarize(rows, fields)
    for result in summary:
        values = [r for r in rows if all(r[k] == result[k] for k in fields)]
        contact = [r for r in values if r["native_contact"]]
        for name in (
            "closest_witness_native_point_error_mm",
            "closest_witness_native_normal_error_deg",
        ):
            result[f"native_contact_frames_mean_{name}"] = (
                float(np.mean([r[name] for r in contact])) if contact else None
            )
    return summary


def case_state(x, v, p, rv, target=None, target_v=None):
    return (
        x,
        v,
        p,
        rv,
        p.clone() if target is None else target,
        torch.zeros_like(rv) if target_v is None else target_v,
    )


def static_cases(mesh, base, depths, *, kind=None):
    x, v = (
        torch.tensor(mesh["rest_positions_world"]),
        torch.zeros_like(torch.tensor(mesh["rest_positions_world"])),
    )
    radius, skin = base["sphere_radius_m"], base["skin_radius_m"]
    for feature in rest_features(mesh, base["static"]["offset_grid"]):
        if kind is not None and feature["kind"] != kind:
            continue
        for depth in depths:
            p = torch.tensor(feature["point"] + (radius + skin - depth) * feature["normal"])
            meta = {k: feature[k] for k in ("feature", "kind", "region")}
            yield {**meta, "depth_mm": depth * 1e3}, case_state(x, v, p, torch.zeros(3).double())


def trajectory_cases(mesh, base, data):
    for split, names in base["splits"].items():
        for name in names:
            with np.load(data / f"{name}.npz", allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in (*CURRENT_FIELDS, "node_ids", "time", "phase_id")}
            if not np.array_equal(arrays["node_ids"], mesh["node_ids"]):
                raise ValueError("Node order changed")
            for frame in range(len(arrays["time"]) - 1):
                yield (
                    {
                        "trajectory": name,
                        "split": split,
                        "frame": frame,
                        "time_s": float(arrays["time"][frame]),
                        "phase": PHASES[int(arrays["phase_id"][frame])],
                    },
                    current_inputs(arrays, frame),
                )


@torch.no_grad()
def evaluate(cases, mesh, native, predictors, base, output, prefix):
    rows, face_rows, manifest, witness = [], [], [], []
    contacts = {name: [] for name in (*METHODS, "native", "patch_face_candidates")}
    radius, skin = base["sphere_radius_m"], base["skin_radius_m"]
    robot_positions, robot_velocities = [], []
    for case, (metadata, inputs) in enumerate(cases):
        x, v, p, rv, target, target_v = inputs
        state = dynamics_state(x, v, mesh)
        robot = IndenterInput(p.numpy(), rv.numpy(), target.numpy(), target_v.numpy())
        truth = native_query(native, state, robot)
        reference = closest_triangle_surface(
            state.centers.numpy(), mesh["surface_triangles"], p.numpy(), radius, skin
        )
        manifest.append({"case": case, **metadata})
        robot_positions.append(p.numpy())
        robot_velocities.append(rv.numpy())
        contacts["native"].append(truth)
        for method, predictor in predictors.items():
            query = predictor.query(x, v, p, rv, radius)
            row = scored_row(
                query,
                truth,
                reference,
                state.centers.numpy(),
                mesh,
                radius,
                skin,
                robot_position=p.numpy(),
                method=method,
                case=case,
                **metadata,
            )
            rows.append(row)
            contacts[method].append(query.contacts)
            if method == "continuous_patch":
                contacts["patch_face_candidates"].append(query.face_candidates)
                face_rows.append(
                    scored_row(
                        replace(query, contacts=query.face_candidates),
                        truth,
                        reference,
                        state.centers.numpy(),
                        mesh,
                        radius,
                        skin,
                        robot_position=p.numpy(),
                        method="patch_face_candidates",
                        case=case,
                        **metadata,
                    )
                )
                witness.append(
                    np.r_[
                        int(query.triangle_index),
                        query.barycentric_weights.numpy(),
                        query.material_point.numpy(),
                        query.material_velocity.numpy(),
                        query.closest_normal.numpy(),
                        float(query.closest_gap),
                        query.roundoff_tolerance_m,
                    ]
                )
    write_csv(output / f"{prefix}_per_query.csv", rows)
    write_csv(output / f"{prefix}_face_candidates.csv", face_rows)
    write_csv(output / f"{prefix}_manifest.csv", manifest)
    np.savez_compressed(
        output / f"{prefix}_witnesses.npz",
        witness=np.asarray(witness),
        robot_position=np.asarray(robot_positions),
        robot_velocity=np.asarray(robot_velocities),
    )
    for method, records in contacts.items():
        np.savez_compressed(output / f"{prefix}_{method}_contacts.npz", **pack_contacts(records))
    return rows, face_rows


@torch.no_grad()
def onset_audit(mesh, corrected, base, config, output):
    old = csv_rows(ROOT / config["baseline_run"] / "static_onset.csv")
    old = {int(r["feature"]): r for r in old if r["method"] == "dense"}
    x = torch.tensor(mesh["rest_positions_world"])
    v, rv = torch.zeros_like(x), torch.zeros(3).double()
    radius, skin = base["sphere_radius_m"], base["skin_radius_m"]
    result = []
    for feature in rest_features(mesh, base["static"]["offset_grid"]):
        lo, hi = config["onset_bracket_m"]

        def detected(depth, feature=feature):
            p = torch.tensor(feature["point"] + (radius + skin - depth) * feature["normal"])
            return bool(corrected.query(x, v, p, rv, radius).contacts.num_contacts)

        if detected(lo) or not detected(hi):
            raise AssertionError("Corrected contact onset not bracketed")
        for _ in range(config["onset_bisection_iterations"]):
            mid = 0.5 * (lo + hi)
            if detected(mid):
                hi = mid
            else:
                lo = mid
        previous = old[feature["feature"]]
        native_mid_mm = 0.5 * (
            float(previous["native_onset_low_mm"]) + float(previous["native_onset_high_mm"])
        )
        result.append(
            {
                **{k: feature[k] for k in ("feature", "kind", "region")},
                "dense_ball_onset_mm": float(previous["proxy_onset_mm"]),
                "corrected_onset_low_mm": lo * 1e3,
                "corrected_onset_high_mm": hi * 1e3,
                "corrected_onset_mid_mm": 0.5 * (lo + hi) * 1e3,
                "stored_native_onset_mid_mm": native_mid_mm,
                "corrected_delay_vs_native_mid_mm": 0.5 * (lo + hi) * 1e3 - native_mid_mm,
            }
        )
    write_csv(output / "static_onset.csv", result)
    return result


def compare_historical(static, trajectories, config):
    previous = ROOT / config["baseline_run"]
    max_gap_difference, checked = 0.0, 0
    for rows, source, keys in (
        (static, "static_per_query.csv", ("feature", "depth_mm")),
        (trajectories, "trajectory_per_frame.csv", ("trajectory", "frame")),
    ):
        old = {
            tuple(r[k] for k in keys): r
            for r in csv_rows(previous / source)
            if r["method"] == "dense"
        }
        for row in rows:
            if row["method"] != "dense_balls":
                continue
            source_row = old[tuple(str(row[k]) for k in keys)]
            assert row["predicted_count"] == int(source_row["predicted_count"])
            difference = abs(row["closest_gap_mm"] - float(source_row["closest_gap_mm"]))
            max_gap_difference = max(max_gap_difference, difference)
            checked += 1
    assert max_gap_difference == 0
    return {
        "historical_dense_queries_exactly_reproduced": checked,
        "historical_dense_max_gap_difference_mm": max_gap_difference,
    }


def contract_checks(binding, corrected, mesh, data, output, base):
    with np.load(data / "center_deep.npz") as saved:
        arrays = {k: saved[k] for k in CURRENT_FIELDS}
    x, v, p, rv, _, _ = current_inputs(arrays, 110)
    a = corrected.query(x, v, p, rv, base["sphere_radius_m"])
    reloaded = ContinuousContactSurface(BoundSurfaceGaussians.load(output / "binding.npz"))
    b = reloaded.query(x, v, p, rv, base["sphere_radius_m"])
    for name in (
        "closest_gap",
        "closest_normal",
        "closest_point",
        "material_velocity",
        "barycentric_weights",
        "triangle_index",
    ):
        assert torch.equal(getattr(a, name), getattr(b, name))
    for key in CURRENT_FIELDS:
        arrays[key][111:] = np.nan
    arrays["contact_force_on_block_world"] = np.full((1, 3), np.nan)
    xx, vv, pp, rr, _, _ = current_inputs(arrays, 110)
    c = corrected.query(xx, vv, pp, rr, base["sphere_radius_m"])
    assert torch.equal(a.closest_gap, c.closest_gap) and torch.equal(
        a.closest_normal, c.closest_normal
    )
    pin = torch.tensor(mesh["pinned_mask"])
    before_x, before_v = x.clone(), v.clone()
    x[pin] += 0.01
    v[pin] += 1
    after = binding.cloud(x, v)
    before = binding.cloud(before_x, before_v)
    assert torch.equal(after.centers, before.centers) and torch.equal(
        after.velocities, before.velocities
    )
    x, v, p = x.requires_grad_(), v.requires_grad_(), p.requires_grad_()
    result = corrected.query(x, v, p, rv, base["sphere_radius_m"])
    distance_gradient = torch.autograd.grad(result.closest_gap, p, retain_graph=True)[0]
    gradients = torch.autograd.grad(
        result.closest_gap + result.material_velocity.square().sum(), (x, v)
    )
    assert all(torch.isfinite(g).all() for g in (*gradients, distance_gradient))
    assert not torch.any(gradients[0][pin]) and not torch.any(gradients[1][pin])
    error = float((distance_gradient + result.closest_normal).abs().max().detach())
    assert error < 1e-10
    archives = 0
    for path in output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as saved:
            assert all(np.isfinite(saved[key]).all() for key in saved.files)
        archives += 1
    return {
        "binding_reload_bitwise_equal": True,
        "future_and_force_label_poisoning_invariant": True,
        "fixed_node_input_errors_removed_before_interpolation": True,
        "fixed_node_gradients_zero": True,
        "finite_gradients": True,
        "distance_gradient_plus_normal_max_abs": error,
        "numeric_archives_reloaded_finite": archives,
        "physical_contact_margin_m": 0,
        "float64_roundoff_band_m": a.roundoff_tolerance_m,
        "sphere_center_required_exterior": True,
        "native_force_and_time_integration_calls_forbidden": True,
        "pure_gaussian_ellipsoid_union_query": False,
        "training_updates": 0,
    }


@torch.no_grad()
def timing(predictors, data, base):
    with np.load(data / "center_deep.npz") as saved:
        arrays = {k: saved[k] for k in CURRENT_FIELDS}
    x, v, p, rv, _, _ = current_inputs(arrays, base["benchmark"]["frame"])
    rows = []
    for name, model in predictors.items():
        for _ in range(base["benchmark"]["warmup"]):
            model.query(x, v, p, rv, base["sphere_radius_m"])
        samples = []
        for _ in range(base["benchmark"]["repetitions"]):
            start = time.perf_counter()
            model.query(x, v, p, rv, base["sphere_radius_m"])
            samples.append((time.perf_counter() - start) * 1e3)
        rows.append(
            {
                "method": name,
                "median_ms": float(np.median(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
                "queries": len(samples),
            }
        )
    return rows


def plots(trajectories, critical, common, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, name in zip(axes, ("left_shallow", "left_deep"), strict=True):
        for method in METHODS:
            rows = [
                r
                for r in trajectories
                if r["method"] == method and r["trajectory"] == name and 140 <= r["frame"] <= 160
            ]
            axis.plot(
                [r["time_s"] for r in rows],
                [r["closest_gap_mm"] * 1e3 for r in rows],
                "o-",
                markersize=3,
                label=method,
            )
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(title=name, xlabel="Time (s)", ylabel="Minimum gap (um)", ylim=(-50, 50))
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "release_gap_correction.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    depths = sorted({r["depth_mm"] for r in critical if r["depth_mm"] > 0})
    for method in METHODS:
        recall = [
            np.mean(
                [
                    r["predicted_contact"]
                    for r in critical
                    if r["method"] == method and r["depth_mm"] == depth
                ]
            )
            for depth in depths
        ]
        axes[0].plot(np.array(depths) * 1e3, recall, "o-", label=method)
    axes[0].set(xlabel="Static penetration (um)", ylabel="Recall", ylim=(-0.02, 1.02))
    axes[0].legend()
    left = [r for r in common if r["split"] == "validation"]
    axes[1].bar([r["method"] for r in left], [r["mean_matched_normal_error_deg"] for r in left])
    axes[1].set(ylabel="Candidate-set normal error (deg)", title="Same 172 detected frames")
    axes[1].tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(output / "critical_detection_and_normals.png", dpi=160)
    plt.close(figure)


def forbidden(*args, **kwargs):
    raise AssertionError("Geometry-only experiment called a force solver or integrator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/correct_contact_surface.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / config["base_config"]).read_text())
    if (
        config["seed"] != 7
        or tuple(config["methods"]) != METHODS
        or config["contact_margin_m"] != 0
    ):
        raise ValueError("Keep the fixed correction protocol")
    old_config = yaml.safe_load((ROOT / config["baseline_run"] / "config.yaml").read_text())
    if base != old_config:
        raise ValueError("Keep the previous density, samples, skin and splits")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    data = ROOT / base["data_directory"]
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
    write_json(output / "run_status.json", {"complete": False})
    write_json(output / "historical_hashes.json", before)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "base_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    sources = [
        "scripts/correct_contact_surface.py",
        "scripts/validate_contact_surface.py",
        "src/contact_gwm/continuous_contact_surface.py",
        "src/contact_gwm/surface_proxies.py",
        "configs/correct_contact_surface.yaml",
        "tests/test_continuous_contact_surface.py",
    ]
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    try:
        mesh = read_npz(data / "mesh.npz")
        native = NativeCurrentContact(data / "scene.xml", mesh)
        binding = BoundSurfaceGaussians.load(ROOT / config["baseline_run"] / "dense_binding.npz")
        if not np.array_equal(binding.rest, mesh["rest_positions_world"]):
            raise ValueError("Rest node order changed")
        binding.save(output / "binding.npz")
        write_json(
            output / "witness_schema.json",
            {
                "witness_columns": [
                    "triangle_index",
                    "barycentric_w0",
                    "barycentric_w1",
                    "barycentric_w2",
                    "material_point_x_m",
                    "material_point_y_m",
                    "material_point_z_m",
                    "material_velocity_x_m_s",
                    "material_velocity_y_m_s",
                    "material_velocity_z_m_s",
                    "normal_x",
                    "normal_y",
                    "normal_z",
                    "signed_gap_m",
                    "roundoff_tolerance_m",
                ]
            },
        )
        corrected = ContinuousContactSurface(binding)
        predictors = dict(zip(METHODS, (binding, corrected), strict=True))
        with (
            patch.object(mujoco, "mj_forward", forbidden),
            patch.object(mujoco, "mj_contactForce", forbidden),
            patch.object(mujoco, "mj_step", forbidden),
        ):
            print("Evaluating the original 5634 static queries per method.", flush=True)
            static, _ = evaluate(
                static_cases(mesh, base, base["static"]["penetration_depths_m"]),
                mesh,
                native,
                predictors,
                base,
                output,
                "static",
            )
            print("Evaluating fixed near-zero contact probes.", flush=True)
            critical, _ = evaluate(
                static_cases(
                    mesh, base, config["critical_depths_m"], kind=config["critical_feature_kind"]
                ),
                mesh,
                native,
                predictors,
                base,
                output,
                "critical",
            )
            print("Evaluating all 1500 current trajectory states.", flush=True)
            trajectories, faces = evaluate(
                trajectory_cases(mesh, base, data),
                mesh,
                native,
                predictors,
                base,
                output,
                "trajectory",
            )
            print(
                "Checking contact onset, reload, gradients and future-label isolation.", flush=True
            )
            onset = onset_audit(mesh, corrected, base, config, output)
            checks = contract_checks(binding, corrected, mesh, data, output, base)
            speeds = timing(predictors, data, base)
        checks.update(compare_historical(static, trajectories, config))
        groups = extra_summary(trajectories, ("method", "split"))
        write_csv(output / "split_metrics.csv", groups)
        write_csv(
            output / "face_candidate_split_metrics.csv", extra_summary(faces, ("method", "split"))
        )
        write_csv(
            output / "face_candidate_phase_metrics.csv",
            extra_summary(faces, ("method", "split", "phase")),
        )
        write_csv(
            output / "phase_metrics.csv", extra_summary(trajectories, ("method", "split", "phase"))
        )
        write_csv(
            output / "trajectory_metrics.csv",
            extra_summary(trajectories, ("method", "trajectory", "split")),
        )
        write_csv(
            output / "static_metrics.csv", extra_summary(static, ("method", "depth_mm", "region"))
        )
        write_csv(output / "critical_metrics.csv", extra_summary(critical, ("method", "depth_mm")))
        common_cases = {
            r["case"]
            for r in trajectories
            if r["method"] == "dense_balls" and r["native_contact"] and r["predicted_contact"]
        }
        common_rows = [r for r in trajectories if r["case"] in common_cases]
        common_faces = [r for r in faces if r["case"] in common_cases]
        common = extra_summary([*common_rows, *common_faces], ("method", "split"))
        write_csv(output / "common_detected_frame_metrics.csv", common)
        # Preserve exactly the historical 14-frame subset as an additional view.
        old = csv_rows(ROOT / config["baseline_run"] / "trajectory_per_frame.csv")
        fourteen = {
            (r["trajectory"], int(r["frame"]))
            for r in old
            if r["method"] == "sparse"
            and r["split"] == "validation"
            and r["predicted_contact"] == "True"
            and r["native_contact"] == "True"
        }
        selected = [
            r
            for r in [*trajectories, *faces]
            if "trajectory" in r and (r["trajectory"], r["frame"]) in fourteen
        ]
        write_csv(
            output / "historical_14_frame_metrics.csv", extra_summary(selected, ("method", "split"))
        )
        missed = [r for r in trajectories if r["native_contact"] and not r["predicted_contact"]]
        if missed:
            write_csv(output / "missed_contact_frames.csv", missed)
            missed_cases = {r["case"] for r in missed if r["method"] == "dense_balls"}
            recovered = [
                r
                for r in trajectories
                if r["method"] == "continuous_patch" and r["case"] in missed_cases
            ]
            write_csv(output / "corrected_previous_misses.csv", recovered)
        write_csv(output / "query_timing.csv", speeds)
        plots(trajectories, critical, common, output)
        assert before == {name: file_sha256(ROOT / name) for name in before}
        checks["historical_files_unchanged_sha256"] = len(before)
        checks["test_trajectory_arrays_loaded"] = False
        left = {r["method"]: r for r in groups if r["split"] == "validation"}
        common_left = {r["method"]: r for r in common if r["split"] == "validation"}
        criteria = config["acceptance"]
        closest_normal = "native_contact_frames_mean_closest_witness_native_normal_error_deg"
        acceptance = {
            "validation_recall": left["continuous_patch"]["native_recall"]
            >= criteria["min_validation_recall"],
            "validation_false_positives": left["continuous_patch"]["native_fp"]
            <= criteria["max_validation_false_positive_frames"],
            "gap_rmse": left["continuous_patch"]["gap_rmse_vs_mesh_mm"]
            <= criteria["max_gap_rmse_vs_mesh_mm"],
            "same_closest_witness_normal_protocol_not_worse": common_left["continuous_patch"][
                closest_normal
            ]
            <= common_left["dense_balls"][closest_normal],
        }
        environment = {
            "experiment_date": config["experiment_date"],
            "host_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "cpu_threads": config["cpu_threads"],
            "elapsed_seconds": time.perf_counter() - started,
            "output_bytes_before_summary": sum(
                f.stat().st_size for f in output.rglob("*") if f.is_file()
            ),
        }
        summary = {
            "environment": environment,
            "checks": checks,
            "acceptance": acceptance,
            "proxy_count": len(binding.parents),
            "dynamics_node_count": len(binding.rest),
            "patch_count": len(binding.triangles),
            "queries_per_method": {
                "static": len(static) // 2,
                "critical": len(critical) // 2,
                "trajectory": len(trajectories) // 2,
            },
            "split_metrics": groups,
            "common_frame_metrics": common,
            "timing": speeds,
            "max_abs_corrected_onset_mid_mm": max(abs(r["corrected_onset_mid_mm"]) for r in onset),
            "source_hashes": hashes(
                [*(ROOT / s for s in sources), data / "mesh.npz", data / "scene.xml"]
            ),
            "scope": "known_topology_continuous_surface_closest_witness_geometry_only",
        }
        write_json(output / "summary.json", summary)
        write_json(output / "checks.json", checks)
        write_json(output / "failures.json", [])
        write_json(output / "environment.json", environment)
        write_json(output / "run_status.json", {"complete": True})
        print(
            json.dumps(
                {
                    "output": str(output),
                    "left": left,
                    "acceptance": acceptance,
                    "elapsed_seconds": environment["elapsed_seconds"],
                },
                indent=2,
            )
        )
    except Exception as error:
        write_json(
            output / "failures.json", [{"error": type(error).__name__, "message": str(error)}]
        )
        raise


if __name__ == "__main__":
    main()
