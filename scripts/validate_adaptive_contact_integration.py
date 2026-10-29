"""Clipped/adaptive integration pilot, fixed samples and independent references."""

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
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from audit_contact_sampling import selected_static_cases
from correct_contact_surface import case_state, trajectory_cases
from validate_contact_pipeline import hashes, write_csv, write_json
from validate_contact_surface import CURRENT_FIELDS, current_inputs, rest_features

from contact_gwm.contact_integration import ClippedSurfaceQuadrature, IntegrationSettings
from contact_gwm.contact_sampling import SurfaceQuadrature
from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.integration_reference import surface_disk_reference
from contact_gwm.surface_proxies import BoundSurfaceGaussians

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def origin_for(x, mesh):
    pin = torch.tensor(mesh["pinned_mask"])
    rest = torch.tensor(mesh["rest_positions_world"])
    mass = torch.tensor(mesh["node_masses"])
    return (torch.where(pin[:, None], rest, x) * mass[:, None]).sum(0) / mass.sum()


def static_stress(mesh, base, config):
    features = rest_features(mesh, base["static"]["offset_grid"])
    x = torch.tensor(mesh["rest_positions_world"])
    v = torch.zeros_like(x)
    total = base["sphere_radius_m"] + base["skin_radius_m"]
    for region in config["stress_regions"]:
        feature = next(f for f in features if f["region"] == region)
        for depth in config["stress_depths_m"]:
            for signed in (-depth, depth):
                p = torch.tensor(feature["point"] + (total - signed) * feature["normal"])
                yield (
                    {"region": region, "feature": feature["feature"], "depth_mm": signed * 1e3},
                    case_state(x, v, p, torch.zeros(3).double()),
                )


def fixed_rows(directory, prefix, n):
    return {
        int(r["case"]): r
        for r in read_rows(directory / f"{prefix}_quadrature_per_case.csv")
        if int(r["subdivisions"]) == n
    }


@torch.no_grad()
def evaluate(
    cases,
    model,
    mesh,
    base,
    config,
    output,
    prefix,
    old=None,
    fixed_model=None,
    all_references=False,
):
    rows, face_rows, numeric = [], [], []
    last_name = None
    for case, (meta, inputs) in enumerate(cases):
        x, v, p, rv, _, _ = inputs
        if meta.get("trajectory") != last_name:
            last_name = meta.get("trajectory")
            if last_name:
                print(f"  {last_name}: integrating current states", flush=True)
        origin = origin_for(x, mesh)
        started = time.perf_counter()
        result = model.normal_bases(
            x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
        )
        elapsed = (time.perf_counter() - started) * 1e3
        geometric = result["positive_contact_faces"] > 0
        if old is not None:
            control = old[case]
            for key, value in meta.items():
                assert str(value) == control[key], (prefix, case, key)
            fixed_area = float(control["contact_area_m2"])
            fixed_vec = np.array([float(control[f"elastic_{a}_m3"]) for a in "xyz"])
            fixed_count = int(control["active_samples"])
        else:
            control = fixed_model.normal_bases(
                x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
            )
            fixed_area = float(control["contact_area_m2"])
            fixed_vec = control["elastic_vector_m3"].numpy()
            fixed_count = int(control["active_sample_count"])
        vector = result["elastic_vector_m3"].numpy()
        row = {
            "case": case,
            **meta,
            "geometric_positive_area": geometric,
            "contact_area_m2": float(result["contact_area_m2"]),
            "elastic_scalar_m3": float(result["elastic_scalar_m3"]),
            "damping_scalar_m3_s": float(result["damping_scalar_m3_s"]),
            **{f"elastic_{a}_m3": float(z) for a, z in zip("xyz", vector, strict=True)},
            **{
                f"damping_{a}_m3_s": float(z)
                for a, z in zip("xyz", result["damping_vector_m3_s"], strict=True)
            },
            "fixed_area_m2": fixed_area,
            **{f"fixed_elastic_{a}_m3": float(z) for a, z in zip("xyz", fixed_vec, strict=True)},
            "fixed_active_samples": fixed_count,
            "active_samples": result["active_sample_count"],
            "evaluated_samples": result["evaluated_samples"],
            "evaluated_intervals": result["evaluated_intervals"],
            "interval_budget_exhausted": result["interval_budget_exhausted"],
            "positive_faces": result["positive_contact_faces"],
            "zero_integral_positive_faces": result["zero_integral_positive_faces"],
            "unresolved_intervals": result["unresolved_intervals"],
            "unresolved_area_intervals": result["unresolved_area_intervals"],
            "unresolved_elastic_intervals": result["unresolved_elastic_intervals"],
            "unresolved_damping_intervals": result["unresolved_damping_intervals"],
            "leaf_intervals": result["leaf_intervals"],
            "max_subdivision_depth": result["max_subdivision_depth"],
            "estimated_area_error_m2": float(result["estimated_area_error_m2"]),
            "estimated_elastic_error_m3": float(result["estimated_elastic_error_m3"]),
            "estimated_damping_error_m3_s": float(result["estimated_damping_error_m3_s"]),
            "query_ms": elapsed,
            "reference_area_m2": None,
            "reference_elastic_norm_m3": None,
            "reference_area_relative_error": None,
            "reference_elastic_relative_error": None,
            "reference_order_area_difference": None,
            "reference_order_elastic_difference": None,
        }
        use_reference = all_references or meta.get("frame") in config["reference_trajectory_frames"]
        if use_reference:
            pin = mesh["pinned_mask"]
            current = x.numpy().copy()
            current[pin] = mesh["rest_positions_world"][pin]
            references = [
                surface_disk_reference(
                    current,
                    mesh["surface_triangles"],
                    p.numpy(),
                    base["sphere_radius_m"] + base["skin_radius_m"],
                    n,
                )
                for n in config["reference_orders"]
            ]
            coarse, fine = references
            if not all(
                np.isfinite(
                    [ref["contact_area_m2"], ref["elastic_scalar_m3"], *ref["elastic_vector_m3"]]
                ).all()
                for ref in references
            ):
                raise FloatingPointError("Non-finite independent reference")
            area = fine["contact_area_m2"]
            elastic = fine["elastic_vector_m3"]
            scale = np.linalg.norm(elastic)
            row.update(
                {
                    "reference_area_m2": area,
                    "reference_elastic_norm_m3": float(scale),
                    "reference_area_relative_error": abs(float(result["contact_area_m2"]) - area)
                    / area
                    if area > 0
                    else None,
                    "reference_elastic_relative_error": float(
                        np.linalg.norm(vector - elastic) / scale
                    )
                    if scale > 0
                    else None,
                    "reference_order_area_difference": abs(coarse["contact_area_m2"] - area) / area
                    if area > 0
                    else None,
                    "reference_order_elastic_difference": float(
                        np.linalg.norm(coarse["elastic_vector_m3"] - elastic) / scale
                    )
                    if scale > 0
                    else None,
                }
            )
        rows.append(row)
        face_rows.extend(
            {
                "case": case,
                "face": int(f[0]),
                "area_m2": f[1],
                "elastic_scalar_m3": f[2],
                "final_samples": f[3],
                "unresolved_intervals": f[4],
            }
            for f in result["face_diagnostics"]
        )
        numeric.append(
            [
                case,
                float(result["contact_area_m2"]),
                float(result["elastic_scalar_m3"]),
                *vector,
                *result["damping_vector_m3_s"].numpy(),
                *result["elastic_torque_m4"].numpy(),
                result["evaluated_samples"],
            ]
        )
    write_csv(output / f"{prefix}_per_case.csv", rows)
    if face_rows:
        write_csv(output / f"{prefix}_per_face.csv", face_rows)
    np.savez_compressed(output / f"{prefix}_integrals.npz", values=np.asarray(numeric))
    problematic = [
        r
        for r in rows
        if r["unresolved_intervals"]
        or r["zero_integral_positive_faces"]
        or (r["geometric_positive_area"] and r["contact_area_m2"] <= 0)
    ]
    if problematic:
        write_csv(output / f"{prefix}_unresolved_cases.csv", problematic)
    return rows


def summarize(rows, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    result = []
    for key, values in sorted(groups.items()):
        row = {
            **dict(zip(fields, key, strict=True)),
            "cases": len(values),
            "positive_cases": sum(r["geometric_positive_area"] for r in values),
            "fixed_zero_positive_cases": sum(
                r["geometric_positive_area"] and r["fixed_area_m2"] == 0 for r in values
            ),
            "clipped_zero_positive_cases": sum(
                r["geometric_positive_area"]
                and (r["contact_area_m2"] <= 0 or r["elastic_scalar_m3"] <= 0)
                for r in values
            ),
            "separated_cases_with_area": sum(
                not r["geometric_positive_area"] and r["contact_area_m2"] > 0 for r in values
            ),
            "unresolved_cases": sum(r["unresolved_intervals"] > 0 for r in values),
            "interval_budget_exhausted_cases": sum(r["interval_budget_exhausted"] for r in values),
            "median_query_ms": float(np.median([r["query_ms"] for r in values])),
            "p95_query_ms": float(np.percentile([r["query_ms"] for r in values], 95)),
            "max_evaluated_samples": max(r["evaluated_samples"] for r in values),
            "max_subdivision_depth": max(r["max_subdivision_depth"] for r in values),
        }
        for name in (
            "reference_area_relative_error",
            "reference_elastic_relative_error",
            "reference_order_area_difference",
            "reference_order_elastic_difference",
        ):
            vals = [r[name] for r in values if r[name] is not None]
            row[f"max_{name}"] = max(vals) if vals else None
        result.append(row)
    return result


@torch.no_grad()
def closed_form(model, mesh, base, config, output):
    x = torch.tensor(mesh["rest_positions_world"])
    v = torch.zeros_like(x)
    radius = base["sphere_radius_m"] + base["skin_radius_m"]
    speed = config["flat_plane_robot_speed_m_s"]
    rows = []
    for depth in config["flat_plane_depths_m"]:
        h = radius - depth
        p = torch.tensor([0, 0, float(x[:, 2].max()) + h]).double()
        rv = torch.tensor([0, 0, -speed]).double()
        result = model.normal_bases(
            x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], torch.zeros(3).double()
        )
        expected_area = np.pi * (radius - h) * (radius + h)
        expected_elastic = -np.pi * h * depth**2
        expected_damping = -2 * np.pi * speed * h * h * np.log(radius / h)
        rows.append(
            {
                "depth_m": depth,
                "area_m2": float(result["contact_area_m2"]),
                "area_relative_error": abs(float(result["contact_area_m2"]) - expected_area)
                / expected_area,
                "elastic_z_m3": float(result["elastic_vector_m3"][2]),
                "elastic_relative_error": abs(
                    float(result["elastic_vector_m3"][2]) - expected_elastic
                )
                / abs(expected_elastic),
                "damping_z_m3_s": float(result["damping_vector_m3_s"][2]),
                "damping_relative_error": abs(
                    float(result["damping_vector_m3_s"][2]) - expected_damping
                )
                / abs(expected_damping),
                "unresolved_intervals": result["unresolved_intervals"],
            }
        )
    write_csv(output / "flat_plane_closed_form.csv", rows)
    return rows


@torch.no_grad()
def benchmark(model, fixed, mesh, base, config, data, output):
    settings = config["benchmark"]
    with np.load(data / f"{settings['trajectory']}.npz") as f:
        arrays = {k: f[k] for k in CURRENT_FIELDS}
    rows = []
    for frame in settings["frames"]:
        x, v, p, rv, _, _ = current_inputs(arrays, frame)
        origin = origin_for(x, mesh)
        for name, rule in [("fixed_n32", fixed), ("clipped_adaptive", model)]:
            for _ in range(settings["warmup"]):
                rule.normal_bases(
                    x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
                )
            values = []
            for _ in range(settings["repetitions"]):
                started = time.perf_counter()
                rule.normal_bases(
                    x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
                )
                values.append((time.perf_counter() - started) * 1e3)
            rows.append(
                {
                    "trajectory": settings["trajectory"],
                    "frame": frame,
                    "method": name,
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "repetitions": len(values),
                }
            )
    write_csv(output / "timing.csv", rows)
    return rows


def contracts(model, mesh, base, config, data, output):
    with np.load(data / "center_deep.npz") as f:
        arrays = {k: f[k] for k in CURRENT_FIELDS}
    x, v, p, rv, _, _ = current_inputs(arrays, 110)
    origin = origin_for(x, mesh)
    with torch.no_grad():
        a = model.normal_bases(x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin)
        saved = BoundSurfaceGaussians.load(output / "binding.npz")
        restored = ClippedSurfaceQuadrature(
            saved.rest, saved.triangles, saved.pinned, IntegrationSettings(**config["integration"])
        )
        b = restored.normal_bases(
            x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
        )
        for name in ("contact_area_m2", "elastic_vector_m3", "damping_vector_m3_s"):
            assert torch.equal(a[name], b[name])
        for key in CURRENT_FIELDS:
            arrays[key][111:] = np.nan
        arrays["contact_force_on_block_world"] = np.full((1, 3), np.nan)
        xx, vv, pp, rr, _, _ = current_inputs(arrays, 110)
        c = model.normal_bases(
            xx, vv, pp, rr, base["sphere_radius_m"], base["skin_radius_m"], origin
        )
        assert torch.equal(a["elastic_vector_m3"], c["elastic_vector_m3"])
        before_x, before_v = x.clone(), v.clone()
        pin = torch.tensor(mesh["pinned_mask"])
        x[pin] += 0.1
        v[pin] += 1
        d = model.normal_bases(x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin)
        assert torch.equal(a["elastic_vector_m3"], d["elastic_vector_m3"])
        x, v = before_x.requires_grad_(), before_v.requires_grad_()
    result = model.normal_bases(x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin)
    gradients = torch.autograd.grad(
        result["elastic_vector_m3"].sum() + result["damping_vector_m3_s"].sum(), (x, v)
    )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert not torch.any(gradients[0][pin]) and not torch.any(gradients[1][pin])
    archives = 0
    for path in output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as f:
            assert all(np.isfinite(f[k]).all() for k in f.files)
        archives += 1
    return {
        "recreation_from_saved_config_and_binding_bitwise_equal": True,
        "future_and_force_label_poisoning_invariant": True,
        "fixed_boundary_preserved": True,
        "finite_selected_branch_gradients": True,
        "fixed_node_gradients_zero": True,
        "numeric_archives_reloaded_finite": archives,
        "native_calls_forbidden": True,
        "force_coefficients_fitted": False,
        "next_state_predicted": False,
        "test_trajectory_arrays_loaded": False,
    }


def plots(static, stress, plane, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = [r for r in static if r["region"] == "corner" and r["depth_mm"] == 0.05]
    labels = [str(r["case"]) for r in rows]
    positions = np.arange(len(rows))
    axes[0].bar(
        positions - 0.18, [r["fixed_area_m2"] * 1e9 for r in rows], width=0.36, label="fixed n=64"
    )
    axes[0].bar(
        positions + 0.18,
        [r["contact_area_m2"] * 1e9 for r in rows],
        width=0.36,
        label="clipped adaptive",
    )
    axes[0].set(
        xticks=positions,
        xticklabels=labels,
        xlabel="Previously missed corner case",
        ylabel="Contact area (1e-9 m^2)",
    )
    axes[0].legend()
    for region in sorted({r["region"] for r in stress}):
        chosen = [r for r in stress if r["region"] == region and r["depth_mm"] > 0]
        axes[1].loglog(
            [r["depth_mm"] * 1e3 for r in chosen],
            [r["contact_area_m2"] for r in chosen],
            "o-",
            label=region,
        )
    axes[1].set(xlabel="Positive penetration (um)", ylabel="Geometrically clipped area (m^2)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "tiny_contact_recovery.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for field, label in [
        ("area_relative_error", "area"),
        ("elastic_relative_error", "elastic"),
        ("damping_relative_error", "damping"),
    ]:
        axis.semilogy(
            [r["depth_m"] * 1e6 for r in plane], [r[field] for r in plane], "o-", label=label
        )
    axis.set(xlabel="Plane penetration (um)", ylabel="Relative error vs closed form")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "closed_form_accuracy.png", dpi=160)
    plt.close(fig)


def forbidden(*args, **kwargs):
    raise AssertionError("Integration experiment used native geometry/force/integration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/validate_adaptive_contact_integration.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / config["base_config"]).read_text())
    sampling = yaml.safe_load((ROOT / config["sampling_config"]).read_text())
    if config["seed"] != 7:
        raise ValueError("Keep the fixed pilot seed")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    data = ROOT / base["data_directory"]
    prior = ROOT / config["baseline_run"]
    assert sampling == yaml.safe_load((prior / "config.yaml").read_text())
    assert base == yaml.safe_load((prior / "base_config.yaml").read_text())
    previous = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
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
    write_json(output / "historical_hashes.json", previous)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "base_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    sources = [
        "scripts/validate_adaptive_contact_integration.py",
        "src/contact_gwm/contact_integration.py",
        "src/contact_gwm/integration_reference.py",
        "configs/validate_adaptive_contact_integration.yaml",
        "tests/test_contact_integration.py",
    ]
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    try:
        mesh = read_npz(data / "mesh.npz")
        saved = BoundSurfaceGaussians.load(prior / "binding.npz")
        saved.save(output / "binding.npz")
        assert np.array_equal(saved.rest, mesh["rest_positions_world"])
        model = ClippedSurfaceQuadrature(
            saved.rest, saved.triangles, saved.pinned, IntegrationSettings(**config["integration"])
        )
        fixed32 = SurfaceQuadrature(saved.rest, saved.triangles, saved.pinned, 32)
        fixed64 = SurfaceQuadrature(saved.rest, saved.triangles, saved.pinned, 64)
        with (
            patch.object(mujoco, "mj_fwdPosition", forbidden),
            patch.object(mujoco, "mj_forward", forbidden),
            patch.object(mujoco, "mj_contactForce", forbidden),
            patch.object(mujoco, "mj_step", forbidden),
        ):
            print(
                "Evaluating 48 boundary cases against fixed n64 and independent slice references.",
                flush=True,
            )
            static = evaluate(
                selected_static_cases(mesh, base, sampling),
                model,
                mesh,
                base,
                config,
                output,
                "selected_static",
                fixed_rows(prior, "selected_static", 64),
                all_references=True,
            )
            print("Evaluating all original 1500 current trajectory states.", flush=True)
            trajectory = evaluate(
                trajectory_cases(mesh, base, data),
                model,
                mesh,
                base,
                config,
                output,
                "trajectory",
                fixed_rows(prior, "trajectory", 32),
            )
            print(
                "Evaluating predetermined tiny cap/separation stress cases and plane closed forms.",
                flush=True,
            )
            stress = evaluate(
                static_stress(mesh, base, config),
                model,
                mesh,
                base,
                config,
                output,
                "stress",
                fixed_model=fixed64,
                all_references=True,
            )
            plane = closed_form(model, mesh, base, config, output)
            speeds = benchmark(model, fixed32, mesh, base, config, data, output)
            checks = contracts(model, mesh, base, config, data, output)
        write_csv(output / "static_by_region.csv", summarize(static, ("region",)))
        write_csv(output / "trajectory_by_split.csv", summarize(trajectory, ("split",)))
        write_csv(output / "trajectory_by_phase.csv", summarize(trajectory, ("split", "phase")))
        write_csv(output / "stress_by_region.csv", summarize(stress, ("region",)))
        recovered = [r for r in static if r["geometric_positive_area"] and r["fixed_area_m2"] == 0]
        assert len(recovered) == 3, "Keep the original three failed static cases"
        write_csv(output / "previously_missed_cases.csv", recovered)
        plots(static, stress, plane, output)
        assert previous == {name: file_sha256(ROOT / name) for name in previous}
        checks["historical_files_sha256_unchanged"] = len(previous)
        all_rows = [*static, *trajectory, *stress]
        criteria = config["acceptance"]
        referenced = [r for r in all_rows if r["reference_elastic_relative_error"] is not None]
        acceptance = {
            "original_three_corner_cases_positive": all(
                r["contact_area_m2"] > 0 and r["elastic_scalar_m3"] > 0 for r in recovered
            ),
            "positive_faces_never_zero": all(
                r["zero_integral_positive_faces"] == 0 for r in all_rows
            ),
            "separated_cases_zero": all(
                r["contact_area_m2"] == 0 for r in all_rows if not r["geometric_positive_area"]
            ),
            "adaptive_estimates_converged_all_cases": all(
                r["unresolved_intervals"] == 0 for r in all_rows
            ),
            "independent_reference_agreement": all(
                r["reference_area_relative_error"] <= criteria["max_reference_relative_error"]
                and r["reference_elastic_relative_error"]
                <= criteria["max_reference_relative_error"]
                for r in referenced
            ),
            "independent_reference_order_stable": all(
                r["reference_order_area_difference"] <= criteria["max_reference_order_difference"]
                and r["reference_order_elastic_difference"]
                <= criteria["max_reference_order_difference"]
                for r in referenced
            ),
            "plane_closed_form_agreement": all(
                max(
                    r["area_relative_error"],
                    r["elastic_relative_error"],
                    r["damping_relative_error"],
                )
                <= criteria["max_closed_form_relative_error"]
                for r in plane
            ),
        }
        env = {
            "experiment_date": config["experiment_date"],
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_threads": config["cpu_threads"],
            "elapsed_seconds": time.perf_counter() - started,
            "output_bytes_before_summary": sum(
                f.stat().st_size for f in output.rglob("*") if f.is_file()
            ),
        }
        summary = {
            "environment": env,
            "checks": checks,
            "acceptance": acceptance,
            "settings": asdict(model.settings),
            "static": summarize(static, ("region",)),
            "trajectory": summarize(trajectory, ("split",)),
            "stress": summarize(stress, ("region",)),
            "closed_form": plane,
            "timing": speeds,
            "source_hashes": hashes([ROOT / name for name in sources]),
            "scope": "topology_assisted_area_and_normal_pressure_bases_no_force_fitting",
        }
        write_json(output / "summary.json", summary)
        write_json(output / "checks.json", checks)
        write_json(output / "environment.json", env)
        write_json(output / "failures.json", [])
        write_json(output / "run_status.json", {"complete": True})
        print(
            json.dumps(
                {
                    "output": str(output),
                    "acceptance": acceptance,
                    "static": summary["static"],
                    "trajectory": summary["trajectory"],
                    "elapsed_seconds": env["elapsed_seconds"],
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
