"""CPU audit of contact witness grouping, velocity definitions and area integration.

Current states only. No training, solver-force labels, stiffness fitting, state
integration or right-side trajectory arrays. Area bases are not calibrated force.
"""

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
from correct_contact_surface import static_cases, trajectory_cases
from validate_contact_pipeline import hashes, write_csv, write_json
from validate_contact_surface import CURRENT_FIELDS, current_inputs, dynamics_state, rest_features

from contact_gwm.contact_force import ContactMaterial, compute_contact_forces
from contact_gwm.contact_sampling import (
    SurfaceQuadrature,
    group_material_contacts,
    material_samples,
)
from contact_gwm.continuous_contact_surface import ContinuousContactSurface
from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.data.mujoco_soft import node_velocities
from contact_gwm.pinned_contact_pipeline import IndenterInput, NativeCurrentContact
from contact_gwm.surface_proxies import BoundSurfaceGaussians

ROOT = Path(__file__).resolve().parents[1]


def force(geometry, origin, material):
    return compute_contact_forces(geometry, origin, material).forces.sum(0)


def basis_row(values, n, **metadata):
    return {
        **metadata,
        "subdivisions": n,
        "active_samples": int(values["active_sample_count"]),
        "contact_area_m2": float(values["contact_area_m2"]),
        "total_surface_area_m2": float(values["total_surface_area_m2"]),
        **{
            f"elastic_{axis}_m3": float(v)
            for axis, v in zip("xyz", values["elastic_vector_m3"], strict=True)
        },
        **{
            f"damping_{axis}_m3_s": float(v)
            for axis, v in zip("xyz", values["damping_vector_m3_s"], strict=True)
        },
        **{
            f"torque_{axis}_m4": float(v)
            for axis, v in zip("xyz", values["elastic_torque_m4"], strict=True)
        },
        **{
            f"unweighted_{axis}_m": float(v)
            for axis, v in zip("xyz", values["unweighted_elastic_vector_m"], strict=True)
        },
    }


def grouped_summary(rows, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    output = []
    for key, values in sorted(groups.items()):
        paired = sum(r["paired_contact_count"] for r in values)
        result = {
            **dict(zip(fields, key, strict=True)),
            "frames": len(values),
            "geometric_contact_frames": sum(r["geometric_contact"] for r in values),
            "face_candidates": sum(r["face_count"] for r in values),
            "grouped_candidates": sum(r["group_count"] for r in values),
            "removed_repeated_witnesses": sum(r["face_count"] - r["group_count"] for r in values),
            "native_raw_count_disagreement_frames": sum(
                r["native_count"] != r["face_count"] for r in values
            ),
            "native_group_count_disagreement_frames": sum(
                r["native_group_count"] != r["group_count"] for r in values
            ),
            "paired_contacts": paired,
            "unmatched_geometric_candidates": sum(r["unmatched_geometry_count"] for r in values),
            "frozen_force_change_rms_N": float(
                np.sqrt(np.mean([r["frozen_force_change_squared_N2"] for r in values]))
            ),
            "raw_to_grouped_frozen_force_change_rms_N": float(
                np.sqrt(np.mean([r["group_force_change_squared_N2"] for r in values]))
            ),
        }
        for mode in ("legacy", "aligned"):
            for component in ("vector", "normal", "tangent"):
                name = f"{mode}_{component}_velocity_error_squared"
                result[f"{mode}_{component}_velocity_rmse_mm_s"] = (
                    float(np.sqrt(sum(r[name] for r in values) / paired) * 1e3) if paired else None
                )
        output.append(result)
    return output


@torch.no_grad()
def witness_audit(cases, mesh, native, corrected, material, config, base, output, prefix):
    rows, details, samples, jacobians = [], [], [], []
    tri = torch.tensor(mesh["surface_triangles"], dtype=torch.long)
    mass = torch.tensor(mesh["node_masses"])
    for case, (meta, inputs) in enumerate(cases):
        x, v, p, rv, target, target_v = inputs
        state = dynamics_state(x, v, mesh)
        robot = IndenterInput(p.numpy(), rv.numpy(), target.numpy(), target_v.numpy())
        default, _ = native.query(state, robot, with_force=False)
        legacy, _ = native.query(
            state, robot, with_force=False, velocity_mode="legacy_midpoint_tetra"
        )
        aligned, _ = native.query(state, robot, with_force=False, velocity_mode="material_surface")
        assert torch.equal(default.relative_velocities, legacy.relative_velocities)
        for field in ("points", "normals", "signed_gaps", "object_indices"):
            assert torch.equal(getattr(legacy, field), getattr(aligned, field))
        query = corrected.query(x, v, p, rv, base["sphere_radius_m"])
        face = query.face_candidates
        groups = group_material_contacts(
            face, query.face_indices, query.face_weights, tri, **config["grouping"]
        )
        q_native = (
            aligned.points
            + (base["skin_radius_m"] + 0.5 * aligned.signed_gaps[:, None]) * aligned.normals
        )
        mapping = material_samples(state.centers, state.velocities, tri, q_native)
        native_groups = group_material_contacts(
            aligned, mapping.face_indices, mapping.weights, tri, **config["grouping"]
        )
        origin = (state.centers * mass[:, None]).sum(0) / mass.sum()
        old_force, new_force = force(legacy, origin, material), force(aligned, origin, material)
        raw_force, group_force = (
            force(face, origin, material),
            force(groups.contacts, origin, material),
        )
        row = {
            "case": case,
            **meta,
            "geometric_contact": bool(query.contacts.num_contacts),
            "native_count": aligned.num_contacts,
            "face_count": face.num_contacts,
            "group_count": groups.contacts.num_contacts,
            "native_group_count": native_groups.contacts.num_contacts,
            "max_group_multiplicity": int(groups.multiplicities.max())
            if len(groups.multiplicities)
            else 0,
            "paired_contact_count": 0,
            "unmatched_geometry_count": 0,
            **{
                f"{mode}_{component}_velocity_error_squared": 0.0
                for mode in ("legacy", "aligned")
                for component in ("vector", "normal", "tangent")
            },
            "frozen_force_change_squared_N2": float((old_force - new_force).square().sum()),
            "group_force_change_squared_N2": float((raw_force - group_force).square().sum()),
            **{f"legacy_force_{a}_N": float(z) for a, z in zip("xyz", old_force, strict=True)},
            **{f"aligned_force_{a}_N": float(z) for a, z in zip("xyz", new_force, strict=True)},
        }
        if face.num_contacts and aligned.num_contacts:
            distance = torch.cdist(face.points, aligned.points)
            best, nearest = distance.min(1)
            paired = best <= 1e-8  # 10 nm, fixed geometry-pairing tolerance, not a contact margin.
            row["paired_contact_count"] = int(paired.sum())
            row["unmatched_geometry_count"] = int((~paired).sum())
            for mode, contact in (("legacy", legacy), ("aligned", aligned)):
                dv = face.relative_velocities[paired] - contact.relative_velocities[nearest[paired]]
                vn = (dv * face.normals[paired]).sum(-1)
                row[f"{mode}_vector_velocity_error_squared"] = float(dv.square().sum())
                row[f"{mode}_normal_velocity_error_squared"] = float(vn.square().sum())
                row[f"{mode}_tangent_velocity_error_squared"] = float(
                    (dv.square().sum() - vn.square().sum()).clamp_min(0)
                )
        elif face.num_contacts:
            row["unmatched_geometry_count"] = face.num_contacts
        assert bool(face.num_contacts) == bool(groups.contacts.num_contacts)
        rows.append(row)
        for i in range(face.num_contacts):
            details.append(
                {
                    "case": case,
                    "candidate": i,
                    "face_index": int(query.face_indices[i]),
                    "group": int(groups.group_for_candidate[i]),
                    **{f"w{k}": float(query.face_weights[i, k]) for k in range(3)},
                    **{
                        f"material_{a}_m": float(z)
                        for a, z in zip("xyz", query.face_material_points[i], strict=True)
                    },
                    **{
                        f"material_velocity_{a}_m_s": float(z)
                        for a, z in zip("xyz", query.face_material_velocities[i], strict=True)
                    },
                }
            )
            samples.append(
                np.r_[
                    case,
                    i,
                    int(query.face_indices[i]),
                    int(groups.group_for_candidate[i]),
                    query.face_weights[i].numpy(),
                    query.face_material_points[i].numpy(),
                    query.face_material_velocities[i].numpy(),
                    face.relative_velocities[i].numpy(),
                ]
            )
        if prefix == "trajectory" and meta["frame"] in config["jacobian_frames"]:
            independent = torch.tensor(node_velocities(native.model, native.data))
            expected = (independent[tri[mapping.face_indices]] * mapping.weights[..., None]).sum(1)
            error = float((mapping.velocities - expected).abs().max()) if len(expected) else 0.0
            jac = np.zeros((3, native.model.nv))
            mujoco.mj_jac(
                native.model,
                native.data,
                jac,
                None,
                p.numpy(),
                int(native.model.geom_bodyid[native.sphere]),
            )
            robot_error = float(np.max(np.abs(jac @ native.data.qvel - rv.numpy())))
            jacobians.append(
                {
                    **meta,
                    "material_velocity_max_abs_error_m_s": error,
                    "robot_velocity_max_abs_error_m_s": robot_error,
                }
            )
    write_csv(output / f"{prefix}_witness_per_case.csv", rows)
    if details:
        write_csv(output / f"{prefix}_candidate_groups.csv", details)
    np.savez_compressed(output / f"{prefix}_material_samples.npz", samples=np.asarray(samples))
    if jacobians:
        write_csv(output / "jacobian_velocity_checks.csv", jacobians)
    return rows, jacobians


@torch.no_grad()
def quadrature_audit(cases, mesh, rules, base, output, prefix, reference_rule=None):
    rows = []
    mass = torch.tensor(mesh["node_masses"])
    for case, (meta, inputs) in enumerate(cases):
        x, v, p, rv, _, _ = inputs
        state = dynamics_state(x, v, mesh)
        origin = (state.centers * mass[:, None]).sum(0) / mass.sum()
        for n, rule in rules.items():
            value = rule.normal_bases(
                x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
            )
            rows.append(basis_row(value, n, case=case, **meta))
        if reference_rule is not None:
            value = reference_rule.normal_bases(
                x, v, p, rv, base["sphere_radius_m"], base["skin_radius_m"], origin
            )
            rows.append(basis_row(value, reference_rule.subdivisions, case=case, **meta))
    write_csv(output / f"{prefix}_quadrature_per_case.csv", rows)
    numeric = np.array(
        [
            [
                r["case"],
                r["subdivisions"],
                r["active_samples"],
                r["contact_area_m2"],
                *[r[f"elastic_{a}_m3"] for a in "xyz"],
                *[r[f"damping_{a}_m3_s"] for a in "xyz"],
                *[r[f"torque_{a}_m4"] for a in "xyz"],
                *[r[f"unweighted_{a}_m"] for a in "xyz"],
            ]
            for r in rows
        ]
    )
    np.savez_compressed(output / f"{prefix}_quadrature_bases.npz", values=numeric)
    return rows


def quadrature_summary(rows, witness, reference_n, fields, stiffness):
    reference = {r["case"]: r for r in rows if r["subdivisions"] == reference_n}
    truth = {r["case"]: r["geometric_contact"] for r in witness}
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in (*fields, "subdivisions"))].append(r)
    output = []
    for key, values in sorted(groups.items()):
        estimate = np.array([[r[f"elastic_{a}_m3"] for a in "xyz"] for r in values])
        target = np.array(
            [[reference[r["case"]][f"elastic_{a}_m3"] for a in "xyz"] for r in values]
        )
        error = float(np.sqrt(np.mean(np.square(estimate - target).sum(-1))))
        scale = float(np.sqrt(np.mean(np.square(target).sum(-1))))
        raw = np.array([[r[f"unweighted_{a}_m"] for a in "xyz"] for r in values])
        output.append(
            {
                **dict(zip((*fields, "subdivisions"), key, strict=True)),
                "frames": len(values),
                "reference_subdivisions": reference_n,
                "reference_is_truth": False,
                "elastic_vector_rmse_m3": error,
                "reference_vector_rms_m3": scale,
                "relative_vector_l2_error": error / scale if scale > 0 else None,
                "elastic_vector_rms_m3": float(np.sqrt(np.mean(np.square(estimate).sum(-1)))),
                "frozen_unweighted_normal_force_rms_N": float(
                    np.sqrt(np.mean(np.square(raw).sum(-1))) * stiffness
                ),
                "geometric_contact_frames": sum(truth[r["case"]] for r in values),
                "geometric_contact_but_zero_quadrature_frames": sum(
                    truth[r["case"]] and r["active_samples"] == 0 for r in values
                ),
            }
        )
    return output


def selected_static_cases(mesh, base, config):
    selected = []
    features = rest_features(mesh, base["static"]["offset_grid"])
    for region in config["static_regions"]:
        selected.extend(
            [f["feature"] for f in features if f["region"] == region][
                : config["static_features_per_region"]
            ]
        )
    keep = set(selected)
    return (
        (meta, inputs)
        for meta, inputs in static_cases(mesh, base, config["static_depths_m"])
        if meta["feature"] in keep
    )


def save_zero_integral_cases(rows, witness, output, prefix):
    geometric = {r["case"]: r["geometric_contact"] for r in witness}
    missed = [r for r in rows if geometric[r["case"]] and r["active_samples"] == 0]
    if missed:
        write_csv(output / f"{prefix}_zero_integral_contact_cases.csv", missed)
    return len(missed)


@torch.no_grad()
def planar_check(mesh, base, config, output):
    x = torch.tensor(mesh["rest_positions_world"])
    v = torch.zeros_like(x)
    rows = []
    total = base["sphere_radius_m"] + base["skin_radius_m"]
    for n in config["flat_plane_subdivisions"]:
        rule = SurfaceQuadrature(
            mesh["rest_positions_world"], mesh["surface_triangles"], mesh["pinned_mask"], n
        )
        for depth in config["flat_plane_depths_m"]:
            height = total - depth
            p = torch.tensor([0, 0, float(x[:, 2].max()) + height]).double()
            result = rule.normal_bases(
                x,
                v,
                p,
                torch.zeros(3).double(),
                base["sphere_radius_m"],
                base["skin_radius_m"],
                torch.zeros(3).double(),
            )
            expected = -np.pi * height * depth**2
            expected_area = np.pi * (total**2 - height**2)
            estimate = float(result["elastic_vector_m3"][2])
            rows.append(
                {
                    "subdivisions": n,
                    "depth_m": depth,
                    "exact_normal_basis_z_m3": expected,
                    "estimated_normal_basis_z_m3": estimate,
                    "relative_normal_error": abs(estimate - expected) / abs(expected),
                    "exact_area_m2": expected_area,
                    "estimated_area_m2": float(result["contact_area_m2"]),
                    "active_samples": int(result["active_sample_count"]),
                }
            )
    write_csv(output / "flat_plane_closed_form.csv", rows)
    return rows


def checks(corrected, mesh, data, base, output):
    with np.load(data / "center_deep.npz") as f:
        arrays = {k: f[k] for k in CURRENT_FIELDS}
    x, v, p, rv, _, _ = current_inputs(arrays, 110)
    a = corrected.query(x, v, p, rv, base["sphere_radius_m"])
    reload = ContinuousContactSurface(BoundSurfaceGaussians.load(output / "binding.npz"))
    b = reload.query(x, v, p, rv, base["sphere_radius_m"])
    for field in (
        "face_indices",
        "face_weights",
        "face_material_points",
        "face_material_velocities",
    ):
        assert torch.equal(getattr(a, field), getattr(b, field))
    for key in CURRENT_FIELDS:
        arrays[key][111:] = np.nan
    arrays["contact_force_on_block_world"] = np.full((1, 3), np.nan)
    xx, vv, pp, rr, _, _ = current_inputs(arrays, 110)
    c = corrected.query(xx, vv, pp, rr, base["sphere_radius_m"])
    assert torch.equal(a.face_candidates.relative_velocities, c.face_candidates.relative_velocities)
    archives = 0
    for file in output.glob("*.npz"):
        with np.load(file, allow_pickle=False) as saved:
            assert all(np.isfinite(saved[k]).all() for k in saved.files)
        archives += 1
    return {
        "metadata_reload_bitwise_equal": True,
        "future_and_force_poisoning_invariant": True,
        "finite_numeric_archives": archives,
        "native_force_solver_and_integration_forbidden": True,
        "new_force_coefficients_fitted": False,
        "young_modulus_used_as_contact_stiffness": False,
        "next_state_or_velocity_predicted": False,
        "test_trajectory_arrays_loaded": False,
    }


def plots(groups, density, planar, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "validation"):
        rows = [r for r in density if r["split"] == split]
        axes[0].plot(
            [r["subdivisions"] for r in rows],
            [r["frozen_unweighted_normal_force_rms_N"] for r in rows],
            "o-",
            label=split,
        )
        axes[1].plot(
            [r["subdivisions"] for r in rows],
            [r["elastic_vector_rms_m3"] * 1e12 for r in rows],
            "o-",
            label=split,
        )
    axes[0].set(
        xlabel="Quadrature subdivision n",
        ylabel="Copied per-point spring force RMS (N)",
        title="Density dependence diagnostic",
    )
    axes[1].set(
        xlabel="Quadrature subdivision n",
        ylabel="Area integral vector RMS (1e-12 m^3)",
        title="Not a calibrated force",
    )
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "density_dependence.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for depth in sorted({r["depth_m"] for r in planar}):
        rows = [r for r in planar if r["depth_m"] == depth]
        axes[0].plot(
            [r["subdivisions"] for r in rows],
            [r["relative_normal_error"] for r in rows],
            "o-",
            label=f"{depth * 1e6:g} um",
        )
    axes[0].set(
        xlabel="Quadrature subdivision n", ylabel="Relative error vs closed form", yscale="log"
    )
    axes[0].legend()
    row = next(r for r in groups if r["split"] == "validation")
    axes[1].bar(
        ["legacy midpoint", "material surface"],
        [row["legacy_vector_velocity_rmse_mm_s"], row["aligned_vector_velocity_rmse_mm_s"]],
    )
    axes[1].set(
        ylabel="Matched current velocity RMSE (mm/s)", title="Same material-point convention"
    )
    fig.tight_layout()
    fig.savefig(output / "velocity_and_planar_convergence.png", dpi=160)
    plt.close(fig)


def forbidden(*args, **kwargs):
    raise AssertionError("Audit called force solver or state integrator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/audit_contact_sampling.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / config["base_config"]).read_text())
    if (
        config["seed"] != 7
        or config["quadrature_subdivisions"] != [4, 8, 16, 32]
        or config["velocity_modes"] != ["legacy_midpoint_tetra", "material_surface"]
    ):
        raise ValueError("Keep the fixed audit grid")
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
    write_json(output / "historical_hashes.json", before)
    write_json(output / "run_status.json", {"complete": False})
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "base_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    sources = [
        "scripts/audit_contact_sampling.py",
        "src/contact_gwm/contact_sampling.py",
        "src/contact_gwm/continuous_contact_surface.py",
        "src/contact_gwm/pinned_contact_pipeline.py",
        "configs/audit_contact_sampling.yaml",
        "tests/test_contact_sampling.py",
    ]
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    try:
        mesh = read_npz(data / "mesh.npz")
        native = NativeCurrentContact(data / "scene.xml", mesh)
        binding = BoundSurfaceGaussians.load(ROOT / config["surface_run"] / "binding.npz")
        if (
            not np.array_equal(binding.rest, mesh["rest_positions_world"])
            or not np.array_equal(binding.triangles, mesh["surface_triangles"])
            or not np.array_equal(binding.pinned, mesh["pinned_mask"])
        ):
            raise ValueError("Keep the original surface binding and node order")
        binding.save(output / "binding.npz")
        corrected = ContinuousContactSurface(binding)
        calibration = json.loads((ROOT / config["frozen_material"]).read_text())
        material = ContactMaterial(**calibration["material"])
        shutil.copy2(ROOT / config["frozen_material"], output / "frozen_material.json")
        rules = {
            n: SurfaceQuadrature(
                mesh["rest_positions_world"], mesh["surface_triangles"], mesh["pinned_mask"], n
            )
            for n in config["quadrature_subdivisions"]
        }
        fine = SurfaceQuadrature(
            mesh["rest_positions_world"],
            mesh["surface_triangles"],
            mesh["pinned_mask"],
            config["static_reference_subdivisions"],
        )
        with (
            patch.object(mujoco, "mj_forward", forbidden),
            patch.object(mujoco, "mj_contactForce", forbidden),
            patch.object(mujoco, "mj_step", forbidden),
        ):
            print("Auditing candidates and velocities on all 1500 trajectory states.", flush=True)
            rows, jac = witness_audit(
                trajectory_cases(mesh, base, data),
                mesh,
                native,
                corrected,
                material,
                config,
                base,
                output,
                "trajectory",
            )
            print(
                "Auditing shared-edge/corner candidate grouping on all original static queries.",
                flush=True,
            )
            static, _ = witness_audit(
                static_cases(mesh, base, base["static"]["penetration_depths_m"]),
                mesh,
                native,
                corrected,
                material,
                config,
                base,
                output,
                "static",
            )
            print(
                "Integrating material area at fixed resolutions 4/8/16/32 on all trajectories.",
                flush=True,
            )
            density = quadrature_audit(
                trajectory_cases(mesh, base, data), mesh, rules, base, output, "trajectory"
            )
            print(
                "Checking predetermined boundary examples and the independent planar closed form.",
                flush=True,
            )
            selected, _ = witness_audit(
                selected_static_cases(mesh, base, config),
                mesh,
                native,
                corrected,
                material,
                config,
                base,
                output,
                "selected_static",
            )
            static_density = quadrature_audit(
                selected_static_cases(mesh, base, config),
                mesh,
                rules,
                base,
                output,
                "selected_static",
                fine,
            )
            planar = planar_check(mesh, base, config, output)
            contracts = checks(corrected, mesh, data, base, output)
        groups = grouped_summary(rows, ("split",))
        phases = grouped_summary(rows, ("split", "phase"))
        summary = quadrature_summary(
            density, rows, max(rules), ("split",), material.normal_stiffness
        )
        write_csv(output / "velocity_grouping_by_split.csv", groups)
        write_csv(output / "velocity_grouping_by_phase.csv", phases)
        write_csv(output / "static_grouping_by_region.csv", grouped_summary(static, ("region",)))
        write_csv(output / "quadrature_by_split.csv", summary)
        write_csv(
            output / "quadrature_by_phase.csv",
            quadrature_summary(
                density, rows, max(rules), ("split", "phase"), material.normal_stiffness
            ),
        )
        write_csv(
            output / "selected_static_quadrature_metrics.csv",
            quadrature_summary(
                static_density, selected, fine.subdivisions, ("region",), material.normal_stiffness
            ),
        )
        plots(groups, summary, planar, output)
        zero_trajectory = save_zero_integral_cases(density, rows, output, "trajectory")
        zero_static = save_zero_integral_cases(static_density, selected, output, "selected_static")
        write_json(
            output / "limitations.json",
            {
                "trajectory_zero_integral_contact_records_across_resolutions": zero_trajectory,
                "selected_static_zero_integral_contact_records_across_resolutions": zero_static,
                "finite_quadrature_is_not_a_contact_detector": True,
                "witness_grouping_does_not_reproduce_all_native_manifolds": True,
                "area_force_coefficients_not_calibrated": True,
            },
        )
        assert before == {name: file_sha256(ROOT / name) for name in before}
        contracts["historical_files_sha256_unchanged"] = len(before)
        contracts["legacy_default_equals_explicit_legacy_all_queries"] = True
        contracts["velocity_modes_have_bitwise_identical_geometry"] = True
        contracts["jacobian_material_velocity_max_abs_error_m_s"] = max(
            r["material_velocity_max_abs_error_m_s"] for r in jac
        )
        contracts["jacobian_robot_velocity_max_abs_error_m_s"] = max(
            r["robot_velocity_max_abs_error_m_s"] for r in jac
        )
        environment = {
            "experiment_date": config["experiment_date"],
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "cpu_threads": config["cpu_threads"],
            "elapsed_seconds": time.perf_counter() - started,
            "output_bytes_before_summary": sum(
                f.stat().st_size for f in output.rglob("*") if f.is_file()
            ),
        }
        write_json(
            output / "summary.json",
            {
                "environment": environment,
                "checks": contracts,
                "velocity_grouping": groups,
                "quadrature": summary,
                "source_hashes": hashes([ROOT / n for n in sources]),
                "area_stiffness_unit": "N/m^3, not fitted",
                "elastic_basis_unit": "m^3",
                "damping_basis_unit": "m^3/s",
                "normal_torque_basis_unit": "m^4",
            },
        )
        write_json(output / "checks.json", contracts)
        write_json(output / "environment.json", environment)
        write_json(output / "failures.json", [])
        write_json(output / "run_status.json", {"complete": True})
        print(
            json.dumps(
                {
                    "output": str(output),
                    "velocity_grouping": groups,
                    "quadrature": summary,
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
