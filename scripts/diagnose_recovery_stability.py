"""Frozen-network finite-perturbation response against native MuJoCo integration."""

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
from contact_gwm.data.mujoco_soft import tetra_volumes
from contact_gwm.neighbor_dynamics import make_neighbor_model
from contact_gwm.recovery_multistep import autoregressive_recovery
from contact_gwm.recovery_stability import (
    NativeRecoveryReplay,
    make_perturbations,
    normalized_state_norm,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mujoco", "constant_velocity", "one_step", "five_step")


def read_inputs(config):
    data, source = ROOT / config["data_directory"], ROOT / config["source_run"]
    mesh = read_npz(data / "mesh.npz")
    arrays = read_npz(data / "center_deep.npz")
    manifest = json.loads((data / "manifest.json").read_text())
    sha = next(t["sha256"] for t in manifest["trajectories"] if t["file"] == "center_deep.npz")
    if (
        file_sha256(data / "center_deep.npz") != sha
        or file_sha256(data / "mesh.npz") != manifest["shared_sha256"]["mesh.npz"]
    ):
        raise ValueError("Independent data checksum mismatch")
    first, steps = config["window"]["first_frame"], config["window"]["steps"]
    command = arrays["action_target_position"][first].copy()
    if not np.array_equal(
        arrays["action_target_position"][first : first + steps + 1],
        np.broadcast_to(command, (steps + 1, 3)),
    ):
        raise ValueError("This fixed recovery replay requires a constant known robot command")
    window = read_npz(source / "fixed_window.npz")
    if (
        not np.array_equal(window["node_ids"], mesh["node_ids"])
        or not np.array_equal(window["x"], arrays["node_position_world"][first : first + steps + 1])
        or not np.array_equal(window["v"], arrays["node_velocity_world"][first : first + steps + 1])
    ):
        raise ValueError("Keep the original fitted window and stable IDs")
    normalization = yaml.safe_load((ROOT / "configs/train_soft_local.yaml").read_text())[
        "normalization"
    ]
    if [normalization[k] for k in ("rest_position", "displacement", "velocity")] != [
        0.03,
        0.01,
        0.05,
    ]:
        raise ValueError("Keep the existing network input normalization")
    return data, source, mesh, arrays, window, command, normalization


@torch.no_grad()
def collect(config, data, source, mesh, arrays, window, command, normalization, output):
    first, steps = config["window"]["first_frame"], config["window"]["steps"]
    dt = config["window"]["record_dt"]
    cases, dx, dv, directions = make_perturbations(
        mesh["pinned_mask"], config["perturbations"], config["seed"]
    )
    np.savez_compressed(
        output / "perturbations.npz",
        node_ids=mesh["node_ids"],
        delta_position=dx,
        delta_velocity=dv,
        directions=directions,
    )
    write_json(output / "cases.json", cases)
    x0, v0 = window["x"][0], window["v"][0]
    saved, native_metadata = {}, []
    replay = NativeRecoveryReplay(data / "scene.xml", mesh)
    robot_positions, robot_velocities = [], []
    px, pv = [], []
    max_initial_mismatch = 0.0
    for i, case in enumerate(cases):
        x, v, stats = replay.rollout(
            arrays["integration_state"][first].copy(), command, dx[i], dv[i], steps, dt
        )
        robot_positions.append(stats.pop("robot_positions"))
        robot_velocities.append(stats.pop("robot_velocities"))
        native_metadata.append({"case": case["case"], **stats})
        max_initial_mismatch = max(
            max_initial_mismatch,
            float(np.abs(x[0] - x0 - dx[i]).max()),
            float(np.abs(v[0] - v0 - dv[i]).max()),
        )
        px.append(x)
        pv.append(v)
    saved["mujoco__position"], saved["mujoco__velocity"] = np.asarray(px), np.asarray(pv)
    repeat_x, repeat_v, _ = replay.rollout(
        arrays["integration_state"][first].copy(), command, dx[0], dv[0], steps, dt
    )
    assert np.array_equal(repeat_x, px[0]) and np.array_equal(repeat_v, pv[0])
    position_replay_error = float(np.abs(px[0] - window["x"]).max())
    velocity_replay_error = float(np.abs(pv[0] - window["v"]).max())
    if (
        position_replay_error > config["evaluation"]["replay_position_tolerance_m"]
        or velocity_replay_error > config["evaluation"]["replay_velocity_tolerance_m_s"]
    ):
        raise AssertionError("Unperturbed native rollout does not reproduce saved labels")
    if max_initial_mismatch > 1e-12:
        raise AssertionError("Initial perturbations differ across simulator and learned models")
    robots = np.asarray(robot_positions)
    robot_v = np.asarray(robot_velocities)
    assert np.array_equal(robots, np.broadcast_to(robots[:1], robots.shape))
    assert np.array_equal(robot_v, np.broadcast_to(robot_v[:1], robot_v.shape))
    if any(row["max_contact_count_including_internal_steps"] for row in native_metadata):
        raise AssertionError("Perturbed native rollout leaves the no-contact regime")
    print(
        f"MuJoCo completed: {len(cases)} matched cases, 20 steps each; baseline replay exact",
        flush=True,
    )
    t = np.arange(steps + 1)[:, None, None] * dt
    saved["constant_velocity__position"] = np.asarray(
        [x0 + dx[i] + t * (v0 + dv[i]) for i in range(len(cases))]
    )
    saved["constant_velocity__velocity"] = np.asarray(
        [np.broadcast_to(v0 + dv[i], (steps + 1, *v0.shape)) for i in range(len(cases))]
    )
    rest = torch.tensor(mesh["rest_positions_world"], dtype=torch.float64)
    pinned = torch.tensor(mesh["pinned_mask"])
    original_predictions = read_npz(source / "predictions.npz")
    same_checkpoint_baseline, checkpoint_hashes, separation, volume_checks = {}, {}, {}, {}
    for method in config["models"]:
        checkpoint = source / method / "checkpoint.pt"
        checkpoint_hashes[str(checkpoint.relative_to(ROOT))] = file_sha256(checkpoint)
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = make_neighbor_model("B", mesh, artifact["config"])
        model.load_state_dict(artifact["model_state"], strict=True)
        model.eval()
        positions, velocities = [], []
        # Deliberately retain the original UNBATCHED numerical path for every
        # case; batch-size-dependent GEMM rounding is not an intended perturbation.
        for i in range(len(cases)):
            initial_x, initial_v = torch.tensor(x0 + dx[i]), torch.tensor(v0 + dv[i])
            x, v = autoregressive_recovery(
                model, initial_x, initial_v, rest, pinned, normalization, steps
            )
            positions.append(torch.cat((initial_x[None], x)).numpy())
            velocities.append(torch.cat((initial_v[None], v)).numpy())
        x, v = np.asarray(positions), np.asarray(velocities)
        saved[f"{method}__position"], saved[f"{method}__velocity"] = x, v
        same_checkpoint_baseline[method] = bool(
            np.array_equal(x[0], original_predictions[f"{method}__position"])
            and np.array_equal(v[0], original_predictions[f"{method}__velocity"])
        )
        assert same_checkpoint_baseline[method]
        if file_sha256(checkpoint) != checkpoint_hashes[str(checkpoint.relative_to(ROOT))]:
            raise AssertionError("Frozen checkpoint changed")
        print(f"{method} completed: {len(cases)} matched cases, no training", flush=True)
    sphere_radius = json.loads((data / "manifest.json").read_text())["config"]["indenter"]["radius"]
    skin = json.loads((data / "manifest.json").read_text())["config"]["block"]["collision_radius"]
    max_pin = 0.0
    for method in METHODS:
        x, v = saved[f"{method}__position"], saved[f"{method}__velocity"]
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise FloatingPointError("Non-finite perturbation trajectory")
        max_pin = max(
            max_pin,
            float(
                np.abs(
                    x[..., mesh["pinned_mask"], :]
                    - mesh["rest_positions_world"][mesh["pinned_mask"]]
                ).max()
            ),
            float(np.abs(v[..., mesh["pinned_mask"], :]).max()),
        )
        separation[method] = float(
            (robots[..., 2] - sphere_radius - skin - x[..., 2].max(-1)).min()
        )
        if separation[method] <= 0:
            raise AssertionError("No-contact network specialization no longer certified")
        ratio = tetra_volumes(x, mesh["tetrahedra"]) / mesh["rest_signed_tetra_volumes"]
        volume_checks[method] = float(ratio.min())
    saved.update(node_ids=mesh["node_ids"], robot_position=robots, robot_velocity=robot_v)
    np.savez_compressed(output / "trajectories.npz", **saved)
    loaded = read_npz(output / "trajectories.npz")
    assert all(np.array_equal(value, loaded[key]) for key, value in saved.items())
    write_json(output / "native_replay_checks.json", native_metadata)
    checks = {
        "native_unperturbed_position_replay_max_error_m": position_replay_error,
        "native_unperturbed_velocity_replay_max_error_m_s": velocity_replay_error,
        "native_repeat_bitwise_equal": True,
        "initial_state_max_difference_across_methods": max_initial_mismatch,
        "model_unperturbed_bitwise_equal_previous_run": same_checkpoint_baseline,
        "same_external_robot_motion_all_perturbations": True,
        "max_fixed_position_or_velocity_error": max_pin,
        "all_saved_trajectories_finite_and_reloaded_equal": True,
        "minimum_robot_separation_margin_m": separation,
        "minimum_tetra_volume_ratio": volume_checks,
        "case_count_per_method_including_base": len(cases),
        "native_physics_steps_excluding_repeat": len(cases)
        * steps
        * round(dt / replay.model.opt.timestep),
        "training_updates": 0,
        "frozen_checkpoint_hashes": checkpoint_hashes,
    }
    assert max_pin == 0
    return cases, saved, checks


def metrics(config, cases, saved, window, output, *, methods=METHODS):
    units = config["state_norm"]
    xu, vu = units["position_unit_m"], units["velocity_unit_m_s"]
    regions = {"active": ~window["pinned"], "contact_neighborhood": window["neighborhood"]}

    def norm(x, v, mask):
        return normalized_state_norm(x, v, mask, xu, vu)

    def rms(values, mask, multiplier):
        return np.sqrt(np.square(values[..., mask, :]).sum(-1).mean(-1)) * multiplier

    baseline, one_sided, central, linearity = [], [], [], []
    native_x, native_v = saved["mujoco__position"], saved["mujoco__velocity"]
    lookup = {
        (c["kind"], c["level"], c["direction"], c["sign"]): i
        for i, c in enumerate(cases)
        if c["kind"] != "none"
    }
    responses = {}
    for method in methods:
        x, v = saved[f"{method}__position"], saved[f"{method}__velocity"]
        for region, mask in regions.items():
            bx = np.sqrt(np.square((x[0] - native_x[0])[:, mask]).sum(-1).mean(-1)) * 1e6
            bv = np.sqrt(np.square((v[0] - native_v[0])[:, mask]).sum(-1).mean(-1)) * 1e3
            for h in range(x.shape[1]):
                baseline.append(
                    {
                        "method": method,
                        "region": region,
                        "horizon": h,
                        "position_rmse_um": float(bx[h]),
                        "velocity_rmse_mm_s": float(bv[h]),
                    }
                )
            for i, case in enumerate(cases[1:], 1):
                rx, rv = x[i] - x[0], v[i] - v[0]
                nx, nv = native_x[i] - native_x[0], native_v[i] - native_v[0]
                initial = float(norm(rx[0], rv[0], mask))
                gain, mismatch = (
                    norm(rx, rv, mask) / initial,
                    norm(rx - nx, rv - nv, mask) / initial,
                )
                components = {
                    "response_position_rms_um": rms(rx, mask, 1e6),
                    "response_velocity_rms_mm_s": rms(rv, mask, 1e3),
                    "perturbed_position_error_vs_mujoco_um": rms(x[i] - native_x[i], mask, 1e6),
                    "perturbed_velocity_error_vs_mujoco_mm_s": rms(v[i] - native_v[i], mask, 1e3),
                }
                for h in range(x.shape[1]):
                    one_sided.append(
                        {
                            **case,
                            "method": method,
                            "region": region,
                            "horizon": h,
                            "initial_normalized_perturbation": initial,
                            "gain": float(gain[h]),
                            "response_mismatch_vs_mujoco": float(mismatch[h]),
                            **{key: float(value[h]) for key, value in components.items()},
                        }
                    )
            for kind in ("position", "velocity"):
                for level in range(2):
                    for direction in range(config["perturbations"]["direction_count"]):
                        ip = lookup[kind, level, direction, 1]
                        im = lookup[kind, level, direction, -1]
                        rx, rv = (x[ip] - x[im]) / 2, (v[ip] - v[im]) / 2
                        nx, nv = (
                            (native_x[ip] - native_x[im]) / 2,
                            (native_v[ip] - native_v[im]) / 2,
                        )
                        initial, native_initial = (
                            float(norm(rx[0], rv[0], mask)),
                            float(norm(nx[0], nv[0], mask)),
                        )
                        qx, qv = rx / (xu * initial), rv / (vu * initial)
                        nqx, nqv = nx / (xu * native_initial), nv / (vu * native_initial)
                        gain = normalized_state_norm(qx, qv, mask, 1.0, 1.0)
                        mismatch = normalized_state_norm(qx - nqx, qv - nqv, mask, 1.0, 1.0)
                        even = (
                            norm((x[ip] + x[im]) / 2 - x[0], (v[ip] + v[im]) / 2 - v[0], mask)
                            / initial
                        )
                        components = {
                            "response_position_rms_um": rms(rx, mask, 1e6),
                            "response_velocity_rms_mm_s": rms(rv, mask, 1e3),
                            "response_position_error_vs_mujoco_um": rms(rx - nx, mask, 1e6),
                            "response_velocity_error_vs_mujoco_mm_s": rms(rv - nv, mask, 1e3),
                        }
                        responses[method, region, kind, level, direction] = (qx, qv)
                        for h in range(x.shape[1]):
                            central.append(
                                {
                                    "method": method,
                                    "region": region,
                                    "kind": kind,
                                    "level": level,
                                    "amplitude": cases[ip]["amplitude"],
                                    "direction": direction,
                                    "horizon": h,
                                    "central_gain": float(gain[h]),
                                    "response_mismatch_vs_mujoco": float(mismatch[h]),
                                    "even_over_odd_response": float(even[h] / max(gain[h], 1e-30)),
                                    **{key: float(value[h]) for key, value in components.items()},
                                }
                            )
                for direction in range(config["perturbations"]["direction_count"]):
                    small_x, small_v = responses[method, region, kind, 0, direction]
                    large_x, large_v = responses[method, region, kind, 1, direction]
                    deviation = normalized_state_norm(
                        small_x - large_x, small_v - large_v, mask, 1.0, 1.0
                    ) / np.maximum(normalized_state_norm(large_x, large_v, mask, 1.0, 1.0), 1e-30)
                    for h in range(len(deviation)):
                        linearity.append(
                            {
                                "method": method,
                                "region": region,
                                "kind": kind,
                                "direction": direction,
                                "horizon": h,
                                "relative_response_difference_between_amplitudes": float(
                                    deviation[h]
                                ),
                            }
                        )
    groups = defaultdict(list)
    for row in central:
        groups[row["method"], row["region"], row["kind"], row["level"], row["horizon"]].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        method, region, kind, level, horizon = key
        physical = groups["mujoco", region, kind, level, horizon]
        gains = [r["central_gain"] for r in rows]
        summary.append(
            {
                "method": method,
                "region": region,
                "kind": kind,
                "level": level,
                "amplitude": rows[0]["amplitude"],
                "horizon": horizon,
                "direction_count": len(rows),
                "mean_gain": float(np.mean(gains)),
                "min_gain": float(min(gains)),
                "max_gain": float(max(gains)),
                "gain_ratio_of_means_vs_mujoco": float(
                    np.mean(gains) / np.mean([r["central_gain"] for r in physical])
                ),
                "mean_response_mismatch_vs_mujoco": float(
                    np.mean([r["response_mismatch_vs_mujoco"] for r in rows])
                ),
                **{
                    f"mean_{field}": float(np.mean([r[field] for r in rows]))
                    for field in (
                        "response_position_rms_um",
                        "response_velocity_rms_mm_s",
                        "response_position_error_vs_mujoco_um",
                        "response_velocity_error_vs_mujoco_mm_s",
                    )
                },
            }
        )
    write_csv(output / "baseline_accuracy.csv", baseline)
    write_csv(output / "one_sided_responses.csv", one_sided)
    write_csv(output / "central_responses.csv", central)
    write_csv(output / "amplitude_linearity.csv", linearity)
    write_csv(output / "gain_summary_all_steps.csv", summary)
    write_csv(
        output / "gain_summary.csv",
        [r for r in summary if r["horizon"] in config["evaluation"]["horizons"]],
    )
    return baseline, summary, linearity


def plots(summary, baseline, output, *, methods=METHODS):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for region in ("active", "contact_neighborhood"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for column, kind in enumerate(("position", "velocity")):
            for level in range(2):
                ax = axes[level, column]
                for method in methods:
                    rows = [
                        r
                        for r in summary
                        if r["region"] == region
                        and r["kind"] == kind
                        and r["level"] == level
                        and r["method"] == method
                    ]
                    ax.plot(
                        [r["horizon"] for r in rows], [r["mean_gain"] for r in rows], label=method
                    )
                    ax.fill_between(
                        [r["horizon"] for r in rows],
                        [r["min_gain"] for r in rows],
                        [r["max_gain"] for r in rows],
                        alpha=0.12,
                    )
                amplitude = ((0.1, 1.0) if kind == "position" else (0.01, 0.1))[level]
                unit = "um" if kind == "position" else "mm/s"
                ax.set(
                    title=f"{region}: {kind}, level {level} ({amplitude} {unit})",
                    xlabel="Prediction steps",
                    ylabel="Normalized central response gain",
                    yscale="log",
                )
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
        fig.savefig(output / f"{region}_perturbation_gains.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric in zip(axes, ("position_rmse_um", "velocity_rmse_mm_s"), strict=True):
        for method in methods:
            rows = [r for r in baseline if r["method"] == method and r["region"] == "active"]
            ax.plot([r["horizon"] for r in rows], [r[metric] for r in rows], label=method)
        ax.set(xlabel="Prediction steps", ylabel=metric, title="Unperturbed baseline error")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.savefig(output / "baseline_accuracy.png", dpi=150)
    plt.close(fig)


def representative_curve(cases, saved, window, output, *, methods=METHODS):
    """Predeclared first direction, smallest positive position perturbation."""
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = next(
        i
        for i, c in enumerate(cases)
        if c["kind"] == "position" and c["level"] == 0 and c["direction"] == 0 and c["sign"] == 1
    )
    rest = window["rest"]
    top = np.flatnonzero(np.isclose(rest[:, 2], rest[:, 2].max()))
    xy = saved["robot_position"][0, 0, :2]
    node = int(top[np.argmin(np.linalg.norm(rest[top, :2] - xy, axis=1))])
    rows = []
    for method in methods:
        dx = saved[f"{method}__position"][index] - saved[f"{method}__position"][0]
        dv = saved[f"{method}__velocity"][index] - saved[f"{method}__velocity"][0]
        for h in range(len(dx)):
            rows.append(
                {
                    "method": method,
                    "case": cases[index]["case"],
                    "node_id": node,
                    "horizon": h,
                    "response_z_um": float(dx[h, node, 2] * 1e6),
                    "response_vz_mm_s": float(dv[h, node, 2] * 1e3),
                }
            )
    write_csv(output / "representative_response_curve.csv", rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for column, last in enumerate((5, 20)):
        for axis, metric in zip(
            axes[:, column], ("response_z_um", "response_vz_mm_s"), strict=True
        ):
            for method in methods:
                values = [r for r in rows if r["method"] == method and r["horizon"] <= last]
                axis.plot([r["horizon"] for r in values], [r[metric] for r in values], label=method)
            axis.set(
                title=f"Node {node}, first {last} steps", xlabel="Prediction steps", ylabel=metric
            )
            axis.legend(fontsize=7)
            axis.grid(alpha=0.3)
    fig.savefig(output / "representative_response_curve.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/diagnose_recovery_stability.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (
        config["seed"] != 7
        or config["window"]["first_frame"] != 210
        or config["window"]["steps"] != 20
        or config["window"]["record_dt"] != 0.01
    ):
        raise ValueError("Keep the existing fixed-seed recovery window")
    torch.set_num_threads(config["cpu_threads"])
    data, source, mesh, arrays, window, command, normalization = read_inputs(config)
    before = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
    output = args.output
    if output is None:
        parent, index = ROOT / config["output_root"], 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        output = parent / f"run_{index:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve existing results: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for path in ("scripts/diagnose_recovery_stability.py", "src/contact_gwm/recovery_stability.py"):
        shutil.copy2(ROOT / path, output / Path(path).name)
    cases, saved, checks = collect(
        config, data, source, mesh, arrays, window, command, normalization, output
    )
    baseline, summary, linearity = metrics(config, cases, saved, window, output)
    plots(summary, baseline, output)
    representative_curve(cases, saved, window, output)
    assert before == {path: file_sha256(ROOT / path) for path in before}
    checks["historical_file_count_unchanged_sha256"] = len(before)
    checks["only_center_deep_source_array_loaded"] = True
    write_json(output / "checks.json", checks)
    write_json(output / "historical_hashes.json", before)
    environment = {
        "client_experiment_date": config["experiment_date"],
        "host_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "seed": config["seed"],
        "cpu_threads": config["cpu_threads"],
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "training_updates": 0,
        "not_a_generalization_or_global_stability_proof": True,
    }
    write_json(output / "environment.json", environment)
    write_json(
        output / "summary.json",
        {
            "environment": environment,
            "checks": checks,
            "gain_at_reported_horizons": [
                r for r in summary if r["horizon"] in config["evaluation"]["horizons"]
            ],
        },
    )
    print(
        json.dumps({"output": str(output), "environment": environment, "checks": checks}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
