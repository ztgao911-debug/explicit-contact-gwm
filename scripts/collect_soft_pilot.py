"""Collect exactly seven independent MuJoCo CPU soft-block trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import yaml

from contact_gwm.data import mujoco_soft

ROOT = Path(__file__).resolve().parents[1]


def plot_batch(directory: Path, mesh: dict, items: list[dict], config: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    control = mujoco_soft.load_npz(directory / "no_contact.npz")
    for row, position in enumerate(("left", "center", "right")):
        for item in items:
            if item["case"]["position"] != position or item["case"]["no_contact"]:
                continue
            arrays = mujoco_soft.load_npz(directory / item["file"])
            magnitude = np.linalg.norm(
                arrays["node_position_world"] - mesh["rest_positions_world"], axis=-1
            ).max(axis=1)
            label = item["case"]["depth_name"]
            axes[row, 0].plot(arrays["time"], magnitude * 1000, label=label)
            axes[row, 1].plot(
                arrays["time"], -arrays["net_contact_force_on_block_world"][:, 2], label=label
            )
        for column, ylabel in enumerate(
            ("Max node displacement (mm)", "Downward contact force (N)")
        ):
            ax = axes[row, column]
            ax.plot(
                control["time"],
                np.zeros(len(control["time"])),
                "--",
                color="gray",
                label="No-contact control",
            )
            for boundary in np.cumsum([config["phases"][name] for name in mujoco_soft.PHASES])[:-1]:
                ax.axvline(boundary, linestyle=":", color="gray", alpha=0.5)
            ax.set(title=position, xlabel="Time (s)", ylabel=ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("MuJoCo CPU FEM: approach / press / hold / release / recover")
    fig.savefig(directory / "deformation_curves.png", dpi=160)
    plt.close(fig)

    item = next(item for item in items if item["case"]["name"] == "center_deep")
    arrays = mujoco_soft.load_npz(directory / item["file"])
    peak = item["validation"]["peak_frame"]
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    for column, (positions, title) in enumerate(
        (
            (mesh["rest_positions_world"], "Initial shape"),
            (
                arrays["node_position_world"][peak],
                f"Peak deformation, t={arrays['time'][peak]:.2f} s",
            ),
        )
    ):
        ax = fig.add_subplot(1, 2, column + 1, projection="3d")
        surface = Poly3DCollection(
            positions[mesh["surface_triangles"]] * 1000,
            facecolor="tab:blue",
            edgecolor="navy",
            alpha=0.45,
            linewidth=0.3,
        )
        ax.add_collection3d(surface)
        pinned = positions[mesh["pinned_mask"]] * 1000
        ax.scatter(*pinned.T, color="black", s=8, label="Fixed nodes")
        if column:
            center = arrays["indenter_position_world"][peak] * 1000
            u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 15))
            radius = config["indenter"]["radius"] * 1000
            ax.plot_surface(
                center[0] + radius * np.cos(u) * np.sin(v),
                center[1] + radius * np.sin(u) * np.sin(v),
                center[2] + radius * np.cos(v),
                color="tab:red",
                alpha=0.2,
            )
        ax.set(
            xlim=(-35, 35),
            ylim=(-25, 25),
            zlim=(-2, 60),
            title=title,
            xlabel="x (mm)",
            ylabel="y (mm)",
            zlabel="z (mm)",
        )
        ax.set_box_aspect((70, 50, 62))
        ax.view_init(elev=22, azim=-65)
    fig.savefig(directory / "center_deep_shape.png", dpi=160)
    plt.close(fig)


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mujoco_soft_pilot.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001"
    )
    parser.add_argument(
        "--validate-only", type=Path, help="Reload and replay an existing seven-trajectory batch"
    )
    args = parser.parse_args()
    if args.validate_only:
        report = mujoco_soft.validate_batch(args.validate_only)
        print(f"Reload and independent MuJoCo replay: {report['trajectories_checked']}/7 passed")
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mujoco_soft.validate_config(config)
    directory = args.output_dir.resolve()
    if directory.exists():
        parser.error(f"Output directory already exists; select a new --output-dir: {directory}")
    xml = mujoco_soft.build_xml(config)
    start = time.perf_counter()
    model = mujoco.MjModel.from_xml_string(xml)
    compile_seconds = time.perf_counter() - start
    directory.mkdir(parents=True)
    (directory / "scene.xml").write_text(xml, encoding="utf-8")
    mujoco.mj_saveLastXML(str(directory / "compiled_scene.xml"), model)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(Path(mujoco_soft.__file__), directory / "collector_backend.py")
    shutil.copy2(Path(__file__), directory / "collector_entrypoint.py")
    protocol_cases = mujoco_soft.cases(config)
    mesh = mujoco_soft.mesh_from_model(
        model, mujoco_soft.initialize(model, protocol_cases[0], config)
    )
    np.savez_compressed(directory / "mesh.npz", **mesh)
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    environment = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "mujoco": mujoco.__version__,
        "device": "native_cpu",
        "git_revision": git_revision,
        "git_status": git_status,
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in metadata.distributions()
        },
    }
    write_json(directory / "environment.json", environment)
    shared_names = (
        "scene.xml",
        "compiled_scene.xml",
        "config.yaml",
        "mesh.npz",
        "environment.json",
        "collector_backend.py",
        "collector_entrypoint.py",
    )
    manifest = {
        "schema_version": 1,
        "config": config,
        "environment": environment,
        "state_spec": int(mujoco_soft.STATE_SPEC),
        "state_spec_name": "mjSTATE_INTEGRATION",
        "phase_names": list(mujoco_soft.PHASES),
        "model_compile_seconds": compile_seconds,
        "shared_sha256": {name: mujoco_soft.digest(directory / name) for name in shared_names},
        "label_source": "MuJoCo native 3D flex Saint Venant-Kirchhoff FEM + contact solver",
        "sampling": "All frame fields describe time[k]. mj_forward precedes sampling. "
        "Action target position is servo ctrl, target velocity is its analytic derivative. "
        "Actual indenter velocity and node velocities are engine Jacobian @ qvel.",
        "contact_convention": "Packed offsets index per-frame contacts. Frame rows are world "
        "basis vectors, row 0 is side0-to-side1 normal. Local wrench is on "
        "side1; world block force flips sign if the block is on side0. "
        "Geom/flex/element/vertex IDs are engine IDs; -1 means not applicable. "
        "Contacts are sampled at 100 Hz, not an exhaustive substep event log.",
        "trajectories": [],
        "failures": [],
    }
    write_json(directory / "manifest.json", manifest)
    for case in protocol_cases:
        print(f"Collecting {case['name']} ...", flush=True)
        try:
            arrays, timing = mujoco_soft.collect_trajectory(model, case, config, mesh)
            path = directory / f"{case['name']}.npz"
            tick = time.perf_counter()
            np.savez_compressed(path, **arrays)
            save_seconds = time.perf_counter() - tick
            reloaded = mujoco_soft.load_npz(path)
            if any(not np.array_equal(arrays[key], reloaded[key]) for key in arrays):
                raise ValueError("NPZ round trip changed an array")
            validation = mujoco_soft.validate_trajectory(reloaded, mesh, config, case)
            replay = mujoco_soft.replay_check(model, reloaded, validation["peak_frame"])
            item = {
                "case": case,
                "file": path.name,
                "sha256": mujoco_soft.digest(path),
                "compressed_bytes": path.stat().st_size,
                "uncompressed_array_bytes": sum(array.nbytes for array in arrays.values()),
                "save_wall_seconds": save_seconds,
                "timing": timing,
                "validation": validation,
                "replay": replay,
            }
            manifest["trajectories"].append(item)
            print(
                f"PASS {case['name']}: peak={validation['peak_displacement_m'] * 1000:.3f} mm, "
                f"residual={validation['final_residual_m'] * 1000:.3f} mm, "
                f"wall={timing['collection_wall_seconds']:.2f} s",
                flush=True,
            )
        except Exception as error:
            manifest["failures"].append({"case": case, "error": repr(error)})
            print(f"FAIL {case['name']}: {error}", flush=True)
        write_json(directory / "manifest.json", manifest)
    if manifest["failures"]:
        raise SystemExit("Pilot rejected; failures recorded in manifest.json")
    reload_start = time.perf_counter()
    mujoco_soft.validate_batch(directory)
    manifest["reload_validation_wall_seconds"] = time.perf_counter() - reload_start
    plot_start = time.perf_counter()
    plot_batch(directory, mesh, manifest["trajectories"], config)
    manifest["plot_wall_seconds"] = time.perf_counter() - plot_start
    manifest["trajectory_compressed_bytes"] = sum(
        item["compressed_bytes"] for item in manifest["trajectories"]
    )
    manifest["collection_wall_seconds"] = sum(
        item["timing"]["collection_wall_seconds"] for item in manifest["trajectories"]
    )
    manifest["simulated_seconds"] = sum(
        item["timing"]["simulated_seconds"] for item in manifest["trajectories"]
    )
    rows = []
    for item in manifest["trajectories"]:
        rows.append(
            {
                "case": item["case"]["name"],
                **item["validation"],
                **item["timing"],
                "compressed_bytes": item["compressed_bytes"],
            }
        )
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "manifest.json", manifest)
    print("\nAccepted: 7/7; reload and solver replay passed; failures: 0")
    print(f"Trajectory bytes: {manifest['trajectory_compressed_bytes']:,}")
    print(f"Output: {directory}")


if __name__ == "__main__":
    main()
