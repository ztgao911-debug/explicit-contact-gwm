"""Inspect signed, instantaneous node recovery from already-saved rollouts.

This diagnostic does not train, select weights, or select nodes by prediction
error. The probe is the rest top-face node nearest the known indenter XY at
each fixed window start. Thresholded sign changes are counts, not frequencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("constant_velocity", "frozen_legacy", "gated_joint", "autonomous_joint")
STARTS = (170, 210)
THRESHOLD_MM_S = 0.01


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _count_changes(velocity_mm_s):
    above = velocity_mm_s[np.abs(velocity_mm_s) > THRESHOLD_MM_S]
    return int(np.count_nonzero(np.diff(np.sign(above))))


def analyze(run_dir: Path, methods=METHODS) -> dict:
    run_dir = run_dir.resolve()
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = ROOT / config["data_directory"]
    with np.load(data / "mesh.npz") as archive:
        mesh = {key: archive[key] for key in archive.files}
    with np.load(data / "left_deep.npz") as archive:
        trajectory = {key: archive[key] for key in archive.files}
    with np.load(run_dir / "rollout_predictions.npz") as archive:
        saved = {key: archive[key] for key in archive.files}
    if not np.array_equal(saved["node_ids"], mesh["node_ids"]):
        raise ValueError("Saved rollout node ordering differs from the source mesh")
    rest = mesh["rest_positions_world"]
    top = np.flatnonzero(np.isclose(rest[:, 2], rest[:, 2].max(), rtol=0, atol=1e-12))
    rows, summaries, selected_nodes = [], [], []
    for start in STARTS:
        xy = trajectory["indenter_position_world"][start, :2]
        index = top[np.argmin(np.linalg.norm(rest[top, :2] - xy, axis=1))]
        node_id = int(mesh["node_ids"][index])
        selected_nodes.append(
            {
                "start": start,
                "node_id": node_id,
                "rest_position_m": rest[index].tolist(),
                "indenter_xy_m": xy.tolist(),
            }
        )
        for method_index, method in enumerate(methods):
            prefix = f"left_deep__{start}__{method}"
            x, v = saved[f"{prefix}__position"], saved[f"{prefix}__velocity"]
            if x.shape != v.shape or not np.isfinite(x).all() or not np.isfinite(v).all():
                raise ValueError(f"Invalid saved predictions: {prefix}")
            frames = np.arange(start, start + len(x))
            true_z = (trajectory["node_position_world"][frames, index, 2] - rest[index, 2]) * 1e6
            pred_z = (x[:, index, 2] - rest[index, 2]) * 1e6
            true_v = trajectory["node_velocity_world"][frames, index, 2] * 1e3
            pred_v = v[:, index, 2] * 1e3
            for offset, frame in enumerate(frames):
                rows.append(
                    {
                        "time_s": float(trajectory["time"][frame]),
                        "method": method,
                        "start": start,
                        "node_id": node_id,
                        "true_displacement_z_um": float(true_z[offset]),
                        "predicted_displacement_z_um": float(pred_z[offset]),
                        "true_velocity_z_mm_s": float(true_v[offset]),
                        "predicted_velocity_z_mm_s": float(pred_v[offset]),
                    }
                )
            series = [(method, pred_v, pred_z)]
            if method_index == 0:
                series.insert(0, ("mujoco_label", true_v, true_z))
            for label, velocity, displacement in series:
                summaries.append(
                    {
                        "start": start,
                        "node_id": node_id,
                        "method": label,
                        "threshold_mm_s": THRESHOLD_MM_S,
                        "above_threshold_samples": int((np.abs(velocity) > THRESHOLD_MM_S).sum()),
                        "velocity_sign_changes_after_threshold": _count_changes(velocity),
                        "peak_absolute_velocity_z_mm_s": float(np.abs(velocity).max()),
                        "initial_displacement_z_um": float(displacement[0]),
                        "final_displacement_z_um": float(displacement[-1]),
                        "initial_velocity_z_mm_s": float(velocity[0]),
                        "final_velocity_z_mm_s": float(velocity[-1]),
                    }
                )
    _write_csv(output / "signed_node_recovery.csv", rows)
    _write_csv(output / "signed_node_sign_changes.csv", summaries)
    result = {
        "source_run": str(run_dir),
        "trajectory": "left_deep",
        "selected_nodes": selected_nodes,
        "node_selection": "nearest rest top-face node to prescribed indenter XY at window start",
        "threshold_rule": (
            "remove |instantaneous vz| <= 0.01 mm/s; count adjacent remaining sign changes"
        ),
        "interpretation": (
            "Counts describe sampled windows, not a continuous frequency or energy law."
        ),
        "training_or_weight_selection_performed": False,
        "sign_changes": summaries,
    }
    (output / "signed_node_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for start in STARTS:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
        for method_index, method in enumerate(methods):
            values = [row for row in rows if row["start"] == start and row["method"] == method]
            times = [row["time_s"] for row in values]
            for ax, quantity in zip(axes, ("displacement_z_um", "velocity_z_mm_s"), strict=True):
                if method_index == 0:
                    ax.plot(
                        times,
                        [row[f"true_{quantity}"] for row in values],
                        "k--",
                        label="MuJoCo label",
                    )
                ax.plot(times, [row[f"predicted_{quantity}"] for row in values], label=method)
                ax.axhline(0, color="grey", linewidth=0.6)
                ax.set_ylabel(quantity)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
        axes[0].set_title(f"left_deep / probe node {values[0]['node_id']} / start frame {start}")
        axes[1].set_xlabel("Time (s)")
        fig.savefig(output / f"signed_node_recovery_start_{start}.png", dpi=150)
        plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/joint_state_recovery/run_001"
    )
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()
    analyze(args.run_dir, methods=tuple(args.methods))
