"""Analyze saved fixed-window fits without changing weights or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from validate_contact_pipeline import write_csv, write_json

from contact_gwm.data.gaussian_pairs import read_npz

ROOT = Path(__file__).resolve().parents[1]


def analyze(run_dir: Path):
    run_dir = run_dir.resolve()
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    checkpoints = [run_dir / arm / "checkpoint.pt" for arm in ("one_step", "five_step")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
    saved = read_npz(run_dir / "predictions.npz")
    window = read_npz(run_dir / "fixed_window.npz")
    checks = json.loads((run_dir / "checks.json").read_text())
    node = checks["selected_probe_node"]
    methods = [key.removesuffix("__position") for key in saved if key.endswith("__position")]
    true_x, true_v = saved["true_position"], saved["true_velocity"]
    rows, probes, growth = [], [], []
    for method in methods:
        x, v = saved[f"{method}__position"], saved[f"{method}__velocity"]
        if x.shape != true_x.shape or v.shape != true_v.shape or not np.isfinite(v).all():
            raise ValueError("Invalid saved fixed-window prediction")
        for step in range(1, len(true_x)):
            for region, mask in (
                ("active", ~window["pinned"]),
                ("contact_neighborhood", window["neighborhood"]),
            ):
                rows.append(
                    {
                        "method": method,
                        "horizon": step,
                        "region": region,
                        "position_rmse_um": float(
                            np.sqrt(np.square(x[step, mask] - true_x[step, mask]).sum(-1).mean())
                            * 1e6
                        ),
                        "velocity_rmse_mm_s": float(
                            np.sqrt(np.square(v[step, mask] - true_v[step, mask]).sum(-1).mean())
                            * 1e3
                        ),
                        "predicted_rms_speed_mm_s": float(
                            np.sqrt(np.square(v[step, mask]).sum(-1).mean()) * 1e3
                        ),
                        "true_rms_speed_mm_s": float(
                            np.sqrt(np.square(true_v[step, mask]).sum(-1).mean()) * 1e3
                        ),
                    }
                )
        for horizon in (5, 10, 20):
            tv = true_v[1 : horizon + 1, node, 2] * 1e3
            pv = v[1 : horizon + 1, node, 2] * 1e3
            probes.append(
                {
                    "method": method,
                    "node_id": node,
                    "horizon": horizon,
                    "signed_z_rmse_um": float(
                        np.sqrt(
                            np.square(
                                x[1 : horizon + 1, node, 2] - true_x[1 : horizon + 1, node, 2]
                            ).mean()
                        )
                        * 1e6
                    ),
                    "signed_vz_rmse_mm_s": float(np.sqrt(np.square(pv - tv).mean())),
                    "vz_std_ratio": 0.0 if np.ptp(pv) == 0 else float(pv.std() / tv.std()),
                    "vz_correlation": None if np.ptp(pv) == 0 else float(np.corrcoef(tv, pv)[0, 1]),
                }
            )
        active = [r for r in rows if r["method"] == method and r["region"] == "active"]
        r5, r10, r20 = (next(r for r in active if r["horizon"] == h) for h in (5, 10, 20))
        growth.append(
            {
                "method": method,
                "velocity_error_20_over_5": r20["velocity_rmse_mm_s"] / r5["velocity_rmse_mm_s"],
                "velocity_error_20_over_10": r20["velocity_rmse_mm_s"] / r10["velocity_rmse_mm_s"],
                "predicted_rms_speed_at_20_mm_s": r20["predicted_rms_speed_mm_s"],
                "true_rms_speed_at_20_mm_s": r20["true_rms_speed_mm_s"],
                "position_error_at_20_um": r20["position_rmse_um"],
                "velocity_error_at_20_mm_s": r20["velocity_rmse_mm_s"],
            }
        )
    write_csv(output / "per_step_errors_and_speed.csv", rows)
    write_csv(output / "probe_by_prediction_horizon.csv", probes)
    write_csv(output / "error_growth.csv", growth)
    _plots(rows, saved, window, methods, node, output)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
    result = {
        "all_results_are_training_segment_diagnostics": True,
        "no_training_or_checkpoint_selection_performed": True,
        "checkpoint_sha256_unchanged": before,
        "node_id": node,
        "growth": growth,
        "probe_by_horizon": probes,
        "interpretation": (
            "Error-growth ratios are sampled diagnostics, "
            "not Jacobian spectra or Lyapunov exponents"
        ),
    }
    write_json(output / "analysis_summary.json", result)
    print(json.dumps({"output": str(output), "growth": growth}, indent=2), flush=True)
    return result


def _plots(rows, saved, window, methods, node, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, region in enumerate(("active", "contact_neighborhood")):
        for axis, quantity in zip(
            axes[:, column], ("position_rmse_um", "velocity_rmse_mm_s"), strict=True
        ):
            for method in methods:
                values = [r for r in rows if r["method"] == method and r["region"] == region]
                axis.plot(
                    [r["horizon"] for r in values], [r[quantity] for r in values], label=method
                )
            axis.set(title=region, xlabel="Free prediction steps", ylabel=quantity)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
    fig.savefig(output / "per_step_error_growth.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for column, (name, count) in enumerate((("first ten steps", 11), ("all twenty steps", 21))):
        t = np.arange(count) * 0.01
        true_z = (saved["true_position"][:count, node, 2] - window["rest"][node, 2]) * 1e6
        true_v = saved["true_velocity"][:count, node, 2] * 1e3
        axes[0, column].plot(t, true_z, "k--", label="MuJoCo")
        axes[1, column].plot(t, true_v, "k--", label="MuJoCo")
        for method in methods:
            pred_z = (saved[f"{method}__position"][:count, node, 2] - window["rest"][node, 2]) * 1e6
            pred_v = saved[f"{method}__velocity"][:count, node, 2] * 1e3
            axes[0, column].plot(t, pred_z, label=method)
            axes[1, column].plot(t, pred_v, label=method)
        for index, quantity in enumerate(("Signed z (um)", "Instantaneous vz (mm/s)")):
            axes[index, column].set(
                title=name, xlabel="Time since initialization (s)", ylabel=quantity
            )
            axes[index, column].legend(fontsize=7)
            axes[index, column].grid(alpha=0.3)
    fig.savefig(output / "early_and_late_signed_recovery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/recovery_multistep_diagnostic/run_001"
    )
    analyze(parser.parse_args().run_dir)
