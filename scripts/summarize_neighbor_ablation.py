"""Describe saved neighbor ablations without training or selecting checkpoints.

All improvement percentages are 100 * (reference - B) / reference: negative
values are regressions. Signed-node diagnostics use prediction horizons 1--20,
excluding the shared initialization. Correlation is a sampled diagnostic, not
a physical phase angle, and no metric here participates in model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ("constant_velocity", "frozen_legacy", "A", "B", "C")
REFERENCES = ("A", "C", "constant_velocity")
STARTS = (170, 210)
REGIONS = ("active", "contact_neighborhood")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty diagnostic source: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite input metric: {value}")
    return number


def _improvement(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or reference == 0:
        return None
    return 100 * (reference - value) / reference


def _comparisons(rows: list[dict], kind: str, per_trajectory: bool = False) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        if row["split"] != "validation" or row["method"] not in METHODS:
            continue
        if row["region"] not in REGIONS:
            continue
        horizon = int(row["horizon"]) if kind == "rollout" else 1
        if kind == "rollout" and horizon not in (5, 10, 20):
            continue
        trajectory = row.get("trajectory", "all_validation")
        key = (trajectory, row["phase"], row["contact_group"], row["region"], horizon)
        if row["method"] in grouped[key]:
            raise ValueError(f"Duplicate {kind} metric: {key}/{row['method']}")
        grouped[key][row["method"]] = row
    output = []
    for (trajectory, phase, group, region, horizon), methods in sorted(grouped.items()):
        if set(methods) != set(METHODS):
            raise ValueError(f"Unequal method coverage: {kind}/{trajectory}/{phase}/{group}")
        quantities = (
            ("increment_rmse_um" if kind == "one_step" else "position_rmse_um"),
            "velocity_rmse_mm_s",
        )
        for quantity in quantities:
            values = {method: _number(methods[method][quantity]) for method in METHODS}
            output.append(
                {
                    "evaluation": kind,
                    **({"trajectory": trajectory} if per_trajectory else {}),
                    "split": "validation",
                    "phase": phase,
                    "contact_group": group,
                    "region": region,
                    "horizon": horizon,
                    "metric": quantity,
                    **values,
                    **{
                        f"B_improvement_vs_{reference}_percent": _improvement(
                            values[reference], values["B"]
                        )
                        for reference in REFERENCES
                    },
                }
            )
    if not output:
        raise ValueError(f"No validation rows found for {kind}")
    return output


def _correlation(truth: np.ndarray, predicted: np.ndarray) -> float | None:
    if np.ptp(truth) == 0 or np.ptp(predicted) == 0:
        return None
    a, b = truth - truth.mean(), predicted - predicted.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(float).tiny:
        return None
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


def _probe_diagnostics(rows: list[dict], sign_rows: list[dict]) -> tuple[list, list]:
    reported = {
        (int(row["start"]), row["method"]): row
        for row in sign_rows
        if int(row["start"]) in STARTS
    }
    output, comparisons = [], []
    for start in STARTS:
        true_reference = None
        method_metrics = {}
        for method in METHODS:
            selected = sorted(
                [r for r in rows if int(r["start"]) == start and r["method"] == method],
                key=lambda row: float(row["time_s"]),
            )
            if len(selected) != 21 or {int(r["node_id"]) for r in selected} != {31}:
                raise ValueError(
                    f"Expected node 31, initialization + 20 predictions: {start}/{method}"
                )
            times = np.asarray([float(r["time_s"]) for r in selected])
            if not np.allclose(np.diff(times), 0.01, rtol=0, atol=1e-10):
                raise ValueError(f"Probe samples are not adjacent 0.01 s records: {start}/{method}")
            values = {
                key: np.asarray([float(row[key]) for row in selected[1:]])
                for key in (
                    "true_displacement_z_um",
                    "predicted_displacement_z_um",
                    "true_velocity_z_mm_s",
                    "predicted_velocity_z_mm_s",
                )
            }
            if not all(np.isfinite(value).all() for value in values.values()):
                raise ValueError(f"Nonfinite signed probe: {start}/{method}")
            truth = np.column_stack(
                (values["true_displacement_z_um"], values["true_velocity_z_mm_s"])
            )
            if true_reference is None:
                true_reference = truth
            elif not np.array_equal(truth, true_reference):
                raise ValueError("Methods do not share identical signed probe labels")
            tz, pz = values["true_displacement_z_um"], values["predicted_displacement_z_um"]
            tv, pv = values["true_velocity_z_mm_s"], values["predicted_velocity_z_mm_s"]
            true_std = float(tv.std()) if np.ptp(tv) else 0.0
            predicted_std = float(pv.std()) if np.ptp(pv) else 0.0
            original, label = reported[(start, method)], reported[(start, "mujoco_label")]
            metrics = {
                "trajectory": "left_deep",
                "start": start,
                "node_id": 31,
                "method": method,
                "prediction_samples": 20,
                "signed_dz_rmse_um": float(np.sqrt(np.square(pz - tz).mean())),
                "signed_vz_rmse_mm_s": float(np.sqrt(np.square(pv - tv).mean())),
                "true_vz_std_mm_s": true_std,
                "predicted_vz_std_mm_s": predicted_std,
                "vz_std_ratio_predicted_over_true": predicted_std / true_std if true_std else None,
                "demeaned_signed_vz_pearson": _correlation(tv, pv),
                "reported_sign_change_threshold_mm_s": float(original["threshold_mm_s"]),
                "reported_true_sign_changes_including_initial": int(
                    label["velocity_sign_changes_after_threshold"]
                ),
                "reported_predicted_sign_changes_including_initial": int(
                    original["velocity_sign_changes_after_threshold"]
                ),
            }
            output.append(metrics)
            method_metrics[method] = metrics
        b = method_metrics["B"]
        for reference in REFERENCES:
            ref = method_metrics[reference]
            b_ratio, ref_ratio = (
                value["vz_std_ratio_predicted_over_true"] for value in (b, ref)
            )
            b_corr, ref_corr = (value["demeaned_signed_vz_pearson"] for value in (b, ref))
            comparisons.append(
                {
                    "trajectory": "left_deep",
                    "start": start,
                    "node_id": 31,
                    "reference": reference,
                    "B_signed_dz_rmse_improvement_percent": _improvement(
                        ref["signed_dz_rmse_um"], b["signed_dz_rmse_um"]
                    ),
                    "B_signed_vz_rmse_improvement_percent": _improvement(
                        ref["signed_vz_rmse_mm_s"], b["signed_vz_rmse_mm_s"]
                    ),
                    "B_velocity_std_ratio_closer_to_one": (
                        abs(b_ratio - 1) < abs(ref_ratio - 1)
                        if b_ratio is not None and ref_ratio is not None
                        else None
                    ),
                    "B_demeaned_signed_vz_correlation_higher": (
                        b_corr > ref_corr if b_corr is not None and ref_corr is not None else None
                    ),
                    "B_sign_change_count_closer_to_true": abs(
                        b["reported_predicted_sign_changes_including_initial"]
                        - b["reported_true_sign_changes_including_initial"]
                    )
                    < abs(
                        ref["reported_predicted_sign_changes_including_initial"]
                        - ref["reported_true_sign_changes_including_initial"]
                    ),
                }
            )
    return output, comparisons


def _phase_plot(rows: list[dict], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, start in zip(axes, STARTS, strict=True):
        for index, method in enumerate(("A", "B", "C")):
            selected = sorted(
                [r for r in rows if int(r["start"]) == start and r["method"] == method],
                key=lambda row: float(row["time_s"]),
            )
            if index == 0:
                ax.plot(
                    [float(row["true_displacement_z_um"]) for row in selected],
                    [float(row["true_velocity_z_mm_s"]) for row in selected],
                    "ko--",
                    markersize=3,
                    label="MuJoCo label",
                )
            ax.plot(
                [float(row["predicted_displacement_z_um"]) for row in selected],
                [float(row["predicted_velocity_z_mm_s"]) for row in selected],
                "o-",
                markersize=3,
                label=method,
            )
        ax.set(
            title=f"left_deep / node 31 / start {start}",
            xlabel="Signed z displacement (um)",
            ylabel="Instantaneous signed vz (mm/s)",
        )
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.axvline(0, color="grey", linewidth=0.6)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / "signed_node_phase_portraits.png", dpi=150)
    plt.close(fig)


def summarize(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        name: run_dir / name
        for name in (
            "one_step_metrics.csv",
            "rollout_metrics.csv",
            "one_step_per_trajectory.csv",
            "rollout_per_trajectory.csv",
            "window_profiles.csv",
            "analysis/signed_node_recovery.csv",
            "analysis/signed_node_sign_changes.csv",
        )
    }
    tables = {name: _read_csv(path) for name, path in sources.items()}
    comparisons = _comparisons(tables["one_step_metrics.csv"], "one_step") + _comparisons(
        tables["rollout_metrics.csv"], "rollout"
    )
    per_trajectory = _comparisons(
        tables["one_step_per_trajectory.csv"], "one_step", True
    ) + _comparisons(tables["rollout_per_trajectory.csv"], "rollout", True)
    probes, probe_comparisons = _probe_diagnostics(
        tables["analysis/signed_node_recovery.csv"],
        tables["analysis/signed_node_sign_changes.csv"],
    )
    _write_csv(output / "ablation_comparison.csv", comparisons)
    _write_csv(output / "ablation_per_trajectory_comparison.csv", per_trajectory)
    _write_csv(output / "signed_node_ablation_metrics.csv", probes)
    _write_csv(output / "signed_node_ablation_comparison.csv", probe_comparisons)
    _phase_plot(tables["analysis/signed_node_recovery.csv"], output)
    result = {
        "run_directory": str(run_dir),
        "training_or_checkpoint_selection_performed": False,
        "scope": "left trajectories, validation split, every available phase/contact group",
        "percentage_definition": "100 * (reference RMSE - B RMSE) / reference RMSE",
        "negative_percentages_are_regressions": True,
        "missing_or_zero_reference_percentage": None,
        "probe_scope": "left_deep, node 31, starts 170/210, predicted horizons 1..20",
        "probe_variance": "population standard deviation, ddof=0",
        "probe_correlation": "demeaned signed vz Pearson; zero variance returns null",
        "sign_change_scope": "existing thresholded counts include common initialization sample",
        "interpretation_limit": (
            "Position improvement alone does not establish elastic oscillation recovery. "
            "Read velocity error, amplitude ratio, signed correlation and sign counts together. "
            "These discrete diagnostics are not physical phase angles or generalization proof."
        ),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
        },
        "comparisons": comparisons,
        "per_trajectory_comparisons": per_trajectory,
        "signed_node_metrics": probes,
        "signed_node_B_comparisons": probe_comparisons,
    }
    (output / "ablation_comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"Neighbor ablation: saved {len(comparisons)} complete comparison rows to {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    summarize(parser.parse_args().run_dir)
