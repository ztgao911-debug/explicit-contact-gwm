"""Readable tables and focused curves for the completed paired-response pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

NAMES = (
    "frozen",
    "original_absolute",
    "original_response",
    "expanded_absolute",
    "expanded_response",
)


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    output = parser.parse_args().run.resolve()
    summary = json.loads((output / "summary.json").read_text())
    one = read(output / "ordinary_evaluation/one_step_metrics.csv")
    roll = read(output / "ordinary_evaluation/rollout_metrics.csv")
    responses = read(output / "response_evaluation/summary.csv")
    physical_validity = {
        "ordinary_inverted_recorded_states": summary["checks"][
            "rollout_states_with_inverted_tetrahedra"
        ],
        "intervention_inverted_recorded_states": summary["response_evaluation"]["inverted_states"],
        "all_recorded_tetrahedra_positive": (
            summary["checks"]["rollout_states_with_inverted_tetrahedra"] == 0
            and summary["response_evaluation"]["inverted_states"] == 0
        ),
        "runtime_completion_does_not_imply_physical_validity": True,
    }
    (output / "physical_validity.json").write_text(json.dumps(physical_validity, indent=2) + "\n")
    lines = ["# Paired-response pilot: saved results", ""]

    def metric(rows, method, phase, region, key, horizon=None):
        return float(
            next(
                r[key]
                for r in rows
                if r["method"] == method
                and r.get("split", "validation") == "validation"
                and r.get("phase", phase) == phase
                and r.get("contact_group", "all") == "all"
                and r["region"] == region
                and (horizon is None or int(r["horizon"]) == horizon)
            )
        )

    def table(title, headers, rows):
        lines.extend(
            [
                "## " + title,
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        lines.append("")

    names = (*NAMES, "constant_velocity")
    result = []
    for name in names:
        method = name if name == "constant_velocity" else name + "__mujoco_force"
        result.append(
            [
                name,
                metric(one, method, "all", "active", "increment_rmse_um"),
                metric(roll, method, "all", "active", "position_rmse_um", 20),
                metric(roll, method, "all", "active", "velocity_rmse_mm_s", 20),
            ]
        )
    table(
        "Ordinary validation / M / active",
        ["model", "one-step dx um", "20-step x um", "20-step v mm/s"],
        [[r[0], *(f"{v:.3f}" for v in r[1:])] for r in result],
    )
    table(
        "Independent interventions / active",
        [
            "model",
            "horizon",
            "response x RMSE um",
            "response v RMSE mm/s",
            "negative / defined direction",
        ],
        [
            [
                r["method"],
                r["horizon"],
                f"{float(r['position_response_rmse']):.3f}",
                f"{float(r['velocity_response_rmse']):.3f}",
                f"{r['position_negative_cosines']} / {r['position_defined_cosines']}",
            ]
            for r in responses
            if r["region"] == "active" and r["horizon"] in ("1", "5", "10", "20")
        ],
    )
    for source, key, horizon, title in (
        (one, "increment_rmse_um", None, "Ordinary one-step phase dx / active and ROI um"),
        (roll, "position_rmse_um", 20, "Ordinary 20-step phase x / active and ROI um"),
    ):
        values = []
        for phase in ("press", "hold", "release", "recover"):
            row = [phase]
            for name in names:
                method = name if name == "constant_velocity" else name + "__mujoco_force"
                row.append(
                    " / ".join(
                        f"{metric(source, method, phase, region, key, horizon):.3f}"
                        for region in ("active", "contact_neighborhood")
                    )
                )
            values.append(row)
        table(title, ["phase", *names], values)
    table(
        "Ordinary validation 20-step / active / all conditions",
        ["model / condition", "x um", "v mm/s"],
        [
            [
                r["method"],
                f"{float(r['position_rmse_um']):.3f}",
                f"{float(r['velocity_rmse_mm_s']):.3f}",
            ]
            for r in roll
            if r["split"] == "validation"
            and r["phase"] == "all"
            and r["contact_group"] == "all"
            and r["region"] == "active"
            and r["horizon"] == "20"
        ],
    )
    (output / "result_tables.md").write_text("\n".join(lines))

    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_roll = read(output / "ordinary_evaluation/rollout_all_horizons.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for col, region in enumerate(("active", "contact_neighborhood")):
        for row_index, key in enumerate(("position_rmse_um", "velocity_rmse_mm_s")):
            ax = axes[row_index, col]
            for name in names:
                method = name if name == "constant_velocity" else name + "__mujoco_force"
                selected = [
                    r
                    for r in all_roll
                    if r["method"] == method
                    and r["split"] == "validation"
                    and r["phase"] == "all"
                    and r["region"] == region
                    and r["contact_group"] == "all"
                ]
                selected.sort(key=lambda r: int(r["horizon"]))
                ax.plot(
                    [int(r["horizon"]) for r in selected],
                    [float(r[key]) for r in selected],
                    label=name,
                )
            ax.set(title=f"M / {region}", xlabel="Steps", ylabel=key)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
    fig.savefig(output / "focused_multistep_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in NAMES[1:]:
        curve = read(output / name / "training_curve.csv")
        for ax, key in zip(
            axes, ("validation_ordinary", "validation_paired_response"), strict=True
        ):
            ax.semilogy([int(r["step"]) for r in curve], [float(r[key]) for r in curve], label=name)
            ax.set(xlabel="Updates", ylabel=key)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
    fig.savefig(output / "training_loss_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in names:
        selected = [r for r in responses if r["method"] == name and r["region"] == "active"]
        selected.sort(key=lambda r: int(r["horizon"]))
        for ax, key in zip(axes, ("position_response_rmse", "velocity_response_rmse"), strict=True):
            ax.plot(
                [int(r["horizon"]) for r in selected],
                [float(r[key]) for r in selected],
                marker="o",
                label=name,
            )
            ax.set(xlabel="Steps", ylabel=key)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
    fig.savefig(output / "intervention_response_curves.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "tables": "result_tables.md",
                "computational_run_complete": json.loads((output / "run_status.json").read_text())[
                    "complete"
                ],
                "training_seconds": summary["environment"].get("training_elapsed_seconds_sum"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
