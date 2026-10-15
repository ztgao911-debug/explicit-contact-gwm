"""Change only the loss multiplier on existing single-frame or fixed-pair CPU probes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from contact_gwm.comparison import prepare_pairs
from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import (
    ConstantVelocityResidualDeformation,
    constant_velocity_increment,
)
from contact_gwm.loss_scale_diagnostics import fit_loss_scale, grouped_probe_metrics
from contact_gwm.optimization_diagnostics import (
    ProbeBatch,
    make_probe_model,
    probe_metrics,
    probe_predict,
)
from contact_gwm.residual_diagnostics import fixed_subset_indices

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def directory_hashes(directory, exclude=None):
    return {
        str(p.relative_to(directory)): file_sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and (exclude is None or not p.is_relative_to(exclude))
    }


def read_batch(path):
    with np.load(path, allow_pickle=False) as archive:
        return ProbeBatch(
            **{key: torch.from_numpy(archive[key]) for key in ProbeBatch.__dataclass_fields__}
        )


def load_config(path):
    addition = yaml.safe_load(path.read_text())
    probe_path = path.parent / addition.pop("base_config")
    probe_addition = yaml.safe_load(probe_path.read_text())
    base = yaml.safe_load((probe_path.parent / probe_addition.pop("base_config")).read_text())
    base.update(probe_addition)
    if addition.get("protocol") == "fixed_eight_pairs":
        if (
            base["device"] != "cpu"
            or base["dt"] != 0.01
            or base["trajectories"] != ["center_shallow", "center_deep"]
            or base["subset_quantiles"] != [0.25, 0.75]
            or addition["loss_scale"]["multipliers"] != [1.0, 100.0]
        ):
            raise ValueError("Keep the original center pairs and preselected loss multipliers")
        base["evaluation"]["protocol"] = "fixed_subset_optimization_diagnostic_not_generalization"
        fit_config = copy.deepcopy(base)
        fit_config["probe"] = {
            "adam_steps": base["overfit"]["epochs"],
            "log_interval": base["overfit"]["log_interval"],
        }
        return {"base": base, "fit_config": fit_config, **addition}
    base["trajectories"] = [base["probe"]["trajectory"]]
    base["evaluation"]["protocol"] = "single_frame_optimization_diagnostic_only"
    if base["device"] != "cpu" or base["dt"] != 0.01 or base["trajectories"] != ["center_deep"]:
        raise ValueError("Use the existing CPU center-deep probe at dt=0.01 s")
    if base["probe"]["frame"] != 71 or addition["loss_scale"]["multipliers"] != [
        1.0,
        100.0,
        10000.0,
    ]:
        raise ValueError("Keep the fixed frame and predeclared three loss multipliers")
    return {"base": base, **addition}


def make_current_batch(data_dir, base):
    dataset = GaussianPairDataset(data_dir, base)
    index = dataset.samples.index((base["probe"]["trajectory"], base["probe"]["frame"]))
    pair = dataset[index]
    encoder = ConstantVelocityResidualDeformation(**base["model"], dt=base["dt"])
    with torch.no_grad():
        context, gates = encoder.encode(pair.inputs)
    return ProbeBatch(
        context[None],
        gates[None],
        pair.inputs.gaussians.velocities[None],
        pair.target_delta[None],
        dataset.pinned,
        pair.inputs.neighborhood_mask[None],
        ~dataset.pinned,
    ), dataset


def make_eight_pair_batch(data_dir, base, reference):
    dataset = GaussianPairDataset(data_dir, base)  # Only base['trajectories'], the two centers.
    encoder = ConstantVelocityResidualDeformation(**base["model"], dt=base["dt"])
    pack = prepare_pairs(dataset, encoder)
    indices = fixed_subset_indices(pack, base["subset_quantiles"])
    samples = [
        {
            "index": i,
            "trajectory": pack.names[i],
            "frame": pack.frames[i],
            "time": pack.times[i],
            "phase": "press" if pack.phases[i] == 1 else "hold",
        }
        for i in indices.tolist()
    ]
    if len(samples) != 8 or samples != json.loads((reference / "fixed_samples.json").read_text()):
        raise ValueError("The fixed eight samples differ from the original diagnostic")
    batch = ProbeBatch(
        pack.context[indices],
        pack.gates[indices],
        pack.velocity[indices],
        pack.target[indices],
        dataset.pinned,
        pack.neighborhood[indices],
        ~dataset.pinned,
    )
    with np.load(reference / "train_residuals.npz", allow_pickle=False) as previous:
        for key, value in (
            ("gate", batch.gates),
            ("current_velocity", batch.velocity),
            ("target_delta", batch.target),
            ("contact_neighborhood_mask", batch.neighborhood),
        ):
            np.testing.assert_array_equal(previous[key][indices.numpy()], value.numpy())
        np.testing.assert_array_equal(previous["node_ids"], dataset.node_ids.numpy())
    with torch.no_grad():
        for j, i in enumerate(indices.tolist()):
            current_context, current_gate = encoder.encode(dataset[i].inputs)
            if not torch.equal(current_context, batch.context[j]) or not torch.equal(
                current_gate, batch.gates[j]
            ):
                raise ValueError("Cached contexts differ from live current-frame inputs")
    return batch, dataset, samples, [pack.phases[i] for i in indices]


def check_reference_control(directory, reference, eight_pairs=False):
    new = torch.load(directory / "loss_x1/checkpoint.pt", weights_only=True)
    old_model_dir = reference / ("unchanged_model" if eight_pairs else "single_frame_adam")
    old = torch.load(old_model_dir / "checkpoint.pt", weights_only=True)
    if new["step"] != old["epoch" if eight_pairs else "step"] or any(
        not torch.equal(value, old["model_state"][key]) for key, value in new["model_state"].items()
    ):
        raise ValueError("Instrumentation changed the original unscaled training result")
    with (directory / "loss_x1/history.csv").open() as handle:
        current = list(csv.DictReader(handle))
    with (old_model_dir / "history.csv").open() as handle:
        previous = list(csv.DictReader(handle))
    if len(current) != len(previous) or any(
        float(a["canonical_loss"]) != float(b["subset_loss" if eight_pairs else "loss"])
        for a, b in zip(current, previous, strict=True)
    ):
        raise ValueError("Unscaled per-step losses differ from the reference")
    return {"selected_weights_exact": True, "all_2001_canonical_losses_exact": True}


@torch.no_grad()
def verify_run(directory):
    summary = json.loads((directory / "summary.json").read_text())
    config = yaml.safe_load((directory / "config.yaml").read_text())
    fit_config = config.get("fit_config", config["base"])
    for relative, expected in summary["artifact_sha256"].items():
        if file_sha256(directory / relative) != expected:
            raise ValueError(f"Changed artifact: {relative}")
    batch = read_batch(directory / "sample.npz")
    grouped = []
    if summary.get("grouped_metrics"):
        samples = summary["fixed_samples"]
        names = [item["trajectory"] for item in samples]
        phases = [1 if item["phase"] == "press" else 2 for item in samples]
        grouped.extend(
            grouped_probe_metrics(
                batch,
                constant_velocity_increment(batch.velocity, batch.pinned),
                names,
                phases,
                "constant_velocity",
            )
        )
    for name, result in summary["experiments"].items():
        checkpoint = torch.load(directory / name / "checkpoint.pt", weights_only=True)
        if (
            checkpoint["config"] != fit_config
            or checkpoint["loss_multiplier"] != result["loss_multiplier"]
        ):
            raise ValueError("Checkpoint config/scale mismatch")
        model = make_probe_model("mlp", batch, fit_config)
        model.load_state_dict(checkpoint["model_state"])
        prediction = probe_predict(model, batch)
        with np.load(directory / name / "predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["prediction"], prediction.numpy())
            np.testing.assert_array_equal(archive["target"], batch.target.numpy())
        if probe_metrics(prediction, batch) != result["selected_metrics"]:
            raise ValueError("Reloaded metrics differ")
        if summary.get("grouped_metrics"):
            grouped.extend(grouped_probe_metrics(batch, prediction, names, phases, name))
        with (directory / name / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        if (
            int(min(history, key=lambda row: float(row["canonical_loss"]))["step"])
            != checkpoint["step"]
        ):
            raise ValueError("Checkpoint does not match canonical fitted-loss selection")
    if grouped != summary.get("grouped_metrics", []):
        raise ValueError("Reloaded trajectory/phase metrics differ")
    return {
        "npz_files_reloaded": 1 + len(summary["experiments"]),
        "checkpoints_reloaded": len(summary["experiments"]),
        "predictions_and_metrics_exact": True,
        "canonical_selection_argmin_matches": True,
        "grouped_metric_rows_exact": len(grouped),
    }


def make_plots(output, histories, updates, required_reduction=0.9):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, history in histories.items():
        for ax, key in zip(axes, ("fit_rmse_um", "fit_neighborhood_rmse_um"), strict=True):
            ax.plot([row["step"] for row in history], [row[key] for row in history], label=name)
    for ax, key, label in zip(
        axes,
        ("fit_rmse_um", "fit_neighborhood_rmse_um"),
        ("All 105 active nodes", "Contact neighborhood"),
        strict=True,
    ):
        baseline = next(iter(histories.values()))[0][key]
        ax.axhline(baseline, color="black", linestyle=":", label="CV baseline")
        ax.axhline(
            baseline * (1 - required_reduction),
            color="gray",
            linestyle="--",
            label=f"{required_reduction:.0%} reduction reference",
        )
        ax.set(title=label, xlabel="Adam step", ylabel="Increment vector RMSE (um)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "fitting_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name, history in histories.items():
        values = history[1:]
        for ax, key in zip(
            axes[0], ("raw_gradient_l2_for_update", "actual_parameter_update_l2"), strict=True
        ):
            ax.loglog([r["step"] for r in values], [r[key] for r in values], label=name, alpha=0.8)
        for ax, tensor in zip(
            axes[1], ("deformation.network.0.weight", "deformation.network.4.weight"), strict=True
        ):
            rows = [r for r in updates[name] if r["tensor"] == tensor]
            ax.plot(
                [r["step"] for r in rows],
                [r["actual_update_l2"] for r in rows],
                marker="o",
                label=name,
            )
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-12)
    axes[0, 0].axhline(1.0, linestyle=":", color="black", label="Clip threshold")
    for ax, title, ylabel in zip(
        axes.flat,
        (
            "Raw gradient: entire model",
            "Measured update: entire model",
            "Measured update: first hidden weight",
            "Measured update: output weight",
        ),
        (
            "Gradient L2 norm",
            "Parameter update L2 norm",
            "Parameter update L2 norm",
            "Parameter update L2 norm",
        ),
        strict=True,
    ):
        ax.set(title=title, xlabel="Adam step", ylabel=ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.savefig(output / "gradients_and_updates.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, rows in updates.items():
        for ax, tensor in zip(
            axes, ("deformation.network.0.weight", "deformation.network.2.weight"), strict=True
        ):
            chosen = [r for r in rows if r["tensor"] == tensor and r["positive_adam_rms_count"]]
            ax.semilogx(
                [r["step"] for r in chosen],
                [r["positive_adam_rms_below_epsilon_fraction"] for r in chosen],
                marker="o",
                label=name,
            )
    for ax, title in zip(axes, ("First hidden weight", "Second hidden weight"), strict=True):
        ax.set(
            title=title,
            xlabel="Adam step",
            ylabel="Fraction of positive moment RMS below epsilon",
            ylim=(-0.03, 1.03),
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "adam_epsilon_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/diagnose_soft_loss_scale.yaml"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_loss_scale_diagnostics/run_001"
    )
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        config = yaml.safe_load((args.verify_only / "config.yaml").read_text())
        torch.set_num_threads(config["base"]["cpu_threads"])
        print(verify_run(args.verify_only.resolve()))
        return
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output exists; choose another --output-dir: {output}")
    config = load_config(args.config)
    base = config["base"]
    eight_pairs = config.get("protocol") == "fixed_eight_pairs"
    fit_config = config.get("fit_config", base)
    required_reduction = (
        base["overfit"]["required_rmse_reduction"]
        if eight_pairs
        else base["probe"]["required_rmse_reduction"]
    )
    reference = ROOT / config["reference_run"]
    if yaml.safe_load((reference / "config.yaml").read_text()) != base:
        raise ValueError("Settings changed from the reference beyond the loss multiplier")
    torch.set_num_threads(base["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    protected = {
        "data": directory_hashes(args.data_dir),
        "existing_outputs": directory_hashes(ROOT / "outputs", exclude=output),
    }
    samples, phases = [], []
    if eight_pairs:
        batch, dataset, samples, phases = make_eight_pair_batch(args.data_dir, base, reference)
    else:
        batch, dataset = make_current_batch(args.data_dir, base)
        previous = read_batch(reference / "sample.npz")
        if any(
            not torch.equal(getattr(batch, key), getattr(previous, key))
            for key in batch.__dataclass_fields__
        ):
            raise ValueError("Current encoded frame or targets differ from the reference")
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    np.savez_compressed(
        output / "sample.npz",
        **{key: getattr(batch, key).numpy() for key in batch.__dataclass_fields__},
        node_ids=dataset.node_ids.numpy(),
        rest_positions=dataset.rest.numpy(),
    )
    if eight_pairs:
        write_json(output / "fixed_samples.json", samples)
    results, histories, updates, grouped_metrics = {}, {}, {}, []
    baseline = constant_velocity_increment(batch.velocity, batch.pinned)
    if eight_pairs:
        names = [sample["trajectory"] for sample in samples]
        grouped_metrics.extend(
            grouped_probe_metrics(batch, baseline, names, phases, "constant_velocity")
        )
    for multiplier in config["loss_scale"]["multipliers"]:
        name = f"loss_x{multiplier:g}"
        result = fit_loss_scale(
            batch, fit_config, multiplier, config["loss_scale"]["update_steps"], output / name
        )
        history, layer_updates, prediction = (
            result.pop("history"),
            result.pop("parameter_updates"),
            result.pop("prediction"),
        )
        histories[name], updates[name] = history, layer_updates
        write_csv(output / name / "history.csv", history)
        write_csv(output / name / "parameter_updates.csv", layer_updates)
        np.savez_compressed(
            output / name / "predictions.npz",
            prediction=prediction.numpy(),
            target=batch.target.numpy(),
            node_ids=dataset.node_ids.numpy(),
            constant_velocity=constant_velocity_increment(batch.velocity, batch.pinned).numpy(),
        )
        if eight_pairs:
            grouped_metrics.extend(grouped_probe_metrics(batch, prediction, names, phases, name))
            region_reductions = {
                key: 1 - result["selected_metrics"][key] / result["baseline_metrics"][key]
                for key in ("fit_rmse_um", "fit_neighborhood_rmse_um")
            }
            result["region_rmse_reductions"] = region_reductions
            result["meets_predeclared_fitting_criterion"] = all(
                value >= required_reduction for value in region_reductions.values()
            )
        else:
            result["meets_90_percent_fitting_criterion"] = (
                result["fit_rmse_reduction"] >= required_reduction
            )
        results[name] = result
    if len({r["initial_parameter_sha256"] for r in results.values()}) != 1:
        raise ValueError("Loss scales did not start from identical model parameters")
    reference_check = check_reference_control(output, reference, eight_pairs)
    rows = [
        {
            "experiment": name,
            **{
                key: r[key]
                for key in (
                    "loss_multiplier",
                    "equivalent_loss_unit_m",
                    "selected_step",
                    "canonical_selected_loss",
                    "clipped_steps",
                    "training_wall_seconds",
                    "fit_rmse_reduction",
                )
            },
            **r["selected_metrics"],
        }
        for name, r in results.items()
    ]
    write_csv(output / "comparison.csv", rows)
    if eight_pairs:
        write_csv(output / "grouped_metrics.csv", grouped_metrics)
    make_plots(output, histories, updates, required_reduction)
    sources = {
        "loss_scale_entrypoint.py": Path(__file__),
        **{
            f"{name}.py": ROOT / f"src/contact_gwm/{name}.py"
            for name in (
                "loss_scale_diagnostics",
                "optimization_diagnostics",
                "local_training",
                "deformation",
                "comparison",
                "residual_diagnostics",
            )
        },
        "gaussian_pairs.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
    }
    for name, source in sources.items():
        shutil.copy2(source, output / name)
    unchanged = {
        "data": directory_hashes(args.data_dir) == protected["data"],
        "existing_outputs": directory_hashes(ROOT / "outputs", exclude=output)
        == protected["existing_outputs"],
    }
    if not all(unchanged.values()):
        raise ValueError("Existing data or outputs changed")
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": "fixed_eight_pairs_loss_multiplier_only_diagnostic"
        if eight_pairs
        else "single_frame_loss_multiplier_only_diagnostic",
        "experiments": results,
        "reference_control": reference_check,
        "reference_sample_check": {
            "fixed_samples_and_stored_labels_velocities_gates_masks_exact": True,
            "cached_inputs_match_live_current_frames": True,
        }
        if eight_pairs
        else {"current_batch_exactly_matches_reference": True},
        "fixed_samples": samples,
        "grouped_metrics": grouped_metrics,
        "required_rmse_reduction": required_reduction,
        "all_initial_weights_identical": True,
        "validation_arrays_loaded": False,
        "test_arrays_loaded": False,
        "future_inputs_used": False,
        "free_rollout": False,
        "source_hashes": dataset.source_hashes,
        "protected_unchanged": unchanged,
        "protected_sha256": protected,
        "artifact_sha256": directory_hashes(output),
        "total_wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
    }
    write_json(output / "summary.json", summary)
    verification = verify_run(output)
    write_json(output / "verification.json", verification)
    print("Reference reproduction:", reference_check, flush=True)
    print("Independent verification:", verification, flush=True)
    print("Comparison:", rows, flush=True)
    print("Output:", output, flush=True)


if __name__ == "__main__":
    main()
