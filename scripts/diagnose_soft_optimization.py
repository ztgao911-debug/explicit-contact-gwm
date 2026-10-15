"""Run and verify single-node/frame fitting and gradient diagnostics on CPU."""

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

import numpy as np
import torch
import yaml

from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import (
    ConstantVelocityResidualDeformation,
    constant_velocity_increment,
)
from contact_gwm.optimization_diagnostics import (
    ProbeBatch,
    finite_difference_audit,
    fit_probe,
    make_probe_model,
    probe_metrics,
    probe_predict,
)

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
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and (exclude is None or not path.is_relative_to(exclude))
    }


def load_config(path):
    extra = yaml.safe_load(path.read_text())
    config = yaml.safe_load((path.parent / extra.pop("base_config")).read_text())
    config.update(extra)
    if config["device"] != "cpu" or config["dt"] != 0.01:
        raise ValueError("CPU and dt=0.01 s are required")
    if config["probe"]["trajectory"] not in ("center_shallow", "center_deep"):
        raise ValueError("Only center training trajectories may enter this diagnostic")
    config["trajectories"] = [config["probe"]["trajectory"]]
    config["evaluation"]["protocol"] = "single_frame_optimization_diagnostic_only"
    return config


def batch_arrays(batch):
    return {key: getattr(batch, key).numpy() for key in batch.__dataclass_fields__}


def read_batch(path):
    with np.load(path, allow_pickle=False) as archive:
        return ProbeBatch(
            **{key: torch.from_numpy(archive[key]) for key in ProbeBatch.__dataclass_fields__}
        )


@torch.no_grad()
def verify_run(directory):
    """Reload saved current tensors, labels, weights and outputs; write nothing."""
    summary = json.loads((directory / "summary.json").read_text())
    config = yaml.safe_load((directory / "config.yaml").read_text())
    for relative, expected in summary["artifact_sha256"].items():
        if file_sha256(directory / relative) != expected:
            raise ValueError(f"Changed artifact: {relative}")
    batch = read_batch(directory / "sample.npz")
    for name, result in summary["experiments"].items():
        path = directory / name
        checkpoint = torch.load(path / "checkpoint.pt", weights_only=True, map_location="cpu")
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration differs from the saved configuration")
        fit_mask = torch.zeros_like(batch.fit_mask)
        fit_mask[checkpoint["fit_node_indices"]] = True
        local = replace(batch, fit_mask=fit_mask)
        model = make_probe_model(result["kind"], local, config)
        model.load_state_dict(checkpoint["model_state"])
        prediction = probe_predict(model, local)
        with np.load(path / "predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["prediction"], prediction.numpy())
            np.testing.assert_array_equal(archive["target"], local.target.numpy())
            np.testing.assert_array_equal(archive["fit_mask"], fit_mask.numpy())
        if probe_metrics(prediction, local) != result["selected_metrics"]:
            raise ValueError("Saved metrics do not match the reloaded prediction")
        with (path / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        best = min(history, key=lambda r: float(r["loss"]))
        if int(best["step"]) != checkpoint["step"]:
            raise ValueError("Selected weights do not match minimum fitting loss")
    return {
        "npz_files_reloaded": 1 + len(summary["experiments"]),
        "checkpoints_reloaded": len(summary["experiments"]),
        "predictions_and_metrics_exact": True,
        "selection_argmin_matches": True,
    }


def make_plots(output, batch, results, predictions, histories, gradients):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, history in histories.items():
        ax = axes[0 if name == "single_node_adam" else 1]
        ax.semilogy(
            [r["backward_calls"] for r in history],
            [max(r["fit_rmse_um"] / history[0]["fit_rmse_um"], 1e-8) for r in history],
            label=name,
        )
    for ax, title in zip(
        axes, ("One selected node", "All 105 active nodes, one frame"), strict=True
    ):
        ax.axhline(1.0, color="black", linestyle=":", label="Constant velocity")
        ax.axhline(0.1, color="gray", linestyle="--", label="90% reduction criterion")
        ax.set(title=title, xlabel="Optimization backward evaluations", ylabel="Fit RMSE / CV RMSE")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.savefig(output / "fitting_curves.png", dpi=160)
    plt.close(fig)

    baseline = constant_velocity_increment(batch.velocity, batch.pinned)
    truth = (batch.target - baseline)[0, ~batch.pinned].numpy() * 1e6
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, name in zip(
        axes, ("single_frame_adam", "single_frame_node_table", "single_frame_lbfgs"), strict=True
    ):
        correction = (predictions[name] - baseline)[0, ~batch.pinned].numpy() * 1e6
        for axis, label in enumerate("xyz"):
            ax.scatter(truth[:, axis], correction[:, axis], s=9, label=label, alpha=0.6)
        limit = max(float(np.abs(truth).max()), float(np.abs(correction).max()))
        ax.plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
        ax.set(title=name, xlabel="True residual axis (um)", ylabel="Learned correction axis (um)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "frame_corrections.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, name in zip(axes, ("single_node_adam", "single_frame_adam"), strict=True):
        rows = gradients[name]
        for tensor in (
            "deformation.network.0.weight",
            "deformation.network.2.weight",
            "deformation.network.4.weight",
        ):
            values = [r for r in rows if r["tensor"] == tensor]
            ax.plot(
                [r["step"] for r in values],
                [r["gradient_l2"] for r in values],
                marker="o",
                label=tensor,
            )
        ax.set(title=name, xlabel="Adam step", ylabel="Parameter gradient L2 norm")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.savefig(output / "gradient_flow.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/diagnose_soft_optimization.yaml"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_optimization_diagnostics/run_001"
    )
    parser.add_argument(
        "--verify-only", type=Path, help="Reload saved results and check them without training"
    )
    args = parser.parse_args()
    if args.verify_only:
        config = yaml.safe_load((args.verify_only / "config.yaml").read_text())
        torch.set_num_threads(config["cpu_threads"])
        print(verify_run(args.verify_only.resolve()))
        return
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output exists; choose another --output-dir: {output}")
    config = load_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    protected = {
        "data": directory_hashes(args.data_dir.resolve()),
        "existing_outputs": directory_hashes(ROOT / "outputs", exclude=output),
    }
    dataset = GaussianPairDataset(args.data_dir, config)
    selected = next(
        (
            i
            for i, item in enumerate(dataset.samples)
            if item == (config["probe"]["trajectory"], config["probe"]["frame"])
        ),
        None,
    )
    if selected is None:
        raise ValueError("The configured frame is not an eligible contact press/hold pair")
    pair = dataset[selected]
    encoder = ConstantVelocityResidualDeformation(**config["model"], dt=config["dt"])
    with torch.no_grad():
        context, gates = encoder.encode(pair.inputs)
    node = int(
        torch.linalg.vector_norm(
            dataset.rest - torch.tensor(config["probe"]["node_rest_position"]), dim=-1
        ).argmin()
    )
    if dataset.pinned[node] or not pair.inputs.neighborhood_mask[node]:
        raise ValueError(
            "The selected geometric node must be active and inside the contact neighborhood"
        )
    batch = ProbeBatch(
        context=context[None],
        gates=gates[None],
        velocity=pair.inputs.gaussians.velocities[None],
        target=pair.target_delta[None],
        pinned=dataset.pinned,
        neighborhood=pair.inputs.neighborhood_mask[None],
        fit_mask=~dataset.pinned,
    )
    single_mask = torch.zeros_like(batch.fit_mask)
    single_mask[node] = True
    single = replace(batch, fit_mask=single_mask)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    np.savez_compressed(
        output / "sample.npz",
        **batch_arrays(batch),
        node_ids=dataset.node_ids.numpy(),
        rest_positions=dataset.rest.numpy(),
    )
    baseline = constant_velocity_increment(batch.velocity, batch.pinned)
    sample = {
        "trajectory": pair.trajectory,
        "frame": pair.frame,
        "time": pair.time,
        "next_time": pair.next_time,
        "node_index": node,
        "stable_node_id": int(dataset.node_ids[node]),
        "node_selection": "nearest shared rest position to configured point; no label ranking",
        "gate": float(gates[node]),
        "true_node_residual_um": ((batch.target - baseline)[0, node] * 1e6).tolist(),
        "source_hashes": dataset.source_hashes,
    }
    write_json(output / "sample.json", sample)
    audit = finite_difference_audit(batch, config)
    write_json(output / "gradient_audit.json", audit)
    print("Gradient audit:", {k: v for k, v in audit.items() if k != "gradient_rows"}, flush=True)
    plan = (
        ("single_node_adam", "mlp", "adam", single),
        ("single_frame_adam", "mlp", "adam", batch),
        ("single_frame_node_table", "node_table", "adam", batch),
        ("single_frame_lbfgs", "mlp", "lbfgs", batch),
    )
    results, histories, gradients, predictions = {}, {}, {}, {}
    for name, kind, optimizer, local in plan:
        result = fit_probe(kind, optimizer, local, config, output / name)
        histories[name] = result.pop("history")
        gradients[name] = result.pop("gradients")
        predictions[name] = result.pop("prediction")
        write_csv(output / name / "history.csv", histories[name])
        write_csv(output / name / "gradients.csv", gradients[name])
        result["meets_predeclared_fitting_criterion"] = (
            result["fit_rmse_reduction"] >= config["probe"]["required_rmse_reduction"]
        )
        results[name] = result
        np.savez_compressed(
            output / name / "predictions.npz",
            prediction=predictions[name].numpy(),
            target=local.target.numpy(),
            fit_mask=local.fit_mask.numpy(),
            constant_velocity=baseline.numpy(),
            node_ids=dataset.node_ids.numpy(),
        )
    rows = [
        {
            "experiment": name,
            "fitted_nodes": result["fitted_node_count"],
            "baseline_rmse_um": result["baseline"]["fit_rmse_um"],
            "fitted_rmse_um": result["selected_metrics"]["fit_rmse_um"],
            "fitted_near_rmse_um": result["selected_metrics"]["fit_neighborhood_rmse_um"],
            "rmse_reduction": result["fit_rmse_reduction"],
            "selected_step": result["selected_step"],
            "backward_calls": result["backward_calls"],
            "training_wall_seconds": result["training_wall_seconds"],
        }
        for name, result in results.items()
    ]
    write_csv(output / "comparison.csv", rows)
    make_plots(output, batch, results, predictions, histories, gradients)
    source_files = {
        "optimization_entrypoint.py": Path(__file__),
        "optimization_backend.py": ROOT / "src/contact_gwm/optimization_diagnostics.py",
        "local_models.py": ROOT / "src/contact_gwm/local_training.py",
        "deformation.py": ROOT / "src/contact_gwm/deformation.py",
        "gaussian_pairs.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
        "comparison_backend.py": ROOT / "src/contact_gwm/comparison.py",
    }
    for name, path in source_files.items():
        shutil.copy2(path, output / name)
    unchanged = {
        "data": directory_hashes(args.data_dir.resolve()) == protected["data"],
        "existing_outputs": directory_hashes(ROOT / "outputs", exclude=output)
        == protected["existing_outputs"],
    }
    if not all(unchanged.values()):
        raise RuntimeError("Existing data or outputs were changed")
    artifacts = directory_hashes(output)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": config["evaluation"]["protocol"],
        "sample": sample,
        "experiments": results,
        "gradient_audit": audit,
        "validation_arrays_loaded": False,
        "test_arrays_loaded": False,
        "future_inputs_used": False,
        "oracle_current_contact_conditioning": True,
        "free_rollout": False,
        "protected_unchanged": unchanged,
        "protected_sha256": protected,
        "artifact_sha256": artifacts,
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
    print("Independent reload:", verification, flush=True)
    print("Fitting results:", rows, flush=True)
    print("Output:", output, flush=True)


if __name__ == "__main__":
    main()
