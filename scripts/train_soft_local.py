"""Fit contact-conditioned local Gaussian increments on two existing CPU trajectories.

This is a teacher-forced training-fit diagnostic with current MuJoCo contacts.
It performs neither a held-out evaluation nor an autonomous future rollout.
"""

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
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import PinnedGaussianDeformation

ROOT = Path(__file__).resolve().parents[1]


def objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    neighborhood: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    squared = ((prediction - target) / config["loss_unit_m"]) ** 2
    weight = config["neighborhood_weight"]
    return (1 - weight) * squared[:, active].mean() + weight * squared[neighborhood].mean()


def region_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)[mask]
    baseline = torch.linalg.vector_norm(target, dim=-1)[mask]
    if not errors.numel():
        raise ValueError("Cannot report error on an empty evaluation mask")
    rmse = float(errors.square().mean().sqrt()) * 1000
    baseline_rmse = float(baseline.square().mean().sqrt()) * 1000
    return {
        "node_frame_count": errors.numel(),
        "vector_rmse_mm": rmse,
        "vector_mae_mm": float(errors.mean()) * 1000,
        "vector_p95_mm": float(torch.quantile(errors, 0.95)) * 1000,
        "zero_increment_vector_rmse_mm": baseline_rmse,
        "rmse_reduction_percent": 100 * (1 - rmse / baseline_rmse) if baseline_rmse else None,
    }


def evaluate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    neighborhood: torch.Tensor,
    names: list[str],
    phases: list[int],
) -> dict:
    all_active = active[None, :].expand(len(target), -1)
    result = {
        "active_nodes": region_metrics(prediction, target, all_active),
        "contact_neighborhood": region_metrics(prediction, target, neighborhood),
        "by_trajectory": {},
        "by_phase": {},
    }
    for name in sorted(set(names)):
        indices = torch.tensor([i for i, value in enumerate(names) if value == name])
        result["by_trajectory"][name] = {
            "active_nodes": region_metrics(
                prediction[indices], target[indices], all_active[indices]
            ),
            "contact_neighborhood": region_metrics(
                prediction[indices], target[indices], neighborhood[indices]
            ),
        }
    for phase, label in ((1, "press"), (2, "hold")):
        indices = torch.tensor([i for i, value in enumerate(phases) if value == phase])
        if len(indices):
            result["by_phase"][label] = {
                "active_nodes": region_metrics(
                    prediction[indices], target[indices], all_active[indices]
                ),
                "contact_neighborhood": region_metrics(
                    prediction[indices], target[indices], neighborhood[indices]
                ),
            }
    return result


def plot_results(directory: Path, history: list[dict], predictions: dict, rest: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].semilogy([r["epoch"] for r in history], [r["loss"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Scaled training MSE", title="Same-pair fitting loss")
    for key, label in (
        ("active_rmse_mm", "Active nodes"),
        ("near_rmse_mm", "Contact neighborhood"),
    ):
        axes[1].semilogy([r["epoch"] for r in history], [r[key] for r in history], label=label)
        axes[1].axhline(history[0][key], linestyle=":", alpha=0.5)
    axes[1].set(xlabel="Epoch", ylabel="Increment vector RMSE (mm)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(directory / "loss_curve.png", dpi=160)
    plt.close(fig)

    center_node = int(np.argmin(np.linalg.norm(rest - [0.0, 0.0, rest[:, 2].max()], axis=1)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, name in enumerate(("center_shallow", "center_deep")):
        selected = predictions["trajectory"] == name
        t = predictions["next_time"][selected]
        for key, label, style in (
            ("true_next_displacement", "True next deformation", "-"),
            ("predicted_next_displacement", "One-step prediction (true current state)", "--"),
            ("zero_increment_next_displacement", "Zero increment baseline", ":"),
        ):
            axes[row, 0].plot(
                t, -predictions[key][selected, center_node, 2] * 1000, style, label=label
            )
        axes[row, 0].set(
            title=name, xlabel="Next frame time (s)", ylabel="Top-center downward deformation (mm)"
        )
        for key, label in (
            ("target_delta", "True increment"),
            ("predicted_delta", "Predicted increment"),
        ):
            axes[row, 1].plot(t, -predictions[key][selected, center_node, 2] * 1000, label=label)
        axes[row, 1].axhline(0, linestyle=":", color="gray", label="Zero increment baseline")
        axes[row, 1].set(
            title=name, xlabel="Next frame time (s)", ylabel="Top-center downward increment (mm)"
        )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Training-fit diagnostic with current solver contacts; not a free rollout")
    fig.savefig(directory / "deformation_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_soft_local.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_local_training/run_001"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        config["device"] != "cpu"
        or config["evaluation"]["protocol"] != "teacher_forced_one_step_training_fit_only"
    ):
        parser.error("This run is explicitly a CPU, teacher-forced, training-fit diagnostic")
    training = config["training"]
    if training["epochs"] < 1 or training["batch_size"] < 1 or training["loss_unit_m"] <= 0:
        parser.error("Training epochs, batch size and loss unit must be positive")
    if not 0 <= training["neighborhood_weight"] <= 1:
        parser.error("Neighborhood loss weight must be in [0,1]")
    directory = args.output_dir.resolve()
    if directory.exists():
        parser.error(f"Output directory exists; choose a new --output-dir: {directory}")
    directory.mkdir(parents=True)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    dataset = GaussianPairDataset(args.data_dir, config)
    print(
        f"CPU pairs: {len(dataset)}; nodes: {len(dataset.rest)}; "
        f"active: {int((~dataset.pinned).sum())}",
        flush=True,
    )
    print(f"Selection: {dataset.selection_counts}", flush=True)
    model = PinnedGaussianDeformation(**config["model"])
    pairs = [dataset[i] for i in range(len(dataset))]
    with torch.no_grad():
        encoded = [model.encode(pair.inputs) for pair in pairs]
    context = torch.stack([value[0] for value in encoded])
    gates = torch.stack([value[1] for value in encoded])
    target = torch.stack([pair.target_delta for pair in pairs])
    neighborhood = torch.stack([pair.inputs.neighborhood_mask for pair in pairs])
    active = ~dataset.pinned
    if not torch.all(neighborhood.any(dim=1)):
        raise ValueError("Every selected pair needs at least one active contact-neighborhood node")
    names = [pair.trajectory for pair in pairs]
    phases = [int(dataset.trajectories[pair.trajectory]["phase_id"][pair.frame]) for pair in pairs]
    with torch.no_grad():
        initial = model.predict_encoded(context, gates, dataset.pinned)
    if torch.count_nonzero(initial):
        raise ValueError("Zero-initialized model should exactly match the zero-increment baseline")
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=training["learning_rate"] * 0.01
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_optimizer = None
    started = time.perf_counter()
    for epoch in range(training["epochs"] + 1):
        if epoch:
            model.train()
            indices = torch.randperm(len(dataset))
            for batch in indices.split(training["batch_size"]):
                prediction = model.predict_encoded(context[batch], gates[batch], dataset.pinned)
                loss = objective(prediction, target[batch], active, neighborhood[batch], training)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = model.predict_encoded(context, gates, dataset.pinned)
            value = float(objective(prediction, target, active, neighborhood, training))
            active_metrics = region_metrics(
                prediction, target, active[None].expand(len(target), -1)
            )
            near_metrics = region_metrics(prediction, target, neighborhood)
        history.append(
            {
                "epoch": epoch,
                "loss": value,
                "active_rmse_mm": active_metrics["vector_rmse_mm"],
                "near_rmse_mm": near_metrics["vector_rmse_mm"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        if epoch % training["log_interval"] == 0 or epoch == training["epochs"]:
            print(
                f"epoch {epoch:4d}: loss={value:.6g}, "
                f"active={active_metrics['vector_rmse_mm']:.6g} mm, "
                f"near={near_metrics['vector_rmse_mm']:.6g} mm",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model.predict_encoded(context, gates, dataset.pinned)
        # Verify cached batching and the live current-frame adapter agree after learning.
        live = torch.stack([model(pair.inputs) for pair in pairs])
    if not torch.allclose(predicted, live, rtol=1e-5, atol=1e-10):
        raise ValueError("Batched context cache differs from live current-frame inference")
    if torch.count_nonzero(predicted[:, dataset.pinned]):
        raise ValueError("Pinned nodes received a learned increment")
    metrics = evaluate(predicted, target, active, neighborhood, names, phases)
    checkpoint = {
        "model_state": best_state,
        "optimizer_state": best_optimizer,
        "config": config,
        "best_epoch": best_epoch,
        "best_training_loss": best_loss,
        "source_hashes": dataset.source_hashes,
        "gaussian_initialization": {
            "node_ids": dataset.node_ids,
            "rest_positions": dataset.rest,
            "scales": dataset.scales,
            "rotations": dataset.rotations,
            "pinned_mask": dataset.pinned,
        },
        "protocol": config["evaluation"]["protocol"],
    }
    torch.save(checkpoint, directory / "checkpoint.pt")
    torch.save(
        {"model_state": last_state, "config": config, "epoch": training["epochs"]},
        directory / "last_checkpoint.pt",
    )
    restored = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    reloaded = PinnedGaussianDeformation(**restored["config"]["model"])
    reloaded.load_state_dict(restored["model_state"])
    with torch.no_grad():
        reloaded_prediction = reloaded.predict_encoded(context, gates, dataset.pinned)
    if not torch.equal(predicted, reloaded_prediction):
        raise ValueError("Reloaded checkpoint predictions changed")
    current_displacement = torch.stack(
        [pair.inputs.gaussians.centers - dataset.rest for pair in pairs]
    )
    true_next_displacement = np.stack(
        [
            dataset.trajectories[pair.trajectory]["node_position_world"][pair.frame + 1]
            - dataset.mesh["rest_positions_world"]
            for pair in pairs
        ]
    )
    predictions = {
        "trajectory": np.asarray(names),
        "frame": np.asarray([pair.frame for pair in pairs]),
        "time": np.asarray([pair.time for pair in pairs]),
        "next_time": np.asarray([pair.next_time for pair in pairs]),
        "node_ids": dataset.node_ids.numpy(),
        "pinned_mask": dataset.pinned.numpy(),
        "contact_neighborhood_mask": neighborhood.numpy(),
        "contact_gates": gates.numpy(),
        "target_delta": target.numpy(),
        "predicted_delta": predicted.numpy(),
        "true_next_displacement": true_next_displacement,
        "predicted_next_displacement": (current_displacement + predicted).numpy(),
        "zero_increment_next_displacement": current_displacement.numpy(),
    }
    np.savez_compressed(directory / "predictions.npz", **predictions)
    np.savez_compressed(
        directory / "gaussian_initialization.npz",
        node_ids=dataset.node_ids.numpy(),
        centers=dataset.rest.numpy(),
        scales=dataset.scales.numpy(),
        rotations_wxyz=dataset.rotations.numpy(),
        pinned_mask=dataset.pinned.numpy(),
    )
    with (directory / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    source_files = {
        "training_entrypoint.py": Path(__file__),
        "gaussian_pair_loader.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
        "pinned_local_model.py": ROOT / "src/contact_gwm/local_training.py",
        "deformation_core.py": ROOT / "src/contact_gwm/deformation.py",
    }
    for name, source in source_files.items():
        shutil.copy2(source, directory / name)
    hashes_after = {name: file_sha256(dataset.directory / name) for name in dataset.source_hashes}
    if hashes_after != dataset.source_hashes:
        raise ValueError("Source dataset changed during training")
    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": config["evaluation"]["protocol"],
        "oracle_current_contact_conditioning": True,
        "future_inputs_used": False,
        "held_out_evaluation": False,
        "selection_counts": dataset.selection_counts,
        "pairs": len(dataset),
        "nodes": len(dataset.rest),
        "active_nodes": int(active.sum()),
        "fixed_nodes": int(dataset.pinned.sum()),
        "dt_s": config["dt"],
        "best_epoch": best_epoch,
        "training_wall_seconds": elapsed,
        "epochs": training["epochs"],
        "initial_loss": history[0]["loss"],
        "best_loss": best_loss,
        "metrics": metrics,
        "checkpoint_reload_exact": True,
        "live_adapter_cache_agree": True,
        "pinned_increment_max_m": float(predicted[:, dataset.pinned].abs().max()),
        "source_hashes_before": dataset.source_hashes,
        "source_hashes_after": hashes_after,
        "source_data_unchanged": hashes_after == dataset.source_hashes,
        "code_snapshot_sha256": {name: file_sha256(directory / name) for name in source_files},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "gaussian_rules": "One node per Gaussian; sigma=0.5*nearest rest-node distance; "
        "identity wxyz; "
        "opacity=0.5; scales/rotations fixed; bottom hard-pinned; free rigid update bypassed.",
    }
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_results(directory, history, predictions, dataset.rest.numpy())
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Best epoch: {best_epoch}; training wall seconds: {elapsed:.2f}; output: {directory}")


if __name__ == "__main__":
    main()
