"""Whole-trajectory comparison utilities; fitting has no access to test samples."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .data.gaussian_pairs import GaussianPairDataset
from .local_training import (
    ConstantVelocityResidualDeformation,
    PinnedGaussianDeformation,
    constant_velocity_increment,
)

METHODS = ("static", "constant_velocity", "direct", "velocity_residual")


def validate_splits(splits: dict) -> None:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test trajectory splits")
    all_names = [name for names in splits.values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Whole-trajectory splits overlap")
    expected = {
        "train": {"center_shallow", "center_deep"},
        "validation": {"left_shallow", "left_deep"},
        "test": {"right_shallow", "right_deep"},
    }
    if any(set(splits[key]) != value for key, value in expected.items()):
        raise ValueError("Pilot comparison requires center/train, left/validation, right/test")


@dataclass
class PreparedPairs:
    dataset: GaussianPairDataset
    context: Tensor
    gates: Tensor
    velocity: Tensor
    target: Tensor
    neighborhood: Tensor
    names: list[str]
    frames: list[int]
    phases: list[int]
    times: list[float]

    @property
    def active(self) -> Tensor:
        return ~self.dataset.pinned


@torch.no_grad()
def prepare_pairs(
    dataset: GaussianPairDataset, encoder: PinnedGaussianDeformation
) -> PreparedPairs:
    pairs = [dataset[i] for i in range(len(dataset))]
    encoded = [encoder.encode(pair.inputs) for pair in pairs]
    result = PreparedPairs(
        dataset=dataset,
        context=torch.stack([item[0] for item in encoded]),
        gates=torch.stack([item[1] for item in encoded]),
        velocity=torch.stack([pair.inputs.gaussians.velocities for pair in pairs]),
        target=torch.stack([pair.target_delta for pair in pairs]),
        neighborhood=torch.stack([pair.inputs.neighborhood_mask for pair in pairs]),
        names=[pair.trajectory for pair in pairs],
        frames=[pair.frame for pair in pairs],
        phases=[
            int(dataset.trajectories[pair.trajectory]["phase_id"][pair.frame]) for pair in pairs
        ],
        times=[pair.time for pair in pairs],
    )
    if not torch.all(result.neighborhood.any(dim=1)):
        raise ValueError("A selected frame has no active contact-neighborhood nodes")
    return result


def make_model(kind: str, config: dict) -> PinnedGaussianDeformation:
    if kind == "direct":
        return PinnedGaussianDeformation(**config["model"])
    if kind == "velocity_residual":
        return ConstantVelocityResidualDeformation(**config["model"], dt=config["dt"])
    raise ValueError(f"Unknown learned method: {kind}")


def predict(
    kind: str,
    model: PinnedGaussianDeformation | None,
    pack: PreparedPairs,
    indices: Tensor | None = None,
) -> Tensor:
    rows = slice(None) if indices is None else indices
    pinned = pack.dataset.pinned
    if kind == "static":
        return torch.zeros_like(pack.velocity[rows])
    if kind == "constant_velocity":
        return constant_velocity_increment(pack.velocity[rows], pinned)
    if kind == "direct":
        return model.predict_encoded(pack.context[rows], pack.gates[rows], pinned)
    if kind == "velocity_residual":
        return model.predict_encoded(
            pack.context[rows], pack.gates[rows], pinned, pack.velocity[rows]
        )
    raise ValueError(f"Unknown method: {kind}")


def balanced_loss(
    prediction: Tensor, target: Tensor, active: Tensor, neighborhood: Tensor, training: dict
) -> Tensor:
    squared = ((prediction - target) / training["loss_unit_m"]) ** 2
    weight = training["neighborhood_weight"]
    return (1 - weight) * squared[:, active].mean() + weight * squared[neighborhood].mean()


def metric(prediction: Tensor, target: Tensor, mask: Tensor) -> dict:
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)[mask]
    if not errors.numel():
        raise ValueError("Cannot score an empty node/frame group")
    return {
        "node_frame_count": errors.numel(),
        "vector_rmse_mm": float(errors.square().mean().sqrt()) * 1000,
        "vector_mae_mm": float(errors.mean()) * 1000,
    }


def metric_rows(split: str, method: str, prediction: Tensor, pack: PreparedPairs) -> list[dict]:
    """Overall, per-trajectory, per-phase and per-trajectory/phase metrics."""
    rows = []
    for trajectory in ("all", *sorted(set(pack.names))):
        for phase_id, phase in ((None, "all"), (1, "press"), (2, "hold")):
            indices = [
                i
                for i, name in enumerate(pack.names)
                if (trajectory == "all" or name == trajectory)
                and (phase_id is None or pack.phases[i] == phase_id)
            ]
            if not indices:
                continue
            p, t = prediction[indices], pack.target[indices]
            active = pack.active[None].expand(len(indices), -1)
            for region, mask in (
                ("active_nodes", active),
                ("contact_neighborhood", pack.neighborhood[indices]),
            ):
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "trajectory": trajectory,
                        "phase": phase,
                        "region": region,
                        "pairs": len(indices),
                        **metric(p, t, mask),
                    }
                )
    return rows


def fit_model(
    kind: str, train: PreparedPairs, validation: PreparedPairs, config: dict, output: Path
) -> dict:
    """Only train and validation packs enter this function. Epoch 0 is a valid candidate."""
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = make_model(kind, config)
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings["learning_rate"] * 0.01
    )
    output.mkdir(parents=True, exist_ok=False)
    history = []
    best_validation = float("inf")
    selected = None
    started = time.perf_counter()
    for epoch in range(settings["epochs"] + 1):
        if epoch:
            model.train()
            for batch in torch.randperm(len(train.target), generator=generator).split(
                settings["batch_size"]
            ):
                value = balanced_loss(
                    predict(kind, model, train, batch),
                    train.target[batch],
                    train.active,
                    train.neighborhood[batch],
                    settings,
                )
                if not torch.isfinite(value):
                    raise RuntimeError(f"Non-finite {kind} loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            train_prediction = predict(kind, model, train)
            val_prediction = predict(kind, model, validation)
            train_loss = float(
                balanced_loss(
                    train_prediction, train.target, train.active, train.neighborhood, settings
                )
            )
            val_loss = float(
                balanced_loss(
                    val_prediction,
                    validation.target,
                    validation.active,
                    validation.neighborhood,
                    settings,
                )
            )
            mask = validation.active[None].expand(len(validation.target), -1)
            val_active = metric(val_prediction, validation.target, mask)["vector_rmse_mm"]
            val_near = metric(val_prediction, validation.target, validation.neighborhood)[
                "vector_rmse_mm"
            ]
        if not all(torch.isfinite(torch.tensor([train_loss, val_loss]))):
            raise RuntimeError("Non-finite train/validation score")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_active_rmse_mm": val_active,
                "validation_neighborhood_rmse_mm": val_near,
            }
        )
        if val_loss < best_validation:
            best_validation = val_loss
            selected = {
                "model_state": copy.deepcopy(model.state_dict()),
                "kind": kind,
                "config": config,
                "epoch": epoch,
                "validation_loss": val_loss,
                "selection_metric": "validation_balanced_coordinate_mse",
                "train_source_hashes": train.dataset.source_hashes,
                "validation_source_hashes": validation.dataset.source_hashes,
            }
        if epoch % settings["log_interval"] == 0 or epoch == settings["epochs"]:
            print(
                f"{kind} epoch {epoch:4}: train={train_loss:.6g}, validation={val_loss:.6g}, "
                f"val active={val_active:.6g} mm, near={val_near:.6g} mm",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    torch.save(selected, output / "checkpoint.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "kind": kind,
            "config": config,
            "epoch": settings["epochs"],
        },
        output / "last_checkpoint.pt",
    )
    # The selected candidate must exactly equal argmin(validation_loss), including epoch zero.
    if selected["epoch"] != min(history, key=lambda row: row["validation_loss"])["epoch"]:
        raise RuntimeError("Checkpoint does not match the validation-only selection rule")
    return {
        "selected_epoch": selected["epoch"],
        "validation_loss": best_validation,
        "training_wall_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
    }
