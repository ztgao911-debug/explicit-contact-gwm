"""Label diagnostics and a fixed-subset optimization probe, without test-set access."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import Tensor

from .comparison import PreparedPairs, balanced_loss, make_model, metric, predict
from .local_training import constant_velocity_increment


def velocity_residual(pack: PreparedPairs, dt: float = 0.01) -> Tensor:
    """Supervision only: the unknown next displacement minus current velocity * dt."""
    return pack.target - constant_velocity_increment(pack.velocity, pack.dataset.pinned, dt)


def fixed_subset_indices(pack: PreparedPairs, quantiles: list[float]) -> Tensor:
    """Choose temporal quantiles within each trajectory/phase, without inspecting labels."""
    if not quantiles or any(not 0 <= value <= 1 for value in quantiles):
        raise ValueError("Subset quantiles must be nonempty and between zero and one")
    selected = []
    for name in sorted(set(pack.names)):
        for phase in (1, 2):
            candidates = sorted(
                (
                    i
                    for i in range(len(pack.names))
                    if pack.names[i] == name and pack.phases[i] == phase
                ),
                key=lambda i: pack.frames[i],
            )
            if not candidates:
                raise ValueError(f"Missing selected phase {phase} in {name}")
            indices = [candidates[round(q * (len(candidates) - 1))] for q in quantiles]
            if len(indices) != len(set(indices)):
                raise ValueError("Quantiles select duplicate frames; use fewer quantiles")
            selected.extend(indices)
    return torch.tensor(selected, dtype=torch.long)


def gated_output_floor(residual: Tensor, gates: Tensor, pinned: Tensor, limit: float) -> Tensor:
    """Optimistic error floor from per-axis tanh bounds, ignoring shared MLP capacity."""
    if limit <= 0:
        raise ValueError("Output limit must be positive")
    bound = gates[..., None] * limit
    best_correction = torch.where(pinned[..., None], 0.0, residual.clamp(-bound, bound))
    return residual - best_correction


def subset_metrics(prediction: Tensor, pack: PreparedPairs, indices: Tensor) -> dict:
    target = pack.target[indices]
    active = pack.active[None].expand(len(indices), -1)
    return {
        "active_nodes": metric(prediction, target, active),
        "contact_neighborhood": metric(prediction, target, pack.neighborhood[indices]),
    }


def fit_fixed_subset(train: PreparedPairs, indices: Tensor, config: dict, output: Path) -> dict:
    """Fit only the fixed center pairs. No validation pack or score enters selection."""
    torch.manual_seed(config["seed"])
    model = make_model("velocity_residual", config)
    settings = config["training"]
    epochs = config["overfit"]["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=settings["learning_rate"] * 0.01
    )
    output.mkdir(parents=True, exist_ok=False)
    history, selected = [], None
    best_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            prediction = predict("velocity_residual", model, train, indices)
            loss = balanced_loss(
                prediction,
                train.target[indices],
                train.active,
                train.neighborhood[indices],
                settings,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite fixed-subset loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = predict("velocity_residual", model, train, indices)
            value = float(
                balanced_loss(
                    prediction,
                    train.target[indices],
                    train.active,
                    train.neighborhood[indices],
                    settings,
                )
            )
            metrics = subset_metrics(prediction, train, indices)
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError("Non-finite fixed-subset evaluation")
        history.append(
            {
                "epoch": epoch,
                "subset_loss": value,
                "active_rmse_mm": metrics["active_nodes"]["vector_rmse_mm"],
                "neighborhood_rmse_mm": metrics["contact_neighborhood"]["vector_rmse_mm"],
            }
        )
        if value < best_loss:
            best_loss = value
            selected = {
                "model_state": copy.deepcopy(model.state_dict()),
                "config": config,
                "epoch": epoch,
                "subset_loss": value,
                "selection_metric": "same_fixed_subset_balanced_coordinate_mse",
                "subset_indices": indices.tolist(),
                "source_hashes": train.dataset.source_hashes,
            }
        if epoch % config["overfit"]["log_interval"] == 0 or epoch == epochs:
            print(
                f"Fixed subset epoch {epoch:4}: loss={value:.7g}; "
                f"active={history[-1]['active_rmse_mm'] * 1000:.5g} um, "
                f"near={history[-1]['neighborhood_rmse_mm'] * 1000:.5g} um",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    if selected["epoch"] != min(history, key=lambda row: row["subset_loss"])["epoch"]:
        raise RuntimeError("Fixed-subset checkpoint selection mismatch")
    torch.save(selected, output / "checkpoint.pt")
    torch.save(
        {"model_state": model.state_dict(), "epoch": epochs, "config": config},
        output / "last_checkpoint.pt",
    )
    # Independently construct/reload the saved model, not the in-memory final weights.
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    loaded = make_model("velocity_residual", config)
    loaded.load_state_dict(checkpoint["model_state"])
    model.load_state_dict(selected["model_state"])
    with torch.no_grad():
        if not torch.equal(
            predict("velocity_residual", model, train, indices),
            predict("velocity_residual", loaded, train, indices),
        ):
            raise RuntimeError("Checkpoint reload changed predictions")
    return {
        "selected_epoch": selected["epoch"],
        "subset_loss": best_loss,
        "training_wall_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_reload_exact": True,
        "history": history,
    }
