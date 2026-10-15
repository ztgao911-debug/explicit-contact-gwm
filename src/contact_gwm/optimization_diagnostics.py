"""Single-frame optimization controls; these probes are not new dynamics models."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from .comparison import balanced_loss
from .local_training import ConstantVelocityResidualDeformation, constant_velocity_increment


@dataclass(frozen=True)
class ProbeBatch:
    context: Tensor
    gates: Tensor
    velocity: Tensor
    target: Tensor
    pinned: Tensor
    neighborhood: Tensor
    fit_mask: Tensor

    def loss(self, prediction: Tensor, training: dict) -> Tensor:
        return balanced_loss(
            prediction,
            self.target,
            self.fit_mask,
            self.neighborhood & self.fit_mask[None],
            training,
        )


class IndependentNodeResidual(nn.Module):
    """Positive control: a separate trainable correction for each stable node index.

    It memorizes one frame; it cannot predict another frame or a new object.
    """

    def __init__(self, nodes: int, max_translation: float) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(nodes, 3))
        self.max_translation = max_translation

    def predict_encoded(self, context, gates, pinned, velocity):
        residual = gates[..., None] * self.raw.tanh() * self.max_translation
        return torch.where(pinned[..., None], 0.0, velocity * 0.01 + residual)


def make_probe_model(kind: str, batch: ProbeBatch, config: dict) -> nn.Module:
    if kind == "mlp":
        return ConstantVelocityResidualDeformation(**config["model"], dt=config["dt"])
    if kind == "node_table":
        return IndependentNodeResidual(len(batch.pinned), config["model"]["max_translation"])
    raise ValueError(f"Unknown diagnostic model: {kind}")


def probe_predict(model: nn.Module, batch: ProbeBatch) -> Tensor:
    return model.predict_encoded(batch.context, batch.gates, batch.pinned, batch.velocity)


def probe_metrics(prediction: Tensor, batch: ProbeBatch) -> dict:
    squared_norm = (prediction - batch.target).square().sum(-1)
    if not torch.isfinite(prediction).all() or torch.count_nonzero(prediction[:, batch.pinned]):
        raise RuntimeError("Non-finite prediction or fixed-node movement")
    return {
        "fit_rmse_um": float(squared_norm[:, batch.fit_mask].mean().sqrt()) * 1e6,
        "fit_neighborhood_rmse_um": float(
            squared_norm[batch.neighborhood & batch.fit_mask[None]].mean().sqrt()
        )
        * 1e6,
        "full_active_rmse_um": float(squared_norm[:, ~batch.pinned].mean().sqrt()) * 1e6,
    }


def gradient_snapshot(model, batch: ProbeBatch, training: dict, step: int) -> list[dict]:
    """Read gradients without modifying optimizer buffers or model parameters."""
    context = batch.context.detach().clone().requires_grad_(True)
    local = replace(batch, context=context)
    parameters = dict(model.named_parameters())
    prediction = probe_predict(model, local)
    loss = local.loss(prediction, training)
    # The node-table control deliberately ignores the context; report unused input as zero.
    gradients = torch.autograd.grad(loss, [*parameters.values(), context], allow_unused=True)
    rows = []
    for name, value, grad in zip(
        [*parameters, "current_context"], [*parameters.values(), context], gradients, strict=True
    ):
        gradient = torch.zeros_like(value) if grad is None else grad
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
        rows.append(
            {
                "step": step,
                "tensor": name,
                "gradient_l2": float(gradient.norm()),
                "gradient_abs_max": float(gradient.abs().max()),
                "zero_fraction": float((gradient == 0).float().mean()),
                "parameter_l2": float(value.detach().norm()),
                "unused": grad is None,
            }
        )
    return rows


def finite_difference_audit(batch: ProbeBatch, config: dict) -> dict:
    """Check head-at-zero and hidden-after-one-update derivatives in float64."""
    torch.manual_seed(config["seed"])
    model = make_probe_model("mlp", batch, config).double()
    double_batch = replace(
        batch,
        context=batch.context.double(),
        gates=batch.gates.double(),
        velocity=batch.velocity.double(),
        target=batch.target.double(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    rows, checks = [], []
    probe = config["probe"]
    for step, name in ((0, "deformation.network.4.bias"), (1, "deformation.network.0.weight")):
        model.zero_grad(set_to_none=True)
        loss = double_batch.loss(probe_predict(model, double_batch), config["training"])
        loss.backward()
        rows.extend(gradient_snapshot(model, double_batch, config["training"], step))
        parameter = dict(model.named_parameters())[name]
        flat_index = int(parameter.grad.abs().argmax())
        derivative = float(parameter.grad.flatten()[flat_index])
        original = float(parameter.detach().flatten()[flat_index])
        epsilon = probe["finite_difference_epsilon"]
        with torch.no_grad():
            try:
                parameter.flatten()[flat_index] = original + epsilon
                plus = float(
                    double_batch.loss(probe_predict(model, double_batch), config["training"])
                )
                parameter.flatten()[flat_index] = original - epsilon
                minus = float(
                    double_batch.loss(probe_predict(model, double_batch), config["training"])
                )
            finally:
                parameter.flatten()[flat_index] = original
        numerical = (plus - minus) / (2 * epsilon)
        difference = abs(derivative - numerical)
        threshold = probe["finite_difference_absolute_tolerance"] + (
            probe["finite_difference_relative_tolerance"] * max(abs(derivative), abs(numerical))
        )
        checks.append(
            {
                "step": step,
                "parameter": name,
                "flat_index": flat_index,
                "autograd": derivative,
                "central_difference": numerical,
                "absolute_error": difference,
                "relative_error": difference / max(abs(derivative), abs(numerical), 1e-30),
                "nonzero_derivative": abs(derivative)
                > probe["finite_difference_absolute_tolerance"],
                "passed": difference <= threshold,
            }
        )
        if step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
    hidden = ("deformation.network.0.", "deformation.network.2.")
    return {
        "precision": "float64 for numerical audit only; training remains float32",
        "finite_differences": checks,
        "gradient_rows": rows,
        "initial_hidden_gradients_zero": all(
            r["gradient_l2"] == 0 for r in rows if r["step"] == 0 and r["tensor"].startswith(hidden)
        ),
        "initial_translation_head_gradient_nonzero": any(
            r["gradient_l2"] > 0
            for r in rows
            if r["step"] == 0 and r["tensor"].startswith("deformation.network.4.")
        ),
        "hidden_gradients_nonzero_after_one_update": all(
            r["gradient_l2"] > 0 for r in rows if r["step"] == 1 and r["tensor"].startswith(hidden)
        ),
        "all_finite_difference_checks_pass": all(
            r["passed"] and r["nonzero_derivative"] for r in checks
        ),
    }


def fit_probe(
    kind: str, optimizer_kind: str, batch: ProbeBatch, config: dict, output: Path
) -> dict:
    """Fit only the explicitly masked one frame. No held-out scores enter this function."""
    torch.manual_seed(config["seed"])
    model = make_probe_model(kind, batch, config)
    probe, training = config["probe"], config["training"]
    if not batch.fit_mask.any() or not (batch.neighborhood & batch.fit_mask[None]).any():
        raise ValueError("Fitting masks must contain active and contact-neighborhood nodes")
    if torch.any(batch.fit_mask & batch.pinned):
        raise ValueError("Do not fit pinned nodes")
    if optimizer_kind == "adam":
        steps = probe["adam_steps"]
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=training["learning_rate"] * 0.01
        )
    elif optimizer_kind == "lbfgs":
        steps = probe["lbfgs_steps"]
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=probe["lbfgs_learning_rate"],
            max_iter=probe["lbfgs_max_iter"],
            max_eval=probe["lbfgs_max_eval"],
            history_size=probe["lbfgs_history_size"],
            tolerance_grad=probe["lbfgs_tolerance_grad"],
            tolerance_change=probe["lbfgs_tolerance_change"],
            line_search_fn="strong_wolfe",
        )
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_kind}")
    output.mkdir(parents=True, exist_ok=False)
    history, gradients = [], []
    selected, best_loss = None, float("inf")
    backward_calls = 0
    started = time.perf_counter()

    def closure():
        nonlocal backward_calls
        optimizer.zero_grad(set_to_none=True)
        loss = batch.loss(probe_predict(model, batch), training)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite optimization probe loss")
        loss.backward()
        backward_calls += 1
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Non-finite optimization probe gradient")
        return loss

    for step in range(steps + 1):
        if step:
            if optimizer_kind == "adam":
                closure()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
                optimizer.step()
                scheduler.step()
            else:
                # L-BFGS line search needs the true gradient of its objective; no clipping.
                optimizer.step(closure)
        with torch.no_grad():
            prediction = probe_predict(model, batch)
            value = float(batch.loss(prediction, training))
            metrics = probe_metrics(prediction, batch)
        history.append({"step": step, "loss": value, "backward_calls": backward_calls, **metrics})
        if value < best_loss:
            best_loss = value
            selected = {
                "model_state": copy.deepcopy(model.state_dict()),
                "kind": kind,
                "optimizer_kind": optimizer_kind,
                "config": config,
                "step": step,
                "loss": value,
                "fit_node_indices": batch.fit_mask.nonzero().flatten().tolist(),
                "selection": "same_fitted_frame_balanced_mse_including_step_zero",
            }
        if step in probe["gradient_steps"] or step == steps:
            gradients.extend(gradient_snapshot(model, batch, training, step))
        if step == 0 or step == steps or step % probe["log_interval"] == 0:
            print(
                f"{output.name} step {step:4}: fit={metrics['fit_rmse_um']:.6g} um, "
                f"near={metrics['fit_neighborhood_rmse_um']:.6g} um, backward={backward_calls}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    if selected["step"] != min(history, key=lambda row: row["loss"])["step"]:
        raise RuntimeError("Selected probe weights do not match the lowest fitted loss")
    torch.save(selected, output / "checkpoint.pt")
    torch.save(
        {"model_state": model.state_dict(), "config": config, "step": steps},
        output / "last_checkpoint.pt",
    )
    model.load_state_dict(selected["model_state"])
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
    loaded = make_probe_model(kind, batch, config)
    loaded.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        prediction = probe_predict(model, batch)
        if not torch.equal(prediction, probe_predict(loaded, batch)):
            raise RuntimeError("Reloading the probe model changed predictions")
        baseline = constant_velocity_increment(batch.velocity, batch.pinned)
        baseline_metrics = probe_metrics(baseline, batch)
        metrics = probe_metrics(prediction, batch)
    return {
        "kind": kind,
        "optimizer": optimizer_kind,
        "selected_step": selected["step"],
        "selected_loss": selected["loss"],
        "backward_calls": backward_calls,
        "training_wall_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "fitted_node_count": int(batch.fit_mask.sum()),
        "baseline": baseline_metrics,
        "selected_metrics": metrics,
        "fit_rmse_reduction": 1 - metrics["fit_rmse_um"] / baseline_metrics["fit_rmse_um"],
        "checkpoint_reload_exact": True,
        "history": history,
        "gradients": gradients,
        "prediction": prediction,
    }
