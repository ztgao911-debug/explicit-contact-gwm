"""Loss-only scale controls with measured gradients and Adam parameter updates."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import torch

from .local_training import constant_velocity_increment
from .optimization_diagnostics import (
    ProbeBatch,
    make_probe_model,
    probe_metrics,
    probe_predict,
)


def grouped_probe_metrics(batch, prediction, trajectories, phases, method):
    """Score fitted current-frame groups, using the same node masks in every method."""
    if len(trajectories) != len(batch.target) or len(phases) != len(batch.target):
        raise ValueError("Group metadata must have one item per fitted frame pair")
    rows = []
    for trajectory in ("all", *sorted(set(trajectories))):
        for phase_id, phase in ((None, "all"), (1, "press"), (2, "hold")):
            selected = [
                i
                for i, (name, p) in enumerate(zip(trajectories, phases, strict=True))
                if (trajectory == "all" or trajectory == name)
                and (phase_id is None or phase_id == p)
            ]
            if not selected:
                continue
            squared = (prediction[selected] - batch.target[selected]).square().sum(-1)
            active = batch.fit_mask[None].expand(len(selected), -1)
            for region, mask in (
                ("active_nodes", active),
                ("contact_neighborhood", batch.neighborhood[selected] & active),
            ):
                if not mask.any():
                    raise ValueError("An evaluation group has no scored nodes")
                rows.append(
                    {
                        "method": method,
                        "trajectory": trajectory,
                        "phase": phase,
                        "region": region,
                        "pairs": len(selected),
                        "node_frames": int(mask.sum()),
                        "vector_rmse_um": float(squared[mask].mean().sqrt()) * 1e6,
                    }
                )
    return rows


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def gradient_statistics(model) -> dict:
    result = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"Missing or non-finite gradient in {name}")
        result[name] = {
            "raw_gradient_l2": float(grad.double().norm()),
            "raw_gradient_abs_max": float(grad.abs().max()),
            "raw_gradient_zero_fraction": float((grad == 0).double().mean()),
        }
    return result


def measured_update_rows(model, before, raw_gradients, optimizer, step, multiplier, lr):
    """Inspect Adam state after stepping, without changing any tensors or buffers."""
    rows = []
    beta2 = optimizer.param_groups[0]["betas"][1]
    epsilon = optimizer.param_groups[0]["eps"]
    for name, parameter in model.named_parameters():
        update = parameter.detach().double() - before[name].double()
        norm_before = float(before[name].double().norm())
        state = optimizer.state[parameter]
        corrected_rms = (state["exp_avg_sq"].double() / (1 - beta2 ** float(state["step"]))).sqrt()
        positive = corrected_rms[corrected_rms > 0]
        rows.append(
            {
                "step": step,
                "loss_multiplier": multiplier,
                "tensor": name,
                "learning_rate": lr,
                **raw_gradients[name],
                "clipped_gradient_l2": float(parameter.grad.double().norm()),
                "parameter_l2_before": norm_before,
                "actual_update_l2": float(update.norm()),
                "actual_update_abs_max": float(update.abs().max()),
                "update_relative_to_parameter_l2": float(update.norm()) / norm_before
                if norm_before
                else None,
                "adam_epsilon": epsilon,
                "positive_adam_rms_count": len(positive),
                "positive_adam_rms_below_epsilon_fraction": float(
                    (positive < epsilon).double().mean()
                )
                if len(positive)
                else None,
                "adam_epsilon_share_median": float((epsilon / (positive + epsilon)).median())
                if len(positive)
                else None,
            }
        )
    return rows


def fit_loss_scale(
    batch: ProbeBatch,
    config: dict,
    multiplier: float,
    update_steps: list[int],
    output: Path,
    *,
    model_factory=None,
) -> dict:
    """Only multiply the original balanced objective; never change inputs or targets."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Loss multiplier must be finite and positive")
    if model_factory is None:

        def model_factory():
            return make_probe_model("mlp", batch, config)

    torch.manual_seed(config["seed"])
    model = model_factory()
    initial_hash = parameter_fingerprint(model)
    training, probe = config["training"], config["probe"]
    steps = probe["adam_steps"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=training["learning_rate"] * 0.01
    )
    output.mkdir(parents=True, exist_ok=False)
    history, updates, selected = [], [], None
    best_loss, clipped_steps = float("inf"), 0
    started = time.perf_counter()
    for step in range(steps + 1):
        raw_norm = clipped_norm = actual_update_norm = None
        clip_coefficient, lr = None, optimizer.param_groups[0]["lr"]
        if step:
            optimizer.zero_grad(set_to_none=True)
            canonical = batch.loss(probe_predict(model, batch), training)
            weighted = canonical * multiplier
            if not torch.isfinite(weighted):
                raise RuntimeError(f"Non-finite scaled loss at step {step}")
            weighted.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing or non-finite gradient")
            before = {
                name: parameter.detach().clone() for name, parameter in model.named_parameters()
            }
            raw_stats = gradient_statistics(model) if step in update_steps else None
            # This is the same clipping call, threshold and Adam step as the reference.
            raw_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            )
            clip_coefficient = min(1.0, training["gradient_clip"] / (raw_norm + 1e-6))
            clipped_steps += int(clip_coefficient < 1.0)
            clipped_norm = math.sqrt(
                sum(float(p.grad.double().square().sum()) for p in model.parameters())
            )
            optimizer.step()
            scheduler.step()
            actual_update_norm = math.sqrt(
                sum(
                    float((parameter.detach().double() - before[name].double()).square().sum())
                    for name, parameter in model.named_parameters()
                )
            )
            if raw_stats is not None:
                updates.extend(
                    measured_update_rows(model, before, raw_stats, optimizer, step, multiplier, lr)
                )
        with torch.no_grad():
            prediction = probe_predict(model, batch)
            canonical_loss = float(batch.loss(prediction, training))
            metrics = probe_metrics(prediction, batch)
        history.append(
            {
                "step": step,
                "canonical_loss": canonical_loss,
                "scaled_loss": canonical_loss * multiplier,
                **metrics,
                "learning_rate_for_update": lr,
                "raw_gradient_l2_for_update": raw_norm,
                "clipped_gradient_l2_for_update": clipped_norm,
                "clip_coefficient": clip_coefficient,
                "actual_parameter_update_l2": actual_update_norm,
            }
        )
        if canonical_loss < best_loss:
            best_loss = canonical_loss
            selected = {
                "model_state": copy.deepcopy(model.state_dict()),
                "config": config,
                "loss_multiplier": multiplier,
                "step": step,
                "canonical_loss": canonical_loss,
                "selection": "same_fitted_frame_canonical_balanced_mse_including_step_zero"
                if len(batch.target) == 1
                else "same_fixed_pairs_canonical_balanced_mse_including_step_zero",
                "fitted_pair_count": len(batch.target),
                "fit_node_indices": batch.fit_mask.nonzero().flatten().tolist(),
            }
        if step == 0 or step == steps or step % probe["log_interval"] == 0:
            print(
                f"loss x{multiplier:g} step {step:4}: active={metrics['fit_rmse_um']:.6g} um, "
                f"near={metrics['fit_neighborhood_rmse_um']:.6g} um, clipped steps={clipped_steps}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    if selected["step"] != min(history, key=lambda row: row["canonical_loss"])["step"]:
        raise RuntimeError("Loss-scale checkpoint does not match minimum canonical fitting loss")
    torch.save(selected, output / "checkpoint.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "loss_multiplier": multiplier,
            "step": steps,
        },
        output / "last_checkpoint.pt",
    )
    model.load_state_dict(selected["model_state"])
    saved = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
    loaded = model_factory()
    loaded.load_state_dict(saved["model_state"])
    with torch.no_grad():
        prediction = probe_predict(model, batch)
        if not torch.equal(prediction, probe_predict(loaded, batch)):
            raise RuntimeError("Reloading scaled-loss weights changed predictions")
        baseline = probe_metrics(constant_velocity_increment(batch.velocity, batch.pinned), batch)
        metrics = probe_metrics(prediction, batch)
    return {
        "loss_multiplier": multiplier,
        "fitted_pair_count": len(batch.target),
        "equivalent_loss_unit_m": training["loss_unit_m"] / math.sqrt(multiplier),
        "initial_parameter_sha256": initial_hash,
        "selected_step": selected["step"],
        "canonical_selected_loss": selected["canonical_loss"],
        "selected_metrics": metrics,
        "baseline_metrics": baseline,
        "clipped_steps": clipped_steps,
        "fit_rmse_reduction": 1 - metrics["fit_rmse_um"] / baseline["fit_rmse_um"],
        "training_wall_seconds": elapsed,
        "checkpoint_reload_exact": True,
        "adam_defaults": {
            "betas": list(optimizer.param_groups[0]["betas"]),
            "eps": optimizer.param_groups[0]["eps"],
        },
        "history": history,
        "parameter_updates": updates,
        "prediction": prediction,
    }
