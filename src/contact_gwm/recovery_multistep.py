"""Differentiable no-contact recovery with the unchanged B state-map network.

This narrow diagnostic uses a certified detached recovery window. It does not
differentiate through MuJoCo or approximate its derivative. Native queries
are checked separately on the resulting predictions to validate no contact.
"""

from __future__ import annotations

import torch
from torch import Tensor


def current_features(positions, velocities, rest, pinned, normalization, dtype):
    """Current x/v only, with the same features/order as the original adapter."""
    x = torch.where(pinned[..., None], rest, positions)
    v = torch.where(pinned[..., None], 0.0, velocities)
    rest_feature = rest.expand_as(x) / normalization["rest_position"]
    pin_feature = pinned.expand(x.shape[:-1])[..., None].to(x.dtype)
    return (
        torch.cat(
            (
                rest_feature,
                (x - rest) / normalization["displacement"],
                v / normalization["velocity"],
                pin_feature,
            ),
            dim=-1,
        ).to(dtype),
        x,
        v,
    )


def no_contact_step(model, positions, velocities, rest, pinned, normalization):
    """A differentiable current-state step; never accepts target/future arrays."""
    dtype = next(model.parameters()).dtype
    features, x, v = current_features(positions, velocities, rest, pinned, normalization, dtype)
    context = features.new_zeros((*features.shape[:-1], 36))
    gates = features.new_zeros(features.shape[:-1])
    increment = model(context, gates, features, v.to(dtype))
    next_x = torch.where(pinned[..., None], rest, x + increment.delta_position.to(x.dtype))
    next_v = torch.where(pinned[..., None], 0.0, v + increment.delta_velocity.to(v.dtype))
    if not torch.isfinite(next_x).all() or not torch.isfinite(next_v).all():
        raise FloatingPointError("Non-finite differentiable no-contact state")
    return next_x, next_v


def autoregressive_recovery(
    model, initial_positions, initial_velocities, rest, pinned, normalization, steps
):
    """No detach/numpy conversion between steps; B recomputes graph features."""
    if steps < 1:
        raise ValueError("Positive rollout length required")
    x, v = initial_positions, initial_velocities
    positions, velocities = [], []
    for _ in range(steps):
        x, v = no_contact_step(model, x, v, rest, pinned, normalization)
        positions.append(x)
        velocities.append(v)
    return torch.stack(positions, dim=-3), torch.stack(velocities, dim=-3)


def teacher_forced_recovery(
    model, current_positions, current_velocities, rest, pinned, normalization
):
    """Independent one-step predictions on the exact same target occurrences.

    Input leading dims [window, step] may be flattened for the unchanged MLP.
    This arm intentionally uses each true current state during TRAINING; the
    free-rollout evaluation of both arms never performs that refill.
    """
    shape = current_positions.shape
    if len(shape) != 4:
        raise ValueError("Teacher-forced states must be [window, step, node, xyz]")
    x, v = no_contact_step(
        model,
        current_positions.reshape(-1, *shape[-2:]),
        current_velocities.reshape(-1, *shape[-2:]),
        rest,
        pinned,
        normalization,
    )
    return x.reshape(shape), v.reshape(shape)


def overlapping_indices(state_count: int, horizon: int) -> tuple[Tensor, Tensor]:
    """Identical current/target occurrence indices for both training arms."""
    if horizon < 1 or state_count <= horizon:
        raise ValueError("Window must contain at least horizon+1 states")
    current = torch.arange(state_count - horizon)[:, None] + torch.arange(horizon)[None]
    return current, current + 1


def joint_endpoint_loss(
    predicted_x, predicted_v, target_x, target_v, pinned, neighborhood, training
):
    """Original coordinate-MSE units and region weights; equal endpoint weights."""
    if predicted_x.shape != target_x.shape or predicted_v.shape != target_v.shape:
        raise ValueError("Predictions and independent endpoint labels must match")
    mask = neighborhood.expand(target_x.shape[:-1])
    active = ~pinned
    components = []
    for error in (
        (predicted_x - target_x) / training["position_loss_unit_m"],
        (predicted_v - target_v) / training["velocity_loss_unit_m_s"],
    ):
        active_mse = error[..., active, :].square().mean()
        near_mse = (error.square() * mask[..., None]).sum() / (3 * mask.sum()).clamp_min(1)
        components.append(
            (1 - training["neighborhood_weight"]) * active_mse
            + training["neighborhood_weight"] * near_mse
        )
    total = (1 - training["velocity_weight"]) * components[0] + training[
        "velocity_weight"
    ] * components[1]
    return total, components
