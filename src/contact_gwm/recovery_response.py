"""Finite native-response supervision, separate from current-state prediction."""

from __future__ import annotations

import torch

from .recovery_multistep import joint_endpoint_loss, no_contact_step


def paired_current_step(model, current_x, current_v, rest, pinned, normalization):
    """Signs are ordered [-,+]. Only perturbed CURRENT states are accepted.

    Leading dimensions may include current frames and direction/kind cases.
    Endpoint labels, native response targets and initial perturbation norms
    deliberately do not appear in this prediction API.
    """
    if current_x.shape != current_v.shape or current_x.shape[-3] != 2:
        raise ValueError("Paired current states must have shape [...,2,node,3]")
    return no_contact_step(model, current_x, current_v, rest, pinned, normalization)


def central_response(paired):
    if paired.shape[-3] != 2 or paired.shape[-1] != 3:
        raise ValueError("Expected ordered [-,+] states [...,2,node,3]")
    return (paired[..., 1, :, :] - paired[..., 0, :, :]) / 2


def response_matching_loss(
    predicted_x,
    predicted_v,
    native_response_x,
    native_response_v,
    initial_norm,
    pinned,
    neighborhood,
    training,
):
    """Match SIGNED finite responses; do not penalize gain above one itself.

    Normalize each pair by its initial active x/v norm, then apply the original
    position/velocity units and region weights. No MuJoCo differentiation or
    higher-order parameter gradients are needed for paired finite differences.
    """
    rx, rv = central_response(predicted_x), central_response(predicted_v)
    if rx.shape != native_response_x.shape or rv.shape != native_response_v.shape:
        raise ValueError("Independent native response labels must match paired predictions")
    if not torch.isfinite(initial_norm).all() or torch.any(initial_norm <= 0):
        raise ValueError("Finite positive initial perturbation norms required")
    scale = initial_norm[..., None, None]
    return joint_endpoint_loss(
        rx / scale,
        rv / scale,
        native_response_x / scale,
        native_response_v / scale,
        pinned,
        neighborhood,
        training,
    )
