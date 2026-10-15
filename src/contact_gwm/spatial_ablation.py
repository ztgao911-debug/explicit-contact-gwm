"""Fixed rest-coordinate features and an integer-width capacity control."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .deformation import ContactGatedDeformation
from .local_training import ConstantVelocityResidualDeformation

FREQUENCIES = (0.5, 1.0, 2.0)
BASE_CONTEXT_DIM = 36
SPATIAL_DIM = 18


def fixed_rest_encoding(rest_positions: Tensor) -> Tensor:
    """Order: frequency, sin/cos, xyz. Eighteen fixed channels, r=rest/0.03."""
    if rest_positions.ndim != 2 or rest_positions.shape[-1] != 3:
        raise ValueError("Rest positions must be [nodes, 3]")
    if not torch.isfinite(rest_positions).all():
        raise ValueError("Rest positions must be finite")
    r = rest_positions / 0.03
    return torch.cat(
        [
            part
            for frequency in FREQUENCIES
            for part in ((math.pi * frequency * r).sin(), (math.pi * frequency * r).cos())
        ],
        dim=-1,
    )


class SpatialEncodedResidual(nn.Module):
    """Append static rest features to the unchanged 36-channel current-frame context.

    The context is produced by the existing adapter, which has no future inputs.
    Only the decoder is used here; no contact geometry or gates are recomputed.
    """

    def __init__(
        self,
        rest_positions: Tensor,
        hidden_dim=128,
        contact_sigma=0.02,
        max_translation=0.001,
        dt=0.01,
    ) -> None:
        super().__init__()
        if abs(dt - 0.01) > 1e-12:
            raise ValueError("The ablation requires dt=0.01 s")
        self.dt = dt
        self.register_buffer("rest_encoding", fixed_rest_encoding(rest_positions).detach().clone())
        # feature_dim sets the decoder's total input width to 36+18. The original
        # 36 channels retain their order; the new 18 are appended at the END.
        self.deformation = ContactGatedDeformation(
            feature_dim=10 + SPATIAL_DIM,
            action_dim=12,
            material_dim=4,
            hidden_dim=hidden_dim,
            contact_sigma=contact_sigma,
            max_translation=max_translation,
            max_log_scale=0.0,
        )

    def augment_context(self, context: Tensor) -> Tensor:
        if context.shape[-2:] != (len(self.rest_encoding), BASE_CONTEXT_DIM):
            raise ValueError("Context must retain stable node order and 36 original channels")
        features = self.rest_encoding.expand(*context.shape[:-1], SPATIAL_DIM)
        return torch.cat((context, features), dim=-1)

    def predict_encoded(self, context, gates, pinned_mask, current_velocity):
        residual, _ = self.deformation.decode_context(self.augment_context(context))
        return torch.where(
            pinned_mask[..., None],
            0.0,
            current_velocity * self.dt + gates[..., None] * residual,
        )


def original_parameter_count(hidden_dim: int) -> int:
    """Two equal hidden widths, 36 inputs and the existing six-output head."""
    return hidden_dim * hidden_dim + (BASE_CONTEXT_DIM + 8) * hidden_dim + 6


def closest_original_width(target_parameters: int) -> int:
    """Globally closest positive integer width; use smaller width for an exact tie."""
    if target_parameters < 1:
        raise ValueError("Target parameter count must be positive")
    upper = 1
    while original_parameter_count(upper) < target_parameters:
        upper += 1
    return min(
        {max(1, upper - 1), upper},
        key=lambda width: (abs(original_parameter_count(width) - target_parameters), width),
    )


def make_ablation_model(group: str, rest_positions: Tensor, base_config: dict) -> nn.Module:
    if group == "A":
        return ConstantVelocityResidualDeformation(**base_config["model"], dt=base_config["dt"])
    if group == "B":
        return SpatialEncodedResidual(rest_positions, **base_config["model"], dt=base_config["dt"])
    if group == "C":
        width = base_config["model"]["hidden_dim"]
        b_count = width * width + (BASE_CONTEXT_DIM + SPATIAL_DIM + 8) * width + 6
        settings = {**base_config["model"], "hidden_dim": closest_original_width(b_count)}
        return ConstantVelocityResidualDeformation(**settings, dt=base_config["dt"])
    raise ValueError(f"Unknown ablation group: {group}")


def model_specifications(rest_positions: Tensor, base_config: dict) -> dict:
    specs = {}
    for group in ("A", "B", "C"):
        model = make_ablation_model(group, rest_positions, base_config)
        specs[group] = {
            "input_dim": model.deformation.network[0].in_features,
            "hidden_dim": model.deformation.network[0].out_features,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "spatial_encoding": group == "B",
        }
    specs["C"]["parameter_difference_from_B"] = (
        specs["C"]["parameter_count"] - specs["B"]["parameter_count"]
    )
    specs["C"]["absolute_relative_parameter_difference_from_B"] = (
        abs(specs["C"]["parameter_difference_from_B"]) / specs["B"]["parameter_count"]
    )
    return specs
