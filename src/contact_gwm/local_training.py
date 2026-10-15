"""Pinned local Gaussian predictor; no free rigid-body integration is performed."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from .data.gaussian_pairs import GaussianFrameInputs
from .deformation import ContactGatedDeformation
from .structures import GaussianSet


class PinnedGaussianDeformation(nn.Module):
    def __init__(
        self, hidden_dim: int = 128, contact_sigma: float = 0.02, max_translation: float = 0.001
    ) -> None:
        super().__init__()
        self.deformation = ContactGatedDeformation(
            feature_dim=10,
            action_dim=12,
            material_dim=4,
            hidden_dim=hidden_dim,
            contact_sigma=contact_sigma,
            max_translation=max_translation,
            max_log_scale=0.0,
        )

    def encode(self, inputs: GaussianFrameInputs) -> tuple[Tensor, Tensor]:
        return self.deformation.encode_context(
            inputs.gaussians,
            inputs.contacts,
            inputs.contact_forces,
            inputs.action,
            inputs.material_features,
        )

    def predict_encoded(self, context: Tensor, gates: Tensor, pinned_mask: Tensor) -> Tensor:
        delta, _ = self.deformation.decode_context(context)
        return torch.where(pinned_mask[..., None], 0.0, gates[..., None] * delta)

    def forward(self, inputs: GaussianFrameInputs) -> Tensor:
        context, gates = self.encode(inputs)
        return self.predict_encoded(context, gates, inputs.pinned_mask)

    def next_gaussians(self, inputs: GaussianFrameInputs) -> GaussianSet:
        delta = self(inputs)
        centers = torch.where(
            inputs.pinned_mask[:, None], inputs.rest_positions, inputs.gaussians.centers + delta
        )
        # Rotations, scales and opacity are copied unchanged; the network learns centers only.
        return replace(inputs.gaussians, centers=centers, velocities=delta / 0.01)
