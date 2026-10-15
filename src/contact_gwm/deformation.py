"""Contact-gated local Gaussian deformation network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .structures import ContactForces, ContactGeometry, GaussianSet


class ContactGatedDeformation(nn.Module):
    """Predict local residuals while exposing the analytic contact gate."""

    def __init__(
        self,
        *,
        feature_dim: int,
        action_dim: int,
        material_dim: int = 4,
        hidden_dim: int = 128,
        contact_sigma: float = 0.25,
        max_translation: float = 0.02,
        max_log_scale: float = 0.03,
    ) -> None:
        super().__init__()
        if contact_sigma <= 0:
            raise ValueError("contact_sigma must be positive")
        self.action_dim = action_dim
        self.material_dim = material_dim
        self.contact_sigma = contact_sigma
        self.max_translation = max_translation
        self.max_log_scale = max_log_scale

        # feature + gate + force + normal + contact offset + action + material
        input_dim = feature_dim + 1 + 3 + 3 + 3 + action_dim + material_dim
        output_layer = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            output_layer,
        )

    def encode_context(
        self,
        gaussians: GaussianSet,
        contacts: ContactGeometry,
        contact_forces: ContactForces,
        action: Tensor,
        material_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build parameter-free current-frame context, reusable across training epochs."""
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        if material_features.shape != (self.material_dim,):
            raise ValueError(f"material_features must have shape ({self.material_dim},)")
        if contact_forces.forces.shape[0] != contacts.num_contacts:
            raise ValueError("contact force count must match contact geometry count")

        count = gaussians.num_gaussians
        active_contacts = contacts.signed_gaps <= 0.0
        if contacts.num_contacts == 0 or not torch.any(active_contacts):
            return (
                gaussians.centers.new_zeros((count, self.network[0].in_features)),
                gaussians.centers.new_zeros(count),
            )

        offsets = contacts.points[None, :, :] - gaussians.centers[:, None, :]
        squared_distances = torch.sum(offsets**2, dim=-1)
        weights = torch.exp(-squared_distances / (2.0 * self.contact_sigma**2))
        weights = weights * active_contacts.to(weights.dtype)[None, :]
        gates = weights.max(dim=1).values
        normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        force_context = normalized @ contact_forces.forces
        normal_context = normalized @ contacts.normals
        offset_context = torch.sum(normalized[..., None] * offsets, dim=1)
        force_context = torch.sign(force_context) * torch.log1p(force_context.abs())

        action_context = action[None, :].expand(count, -1)
        stable_material = torch.sign(material_features) * torch.log1p(material_features.abs())
        material_context = stable_material[None, :].expand(count, -1)
        network_input = torch.cat(
            (
                gaussians.features,
                gates[:, None],
                force_context,
                normal_context,
                offset_context,
                action_context,
                material_context,
            ),
            dim=-1,
        )
        return network_input, gates

    def decode_context(self, network_input: Tensor) -> tuple[Tensor, Tensor]:
        """Predict ungated residuals, including a leading batch dimension if provided."""
        prediction = self.network(network_input)
        delta_centers = torch.tanh(prediction[..., :3]) * self.max_translation
        delta_log_scales = torch.tanh(prediction[..., 3:]) * self.max_log_scale
        return delta_centers, delta_log_scales

    def forward(
        self,
        gaussians: GaussianSet,
        contacts: ContactGeometry,
        contact_forces: ContactForces,
        action: Tensor,
        material_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        network_input, gates = self.encode_context(
            gaussians, contacts, contact_forces, action, material_features
        )
        if not torch.any(gates):
            zeros3 = gaussians.centers.new_zeros((gaussians.num_gaussians, 3))
            return zeros3, zeros3.clone(), gates
        delta_centers, delta_log_scales = self.decode_context(network_input)
        return delta_centers, delta_log_scales, gates
