"""Fixed rest-mesh neighbor information for the internal recovery branch only."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .joint_state_dynamics import JointStateDynamics


def neighbor_mean_matrix(node_count: int, edges: Tensor) -> Tensor:
    """Undirected unique edges, uniform row means, no self-loop or learned weight."""
    edges = torch.as_tensor(edges, dtype=torch.long)
    if node_count < 1 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Expected positive node count and edges [E,2]")
    if edges.numel() and (edges.min() < 0 or edges.max() >= node_count):
        raise ValueError("Mesh edge index outside stable node range")
    weights = torch.zeros((node_count, node_count), dtype=torch.float32)
    if edges.numel():
        first, second = edges.T
        keep = first != second
        weights[first[keep], second[keep]] = 1
        weights[second[keep], first[keep]] = 1
    return weights / weights.sum(-1, keepdim=True).clamp_min(1)


def neighbor_relative_features(features: Tensor, weights: Tensor, pinned_mask: Tensor) -> Tensor:
    """mean_j(u_j-u_i)/.01, mean_j(v_j-v_i)/.05, using CURRENT state only.

    The original ten-channel features already contain these normalizations.
    Fixed nodes remain in the graph, contributing zero displacement/velocity.
    An isolated node has zero neighbor features, not a negative self state.
    """
    count = weights.shape[0]
    if (
        features.shape[-2:] != (count, 10)
        or weights.shape != (count, count)
        or pinned_mask.shape != (count,)
    ):
        raise ValueError("Neighbor features require stable node order and original ten channels")
    dynamic = torch.where(pinned_mask[..., None], 0.0, features[..., 3:9])
    mean = torch.matmul(weights.to(dynamic), dynamic)
    return torch.where((weights.sum(-1) > 0)[..., None], mean - dynamic, 0.0)


def _network(input_dim: int, hidden_dim: int) -> nn.Sequential:
    # Preserve the original output-head-first initialization convention.
    head = nn.Linear(hidden_dim, 6)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        head,
    )


class NeighborInternalNetwork(nn.Sequential):
    """Append six current graph features to the original internal input (28)."""

    def __init__(self, hidden_dim: int, weights: Tensor, pinned_mask: Tensor) -> None:
        super().__init__(*_network(34, hidden_dim))
        self.register_buffer("neighbor_weights", weights.detach().clone())
        self.register_buffer("neighbor_pinned", pinned_mask.detach().bool().clone())

    def forward(self, current_input: Tensor) -> Tensor:
        extra = neighbor_relative_features(
            current_input[..., :10], self.neighbor_weights, self.neighbor_pinned
        )
        # This also appends zeros to the parent's static equilibrium reference.
        return super().forward(torch.cat((current_input, extra), dim=-1))


class NeighborJointStateDynamics(JointStateDynamics):
    """B: keep the entire contact branch and append features to the internal MLP.

    Copy A's common internal initial weights, with zero new input columns. This
    is initialization from scratch, not a fitted checkpoint. Shape-changing C
    cannot share all hidden tensors. Parameter counts are recorded separately.
    """

    def __init__(self, rest_positions, pinned_mask, edges, **kwargs) -> None:
        super().__init__(rest_positions, pinned_mask, **kwargs)
        original = self.internal_network
        hidden = original[0].out_features
        augmented = NeighborInternalNetwork(
            hidden, neighbor_mean_matrix(len(rest_positions), edges), pinned_mask
        )
        with torch.no_grad():
            augmented[0].weight[:, :28].copy_(original[0].weight)
            augmented[0].weight[:, 28:].zero_()
            augmented[0].bias.copy_(original[0].bias)
            for layer in (2, 4):
                augmented[layer].load_state_dict(original[layer].state_dict())
        self.internal_network = augmented


def closest_internal_width(target_parameters: int, contact_parameters: int = 24326) -> int:
    """Match TOTAL parameters; keep the contact branch fixed and internal input=28."""
    if target_parameters <= contact_parameters:
        raise ValueError("Target total must include a positive internal network")

    def count(width):
        return width * width + 36 * width + 6 + contact_parameters

    upper = 1
    while count(upper) < target_parameters:
        upper += 1
    return min(
        {max(1, upper - 1), upper}, key=lambda width: (abs(count(width) - target_parameters), width)
    )


def make_neighbor_model(group: str, mesh: dict, config: dict) -> JointStateDynamics:
    rest = torch.as_tensor(mesh["rest_positions_world"], dtype=torch.float32)
    pinned = torch.as_tensor(mesh["pinned_mask"], dtype=torch.bool)
    kwargs = {**config["model"], "dt": config["dt"], "autonomous": True}
    if group == "B":
        return NeighborJointStateDynamics(rest, pinned, torch.as_tensor(mesh["edges"]), **kwargs)
    model = JointStateDynamics(rest, pinned, **kwargs)
    if group == "A":
        return model
    if group == "C":
        contact_parameters = sum(p.numel() for p in model.contact_network.parameters())
        hidden = config["model"]["hidden_dim"]
        target = sum(p.numel() for p in model.parameters()) + 6 * hidden
        width = closest_internal_width(target, contact_parameters)
        model.internal_network = _network(28, width)
        return model
    raise ValueError(f"Unknown group: {group}")


def model_specification(model: JointStateDynamics) -> dict:
    internal = model.internal_network
    return {
        "internal_input_dim": internal[0].in_features,
        "internal_hidden_dim": internal[0].out_features,
        "contact_input_dim": model.contact_network[0].in_features,
        "contact_hidden_dim": model.contact_network[0].out_features,
        "internal_parameters": sum(p.numel() for p in internal.parameters()),
        "contact_parameters": sum(p.numel() for p in model.contact_network.parameters()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "neighbor_features": isinstance(model, NeighborJointStateDynamics),
    }
