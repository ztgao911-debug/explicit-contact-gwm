from __future__ import annotations

import torch

from contact_gwm import (
    ContactForces,
    ContactGatedDeformation,
    ContactGeometry,
    GaussianSet,
)


def make_object() -> GaussianSet:
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = torch.zeros((2, 4))
    rotations[:, 0] = 1.0
    return GaussianSet(
        centers=centers,
        log_scales=torch.full((2, 3), 0.1).log(),
        rotations=rotations,
        opacity_logits=torch.zeros((2, 1)),
        features=torch.zeros((2, 4)),
        velocities=torch.zeros((2, 3)),
    )


def test_contact_gate_is_local() -> None:
    model = ContactGatedDeformation(
        feature_dim=4,
        action_dim=3,
        contact_sigma=0.1,
    )
    contacts = ContactGeometry(
        object_indices=torch.tensor([0]),
        robot_indices=torch.tensor([0]),
        points=torch.tensor([[0.0, 0.0, 0.0]]),
        normals=torch.tensor([[0.0, 0.0, -1.0]]),
        signed_gaps=torch.tensor([-0.01]),
        penetrations=torch.tensor([0.01]),
        relative_velocities=torch.tensor([[0.0, 0.0, -0.1]]),
    )
    forces = ContactForces(
        forces=torch.tensor([[0.0, 0.0, -1.0]]),
        torques=torch.zeros((1, 3)),
        normal_forces=torch.tensor([[0.0, 0.0, -1.0]]),
        friction_forces=torch.zeros((1, 3)),
    )

    delta_centers, delta_scales, gates = model(
        make_object(),
        contacts,
        forces,
        action=torch.zeros(3),
        material_features=torch.ones(4),
    )

    assert gates[0] > gates[1]
    # The residual head is zero-initialized for a stable no-deformation start.
    torch.testing.assert_close(delta_centers, torch.zeros((2, 3)))
    torch.testing.assert_close(delta_scales, torch.zeros((2, 3)))

