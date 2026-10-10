from __future__ import annotations

import torch

from contact_gwm import ContactGeometry, ContactMaterial, compute_contact_forces


def make_contact(relative_velocity: tuple[float, float, float]) -> ContactGeometry:
    return ContactGeometry(
        object_indices=torch.tensor([0]),
        robot_indices=torch.tensor([0]),
        points=torch.tensor([[0.0, 0.0, 0.0]]),
        normals=torch.tensor([[0.0, 0.0, -1.0]]),
        signed_gaps=torch.tensor([-0.01]),
        penetrations=torch.tensor([0.01]),
        relative_velocities=torch.tensor([relative_velocity]),
    )


def test_normal_force_follows_contact_normal() -> None:
    contacts = make_contact((0.0, 0.0, -0.2))
    material = ContactMaterial(normal_stiffness=100.0, normal_damping=10.0)

    response = compute_contact_forces(contacts, torch.zeros(3), material)

    torch.testing.assert_close(response.normal_forces, torch.tensor([[0.0, 0.0, -3.0]]))
    torch.testing.assert_close(response.friction_forces, torch.zeros((1, 3)))


def test_friction_respects_coulomb_limit() -> None:
    contacts = make_contact((10.0, 0.0, 0.0))
    material = ContactMaterial(
        normal_stiffness=100.0,
        normal_damping=0.0,
        tangent_damping=10.0,
        friction_coefficient=0.5,
    )

    response = compute_contact_forces(contacts, torch.zeros(3), material)

    expected_limit = 0.5 * response.normal_forces.norm(dim=-1)
    torch.testing.assert_close(response.friction_forces.norm(dim=-1), expected_limit)

