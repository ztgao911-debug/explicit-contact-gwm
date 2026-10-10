from __future__ import annotations

import torch

from contact_gwm import GaussianSet, compute_gaussian_contacts


def make_single_gaussian(center: tuple[float, float, float], radius: float) -> GaussianSet:
    centers = torch.tensor([center], dtype=torch.float32)
    return GaussianSet(
        centers=centers,
        log_scales=torch.full((1, 3), radius).log(),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logits=torch.zeros((1, 1)),
        features=torch.zeros((1, 4)),
        velocities=torch.zeros((1, 3)),
    )


def test_overlapping_gaussians_produce_contact() -> None:
    object_gaussian = make_single_gaussian((0.0, 0.0, 0.0), 1.0)
    robot_gaussian = make_single_gaussian((0.0, 0.0, 1.5), 1.0)

    contacts = compute_gaussian_contacts(
        object_gaussian,
        robot_gaussian,
        surface_level=1.0,
    )

    assert contacts.num_contacts == 1
    torch.testing.assert_close(contacts.signed_gaps, torch.tensor([-0.5]))
    torch.testing.assert_close(contacts.penetrations, torch.tensor([0.5]))
    torch.testing.assert_close(contacts.normals, torch.tensor([[0.0, 0.0, -1.0]]))


def test_separated_gaussians_have_no_contact() -> None:
    object_gaussian = make_single_gaussian((0.0, 0.0, 0.0), 1.0)
    robot_gaussian = make_single_gaussian((0.0, 0.0, 3.0), 1.0)

    contacts = compute_gaussian_contacts(
        object_gaussian,
        robot_gaussian,
        surface_level=1.0,
    )

    assert contacts.num_contacts == 0


def test_positive_margin_keeps_near_contact_pair() -> None:
    object_gaussian = make_single_gaussian((0.0, 0.0, 0.0), 1.0)
    robot_gaussian = make_single_gaussian((0.0, 0.0, 2.05), 1.0)

    contacts = compute_gaussian_contacts(
        object_gaussian,
        robot_gaussian,
        surface_level=1.0,
        contact_margin=0.1,
    )

    assert contacts.num_contacts == 1
    torch.testing.assert_close(contacts.signed_gaps, torch.tensor([0.05]))
    torch.testing.assert_close(contacts.penetrations, torch.tensor([0.0]))

