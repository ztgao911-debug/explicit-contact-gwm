from __future__ import annotations

import torch

from contact_gwm import (
    ContactGatedDeformation,
    ContactMaterial,
    ExplicitContactGaussianWorldModel,
    GaussianSet,
    RigidBodyState,
)


def make_gaussian(center: tuple[float, float, float], radius: float) -> GaussianSet:
    centers = torch.tensor([center])
    return GaussianSet(
        centers=centers,
        log_scales=torch.full((1, 3), radius).log(),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logits=torch.zeros((1, 1)),
        features=torch.zeros((1, 4)),
        velocities=torch.zeros((1, 3)),
    )


def test_one_step_combines_contact_physics_and_local_state() -> None:
    object_local = make_gaussian((0.0, 0.0, 0.0), 0.1)
    robot_world = make_gaussian((0.0, 0.0, 0.18), 0.1)
    robot_world = GaussianSet(
        centers=robot_world.centers,
        log_scales=robot_world.log_scales,
        rotations=robot_world.rotations,
        opacity_logits=robot_world.opacity_logits,
        features=robot_world.features,
        velocities=torch.tensor([[0.0, 0.0, -0.1]]),
    )
    rigid_state = RigidBodyState(
        position=torch.zeros(3),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.zeros(3),
        angular_velocity=torch.zeros(3),
        mass=1.0,
        inertia_diag=torch.ones(3),
    )
    deformation = ContactGatedDeformation(feature_dim=4, action_dim=3)
    model = ExplicitContactGaussianWorldModel(
        deformation,
        ContactMaterial(normal_stiffness=100.0, normal_damping=0.0),
        surface_level=1.0,
        gravity=(0.0, 0.0, 0.0),
    )

    output = model(
        object_local,
        robot_world,
        rigid_state,
        action=torch.tensor([0.0, 0.0, -0.1]),
        material_features=torch.tensor([100.0, 0.0, 4.0, 0.45]),
        dt=0.01,
    )

    assert output.contacts_world.num_contacts == 1
    assert output.rigid_state.position[2] < 0.0
    torch.testing.assert_close(output.object_local.centers, object_local.centers)

