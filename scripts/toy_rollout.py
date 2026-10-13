"""Run one synthetic contact step before connecting ManiSkill or a renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from contact_gwm import (
    ContactGatedDeformation,
    ContactMaterial,
    ExplicitContactGaussianWorldModel,
    GaussianSet,
    RigidBodyState,
)


def make_gaussians(
    centers: torch.Tensor,
    scales: torch.Tensor,
    velocities: torch.Tensor,
    feature_dim: int,
) -> GaussianSet:
    count = centers.shape[0]
    rotations = centers.new_zeros((count, 4))
    rotations[:, 0] = 1.0
    return GaussianSet(
        centers=centers,
        log_scales=scales.log(),
        rotations=rotations,
        opacity_logits=centers.new_zeros((count, 1)),
        features=centers.new_zeros((count, feature_dim)),
        velocities=velocities,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/toy.yaml"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    deformation_config = config["deformation"]

    xy = torch.tensor(
        [
            [-0.18, -0.18],
            [0.00, -0.18],
            [0.18, -0.18],
            [-0.18, 0.00],
            [0.00, 0.00],
            [0.18, 0.00],
            [-0.18, 0.18],
            [0.00, 0.18],
            [0.18, 0.18],
        ],
        device=device,
    )
    object_centers = torch.cat((xy, torch.zeros((xy.shape[0], 1), device=device)), dim=1)
    object_local = make_gaussians(
        object_centers,
        scales=torch.full_like(object_centers, 0.11),
        velocities=torch.zeros_like(object_centers),
        feature_dim=deformation_config["feature_dim"],
    )
    robot_centers = torch.tensor([[0.0, 0.0, 0.19]], device=device)
    robot_world = make_gaussians(
        robot_centers,
        scales=torch.tensor([[0.13, 0.13, 0.13]], device=device),
        velocities=torch.tensor([[0.0, 0.0, -0.20]], device=device),
        feature_dim=deformation_config["feature_dim"],
    )
    rigid_state = RigidBodyState(
        position=torch.zeros(3, device=device),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        linear_velocity=torch.zeros(3, device=device),
        angular_velocity=torch.zeros(3, device=device),
        mass=0.25,
        inertia_diag=torch.full((3,), 0.01, device=device),
    )

    deformation = ContactGatedDeformation(**deformation_config).to(device)
    material = ContactMaterial(**config["material"])
    model = ExplicitContactGaussianWorldModel(
        deformation,
        material,
        surface_level=config["surface_level"],
        contact_margin=config["contact_margin"],
        max_contacts=config["max_contacts"],
        gravity=(0.0, 0.0, 0.0),
    ).to(device)
    output = model(
        object_local,
        robot_world,
        rigid_state,
        action=torch.tensor([0.0, 0.0, -0.20], device=device),
        material_features=torch.tensor(
            [
                material.normal_stiffness,
                material.normal_damping,
                material.tangent_damping,
                material.friction_coefficient,
            ],
            device=device,
        ),
        dt=config["time_step"],
    )

    forces = output.contact_forces_world.forces
    net_force = forces.sum(dim=0) if forces.numel() else torch.zeros(3, device=device)
    print(f"contacts: {output.contacts_world.num_contacts}")
    print(f"net force on object: {net_force.detach().cpu().tolist()}")
    print(f"next rigid position: {output.rigid_state.position.detach().cpu().tolist()}")
    print(f"contact gates: {output.contact_gates.detach().cpu().tolist()}")
    local_center_delta = output.object_local.centers - object_local.centers
    local_log_scale_delta = output.object_local.log_scales - object_local.log_scales
    print(f"max local center change: {local_center_delta.detach().abs().max().item()}")
    print(f"max local log-scale change: {local_log_scale_delta.detach().abs().max().item()}")


if __name__ == "__main__":
    main()

