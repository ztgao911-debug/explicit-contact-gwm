"""Composition of analytic contact, explicit physics, and learned deformation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contact_force import ContactMaterial, compute_contact_forces, integrate_rigid_body
from .contact_geometry import compute_gaussian_contacts
from .deformation import ContactGatedDeformation
from .math3d import (
    inverse_rotate_vectors,
    inverse_transform_points,
    normalize_quaternion,
    quaternion_multiply,
    rotate_vectors,
    transform_points,
)
from .structures import ContactForces, ContactGeometry, GaussianSet, RigidBodyState


@dataclass
class WorldModelOutput:
    object_local: GaussianSet
    object_world: GaussianSet
    rigid_state: RigidBodyState
    contacts_world: ContactGeometry
    contact_forces_world: ContactForces
    contact_gates: Tensor


def object_gaussians_to_world(local: GaussianSet, rigid: RigidBodyState) -> GaussianSet:
    centers = transform_points(rigid.position, rigid.orientation, local.centers)
    lever_arms = centers - rigid.position
    rigid_velocities = rigid.linear_velocity + torch.cross(
        rigid.angular_velocity.expand_as(lever_arms), lever_arms, dim=-1
    )
    velocities = rigid_velocities + rotate_vectors(rigid.orientation, local.velocities)
    global_orientation = rigid.orientation[None, :].expand_as(local.rotations)
    rotations = normalize_quaternion(
        quaternion_multiply(global_orientation, local.rotations)
    )
    return GaussianSet(
        centers=centers,
        log_scales=local.log_scales,
        rotations=rotations,
        opacity_logits=local.opacity_logits,
        features=local.features,
        velocities=velocities,
    )


def _contacts_to_local(contacts: ContactGeometry, rigid: RigidBodyState) -> ContactGeometry:
    return ContactGeometry(
        object_indices=contacts.object_indices,
        robot_indices=contacts.robot_indices,
        points=inverse_transform_points(rigid.position, rigid.orientation, contacts.points),
        normals=inverse_rotate_vectors(rigid.orientation, contacts.normals),
        signed_gaps=contacts.signed_gaps,
        penetrations=contacts.penetrations,
        relative_velocities=inverse_rotate_vectors(
            rigid.orientation, contacts.relative_velocities
        ),
    )


def _forces_to_local(forces: ContactForces, rigid: RigidBodyState) -> ContactForces:
    return ContactForces(
        forces=inverse_rotate_vectors(rigid.orientation, forces.forces),
        torques=inverse_rotate_vectors(rigid.orientation, forces.torques),
        normal_forces=inverse_rotate_vectors(rigid.orientation, forces.normal_forces),
        friction_forces=inverse_rotate_vectors(rigid.orientation, forces.friction_forces),
    )


class ExplicitContactGaussianWorldModel(nn.Module):
    """One-step transition model matching modules 3 through 6 of the framework."""

    def __init__(
        self,
        deformation: ContactGatedDeformation,
        material: ContactMaterial,
        *,
        surface_level: float = 2.0,
        contact_margin: float = 0.0,
        max_contacts: int | None = 256,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        super().__init__()
        self.deformation = deformation
        self.material = material
        self.surface_level = surface_level
        self.contact_margin = contact_margin
        self.max_contacts = max_contacts
        self.register_buffer("gravity", torch.tensor(gravity, dtype=torch.float32))

    def forward(
        self,
        object_local: GaussianSet,
        robot_world: GaussianSet,
        rigid_state: RigidBodyState,
        action: Tensor,
        material_features: Tensor,
        dt: float,
    ) -> WorldModelOutput:
        object_world = object_gaussians_to_world(object_local, rigid_state)
        contacts_world = compute_gaussian_contacts(
            object_world,
            robot_world,
            surface_level=self.surface_level,
            contact_margin=self.contact_margin,
            max_contacts=self.max_contacts,
        )
        forces_world = compute_contact_forces(
            contacts_world,
            center_of_mass=rigid_state.position,
            material=self.material,
        )

        next_rigid = integrate_rigid_body(
            rigid_state,
            forces_world,
            dt,
            gravity=self.gravity.to(object_local.device, object_local.dtype),
        )

        contacts_local = _contacts_to_local(contacts_world, rigid_state)
        forces_local = _forces_to_local(forces_world, rigid_state)
        delta_centers, delta_log_scales, gates = self.deformation(
            object_local,
            contacts_local,
            forces_local,
            action,
            material_features,
        )
        gated_delta_centers = gates[:, None] * delta_centers
        gated_delta_log_scales = gates[:, None] * delta_log_scales
        next_local = GaussianSet(
            centers=object_local.centers + gated_delta_centers,
            log_scales=object_local.log_scales + gated_delta_log_scales,
            rotations=object_local.rotations,
            opacity_logits=object_local.opacity_logits,
            features=object_local.features,
            velocities=gated_delta_centers / dt,
        )
        next_world = object_gaussians_to_world(next_local, next_rigid)
        return WorldModelOutput(
            object_local=next_local,
            object_world=next_world,
            rigid_state=next_rigid,
            contacts_world=contacts_world,
            contact_forces_world=forces_world,
            contact_gates=gates,
        )

