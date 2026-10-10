"""Explicit contact response and near-rigid object integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .math3d import integrate_quaternion, quaternion_to_matrix
from .structures import ContactForces, ContactGeometry, RigidBodyState


@dataclass(frozen=True)
class ContactMaterial:
    """Penalty-contact parameters used by the first physics baseline."""

    normal_stiffness: float = 800.0
    normal_damping: float = 12.0
    tangent_damping: float = 4.0
    friction_coefficient: float = 0.45

    def __post_init__(self) -> None:
        values = (
            self.normal_stiffness,
            self.normal_damping,
            self.tangent_damping,
            self.friction_coefficient,
        )
        if any(value < 0 for value in values):
            raise ValueError("contact material parameters must be non-negative")


def compute_contact_forces(
    contacts: ContactGeometry,
    center_of_mass: Tensor,
    material: ContactMaterial,
    *,
    eps: float = 1e-8,
) -> ContactForces:
    """Convert contact geometry to force and torque acting on the object.

    The relative velocity convention is ``robot - object``. Therefore a
    positive normal velocity means that the robot is closing on the object.
    Tangential friction pulls the object along the robot's relative sliding
    direction and is limited by the Coulomb bound.
    """

    if contacts.num_contacts == 0:
        return ContactForces.empty(center_of_mass.device, center_of_mass.dtype)

    normal_speeds = torch.sum(
        contacts.relative_velocities * contacts.normals,
        dim=-1,
    )
    elastic_magnitudes = material.normal_stiffness * contacts.penetrations
    damping_magnitudes = material.normal_damping * normal_speeds.clamp_min(0.0)
    active = (contacts.signed_gaps <= 0.0).to(contacts.points.dtype)
    normal_magnitudes = active * (elastic_magnitudes + damping_magnitudes)
    normal_forces = normal_magnitudes[:, None] * contacts.normals

    tangential_velocities = (
        contacts.relative_velocities
        - normal_speeds[:, None] * contacts.normals
    )
    tangential_speeds = tangential_velocities.norm(dim=-1)
    tangential_directions = (
        tangential_velocities / tangential_speeds[:, None].clamp_min(eps)
    )
    viscous_friction = material.tangent_damping * tangential_speeds
    coulomb_limit = material.friction_coefficient * normal_magnitudes
    friction_magnitudes = torch.minimum(viscous_friction, coulomb_limit)
    friction_forces = friction_magnitudes[:, None] * tangential_directions

    forces = normal_forces + friction_forces
    lever_arms = contacts.points - center_of_mass
    torques = torch.cross(lever_arms, forces, dim=-1)
    return ContactForces(
        forces=forces,
        torques=torques,
        normal_forces=normal_forces,
        friction_forces=friction_forces,
    )


def integrate_rigid_body(
    state: RigidBodyState,
    contact_forces: ContactForces,
    dt: float,
    *,
    gravity: Tensor | None = None,
) -> RigidBodyState:
    """Advance the global object pose using a semi-implicit Euler step."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    if gravity is None:
        gravity = state.position.new_tensor((0.0, 0.0, -9.81))
    if gravity.shape != (3,):
        raise ValueError("gravity must have shape (3,)")

    zero = state.position.new_zeros(3)
    total_force = (
        contact_forces.forces.sum(dim=0)
        if contact_forces.forces.numel()
        else zero
    )
    total_torque = (
        contact_forces.torques.sum(dim=0)
        if contact_forces.torques.numel()
        else zero
    )

    linear_acceleration = total_force / state.mass + gravity
    linear_velocity = state.linear_velocity + dt * linear_acceleration
    position = state.position + dt * linear_velocity

    rotation = quaternion_to_matrix(state.orientation)
    inertia_body = torch.diag(state.inertia_diag)
    inertia_world = rotation @ inertia_body @ rotation.transpose(-1, -2)
    angular_momentum = inertia_world @ state.angular_velocity
    gyroscopic_torque = torch.cross(
        state.angular_velocity,
        angular_momentum,
        dim=-1,
    )
    angular_acceleration = torch.linalg.solve(
        inertia_world,
        total_torque - gyroscopic_torque,
    )
    angular_velocity = state.angular_velocity + dt * angular_acceleration
    orientation = integrate_quaternion(state.orientation, angular_velocity, dt)

    return RigidBodyState(
        position=position,
        orientation=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        mass=state.mass,
        inertia_diag=state.inertia_diag,
    )

