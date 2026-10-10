"""Typed state containers used by the world-model modules.

The first implementation intentionally operates on one scene at a time. This
keeps the geometry and physics code readable. A later data-loader can batch
scenes by padding contact pairs or by using a packed representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_shape(name: str, value: Tensor, suffix: tuple[int, ...]) -> None:
    if value.ndim != len(suffix) or tuple(value.shape) != suffix:
        raise ValueError(f"{name} must have shape {suffix}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class GaussianSet:
    """A set of Gaussian primitives in a common coordinate frame.

    Rotations are stored as quaternions in ``wxyz`` order. ``log_scales`` is
    the unconstrained optimization parameter used by 3D Gaussian Splatting;
    the positive ellipsoid axes are obtained through ``exp(log_scales)``.
    """

    centers: Tensor
    log_scales: Tensor
    rotations: Tensor
    opacity_logits: Tensor
    features: Tensor
    velocities: Tensor

    def __post_init__(self) -> None:
        count = self.centers.shape[0]
        expected = {
            "centers": (count, 3),
            "log_scales": (count, 3),
            "rotations": (count, 4),
            "opacity_logits": (count, 1),
            "velocities": (count, 3),
        }
        for name, shape in expected.items():
            _require_shape(name, getattr(self, name), shape)
        if self.features.ndim != 2 or self.features.shape[0] != count:
            raise ValueError(
                "features must have shape [num_gaussians, feature_dim], "
                f"got {tuple(self.features.shape)}"
            )
        for name in (
            "log_scales",
            "rotations",
            "opacity_logits",
            "features",
            "velocities",
        ):
            value = getattr(self, name)
            if value.device != self.centers.device:
                raise ValueError(f"{name} must be on {self.centers.device}")
            if value.dtype != self.centers.dtype:
                raise ValueError(f"{name} must use dtype {self.centers.dtype}")

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp()

    @property
    def num_gaussians(self) -> int:
        return self.centers.shape[0]

    @property
    def device(self) -> torch.device:
        return self.centers.device

    @property
    def dtype(self) -> torch.dtype:
        return self.centers.dtype

    def to(self, device: torch.device | str) -> GaussianSet:
        return GaussianSet(
            centers=self.centers.to(device),
            log_scales=self.log_scales.to(device),
            rotations=self.rotations.to(device),
            opacity_logits=self.opacity_logits.to(device),
            features=self.features.to(device),
            velocities=self.velocities.to(device),
        )


@dataclass(frozen=True)
class RigidBodyState:
    """Global rigid state of the object.

    ``orientation`` maps object-local vectors to world coordinates and uses
    ``wxyz`` quaternion order. ``inertia_diag`` is expressed in the body frame.
    """

    position: Tensor
    orientation: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    mass: float
    inertia_diag: Tensor

    def __post_init__(self) -> None:
        _require_shape("position", self.position, (3,))
        _require_shape("orientation", self.orientation, (4,))
        _require_shape("linear_velocity", self.linear_velocity, (3,))
        _require_shape("angular_velocity", self.angular_velocity, (3,))
        _require_shape("inertia_diag", self.inertia_diag, (3,))
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if torch.any(self.inertia_diag <= 0):
            raise ValueError("all inertia values must be positive")

    def to(self, device: torch.device | str) -> RigidBodyState:
        return RigidBodyState(
            position=self.position.to(device),
            orientation=self.orientation.to(device),
            linear_velocity=self.linear_velocity.to(device),
            angular_velocity=self.angular_velocity.to(device),
            mass=self.mass,
            inertia_diag=self.inertia_diag.to(device),
        )


@dataclass(frozen=True)
class ContactGeometry:
    """Explicit geometric contact variables for selected Gaussian pairs.

    Normals point from the robot primitive toward the object primitive.
    ``signed_gaps`` are positive for separation and negative for overlap.
    """

    object_indices: Tensor
    robot_indices: Tensor
    points: Tensor
    normals: Tensor
    signed_gaps: Tensor
    penetrations: Tensor
    relative_velocities: Tensor

    def __post_init__(self) -> None:
        count = self.points.shape[0]
        expected = {
            "object_indices": (count,),
            "robot_indices": (count,),
            "points": (count, 3),
            "normals": (count, 3),
            "signed_gaps": (count,),
            "penetrations": (count,),
            "relative_velocities": (count, 3),
        }
        for name, shape in expected.items():
            _require_shape(name, getattr(self, name), shape)

    @property
    def num_contacts(self) -> int:
        return self.points.shape[0]

    @classmethod
    def empty(cls, device: torch.device, dtype: torch.dtype) -> ContactGeometry:
        return cls(
            object_indices=torch.empty(0, device=device, dtype=torch.long),
            robot_indices=torch.empty(0, device=device, dtype=torch.long),
            points=torch.empty((0, 3), device=device, dtype=dtype),
            normals=torch.empty((0, 3), device=device, dtype=dtype),
            signed_gaps=torch.empty(0, device=device, dtype=dtype),
            penetrations=torch.empty(0, device=device, dtype=dtype),
            relative_velocities=torch.empty((0, 3), device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class ContactForces:
    """Forces acting on the object and their moments about its center of mass."""

    forces: Tensor
    torques: Tensor
    normal_forces: Tensor
    friction_forces: Tensor

    def __post_init__(self) -> None:
        count = self.forces.shape[0]
        for name in ("forces", "torques", "normal_forces", "friction_forces"):
            _require_shape(name, getattr(self, name), (count, 3))

    @classmethod
    def empty(cls, device: torch.device, dtype: torch.dtype) -> ContactForces:
        zeros = torch.empty((0, 3), device=device, dtype=dtype)
        return cls(
            forces=zeros,
            torques=zeros.clone(),
            normal_forces=zeros.clone(),
            friction_forces=zeros.clone(),
        )

