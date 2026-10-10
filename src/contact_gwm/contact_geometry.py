"""Analytic contact extraction from object and robot 3D Gaussians.

Each Gaussian is converted to an oriented ellipsoid at a selected iso-surface.
The first baseline evaluates opposing ellipsoid support points along the line
between their centers. It is analytic and differentiable for active pairs, but
it is still an approximation to the exact ellipsoid-to-ellipsoid distance.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .math3d import quaternion_to_matrix
from .structures import ContactGeometry, GaussianSet


def _ellipsoid_support_points(
    cloud: GaussianSet,
    directions: Tensor,
    rotations: Tensor,
    surface_level: float,
    eps: float,
) -> Tensor:
    """Return surface points furthest along world directions.

    ``directions`` has shape ``[N, M, 3]`` and the first dimension must match
    the number of Gaussians in ``cloud``.
    """

    axes = surface_level * cloud.scales
    local_directions = torch.einsum("nmd,ndk->nmk", directions, rotations)
    denominators = torch.sqrt(
        torch.sum((axes[:, None, :] * local_directions) ** 2, dim=-1)
    ).clamp_min(eps)
    local_support = (
        axes[:, None, :] ** 2 * local_directions / denominators[..., None]
    )
    world_offsets = torch.einsum("nmk,ndk->nmd", local_support, rotations)
    return cloud.centers[:, None, :] + world_offsets


def compute_gaussian_contacts(
    object_gaussians: GaussianSet,
    robot_gaussians: GaussianSet,
    *,
    surface_level: float = 2.0,
    contact_margin: float = 0.0,
    max_contacts: int | None = 256,
    eps: float = 1e-8,
) -> ContactGeometry:
    """Compute explicit contact variables from two Gaussian sets.

    Args:
        object_gaussians: Object Gaussians in world coordinates.
        robot_gaussians: Robot Gaussians in world coordinates.
        surface_level: Multiplier from Gaussian scale to ellipsoid semi-axis.
            A Gaussian has infinite support, so this value defines the physical
            iso-surface and should later be calibrated with simulator geometry.
        contact_margin: Keep pairs with a signed gap below this threshold.
        max_contacts: Maximum number of deepest pairs to retain.
        eps: Numerical stability constant.

    Returns:
        Contact points, normals, signed gaps, penetrations, relative velocities,
        and the corresponding object and robot Gaussian indices.
    """

    if surface_level <= 0:
        raise ValueError("surface_level must be positive")
    if max_contacts is not None and max_contacts <= 0:
        raise ValueError("max_contacts must be positive or None")
    if object_gaussians.device != robot_gaussians.device:
        raise ValueError("object and robot Gaussians must be on the same device")
    if object_gaussians.dtype != robot_gaussians.dtype:
        raise ValueError("object and robot Gaussians must use the same dtype")
    if object_gaussians.num_gaussians == 0 or robot_gaussians.num_gaussians == 0:
        return ContactGeometry.empty(object_gaussians.device, object_gaussians.dtype)

    # Pairwise center direction: robot -> object.
    center_offsets = (
        object_gaussians.centers[:, None, :]
        - robot_gaussians.centers[None, :, :]
    )
    center_distances = center_offsets.norm(dim=-1)
    normals = center_offsets / center_distances[..., None].clamp_min(eps)

    # Coincident centers do not define a unique normal. A deterministic fallback
    # prevents NaNs while keeping the exceptional case easy to detect.
    fallback = normals.new_tensor((0.0, 0.0, 1.0))
    normals = torch.where((center_distances > eps)[..., None], normals, fallback)

    object_rotations = quaternion_to_matrix(object_gaussians.rotations, eps)
    robot_rotations = quaternion_to_matrix(robot_gaussians.rotations, eps)

    # The object surface faces the robot (-normal), while the robot surface
    # faces the object (+normal).
    object_surface = _ellipsoid_support_points(
        object_gaussians,
        -normals,
        object_rotations,
        surface_level,
        eps,
    )
    robot_surface = _ellipsoid_support_points(
        robot_gaussians,
        normals.transpose(0, 1),
        robot_rotations,
        surface_level,
        eps,
    ).transpose(0, 1)

    # Positive means separation; negative means overlap.
    signed_gaps = torch.sum((object_surface - robot_surface) * normals, dim=-1)
    pair_indices = torch.nonzero(signed_gaps <= contact_margin, as_tuple=False)
    if pair_indices.numel() == 0:
        return ContactGeometry.empty(object_gaussians.device, object_gaussians.dtype)

    object_indices = pair_indices[:, 0]
    robot_indices = pair_indices[:, 1]
    selected_gaps = signed_gaps[object_indices, robot_indices]

    if max_contacts is not None and selected_gaps.numel() > max_contacts:
        selection = torch.topk(selected_gaps, k=max_contacts, largest=False).indices
        object_indices = object_indices[selection]
        robot_indices = robot_indices[selection]
        selected_gaps = selected_gaps[selection]

    object_points = object_surface[object_indices, robot_indices]
    robot_points = robot_surface[object_indices, robot_indices]

    return ContactGeometry(
        object_indices=object_indices,
        robot_indices=robot_indices,
        points=0.5 * (object_points + robot_points),
        normals=normals[object_indices, robot_indices],
        signed_gaps=selected_gaps,
        penetrations=(-selected_gaps).clamp_min(0.0),
        relative_velocities=(
            robot_gaussians.velocities[robot_indices]
            - object_gaussians.velocities[object_indices]
        ),
    )

