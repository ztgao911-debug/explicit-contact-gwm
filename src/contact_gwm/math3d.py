"""Small differentiable 3D math utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_quaternion(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product for quaternions stored in ``wxyz`` order."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_to_matrix(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert ``wxyz`` quaternions to rotation matrices.

    Input shape is ``[..., 4]`` and output shape is ``[..., 3, 3]``.
    """

    q = normalize_quaternion(quaternion, eps)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def rotate_vectors(quaternion: Tensor, vectors: Tensor) -> Tensor:
    rotation = quaternion_to_matrix(quaternion)
    return vectors @ rotation.transpose(-1, -2)


def inverse_rotate_vectors(quaternion: Tensor, vectors: Tensor) -> Tensor:
    rotation = quaternion_to_matrix(quaternion)
    return vectors @ rotation


def transform_points(position: Tensor, orientation: Tensor, points: Tensor) -> Tensor:
    return rotate_vectors(orientation, points) + position


def inverse_transform_points(position: Tensor, orientation: Tensor, points: Tensor) -> Tensor:
    return inverse_rotate_vectors(orientation, points - position)


def integrate_quaternion(orientation: Tensor, angular_velocity: Tensor, dt: float) -> Tensor:
    """First-order integration for a world-frame angular velocity."""

    zero = angular_velocity.new_zeros((*angular_velocity.shape[:-1], 1))
    omega = torch.cat((zero, angular_velocity), dim=-1)
    derivative = 0.5 * quaternion_multiply(omega, orientation)
    return normalize_quaternion(orientation + dt * derivative)

