"""Topology-bound collision Gaussians, independent of dynamics and force models.

The immutable rest-surface binding is the only topology dependency. A query uses
CURRENT nodal x/v and CURRENT robot x/v; it never queries a simulator. Indices in
returned contacts identify PROXIES, not dynamics nodes. No mass, force, or state
update is introduced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contact_geometry import compute_gaussian_contacts
from .structures import ContactGeometry, GaussianSet


@dataclass(frozen=True)
class SphereSurfaceQuery:
    contacts: ContactGeometry
    closest_gap: Tensor
    closest_point: Tensor
    closest_normal: Tensor


class BoundSurfaceGaussians:
    """Uniform barycentric subdivision, deduplicated by topology, not proximity.

    A common subdivision count makes shared triangle edges conforming. The
    requested step bounds every subtriangle edge IN THE REST SHAPE; deformation
    can stretch it. Skin radius remains fixed and independent of sample spacing.
    """

    def __init__(
        self,
        rest_positions: np.ndarray,
        surface_triangles: np.ndarray,
        pinned_mask: np.ndarray,
        *,
        max_rest_edge_m: float | None,
        skin_radius_m: float = 0.001,
        surface_level: float = 0.2,
    ) -> None:
        rest = np.asarray(rest_positions, dtype=np.float64)
        triangles = np.asarray(surface_triangles, dtype=np.int64)
        pinned = np.asarray(pinned_mask, dtype=bool)
        if (
            rest.ndim != 2
            or rest.shape[1] != 3
            or pinned.shape != (len(rest),)
            or not np.isfinite(rest).all()
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not len(triangles)
            or triangles.min() < 0
            or triangles.max() >= len(rest)
        ):
            raise ValueError("Invalid rest surface or pinned mask")
        if (
            not np.isfinite([skin_radius_m, surface_level]).all()
            or min(skin_radius_m, surface_level) <= 0
        ):
            raise ValueError("Skin radius and surface level must be finite and positive")
        edges = rest[triangles[:, [1, 2, 0]]] - rest[triangles]
        longest = float(np.linalg.norm(edges, axis=-1).max())
        area = np.linalg.norm(
            np.cross(
                rest[triangles[:, 1]] - rest[triangles[:, 0]],
                rest[triangles[:, 2]] - rest[triangles[:, 0]],
            ),
            axis=-1,
        )
        if np.any(area <= 1e-20):
            raise ValueError("Degenerate rest triangle")
        if max_rest_edge_m is not None and (
            not np.isfinite(max_rest_edge_m) or max_rest_edge_m <= 0
        ):
            raise ValueError("Maximum edge step must be finite and positive")
        subdivisions = 1 if max_rest_edge_m is None else int(np.ceil(longest / max_rest_edge_m))
        parents, weights, seen = [], [], set()
        for triangle in triangles:
            for i in range(subdivisions + 1):
                for j in range(subdivisions + 1 - i):
                    counts = (i, j, subdivisions - i - j)
                    key = tuple(
                        sorted(
                            (int(node), weight)
                            for node, weight in zip(triangle, counts, strict=True)
                            if weight
                        )
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    parents.append(triangle)
                    weights.append(np.asarray(counts) / subdivisions)
        self.rest = rest.copy()
        self.triangles = triangles.copy()
        self.pinned = pinned.copy()
        self.parents = np.asarray(parents, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.subdivisions = subdivisions
        self.skin_radius_m = float(skin_radius_m)
        self.surface_level = float(surface_level)
        self.actual_max_rest_edge_m = longest / subdivisions

    def cloud(self, positions: Tensor, velocities: Tensor) -> GaussianSet:
        if positions.shape != self.rest.shape or velocities.shape != positions.shape:
            raise ValueError("Current nodal x/v must preserve the original node order and shape")
        if positions.dtype != velocities.dtype or positions.device != velocities.device:
            raise ValueError("Nodal x/v must share dtype and device")
        if not torch.isfinite(positions).all() or not torch.isfinite(velocities).all():
            raise FloatingPointError("Non-finite current node state")
        rest = torch.as_tensor(self.rest, dtype=positions.dtype, device=positions.device)
        pinned = torch.as_tensor(self.pinned, device=positions.device)
        # The boundary condition is imposed BEFORE interpolation. Caller arrays
        # are not modified, and all-pinned proxies stay at their bound rest point.
        positions = torch.where(pinned[:, None], rest, positions)
        velocities = torch.where(pinned[:, None], 0.0, velocities)
        parents = torch.as_tensor(self.parents, device=positions.device)
        weights = torch.as_tensor(self.weights, dtype=positions.dtype, device=positions.device)
        centers = (positions[parents] * weights[..., None]).sum(1)
        velocity = (velocities[parents] * weights[..., None]).sum(1)
        count = len(centers)
        rotations = positions.new_zeros((count, 4))
        rotations[:, 0] = 1
        return GaussianSet(
            centers,
            positions.new_full((count, 3), self.skin_radius_m / self.surface_level).log(),
            rotations,
            positions.new_zeros((count, 1)),
            positions.new_zeros((count, 0)),
            velocity,
        )

    def query(
        self,
        positions: Tensor,
        velocities: Tensor,
        robot_position: Tensor,
        robot_velocity: Tensor,
        sphere_radius_m: float,
    ) -> SphereSurfaceQuery:
        if not np.isfinite(sphere_radius_m) or sphere_radius_m <= 0:
            raise ValueError("Sphere radius must be finite and positive")
        if robot_position.shape != (3,) or robot_velocity.shape != (3,):
            raise ValueError("Current robot x/v must have shape [3]")
        if not torch.isfinite(robot_position).all() or not torch.isfinite(robot_velocity).all():
            raise FloatingPointError("Non-finite current robot state")
        cloud = self.cloud(positions, velocities)
        sphere = GaussianSet(
            robot_position[None],
            positions.new_full((1, 3), sphere_radius_m / self.surface_level).log(),
            positions.new_tensor([[1.0, 0.0, 0.0, 0.0]]),
            positions.new_zeros((1, 1)),
            positions.new_zeros((1, 0)),
            robot_velocity[None],
        )
        contacts = compute_gaussian_contacts(
            cloud, sphere, surface_level=self.surface_level, max_contacts=None
        )
        offsets = cloud.centers - robot_position
        distances = torch.linalg.vector_norm(offsets, dim=-1)
        index = distances.argmin()
        normal = offsets[index] / distances[index].clamp_min(1e-12)
        normal = torch.where(
            distances[index] > 1e-12, normal, positions.new_tensor([0.0, 0.0, 1.0])
        )
        gap = distances[index] - sphere_radius_m - self.skin_radius_m
        point = 0.5 * (
            cloud.centers[index]
            - self.skin_radius_m * normal
            + robot_position
            + sphere_radius_m * normal
        )
        return SphereSurfaceQuery(contacts, gap, point, normal)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            rest=self.rest,
            triangles=self.triangles,
            pinned=self.pinned,
            parents=self.parents,
            weights=self.weights,
            subdivisions=self.subdivisions,
            skin_radius_m=self.skin_radius_m,
            surface_level=self.surface_level,
            actual_max_rest_edge_m=self.actual_max_rest_edge_m,
        )

    @classmethod
    def load(cls, path: Path) -> BoundSurfaceGaussians:
        with np.load(path, allow_pickle=False) as saved:
            result = cls(
                saved["rest"],
                saved["triangles"],
                saved["pinned"],
                max_rest_edge_m=None,
                skin_radius_m=float(saved["skin_radius_m"]),
                surface_level=float(saved["surface_level"]),
            )
            result.parents = saved["parents"].copy()
            result.weights = saved["weights"].copy()
            result.subdivisions = int(saved["subdivisions"])
            result.actual_max_rest_edge_m = float(saved["actual_max_rest_edge_m"])
        if (
            result.parents.shape != result.weights.shape
            or result.weights.ndim != 2
            or result.weights.shape[1] != 3
            or not np.isfinite(result.weights).all()
            or np.any(result.weights < 0)
            or not np.allclose(result.weights.sum(-1), 1.0, rtol=0, atol=1e-14)
            or result.parents.min() < 0
            or result.parents.max() >= len(result.rest)
        ):
            raise ValueError("Invalid saved proxy binding")
        return result


def closest_triangle_surface(
    positions: np.ndarray,
    triangles: np.ndarray,
    robot_position: np.ndarray,
    sphere_radius_m: float,
    skin_radius_m: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scoring reference ONLY: exact point-to-triangle distance, exterior centers.

    This uses the continuous CURRENT mesh, unlike the proxy predictor. It must
    never be fed into proxy geometry, force computation, or state prediction.
    """
    vertices = positions[triangles]
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ab, ac, ap = b - a, c - a, robot_position - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
    determinant = d00 * d11 - d01**2
    if np.any(determinant <= 1e-30):
        raise ValueError("Degenerate current reference triangle")
    u = (d11 * d20 - d01 * d21) / determinant
    v = (d00 * d21 - d01 * d20) / determinant
    projection = a + u[:, None] * ab + v[:, None] * ac
    valid = (u >= 0) & (v >= 0) & (u + v <= 1)
    candidates = [projection]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.clip(((robot_position - start) * edge).sum(-1) / (edge * edge).sum(-1), 0, 1)
        candidates.append(start + t[:, None] * edge)
    candidates = np.stack(candidates, axis=1)
    squared = np.square(candidates - robot_position).sum(-1)
    squared[~valid, 0] = np.inf
    ti, ci = np.unravel_index(squared.argmin(), squared.shape)
    closest = candidates[ti, ci]
    distance = np.sqrt(squared[ti, ci])
    if distance <= 1e-12:
        raise ValueError("Reference requires an exterior sphere center off the material surface")
    normal = (closest - robot_position) / distance
    point = 0.5 * (closest - skin_radius_m * normal + robot_position + sphere_radius_m * normal)
    return float(distance - sphere_radius_m - skin_radius_m), point, normal
