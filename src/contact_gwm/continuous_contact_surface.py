"""Topology-assisted continuous collision surface over current Gaussian centers.

This is deliberately NOT an ellipsoid-union distance. Known rest connectivity
defines bounded, piecewise-linear patches between existing Gaussian centers.
The skin is an unchanged metric offset. No simulator or reference-distance code
is used. A single closest witness is returned for the pilot's single sphere;
all overlapping face witnesses are retained separately, without summing forces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .structures import ContactGeometry
from .surface_proxies import BoundSurfaceGaussians, SphereSurfaceQuery


def triangle_closest_weights(vertices: Tensor, point: Tensor) -> Tensor:
    """Voronoi-region closest-point solve on bounded triangles, in PyTorch.

    Returns convex barycentric weights. Edges/corners are handled explicitly;
    replacing the triangle by an infinite tangent plane would create contacts
    outside the block. Derivatives are piecewise, with finite branch values.
    """
    a, b, c = vertices.unbind(1)
    ab, ac = b - a, c - a
    area = torch.linalg.vector_norm(torch.cross(ab, ac, dim=-1), dim=-1)
    if torch.any(area <= 1e-20):
        raise ValueError("Degenerate current contact patch")
    ap, bp, cp = point - a, point - b, point - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denominator = (va + vb + vc).clamp_min(1e-30)
    weights = torch.stack((va, vb, vc), dim=-1) / denominator[:, None]
    zero, one = torch.zeros_like(d1), torch.ones_like(d1)

    def choose(mask, new_weights):
        nonlocal weights
        weights = torch.where(mask[:, None], torch.stack(new_weights, dim=-1), weights)

    t = d1 / (d1 - d3).clamp_min(1e-30)
    choose((vc <= 0) & (d1 >= 0) & (d3 <= 0), (one - t, t, zero))
    t = d2 / (d2 - d6).clamp_min(1e-30)
    choose((vb <= 0) & (d2 >= 0) & (d6 <= 0), (one - t, zero, t))
    t = (d4 - d3) / (d4 - d3 + d5 - d6).clamp_min(1e-30)
    choose((va <= 0) & (d4 >= d3) & (d5 >= d6), (zero, one - t, t))
    choose((d1 <= 0) & (d2 <= 0), (one, zero, zero))
    choose((d3 >= 0) & (d4 <= d3), (zero, one, zero))
    choose((d6 >= 0) & (d5 <= d6), (zero, zero, one))
    return weights


@dataclass(frozen=True)
class ContinuousSurfaceQuery(SphereSurfaceQuery):
    triangle_index: Tensor
    barycentric_weights: Tensor
    material_point: Tensor
    material_velocity: Tensor
    face_candidates: ContactGeometry
    roundoff_tolerance_m: float


class ContinuousContactSurface:
    """One current sphere -> continuous closest patch, fixed topology only.

    The full original Gaussian proxy binding/density is retained. Each original
    surface triangle's corners refer to its existing corner proxies. Collinear
    barycentric subdivision does not change a linear patch, so solving on those
    corners is exactly equivalent to solving on all its bound subtriangles.
    Requires the sphere center outside the material, with nondegenerate patches.
    """

    def __init__(self, binding: BoundSurfaceGaussians) -> None:
        self.binding = binding
        corners = {}
        for proxy, (parents, weights) in enumerate(
            zip(binding.parents, binding.weights, strict=True)
        ):
            if np.count_nonzero(weights) == 1:
                corners[int(parents[weights.argmax()])] = proxy
        self.patch_proxy_indices = np.array(
            [[corners[int(node)] for node in face] for face in binding.triangles], dtype=np.int64
        )
        rest_vertices = binding.rest[binding.triangles]
        normal = np.cross(
            rest_vertices[:, 1] - rest_vertices[:, 0], rest_vertices[:, 2] - rest_vertices[:, 0]
        )
        orientation = (normal * (rest_vertices.mean(1) - binding.rest.mean(0))).sum(-1)
        self.outward_sign = np.where(orientation < 0, -1.0, 1.0)

    def query(
        self,
        positions: Tensor,
        velocities: Tensor,
        robot_position: Tensor,
        robot_velocity: Tensor,
        sphere_radius_m: float,
    ) -> ContinuousSurfaceQuery:
        if not np.isfinite(sphere_radius_m) or sphere_radius_m <= 0:
            raise ValueError("Sphere radius must be finite and positive")
        if robot_position.shape != (3,) or robot_velocity.shape != (3,):
            raise ValueError("Current sphere x/v must have shape [3]")
        if (
            robot_position.dtype != positions.dtype
            or robot_velocity.dtype != positions.dtype
            or robot_position.device != positions.device
            or robot_velocity.device != positions.device
        ):
            raise ValueError("Current sphere and node states must share dtype/device")
        if not torch.isfinite(robot_position).all() or not torch.isfinite(robot_velocity).all():
            raise FloatingPointError("Non-finite current sphere state")
        cloud = self.binding.cloud(positions, velocities)
        patch_ids = torch.as_tensor(self.patch_proxy_indices, device=positions.device)
        vertices = cloud.centers[patch_ids]
        weights = triangle_closest_weights(vertices, robot_position)
        q = (vertices * weights[..., None]).sum(1)
        velocity = (cloud.velocities[patch_ids] * weights[..., None]).sum(1)
        offset = q - robot_position
        distance = torch.linalg.vector_norm(offset, dim=-1)
        closest = distance.argmin()
        if distance[closest] <= 1e-12:
            raise ValueError("Sphere center lies on the material surface; normal is undefined")
        normals = offset / distance[:, None].clamp_min(1e-12)
        current_outward = torch.cross(
            vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1
        )
        current_outward = current_outward / current_outward.norm(dim=-1, keepdim=True)
        current_outward = current_outward * positions.new_tensor(self.outward_sign)[:, None]
        if ((robot_position - q[closest]) * current_outward[closest]).sum() < -1e-12:
            raise ValueError("This pilot requires an exterior sphere center")
        skin = self.binding.skin_radius_m
        gaps = distance - sphere_radius_m - skin
        points = 0.5 * (q - skin * normals + robot_position + sphere_radius_m * normals)
        anchor = patch_ids.gather(1, weights.argmax(1)[:, None]).squeeze(1)

        def contacts(ids):
            return ContactGeometry(
                anchor[ids],
                torch.zeros_like(ids),
                points[ids],
                normals[ids],
                gaps[ids],
                (-gaps[ids]).clamp_min(0),
                robot_velocity - velocity[ids],
            )

        # An exact tangency can round to either side of zero. This IEEE-scale
        # band is ~2.31e-17 m in our float64 pilot, not a fitted contact margin.
        # Keep raw metric gaps unchanged; finite penetrations remain max(-gap,0).
        roundoff = 8 * torch.finfo(positions.dtype).eps * (sphere_radius_m + skin)
        active = torch.nonzero(gaps <= roundoff, as_tuple=False).flatten()
        selected = closest[None] if gaps[closest] <= roundoff else closest.new_empty(0)
        return ContinuousSurfaceQuery(
            contacts(selected),
            gaps[closest],
            points[closest],
            normals[closest],
            closest,
            weights[closest],
            q[closest],
            velocity[closest],
            contacts(active),
            roundoff,
        )
