"""Material-point velocity, topology-aware deduplication and surface quadrature.

No simulator, next state, force label, or trainable parameter is used here.
Quadrature returns geometric normal-pressure bases, not a calibrated force.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from .continuous_contact_surface import triangle_closest_weights
from .structures import ContactGeometry


@dataclass(frozen=True)
class MaterialSamples:
    face_indices: Tensor
    weights: Tensor
    points: Tensor
    velocities: Tensor
    projection_errors: Tensor


def material_samples(
    positions: Tensor,
    velocities: Tensor,
    triangles: Tensor,
    target_points: Tensor,
) -> MaterialSamples:
    """Project geometric witnesses onto the current material surface.

    Velocities are interpolated at fixed material barycentrics, not obtained by
    differentiating the moving closest point. Caller must impose fixed nodes.
    """
    vertices = positions[triangles]
    ids, weights, points, speed, error = [], [], [], [], []
    for point in target_points:
        bary = triangle_closest_weights(vertices, point)
        projected = (vertices * bary[..., None]).sum(1)
        distances = torch.linalg.vector_norm(projected - point, dim=-1)
        index = distances.argmin()
        ids.append(index)
        weights.append(bary[index])
        points.append(projected[index])
        speed.append((velocities[triangles[index]] * bary[index, :, None]).sum(0))
        error.append(distances[index])
    if not ids:
        return MaterialSamples(
            triangles.new_empty(0),
            positions.new_empty((0, 3)),
            positions.new_empty((0, 3)),
            positions.new_empty((0, 3)),
            positions.new_empty(0),
        )
    return MaterialSamples(*(torch.stack(value) for value in (ids, weights, points, speed, error)))


@dataclass(frozen=True)
class ContactGroups:
    contacts: ContactGeometry
    representative_indices: Tensor
    group_for_candidate: Tensor
    multiplicities: Tensor


def group_material_contacts(
    contacts: ContactGeometry,
    face_indices: Tensor,
    weights: Tensor,
    triangles: Tensor,
    *,
    point_tolerance_m: float = 1e-9,
    weight_tolerance: float = 1e-10,
    velocity_tolerance_m_s: float = 1e-9,
) -> ContactGroups:
    """Merge only the SAME material support and coincident contact witness.

    Coincident disconnected sheets must remain distinct. Different velocities,
    gaps or normals also prevent merging. No spatial averaging or force scaling
    is introduced; group members retain their original face IDs in the caller.
    Selection is discrete, with gradients preserved through selected records.
    """
    if len(face_indices) != contacts.num_contacts or weights.shape != (len(face_indices), 3):
        raise ValueError("Candidate face mapping must match the contacts")
    tri = triangles.detach().cpu().numpy()[face_indices.detach().cpu().numpy()]
    bary = weights.detach().cpu().numpy()
    cpoints, cnormals = contacts.points.detach(), contacts.normals.detach()
    gaps, speed = contacts.signed_gaps.detach(), contacts.relative_velocities.detach()
    keys, canonical, selected, groups = [], [], [], []
    for i, (nodes, row) in enumerate(zip(tri, bary, strict=True)):
        support = sorted(
            (int(node), float(w))
            for node, w in zip(nodes, row, strict=True)
            if w > weight_tolerance
        )
        key = tuple(node for node, _ in support)
        values = np.array([w for _, w in support])
        group = None
        for k, representative in enumerate(selected):
            if keys[k] != key or not np.allclose(
                canonical[k], values, rtol=0, atol=weight_tolerance
            ):
                continue
            if (
                float((cpoints[i] - cpoints[representative]).norm()) <= point_tolerance_m
                and float((cnormals[i] - cnormals[representative]).norm()) <= 1e-8
                and float((speed[i] - speed[representative]).norm()) <= velocity_tolerance_m_s
                and float((gaps[i] - gaps[representative]).abs()) <= point_tolerance_m
            ):
                group = k
                break
        if group is None:
            group = len(selected)
            selected.append(i)
            keys.append(key)
            canonical.append(values)
        groups.append(group)
    selection = torch.tensor(selected, dtype=torch.long, device=contacts.points.device)
    membership = torch.tensor(groups, dtype=torch.long, device=contacts.points.device)
    result = ContactGeometry(
        **{f.name: getattr(contacts, f.name)[selection] for f in fields(ContactGeometry)}
    )
    return ContactGroups(
        result, selection, membership, torch.bincount(membership, minlength=len(selection))
    )


class SurfaceQuadrature:
    """One centroid per barycentric subtriangle, area in the CURRENT shape.

    All face interiors have unique ownership. Shared vertices/edges have zero
    area. This is a fixed integration rule, NOT additional dynamics/Gaussians.
    Finite resolution can undersample very small contact caps; detection must
    continue using the continuous-surface query, not positive sample count.
    """

    def __init__(
        self, rest: np.ndarray, triangles: np.ndarray, pinned: np.ndarray, subdivisions: int
    ) -> None:
        if not isinstance(subdivisions, int) or subdivisions < 1:
            raise ValueError("Subdivisions must be a positive integer")
        triangles = np.asarray(triangles, dtype=np.int64)
        if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
            raise ValueError("Duplicate material faces would count area twice")
        self.rest, self.triangles, self.pinned = rest.copy(), triangles.copy(), pinned.copy()
        self.subdivisions = subdivisions
        centroids = []
        for i in range(subdivisions):
            for j in range(subdivisions - i):
                # Parametric coordinates (u,v) of a uniform triangular lattice.
                u, v = (i + 1 / 3) / subdivisions, (j + 1 / 3) / subdivisions
                centroids.append((1 - u - v, u, v))
                if i + j < subdivisions - 1:
                    u, v = (i + 2 / 3) / subdivisions, (j + 2 / 3) / subdivisions
                    centroids.append((1 - u - v, u, v))
        self.weights = np.asarray(centroids)
        assert len(self.weights) == subdivisions**2

    def normal_bases(
        self,
        positions: Tensor,
        velocities: Tensor,
        robot_position: Tensor,
        robot_velocity: Tensor,
        radius_m: float,
        skin_m: float,
        torque_origin: Tensor,
    ) -> dict[str, Tensor]:
        """Return area integrals with explicit units, without fitting stiffness.

        elastic vector basis = integral delta*n dA (m^3); multiply by a pressure
        stiffness in N/m^3 for force. Damping basis has units m^3/s, coefficient
        N*s/m^3. Old per-candidate N/m stiffness is NOT interchangeable.
        Torque bases apply normal traction at the material point; a shift along
        its normal to the collision midpoint does not change normal torque.
        """
        if not np.isfinite([radius_m, skin_m]).all() or radius_m <= 0 or skin_m < 0:
            raise ValueError("Invalid radius/skin")
        if not torch.isfinite(positions).all() or not torch.isfinite(velocities).all():
            raise FloatingPointError("Non-finite current state")
        pinned = torch.as_tensor(self.pinned, device=positions.device)
        rest = torch.as_tensor(self.rest, dtype=positions.dtype, device=positions.device)
        x = torch.where(pinned[:, None], rest, positions)
        v = torch.where(pinned[:, None], 0.0, velocities)
        ids = torch.as_tensor(self.triangles, device=positions.device)
        bary = positions.new_tensor(self.weights)
        vertices = x[ids]
        area = 0.5 * torch.linalg.vector_norm(
            torch.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1),
            dim=-1,
        )
        if torch.any(area <= 1e-20):
            raise ValueError("Degenerate current material face")
        q = torch.einsum("fiw,ki->fkw", vertices, bary)
        speed = torch.einsum("fiw,ki->fkw", v[ids], bary)
        offset = q - robot_position
        distance = torch.linalg.vector_norm(offset, dim=-1)
        if torch.any(distance <= 1e-12):
            raise ValueError("Quadrature point coincides with sphere center")
        normal = offset / distance[..., None]
        gap = distance - radius_m - skin_m
        active = gap < 0
        delta = (-gap).clamp_min(0)
        closing = ((robot_velocity - speed) * normal).sum(-1).clamp_min(0) * active
        measure = area[:, None] / self.subdivisions**2
        elastic = measure[..., None] * delta[..., None] * normal
        damping = measure[..., None] * closing[..., None] * normal
        return {
            "elastic_vector_m3": elastic.sum((0, 1)),
            "damping_vector_m3_s": damping.sum((0, 1)),
            "elastic_torque_m4": torch.cross(q - torque_origin, elastic, dim=-1).sum((0, 1)),
            "contact_area_m2": (measure * active).sum(),
            "unweighted_elastic_vector_m": (delta[..., None] * normal).sum((0, 1)),
            "total_surface_area_m2": area.sum(),
            "active_sample_count": active.sum(),
        }
