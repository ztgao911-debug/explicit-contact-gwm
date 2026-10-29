"""Geometrically clipped, adaptive material-surface integration.

A closest material point strictly inside the expanded sphere seeds a fan on
each intersecting triangle. Each ray is clipped analytically to the sphere;
positive quadrature nodes therefore lie inside even tiny corner caps. This is
known-topology geometry, not Gaussian density reconstruction or calibrated force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .continuous_contact_surface import triangle_closest_weights


@dataclass(frozen=True)
class IntegrationSettings:
    orders: tuple[int, ...] = (4, 8, 16, 32)
    relative_tolerance: float = 1e-4
    area_absolute_tolerance_m2: float = 1e-18
    elastic_absolute_tolerance_m3: float = 1e-26
    damping_absolute_tolerance_m3_s: float = 1e-22
    max_depth: int = 12
    max_intervals: int = 512

    def __post_init__(self):
        if (
            len(self.orders) < 2
            or tuple(sorted(set(self.orders))) != tuple(self.orders)
            or min(self.orders) < 2
            or self.max_depth < 0
            or self.max_intervals < 1
        ):
            raise ValueError("Invalid adaptive orders or depth")
        values = (
            self.relative_tolerance,
            self.area_absolute_tolerance_m2,
            self.elastic_absolute_tolerance_m3,
            self.damping_absolute_tolerance_m3_s,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Tolerances must be finite and positive")


def _edge_breakpoints(a: Tensor, b: Tensor, center: Tensor, radius: float) -> list[Tensor]:
    """Split fan boundary parameters at exact edge/sphere intersections."""
    edge, offset = b - a, a - center
    aa, bb = edge.square().sum(), (edge * offset).sum()
    cc = offset.square().sum() - radius**2
    disc = bb.square() - aa * cc
    values = [a.new_tensor(0.0), a.new_tensor(1.0)]
    if float(disc.detach()) > 0:
        root = disc.sqrt()
        sign = 1.0 if float(bb.detach()) >= 0 else -1.0
        stable = -bb - sign * root
        roots = (stable / aa, cc / stable) if abs(float(stable.detach())) > 1e-30 else (-bb / aa,)
        values.extend(t for t in roots if 0 < float(t.detach()) < 1)
    values.sort(key=lambda t: float(t.detach()))
    result = [values[0]]
    for t in values[1:]:
        if float(t.detach()) > float(result[-1].detach()):
            result.append(t)
    return result


class ClippedSurfaceQuadrature:
    """Current areas and velocities; adaptive error estimates are not proofs.

    No force/next-state labels or simulator are accepted. Physical contact margin
    is zero. Float64 is required for thin caps. Each material face owns its area;
    fan edges have zero measure and are never assigned duplicate point forces.
    """

    def __init__(
        self,
        rest: np.ndarray,
        triangles: np.ndarray,
        pinned: np.ndarray,
        settings: IntegrationSettings | None = None,
    ):
        self.rest = np.array(rest, dtype=np.float64, copy=True)
        self.triangles = np.array(triangles, dtype=np.int64, copy=True)
        self.pinned = np.array(pinned, dtype=bool, copy=True)
        if (
            self.rest.ndim != 2
            or self.rest.shape[1] != 3
            or self.pinned.shape != (len(self.rest),)
            or not np.isfinite(self.rest).all()
            or self.triangles.ndim != 2
            or self.triangles.shape[1] != 3
            or not len(self.triangles)
            or self.triangles.min() < 0
            or self.triangles.max() >= len(self.rest)
        ):
            raise ValueError("Invalid material surface")
        if len(np.unique(np.sort(self.triangles, axis=1), axis=0)) != len(self.triangles):
            raise ValueError("Duplicate material faces count area twice")
        self.settings = settings or IntegrationSettings()
        self._gauss = {
            n: tuple(
                (t + 1) / 2 if i == 0 else t / 2
                for i, t in enumerate(np.polynomial.legendre.leggauss(n))
            )
            for n in self.settings.orders
        }

    def normal_bases(
        self,
        positions: Tensor,
        velocities: Tensor,
        robot_position: Tensor,
        robot_velocity: Tensor,
        radius_m: float,
        skin_m: float,
        torque_origin: Tensor,
    ) -> dict:
        if positions.dtype != torch.float64 or velocities.dtype != positions.dtype:
            raise ValueError("Clipped thin-cap integration requires float64 x/v")
        if positions.shape != self.rest.shape or velocities.shape != positions.shape:
            raise ValueError("Keep the material node order and shape")
        if (
            robot_position.shape != (3,)
            or robot_velocity.shape != (3,)
            or torque_origin.shape != (3,)
        ):
            raise ValueError("Current sphere x/v and torque origin must have shape [3]")
        if any(
            t.dtype != positions.dtype or t.device != positions.device
            for t in (velocities, robot_position, robot_velocity, torque_origin)
        ):
            raise ValueError("All current tensors must share dtype/device")
        if not all(
            torch.isfinite(t).all()
            for t in (positions, velocities, robot_position, robot_velocity, torque_origin)
        ):
            raise FloatingPointError("Non-finite current integration input")
        if not np.isfinite([radius_m, skin_m]).all() or radius_m <= 0 or skin_m < 0:
            raise ValueError("Invalid sphere radius/skin")
        pin = torch.as_tensor(self.pinned, device=positions.device)
        x = torch.where(pin[:, None], positions.new_tensor(self.rest), positions)
        v = torch.where(pin[:, None], 0.0, velocities)
        ids = torch.as_tensor(self.triangles, device=positions.device)
        vertices = x[ids]
        area2 = torch.linalg.vector_norm(
            torch.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1),
            dim=-1,
        )
        if torch.any(area2 <= 1e-20):
            raise ValueError("Degenerate current material face")
        bary = triangle_closest_weights(vertices, robot_position)
        seeds = (vertices * bary[..., None]).sum(1)
        distances = torch.linalg.vector_norm(seeds - robot_position, dim=-1)
        if torch.any(distances <= 1e-12):
            raise ValueError("Sphere center on the material surface is unsupported")
        radius = radius_m + skin_m
        roundoff = 8 * torch.finfo(positions.dtype).eps * radius
        active = torch.nonzero(radius - distances > roundoff, as_tuple=False).flatten()
        total = positions.new_zeros(9)
        diagnostics, error_total = [], positions.new_zeros(3)
        face_values = []
        evaluated_samples = final_samples = unresolved = leaves = max_depth = 0
        evaluated_intervals = 0
        failure_components = torch.zeros(3, dtype=torch.long, device=x.device)
        settings = self.settings
        gauss = {n: tuple(positions.new_tensor(t) for t in pair) for n, pair in self._gauss.items()}

        for face in active.tolist():
            seed, seed_bary = seeds[face], bary[face]
            vv, node_v = vertices[face], v[ids[face]]
            offset = seed - robot_position
            # Stable difference of squared radii, positive by the exact closest
            # triangle test. No positive pressure/area floor is introduced.
            slack = (radius - distances[face]) * (radius + distances[face])
            face_value, face_error = positions.new_zeros(9), positions.new_zeros(3)
            face_samples = face_unresolved = 0

            for ia, ib in ((0, 1), (1, 2), (2, 0)):
                determinant = torch.linalg.vector_norm(
                    torch.cross(vv[ia] - seed, vv[ib] - seed, dim=-1)
                )
                if float(determinant.detach()) == 0:
                    continue
                ea, eb = torch.eye(3, dtype=x.dtype, device=x.device)[[ia, ib]]
                breaks = _edge_breakpoints(vv[ia], vv[ib], robot_position, radius)

                def integrate(
                    lo,
                    hi,
                    order,
                    ea=ea,
                    eb=eb,
                    vv=vv,
                    seed=seed,
                    offset=offset,
                    slack=slack,
                    seed_bary=seed_bary,
                    node_v=node_v,
                    determinant=determinant,
                ):
                    nodes, weights = gauss[order]
                    t = lo + (hi - lo) * nodes
                    boundary_bary = (1 - t[:, None]) * ea + t[:, None] * eb
                    boundary = boundary_bary @ vv
                    direction = boundary - seed
                    aa = direction.square().sum(-1)
                    bb = (direction * offset).sum(-1)
                    root = (bb.square() + aa * slack).sqrt()
                    exit_value = torch.where(
                        bb > 0,
                        slack / (root + bb).clamp_min(1e-30),
                        (root - bb) / aa.clamp_min(1e-30),
                    ).clamp_max(1.0)
                    s = exit_value[:, None] * nodes[None, :]
                    point = seed + s[..., None] * direction[:, None, :]
                    distance = torch.linalg.vector_norm(point - robot_position, dim=-1)
                    normal = (point - robot_position) / distance[..., None]
                    delta = (radius - distance).clamp_min(0)
                    measure = (
                        determinant
                        * (hi - lo)
                        * exit_value[:, None].square()
                        * nodes[None, :]
                        * weights[:, None]
                        * weights[None, :]
                    )
                    elastic = measure * delta
                    # Along a ray, material velocity is affine. Its normal
                    # closing numerator is a quadratic in s. Split at its real
                    # roots so a radial max(v_rel.n,0) kink cannot trigger futile
                    # subdivision in the angular parameter.
                    seed_v = seed_bary @ node_v
                    dv = boundary_bary @ node_v - seed_v
                    relative = robot_velocity - seed_v
                    ca = -(dv * direction).sum(-1)
                    cb = (relative * direction).sum(-1) - (dv * offset).sum(-1)
                    cc = (relative * offset).sum().expand_as(ca)
                    eps = (
                        8
                        * torch.finfo(x.dtype).eps
                        * torch.stack((ca.abs(), cb.abs(), cc.abs())).max(0).values
                    )
                    quadratic = ca.abs() > eps
                    disc = cb.square() - 4 * ca * cc
                    real = quadratic & (disc > 0)
                    disc_root = torch.where(real, disc, torch.ones_like(disc)).sqrt()
                    sign = torch.where(cb >= 0, 1.0, -1.0)
                    stable = -0.5 * (cb + sign * disc_root)
                    r1 = stable / torch.where(real, ca, torch.ones_like(ca))
                    r2 = cc / torch.where(real & (stable != 0), stable, torch.ones_like(stable))
                    linear = (~quadratic) & (cb.abs() > eps)
                    linear_root = -cc / torch.where(linear, cb, torch.ones_like(cb))
                    r1 = torch.where(real, r1, torch.where(linear, linear_root, exit_value))
                    r2 = torch.where(real, r2, exit_value)
                    r1 = torch.where((r1 > 0) & (r1 < exit_value), r1, exit_value)
                    r2 = torch.where((r2 > 0) & (r2 < exit_value), r2, exit_value)
                    cuts = torch.sort(
                        torch.stack((torch.zeros_like(exit_value), r1, r2, exit_value), dim=1),
                        dim=1,
                    ).values
                    lower, length = cuts[:, :-1], cuts[:, 1:] - cuts[:, :-1]
                    sd = lower[..., None] + length[..., None] * nodes
                    pd = seed + sd[..., None] * direction[:, None, None, :]
                    bary_d = (1 - sd[..., None]) * seed_bary + sd[..., None] * boundary_bary[
                        :, None, None, :
                    ]
                    distance_d = torch.linalg.vector_norm(pd - robot_position, dim=-1)
                    normal_d = (pd - robot_position) / distance_d[..., None]
                    closing = ((robot_velocity - bary_d @ node_v) * normal_d).sum(-1).clamp_min(0)
                    closing = closing * (distance_d < radius)
                    damping_measure = (
                        determinant
                        * (hi - lo)
                        * weights[:, None, None]
                        * weights[None, None, :]
                        * sd
                        * length[..., None]
                    )
                    damping = damping_measure * closing
                    value = torch.cat(
                        (
                            measure.sum()[None],
                            elastic.sum()[None],
                            (elastic[..., None] * normal).sum((0, 1)),
                            damping.sum()[None],
                            (damping[..., None] * normal_d).sum((0, 1, 2)),
                        )
                    )
                    return value, int((delta > 0).sum())

                def recurse(lo, hi, depth):
                    nonlocal evaluated_samples, final_samples, unresolved, leaves, max_depth
                    nonlocal evaluated_intervals
                    evaluated_intervals += 1
                    previous, _ = integrate(lo, hi, settings.orders[0])
                    evaluated_samples += 4 * settings.orders[0] ** 2
                    last_error = positions.new_zeros(3)
                    for order in settings.orders[1:]:
                        current, count = integrate(lo, hi, order)
                        evaluated_samples += 4 * order**2
                        last_error = torch.stack(
                            (
                                (current[0] - previous[0]).abs(),
                                torch.maximum(
                                    (current[1] - previous[1]).abs(),
                                    (current[2:5] - previous[2:5]).norm(),
                                ),
                                torch.maximum(
                                    (current[5] - previous[5]).abs(),
                                    (current[6:9] - previous[6:9]).norm(),
                                ),
                            )
                        )
                        fraction = float((hi - lo).detach()) / (3 * max(len(active), 1))
                        absolute = (
                            positions.new_tensor(
                                [
                                    settings.area_absolute_tolerance_m2,
                                    settings.elastic_absolute_tolerance_m3,
                                    settings.damping_absolute_tolerance_m3_s,
                                ]
                            )
                            * fraction
                        )
                        scale = current[[0, 1, 5]].abs()
                        if bool(
                            torch.all(last_error <= absolute + settings.relative_tolerance * scale)
                        ):
                            final_samples += count
                            leaves += 1
                            max_depth = max(max_depth, depth)
                            return current, last_error, count, 0
                        previous = current
                    if depth < settings.max_depth and evaluated_intervals < settings.max_intervals:
                        mid = (lo + hi) / 2
                        left = recurse(lo, mid, depth + 1)
                        right = recurse(mid, hi, depth + 1)
                        return (
                            left[0] + right[0],
                            left[1] + right[1],
                            left[2] + right[2],
                            left[3] + right[3],
                        )
                    failure_components.add_(
                        (last_error > absolute + settings.relative_tolerance * scale).long()
                    )
                    final_samples += count
                    unresolved += 1
                    leaves += 1
                    max_depth = max(max_depth, depth)
                    return current, last_error, count, 1

                for lo, hi in zip(breaks[:-1], breaks[1:], strict=True):
                    value, error, count, failed = recurse(lo, hi, 0)
                    face_value = face_value + value
                    face_error = face_error + error
                    face_samples += count
                    face_unresolved += failed
            total = total + face_value
            face_values.append(face_value)
            error_total = error_total + face_error
            diagnostics.append(
                [
                    face,
                    float(face_value[0].detach()),
                    float(face_value[1].detach()),
                    face_samples,
                    face_unresolved,
                ]
            )
        positive_faces_zero = sum(row[1] <= 0 or row[2] <= 0 for row in diagnostics)
        return {
            "contact_area_m2": total[0],
            "elastic_scalar_m3": total[1],
            "elastic_vector_m3": total[2:5],
            "damping_scalar_m3_s": total[5],
            "damping_vector_m3_s": total[6:9],
            "elastic_torque_m4": torch.cross(robot_position - torque_origin, total[2:5], dim=-1),
            "total_surface_area_m2": area2.sum() / 2,
            "active_sample_count": final_samples,
            "evaluated_samples": evaluated_samples,
            "evaluated_intervals": evaluated_intervals,
            "interval_budget_exhausted": evaluated_intervals >= settings.max_intervals,
            "unresolved_area_intervals": int(failure_components[0]),
            "unresolved_elastic_intervals": int(failure_components[1]),
            "unresolved_damping_intervals": int(failure_components[2]),
            "positive_contact_faces": len(active),
            "zero_integral_positive_faces": positive_faces_zero,
            "unresolved_intervals": unresolved,
            "leaf_intervals": leaves,
            "max_subdivision_depth": max_depth,
            "estimated_area_error_m2": error_total[0],
            "estimated_elastic_error_m3": error_total[1],
            "estimated_damping_error_m3_s": error_total[2],
            "face_diagnostics": diagnostics,
            # Additional exports only: the existing totals, error control and
            # summation order above are unchanged. These are loads on ORIGINAL
            # material faces, never one independent spring per quadrature node.
            "face_indices": active,
            "face_elastic_vectors_m3": torch.stack(face_values)[:, 2:5]
            if face_values
            else positions.new_empty((0, 3)),
            "face_contact_areas_m2": torch.stack(face_values)[:, 0]
            if face_values
            else positions.new_empty(0),
        }
