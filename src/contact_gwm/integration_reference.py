"""Evaluation-only independent plane/disk slice integration (NumPy).

The predictor uses clipped fans. This reference instead integrates vertical
slices of a disk/triangle intersection, with analytic inner elastic integrals.
It is a numerical reference, not a force solver or exact physical ground truth.
"""

from __future__ import annotations

import numpy as np


def _inside(point, triangle):
    a, b, c = triangle
    matrix = np.stack((b - a, c - a), axis=1)
    weights = np.linalg.solve(matrix, point - a)
    return weights.min() >= -1e-13 and weights.sum() <= 1 + 1e-13


def _edge_roots(a, b, radius):
    e = b - a
    aa, bb, cc = e @ e, a @ e, a @ a - radius**2
    discriminant = bb * bb - aa * cc
    if discriminant < 0:
        return []
    root = np.sqrt(max(discriminant, 0.0))
    stable = -bb - np.copysign(root, bb)
    values = [stable / aa, cc / stable] if abs(stable) > 1e-30 else [-bb / aa]
    return [a + t * e for t in values if 0 <= t <= 1]


def triangle_disk_reference(vertices, center, radius, order=128):
    a, b, c = np.asarray(vertices)
    e1 = b - a
    e1 = e1 / np.linalg.norm(e1)
    e3 = np.cross(b - a, c - a)
    area2 = np.linalg.norm(e3)
    if area2 <= 1e-20:
        raise ValueError("Degenerate reference triangle")
    e3 = e3 / area2
    e2 = np.cross(e3, e1)
    h = float((a - center) @ e3)
    if abs(h) >= radius:
        return np.zeros(5)
    disk_radius = np.sqrt((radius - abs(h)) * (radius + abs(h)))
    relative = vertices - center
    triangle = np.stack((relative @ e1, relative @ e2), axis=1)
    candidates = [p for p in triangle if p @ p < disk_radius**2]
    crossings = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        crossings.extend(_edge_roots(triangle[i], triangle[j], disk_radius))
    candidates.extend(crossings)
    for point in (np.array([-disk_radius, 0.0]), np.array([disk_radius, 0.0])):
        if _inside(point, triangle):
            candidates.append(point)
    if len(candidates) < 2:
        return np.zeros(5)
    xmin, xmax = min(p[0] for p in candidates), max(p[0] for p in candidates)
    xmin = max(xmin, float(triangle[:, 0].min()), -disk_radius)
    xmax = min(xmax, float(triangle[:, 0].max()), disk_radius)
    if xmax <= xmin:
        return np.zeros(5)
    breaks = sorted(
        set(
            [
                xmin,
                xmax,
                *[p[0] for p in crossings if xmin < p[0] < xmax],
                *[p[0] for p in triangle if xmin < p[0] < xmax],
            ]
        )
    )
    nodes, weights = np.polynomial.legendre.leggauss(order)
    total = np.zeros(5)
    for lo, hi in zip(breaks[:-1], breaks[1:], strict=True):
        xs = (lo + hi) / 2 + (hi - lo) / 2 * nodes
        possible = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = triangle[i], triangle[j]
            if q[0] == p[0]:
                continue
            t = (xs - p[0]) / (q[0] - p[0])
            yy = p[1] + t * (q[1] - p[1])
            possible.append(np.where((t >= 0) & (t <= 1), yy, np.nan))
        yy = np.stack(possible)
        finite = np.isfinite(yy)
        has_slice = finite.any(axis=0)
        tri_lo = np.where(has_slice, np.where(finite, yy, np.inf).min(axis=0), 0.0)
        tri_hi = np.where(has_slice, np.where(finite, yy, -np.inf).max(axis=0), 0.0)
        disk_y = np.sqrt(np.maximum(0.0, disk_radius**2 - xs**2))
        lower, upper = np.maximum(tri_lo, -disk_y), np.minimum(tri_hi, disk_y)
        valid = has_slice & (upper > lower)
        lower = np.where(valid, lower, 0.0)
        upper = np.where(valid, upper, 0.0)
        base2 = xs**2 + h**2
        base = np.sqrt(np.maximum(base2, 1e-300))
        root_upper = np.sqrt(base2 + upper**2)
        root_lower = np.sqrt(base2 + lower**2)
        arsinh = np.arcsinh(upper / base) - np.arcsinh(lower / base)
        length = upper - lower
        common = radius * arsinh - length
        fx = xs * common
        fy = radius * (root_upper - root_lower) - 0.5 * (upper**2 - lower**2)
        fz = h * common
        scalar = radius * length - 0.5 * (upper * root_upper - lower * root_lower + base2 * arsinh)
        local = np.stack((length, scalar, fx, fy, fz), axis=1)
        total += (local * weights[:, None] * (hi - lo) / 2).sum(0)
    total[2:5] = total[2] * e1 + total[3] * e2 + total[4] * e3
    return total


def surface_disk_reference(positions, triangles, center, radius, order=128):
    result = np.zeros(5)
    for ids in triangles:
        result += triangle_disk_reference(positions[ids], center, radius, order)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite independent slice reference")
    return {
        "contact_area_m2": float(result[0]),
        "elastic_scalar_m3": float(result[1]),
        "elastic_vector_m3": result[2:5],
    }
