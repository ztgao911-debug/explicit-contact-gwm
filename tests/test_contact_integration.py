from __future__ import annotations

import warnings
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest
import torch

from contact_gwm.contact_integration import ClippedSurfaceQuadrature, IntegrationSettings
from contact_gwm.contact_sampling import SurfaceQuadrature
from contact_gwm.integration_reference import surface_disk_reference


def square():
    return np.array(
        [[-0.02, -0.02, 0], [0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, 0.02, 0]]
    ), np.array([[0, 1, 2], [1, 3, 2]])


def cube():
    rest = np.array(
        [
            [0, 0, 0],
            [0.06, 0, 0],
            [0, 0.04, 0],
            [0.06, 0.04, 0],
            [0, 0, 0.03],
            [0.06, 0, 0.03],
            [0, 0.04, 0.03],
            [0.06, 0.04, 0.03],
        ]
    )
    tri = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [4, 6, 5],
            [5, 6, 7],
            [0, 4, 1],
            [1, 4, 5],
            [2, 3, 6],
            [3, 7, 6],
            [0, 2, 4],
            [2, 6, 4],
            [1, 5, 3],
            [3, 5, 7],
        ]
    )
    return rest, tri


def evaluate(rule, rest, p, rv=(0, 0, 0)):
    return rule.normal_bases(
        torch.tensor(rest),
        torch.zeros(rest.shape).double(),
        torch.tensor(p, dtype=torch.float64),
        torch.tensor(rv, dtype=torch.float64),
        0.012,
        0.001,
        torch.zeros(3).double(),
    )


def test_clipping_recovers_small_corner_cap_missed_by_uniform_64():
    rest, tri = cube()
    pin = np.isclose(rest[:, 2], 0)
    p = np.full(3, -(0.013 - 0.00005) / np.sqrt(3))
    fixed = evaluate(SurfaceQuadrature(rest, tri, pin, 64), rest, p)
    assert fixed["contact_area_m2"] == 0
    result = evaluate(ClippedSurfaceQuadrature(rest, tri, pin), rest, p)
    reference = surface_disk_reference(rest, tri, p, 0.013, 128)
    assert result["contact_area_m2"] > 0 and result["elastic_scalar_m3"] > 0
    assert result["zero_integral_positive_faces"] == 0 and result["unresolved_intervals"] == 0
    np.testing.assert_allclose(
        result["elastic_vector_m3"].numpy(), reference["elastic_vector_m3"], rtol=1e-4, atol=1e-24
    )
    assert float(result["contact_area_m2"]) == pytest.approx(reference["contact_area_m2"], rel=1e-4)


def test_disk_inside_triangle_is_found_even_when_all_vertices_are_outside_sphere():
    rest = np.array([[-0.04, -0.04, 0], [0.04, -0.04, 0], [0, 0.04, 0]])
    tri = np.array([[0, 1, 2]])
    p = np.array([0, 0, 0.01295])
    assert np.all(np.linalg.norm(rest - p, axis=1) > 0.013)
    result = evaluate(ClippedSurfaceQuadrature(rest, tri, np.zeros(3, bool)), rest, p)
    assert float(result["contact_area_m2"]) == pytest.approx(
        np.pi * (0.013**2 - p[2] ** 2), rel=1e-4
    )
    assert result["positive_contact_faces"] == 1


def test_elastic_and_damping_match_independent_planar_closed_forms():
    rest, tri = square()
    h = 0.0125
    speed = 0.01
    result = evaluate(
        ClippedSurfaceQuadrature(rest, tri, np.zeros(4, bool)), rest, [0, 0, h], [0, 0, -speed]
    )
    assert float(result["elastic_vector_m3"][2]) == pytest.approx(
        -np.pi * h * (0.013 - h) ** 2, rel=2e-4
    )
    exact_damping = -2 * np.pi * speed * h * h * np.log(0.013 / h)
    assert float(result["damping_vector_m3_s"][2]) == pytest.approx(exact_damping, rel=2e-4)
    assert result["elastic_vector_m3"][:2].abs().max() < 1e-16
    assert result["unresolved_intervals"] == 0


@pytest.mark.parametrize("height", [0.013, 0.0130001])
def test_tangency_and_separation_do_not_gain_artificial_contact_area(height):
    rest, tri = square()
    result = evaluate(ClippedSurfaceQuadrature(rest, tri, np.zeros(4, bool)), rest, [0, 0, height])
    assert result["contact_area_m2"] == 0 and result["elastic_scalar_m3"] == 0
    assert result["positive_contact_faces"] == 0


def test_material_tessellation_does_not_multiply_integral():
    rest = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0]])
    refined = np.vstack(
        [rest, (rest[0] + rest[1]) / 2, (rest[1] + rest[2]) / 2, (rest[2] + rest[0]) / 2]
    )
    values = []
    for x, t in [
        (rest, np.array([[0, 1, 2]])),
        (refined, np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])),
    ]:
        values.append(
            evaluate(
                ClippedSurfaceQuadrature(x, t, np.zeros(len(x), bool)), x, [0.005, 0.005, 0.0128]
            )
        )
    # The configured error is a vector norm relative to positive integral
    # scale; near-zero lateral components need not agree to relative precision.
    difference = (values[0]["elastic_vector_m3"] - values[1]["elastic_vector_m3"]).norm()
    assert difference <= 2e-4 * values[1]["elastic_vector_m3"].norm()
    with pytest.raises(ValueError, match="Duplicate"):
        ClippedSurfaceQuadrature(rest, np.array([[0, 1, 2], [0, 2, 1]]), np.zeros(3, bool))


def test_fixed_boundary_finite_gradients_and_no_input_mutation():
    rest, tri = square()
    pin = np.array([True, False, False, False])
    rule = ClippedSurfaceQuadrature(rest, tri, pin)
    x = torch.tensor(rest)
    x[0] += 0.1
    x.requires_grad_()
    v = torch.zeros_like(x)
    v[0] = 1
    v.requires_grad_()
    before_x, before_v = x.detach().clone(), v.detach().clone()
    result = rule.normal_bases(
        x,
        v,
        torch.tensor([0.003, 0.002, 0.0129]).double(),
        torch.tensor([0, 0, -0.01]).double(),
        0.012,
        0.001,
        torch.zeros(3).double(),
    )
    grad = torch.autograd.grad(
        result["elastic_vector_m3"].sum() + result["damping_vector_m3_s"].sum(), (x, v)
    )
    assert all(torch.isfinite(g).all() for g in grad)
    assert not torch.any(grad[0][0]) and not torch.any(grad[1][0])
    assert torch.equal(x.detach(), before_x) and torch.equal(v.detach(), before_v)


def test_recreation_bitwise_and_no_reference_or_solver_dependency():
    rest, tri = square()
    pin = np.zeros(4, bool)
    settings = IntegrationSettings()
    a = evaluate(ClippedSurfaceQuadrature(rest, tri, pin, settings), rest, [0.003, 0.002, 0.0129])

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference read by predictor")

    with patch("contact_gwm.integration_reference.surface_disk_reference", forbidden):
        b = evaluate(
            ClippedSurfaceQuadrature(
                rest.copy(), tri.copy(), pin.copy(), IntegrationSettings(**asdict(settings))
            ),
            rest,
            [0.003, 0.002, 0.0129],
        )
    assert torch.equal(a["elastic_vector_m3"], b["elastic_vector_m3"])
    assert torch.equal(a["contact_area_m2"], b["contact_area_m2"])
    mj = pytest.importorskip("mujoco")
    with (
        patch.object(mj, "mj_fwdPosition", forbidden),
        patch.object(mj, "mj_forward", forbidden),
        patch.object(mj, "mj_contactForce", forbidden),
        patch.object(mj, "mj_step", forbidden),
    ):
        evaluate(ClippedSurfaceQuadrature(rest, tri, pin), rest, [0.003, 0.002, 0.0129])


def test_accuracy_budget_exhaustion_is_reported_with_positive_result():
    rest, tri = square()
    settings = IntegrationSettings(orders=(2, 4), relative_tolerance=1e-12, max_depth=0)
    result = evaluate(
        ClippedSurfaceQuadrature(rest, tri, np.zeros(4, bool), settings),
        rest,
        [0.003, 0.002, 0.0129],
    )
    assert result["contact_area_m2"] > 0 and result["unresolved_intervals"] > 0


def test_independent_reference_agrees_with_known_plane_area_and_elastic_force_basis():
    rest, tri = square()
    h = 0.0125
    result = surface_disk_reference(rest, tri, np.array([0, 0, h]), 0.013, 256)
    assert result["contact_area_m2"] == pytest.approx(np.pi * (0.013**2 - h**2), rel=1e-6)
    assert result["elastic_vector_m3"][2] == pytest.approx(-np.pi * h * (0.013 - h) ** 2, rel=1e-6)


def test_radial_velocity_sign_change_is_split_without_excessive_refinement():
    rest, tri = square()
    pin = np.zeros(4, bool)
    model = ClippedSurfaceQuadrature(rest, tri, pin)
    p = torch.tensor([0.003, 0.002, 0.0129]).double()
    x = torch.tensor(rest)
    v = torch.zeros_like(x)
    v[:, 2] = x[:, 0] - p[0] - 0.0002
    v.requires_grad_()
    result = model.normal_bases(
        x, v, p, torch.zeros(3).double(), 0.012, 0.001, torch.zeros(3).double()
    )
    assert result["unresolved_intervals"] == 0 and not result["interval_budget_exhausted"]
    # Independent one-dimensional integral for a closing half-cap on a plane.
    h = 0.0129
    cap = np.sqrt(0.013**2 - h * h)
    a = 0.0002
    nodes, weights = np.polynomial.legendre.leggauss(256)
    xx = (a + cap) / 2 + (cap - a) / 2 * nodes
    b = np.sqrt(h * h + xx * xx)
    value = -h * h * (xx - a) * 2 / b * np.arctan(np.sqrt(cap * cap - xx * xx) / b)
    expected = np.sum(value * weights) * (cap - a) / 2
    assert float(result["damping_vector_m3_s"][2].detach()) == pytest.approx(expected, rel=3e-4)
    grad = torch.autograd.grad(result["elastic_vector_m3"].sum(), v)[0]
    assert torch.count_nonzero(grad) == 0


def test_independent_reference_zero_width_boundary_slice_stays_finite():
    vertices = np.array(
        [
            [-0.03, -0.02, 0],
            [-0.03, -0.02, 0.009999999999999998],
            [-0.03, -0.01, 0.009999999999999998],
        ]
    )
    center = np.array([-0.041999999999999996, -0.015, 0.004999999999999999])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = surface_disk_reference(vertices, np.array([[0, 1, 2]]), center, 0.013, 128)
    assert np.isfinite(result["contact_area_m2"])
    assert result["contact_area_m2"] > 0
