from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pytest
import torch

from contact_gwm.continuous_contact_surface import ContinuousContactSurface
from contact_gwm.surface_proxies import BoundSurfaceGaussians, closest_triangle_surface


def fixture_surface(pinned=True):
    rest = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0]])
    binding = BoundSurfaceGaussians(
        rest,
        np.array([[0, 1, 2], [1, 3, 2]]),
        np.array([pinned, False, False, False]),
        max_rest_edge_m=0.002,
    )
    return binding, ContinuousContactSurface(binding)


@pytest.mark.parametrize(
    "point",
    [
        [0.0043, 0.0041, 0.0129999],  # A 0.1 um contact between sample centers.
        [-0.003, 0.005, 0.012],  # Edge witness.
        [-0.003, -0.004, 0.012],  # Corner witness.
        [-0.05, 0.005, 0.01],  # Outside the bounded face; no plane extrapolation.
    ],
)
def test_bounded_face_edge_corner_distance_and_normal(point):
    binding, surface = fixture_surface()
    x, v = torch.tensor(binding.rest), torch.zeros((4, 3), dtype=torch.float64)
    p, rv = torch.tensor(point, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    query = surface.query(x, v, p, rv, 0.012)
    gap, q, n = closest_triangle_surface(binding.rest, binding.triangles, p.numpy(), 0.012, 0.001)
    assert float(query.closest_gap) == pytest.approx(gap, abs=1e-15)
    np.testing.assert_allclose(query.closest_point.numpy(), q, atol=1e-15)
    np.testing.assert_allclose(query.closest_normal.numpy(), n, atol=1e-14)
    assert query.contacts.num_contacts == int(gap <= query.roundoff_tolerance_m)
    assert torch.all(query.barycentric_weights >= 0)
    assert float(query.barycentric_weights.sum()) == pytest.approx(1)


def test_critical_contact_fixed_without_margin_radius_or_density_change():
    binding, surface = fixture_surface()
    x, v = torch.tensor(binding.rest), torch.zeros((4, 3), dtype=torch.float64)
    rv = torch.zeros(3, dtype=torch.float64)
    p = torch.tensor([0.0043, 0.0041, 0.0129999], dtype=torch.float64)
    assert binding.query(x, v, p, rv, 0.012).contacts.num_contacts == 0
    fixed = surface.query(x, v, p, rv, 0.012)
    assert fixed.contacts.num_contacts == 1 and fixed.closest_gap == pytest.approx(-1e-7)
    p[2] = 0.0130001
    assert surface.query(x, v, p, rv, 0.012).contacts.num_contacts == 0
    assert surface.binding is binding and len(binding.parents) == 81
    assert binding.skin_radius_m == 0.001


def test_deformed_surface_matches_independent_distance_and_preserves_instant_velocity():
    binding, surface = fixture_surface()
    x = torch.tensor(binding.rest)
    x[3, 2] = 0.002
    v = 0.2 * x
    generator = np.random.default_rng(7)
    for _ in range(20):
        p = torch.tensor([*generator.uniform(-0.003, 0.013, 2), 0.013], dtype=torch.float64)
        rv = torch.tensor([0.01, 0.02, -0.03], dtype=torch.float64)
        result = surface.query(x, v, p, rv, 0.012)
        gap, _, normal = closest_triangle_surface(
            x.numpy(), binding.triangles, p.numpy(), 0.012, 0.001
        )
        assert float(result.closest_gap) == pytest.approx(gap, abs=1e-15)
        np.testing.assert_allclose(result.closest_normal.numpy(), normal, atol=1e-13)
        expected_velocity = (
            v[binding.triangles[int(result.triangle_index)]] * result.barycentric_weights[:, None]
        ).sum(0)
        assert torch.equal(result.material_velocity, expected_velocity)
        if result.contacts.num_contacts:
            torch.testing.assert_close(
                result.contacts.relative_velocities[0], rv - expected_velocity
            )


def test_fixed_boundary_finite_derivatives_and_distance_normal_consistency():
    binding, surface = fixture_surface()
    x = torch.tensor(binding.rest)
    x[0] += 0.1
    x.requires_grad_()
    v = torch.ones_like(x).requires_grad_()
    p = torch.tensor([0.0043, 0.0041, 0.01299], dtype=torch.float64, requires_grad=True)
    result = surface.query(x, v, p, torch.zeros(3, dtype=torch.float64), 0.012)
    distance_gradient = torch.autograd.grad(result.closest_gap, p, retain_graph=True)[0]
    grads = torch.autograd.grad(
        result.closest_gap + result.material_velocity.square().sum(), (x, v, p)
    )
    assert all(torch.isfinite(g).all() for g in grads)
    assert not torch.any(grads[0][0]) and not torch.any(grads[1][0])
    torch.testing.assert_close(distance_gradient, -result.closest_normal)
    assert float(result.material_point[2].detach()) == pytest.approx(0)


def test_world_frame_rotation_translation_equivariance():
    binding, surface = fixture_surface(pinned=False)
    x, v = torch.tensor(binding.rest), torch.zeros((4, 3), dtype=torch.float64)
    p = torch.tensor([0.0043, 0.0041, 0.01299], dtype=torch.float64)
    rv = torch.tensor([0, 0, -0.01], dtype=torch.float64)
    rotation = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)
    shift = torch.tensor([0.02, -0.04, 0.03], dtype=torch.float64)
    a = surface.query(x, v, p, rv, 0.012)
    b = surface.query(
        x @ rotation.T + shift, v @ rotation.T, p @ rotation.T + shift, rv @ rotation.T, 0.012
    )
    torch.testing.assert_close(a.closest_gap, b.closest_gap)
    torch.testing.assert_close(a.closest_normal @ rotation.T, b.closest_normal)
    torch.testing.assert_close(a.closest_point @ rotation.T + shift, b.closest_point)


def test_reload_and_no_reference_or_native_queries(tmp_path):
    binding, surface = fixture_surface()
    binding.save(tmp_path / "binding.npz")
    reloaded = ContinuousContactSurface(BoundSurfaceGaussians.load(tmp_path / "binding.npz"))
    assert set(inspect.signature(surface.query).parameters) == {
        "positions",
        "velocities",
        "robot_position",
        "robot_velocity",
        "sphere_radius_m",
    }
    x, v = torch.tensor(binding.rest), torch.zeros((4, 3), dtype=torch.float64)
    p, rv = torch.tensor([0.0043, 0.0041, 0.01299], dtype=torch.float64), torch.zeros(3).double()
    a = surface.query(x, v, p, rv, 0.012)

    def forbidden(*args, **kwargs):
        raise AssertionError("Continuous query accessed an oracle")

    with patch("contact_gwm.surface_proxies.closest_triangle_surface", forbidden):
        b = reloaded.query(x, v, p, rv, 0.012)
    assert torch.equal(a.closest_gap, b.closest_gap)
    assert torch.equal(a.closest_normal, b.closest_normal)
    assert torch.equal(a.material_velocity, b.material_velocity)
    mj = pytest.importorskip("mujoco")
    with (
        patch.object(mj, "mj_fwdPosition", forbidden),
        patch.object(mj, "mj_forward", forbidden),
        patch.object(mj, "mj_contactForce", forbidden),
        patch.object(mj, "mj_step", forbidden),
    ):
        reloaded.query(x, v, p, rv, 0.012)


def test_degenerate_and_on_surface_states_are_rejected():
    binding, surface = fixture_surface()
    x, v = torch.tensor(binding.rest), torch.zeros((4, 3), dtype=torch.float64)
    rv = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="normal is undefined"):
        surface.query(x, v, torch.tensor([0.003, 0.003, 0.0]).double(), rv, 0.012)
    x[1] = x[0]
    with pytest.raises(ValueError, match="Degenerate"):
        surface.query(x, v, torch.tensor([0.003, 0.003, 0.013]).double(), rv, 0.012)


def test_numerical_tangency_band_does_not_hide_real_separation():
    binding, surface = fixture_surface()
    x, v = torch.tensor(binding.rest), torch.zeros((4, 3), dtype=torch.float64)
    rv = torch.zeros(3, dtype=torch.float64)
    for depth in (0.0, -1e-12, 1e-12):
        p = torch.tensor([0.0043, 0.0041, 0.013 - depth], dtype=torch.float64)
        query = surface.query(x, v, p, rv, 0.012)
        assert query.contacts.num_contacts == int(depth >= 0)
        assert query.roundoff_tolerance_m < 1e-16
