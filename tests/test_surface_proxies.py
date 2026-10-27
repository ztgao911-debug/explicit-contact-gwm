from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from contact_gwm.surface_proxies import BoundSurfaceGaussians, closest_triangle_surface


def surface(step=0.002):
    # Two coplanar triangles with a shared diagonal and four dynamics nodes.
    return BoundSurfaceGaussians(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0]]),
        np.array([[0, 1, 2], [1, 3, 2]]),
        np.array([True, False, False, False]),
        max_rest_edge_m=step,
    )


def test_binding_is_conforming_deduplicated_and_retains_original_nodes():
    dense = surface()
    assert dense.subdivisions == 8 and len(dense.parents) == 81
    assert dense.actual_max_rest_edge_m <= 0.002
    assert np.all(dense.weights >= 0) and np.allclose(dense.weights.sum(-1), 1)
    cloud = dense.cloud(torch.tensor(dense.rest), torch.zeros((4, 3)).double())
    assert len(np.unique(cloud.centers.numpy(), axis=0)) == 81
    for node in dense.rest:
        assert np.min(np.linalg.norm(cloud.centers.numpy() - node, axis=-1)) < 1e-15
    sparse = surface(None)
    assert len(sparse.parents) == 4
    assert torch.allclose(cloud.scales, torch.full((81, 3), 0.005).double())


def test_dense_surface_detects_cell_center_where_sparse_misses_without_radius_inflation():
    dense, sparse = surface(), surface(None)
    x, v = torch.tensor(dense.rest), torch.zeros((4, 3)).double()
    p = torch.tensor([0.005, 0.005, 0.0129], dtype=torch.float64)
    rv = torch.zeros(3, dtype=torch.float64)
    old, new = sparse.query(x, v, p, rv, 0.012), dense.query(x, v, p, rv, 0.012)
    assert old.contacts.num_contacts == 0 and new.contacts.num_contacts > 0
    assert new.closest_gap == pytest.approx(-0.0001)
    assert torch.all(new.contacts.normals[:, 2] < 0)
    away = dense.query(x, v, p + torch.tensor([0, 0, 0.001]), rv, 0.012)
    assert away.contacts.num_contacts == 0


def test_affine_current_deformation_and_instantaneous_velocity_are_interpolated():
    dense = surface()
    x = torch.tensor(dense.rest)
    v = 2 * x
    x = x.clone()
    x[1:, 2] = x[1:, 0] + 2 * x[1:, 1]
    cloud = dense.cloud(x, v)
    expected_x = (x[dense.parents] * torch.tensor(dense.weights)[..., None]).sum(1)
    expected_v = (v[dense.parents] * torch.tensor(dense.weights)[..., None]).sum(1)
    assert torch.equal(cloud.centers, expected_x) and torch.equal(cloud.velocities, expected_v)
    assert cloud.num_gaussians > len(x)


def test_fixed_boundary_no_input_mutation_and_finite_gradients():
    dense = surface()
    x = torch.tensor(dense.rest).requires_grad_()
    v = torch.ones_like(x).requires_grad_()
    with torch.no_grad():
        x[0] += 0.1
    before = x.detach().clone()
    cloud = dense.cloud(x, v)
    # Find the proxy whose complete support is the pinned vertex, regardless of
    # which slot the original triangle assigned to it.
    fixed = np.all(dense.pinned[dense.parents] | (dense.weights == 0), axis=-1)
    assert np.count_nonzero(fixed) == 1
    assert torch.equal(cloud.centers[fixed], torch.zeros((1, 3)).double())
    assert not torch.any(cloud.velocities[fixed])
    p = torch.tensor([0.005, 0.005, 0.0129], requires_grad=True, dtype=torch.float64)
    query = dense.query(x, v, p, torch.zeros(3).double(), 0.012)
    gradients = torch.autograd.grad(query.closest_gap + cloud.velocities.square().sum(), (x, v, p))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert not torch.any(gradients[0][0]) and not torch.any(gradients[1][0])
    assert torch.equal(x.detach(), before)


def test_binding_reload_reproduces_all_contact_fields_bitwise(tmp_path):
    dense = surface()
    path = tmp_path / "binding.npz"
    dense.save(path)
    restored = BoundSurfaceGaussians.load(path)
    x, v = torch.tensor(dense.rest), torch.zeros((4, 3)).double()
    p, rv = torch.tensor([0.005, 0.005, 0.0129]).double(), torch.zeros(3).double()
    a, b = dense.query(x, v, p, rv, 0.012), restored.query(x, v, p, rv, 0.012)
    assert torch.equal(a.closest_gap, b.closest_gap)
    for key in ("points", "normals", "signed_gaps", "relative_velocities", "object_indices"):
        assert torch.equal(getattr(a.contacts, key), getattr(b.contacts, key))


def test_continuous_reference_distance_for_face_edge_and_vertex():
    dense = surface()
    for center, expected in (
        ([0.005, 0.005, 0.013], 0.0),
        ([-0.003, 0.005, 0.004], 0.005 - 0.013),
        ([-0.003, -0.004, 0.012], 0.013 - 0.013),
    ):
        gap, _, normal = closest_triangle_surface(
            dense.rest, dense.triangles, np.array(center), 0.012, 0.001
        )
        assert gap == pytest.approx(expected, abs=1e-15)
        assert np.linalg.norm(normal) == pytest.approx(1.0)


def load_entrypoint():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "surface_entry", scripts / "validate_contact_surface.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(scripts), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_geometry_inputs_have_no_future_contact_force_or_native_solver_dependency():
    pytest.importorskip("mujoco")
    entry = load_entrypoint()
    dense = surface()
    arrays = {
        k: np.ones((3, 4, 3)) if k.startswith("node_") else np.ones((3, 3))
        for k in entry.CURRENT_FIELDS
    }
    current = entry.current_inputs(arrays, 0)
    for values in arrays.values():
        values[1:] = np.nan
    arrays["contact_force_on_block_world"] = np.full((3, 3), np.nan)
    changed = entry.current_inputs(arrays, 0)
    assert all(torch.equal(a, b) for a, b in zip(current, changed, strict=True))
    assert set(inspect.signature(dense.query).parameters) == {
        "positions",
        "velocities",
        "robot_position",
        "robot_velocity",
        "sphere_radius_m",
    }
    import mujoco

    def forbidden(*args, **kwargs):
        raise AssertionError("Gaussian query used native state/force")

    x, v = torch.tensor(dense.rest), torch.zeros((4, 3)).double()
    with (
        patch.object(mujoco, "mj_fwdPosition", forbidden),
        patch.object(mujoco, "mj_forward", forbidden),
        patch.object(mujoco, "mj_contactForce", forbidden),
        patch.object(mujoco, "mj_step", forbidden),
    ):
        result = dense.query(
            x, v, torch.tensor([0.005, 0.005, 0.0129]).double(), torch.zeros(3).double(), 0.012
        )
    assert result.contacts.num_contacts > 0


@pytest.mark.parametrize("step", [0.0, -1.0, float("nan")])
def test_invalid_sampling_step_is_rejected(step):
    with pytest.raises(ValueError):
        surface(step)


def test_nonfinite_current_state_is_rejected():
    dense = surface()
    x = torch.tensor(dense.rest)
    x[1, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        dense.cloud(x, torch.zeros_like(x))
