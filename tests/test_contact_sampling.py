from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from contact_gwm.contact_force import ContactMaterial, compute_contact_forces
from contact_gwm.contact_sampling import (
    SurfaceQuadrature,
    group_material_contacts,
    material_samples,
)
from contact_gwm.pinned_contact_pipeline import IndenterInput, NativeCurrentContact
from contact_gwm.structures import ContactGeometry, GaussianSet


def square():
    return np.array(
        [[-0.02, -0.02, 0], [0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, 0.02, 0]]
    ), np.array([[0, 1, 2], [1, 3, 2]])


def pair():
    gap = torch.tensor([-0.0001, -0.0001], dtype=torch.float64)
    return ContactGeometry(
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([[0, 0, 0], [0, 0, 0]]).double(),
        torch.tensor([[0, 0, -1], [0, 0, -1]]).double(),
        gap,
        -gap,
        torch.tensor([[0, 0, -0.01], [0, 0, -0.01]]).double(),
    )


def test_duplicate_material_witness_grouping_exposes_repeated_force():
    contacts = pair()
    grouped = group_material_contacts(
        contacts,
        torch.tensor([0, 1]),
        torch.tensor([[0, 0.5, 0.5], [0.5, 0, 0.5]]).double(),
        torch.tensor([[0, 1, 2], [1, 3, 2]]),
    )
    assert grouped.contacts.num_contacts == 1 and grouped.multiplicities.tolist() == [2]
    material, origin = ContactMaterial(), torch.zeros(3).double()
    a = compute_contact_forces(contacts, origin, material).forces.sum(0)
    b = compute_contact_forces(grouped.contacts, origin, material).forces.sum(0)
    torch.testing.assert_close(a, 2 * b)


def test_coincident_disconnected_material_or_different_velocity_must_not_merge():
    contacts = pair()
    grouped = group_material_contacts(
        contacts,
        torch.tensor([0, 1]),
        torch.tensor([[1, 0, 0], [1, 0, 0]]).double(),
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    assert grouped.contacts.num_contacts == 2
    changed = contacts.relative_velocities.clone()
    changed[1, 0] += 0.001
    grouped = group_material_contacts(
        replace(contacts, relative_velocities=changed),
        torch.tensor([0, 0]),
        torch.tensor([[1, 0, 0], [1, 0, 0]]).double(),
        torch.tensor([[0, 1, 2]]),
    )
    assert grouped.contacts.num_contacts == 2


def test_material_velocity_is_not_motion_of_the_closest_point():
    rest, tri = square()
    x, v = torch.tensor(rest), torch.zeros((4, 3)).double()
    sample = material_samples(
        x, v, torch.tensor(tri), torch.tensor([[0.001, 0.002, 0.0002]]).double()
    )
    assert torch.count_nonzero(sample.velocities) == 0
    torch.testing.assert_close(sample.points, torch.tensor([[0.001, 0.002, 0.0]]).double())
    assert sample.projection_errors.item() == pytest.approx(0.0002)
    empty = material_samples(x, v, torch.tensor(tri), x.new_empty((0, 3)))
    assert empty.velocities.shape == (0, 3)


def test_area_quadrature_matches_closed_form_flat_plane_normal_integral():
    rest, tri = square()
    rule = SurfaceQuadrature(rest, tri, np.zeros(4, bool), 64)
    x, v = torch.tensor(rest), torch.zeros((4, 3)).double()
    radius, skin, depth = 0.012, 0.001, 0.001
    height = radius + skin - depth
    result = rule.normal_bases(
        x,
        v,
        torch.tensor([0, 0, height]).double(),
        torch.zeros(3).double(),
        radius,
        skin,
        torch.zeros(3).double(),
    )
    expected = -np.pi * height * depth**2
    assert result["elastic_vector_m3"][2].item() == pytest.approx(expected, rel=0.01)
    assert result["elastic_vector_m3"][:2].abs().max() < 1e-15
    assert result["total_surface_area_m2"].item() == pytest.approx(0.04**2)


def test_same_surface_tessellation_with_same_physical_quadrature_gives_same_integral():
    rest = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0]])
    subdivided = np.vstack(
        [rest, (rest[0] + rest[1]) / 2, (rest[1] + rest[2]) / 2, (rest[2] + rest[0]) / 2]
    )
    faces = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
    results = []
    for x, tri, n in [(rest, np.array([[0, 1, 2]]), 8), (subdivided, faces, 4)]:
        rule = SurfaceQuadrature(x, tri, np.zeros(len(x), bool), n)
        results.append(
            rule.normal_bases(
                torch.tensor(x),
                torch.zeros((len(x), 3)).double(),
                torch.tensor([0.005, 0.005, 0.012]).double(),
                torch.zeros(3).double(),
                0.012,
                0.001,
                torch.zeros(3).double(),
            )
        )
    torch.testing.assert_close(results[0]["elastic_vector_m3"], results[1]["elastic_vector_m3"])


def test_current_area_fixed_boundary_and_finite_quadrature_gradients():
    rest, tri = square()
    pin = np.array([True, False, False, False])
    rule = SurfaceQuadrature(rest, tri, pin, 8)
    x = torch.tensor(rest)
    x[0] += 0.1
    x.requires_grad_()
    v = torch.ones_like(x).requires_grad_()
    result = rule.normal_bases(
        x,
        v,
        torch.tensor([0, 0, 0.012]).double(),
        torch.zeros(3).double(),
        0.012,
        0.001,
        torch.zeros(3).double(),
    )
    assert result["total_surface_area_m2"].item() == pytest.approx(0.04**2)
    grads = torch.autograd.grad(
        result["elastic_vector_m3"].sum() + result["damping_vector_m3_s"].sum(), (x, v)
    )
    assert all(torch.isfinite(g).all() for g in grads)
    assert not torch.any(grads[0][0]) and not torch.any(grads[1][0])
    with pytest.raises(ValueError, match="Duplicate"):
        SurfaceQuadrature(rest, np.vstack([tri, tri[0]]), pin, 4)


@pytest.fixture
def native_scene(tmp_path):
    mj = pytest.importorskip("mujoco")
    from contact_gwm.data.mujoco_soft import build_xml, mesh_from_model

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/mujoco_soft_pilot.yaml").read_text())
    scene = tmp_path / "scene.xml"
    scene.write_text(build_xml(config))
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    mj.mj_fwdPosition(model, data)
    mesh = mesh_from_model(model, data)
    native = NativeCurrentContact(scene, mesh)
    x = torch.tensor(mesh["rest_positions_world"])
    v = torch.stack([5 * x[:, 0], -4 * x[:, 1], x[:, 0] * 0 - 0.03], dim=-1) * x[:, 2, None] / 0.03
    q = torch.zeros((len(x), 4)).double()
    q[:, 0] = 1
    state = GaussianSet(
        x,
        x.new_full(x.shape, 0.005).log(),
        q,
        x.new_zeros((len(x), 1)),
        x.new_zeros((len(x), 0)),
        v,
    )
    p = np.array([0.003, 0.002, 0.0429])
    robot = IndenterInput(p, np.array([0, 0, -0.01]), p.copy(), np.zeros(3))
    return mj, native, mesh, state, robot


def test_opt_in_material_velocity_uses_same_geometry_and_legacy_default_is_unchanged(native_scene):
    mj, native, mesh, state, robot = native_scene

    def forbidden(*a, **kw):
        raise AssertionError("Force solver called")

    with patch.object(mj, "mj_forward", forbidden), patch.object(mj, "mj_contactForce", forbidden):
        default, _ = native.query(state, robot, with_force=False)
        explicit, _ = native.query(
            state, robot, with_force=False, velocity_mode="legacy_midpoint_tetra"
        )
        aligned, _ = native.query(state, robot, with_force=False, velocity_mode="material_surface")
    assert torch.equal(default.relative_velocities, explicit.relative_velocities)
    assert torch.equal(default.points, aligned.points) and torch.equal(
        default.normals, aligned.normals
    )
    assert aligned.num_contacts > 0
    material = aligned.points + (0.001 + 0.5 * aligned.signed_gaps[:, None]) * aligned.normals
    expected = torch.stack(
        [5 * material[:, 0], -4 * material[:, 1], material[:, 0] * 0 - 0.03], dim=-1
    )
    torch.testing.assert_close(
        aligned.relative_velocities, torch.tensor(robot.velocity) - expected, atol=1e-10, rtol=0
    )
    with pytest.raises(ValueError):
        native.query(state, robot, with_force=False, velocity_mode="future")


def test_material_velocity_matches_independent_mujoco_node_jacobians(native_scene):
    mj, native, mesh, state, robot = native_scene
    contacts, _ = native.query(state, robot, with_force=False, velocity_mode="material_surface")
    points = contacts.points + (0.001 + 0.5 * contacts.signed_gaps[:, None]) * contacts.normals
    v = state.velocities.clone()
    v[mesh["pinned_mask"]] = 0
    sample = material_samples(state.centers, v, torch.tensor(mesh["surface_triangles"]), points)
    jac = np.zeros((3, native.model.nv))
    nodal = []
    for i, body in enumerate(mesh["vertex_body_ids"]):
        mj.mj_jac(native.model, native.data, jac, None, state.centers[i].numpy(), int(body))
        nodal.append(jac @ native.data.qvel)
    nodal = torch.tensor(np.asarray(nodal))
    independent = (
        nodal[torch.tensor(mesh["surface_triangles"])[sample.face_indices]]
        * sample.weights[..., None]
    ).sum(1)
    torch.testing.assert_close(sample.velocities, independent, atol=1e-14, rtol=0)
