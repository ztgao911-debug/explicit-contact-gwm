from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_contact_mechanism_audit import audit_scene as audit_scene
from test_contact_mechanism_audit import touching_robot

from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.joint_state_dynamics import JointStatePipeline
from contact_gwm.repaired_contact_pipeline import (
    RepairedContactConditions,
    RepairedJointPipeline,
)


def make_conditions(legacy, mesh):
    return RepairedContactConditions(legacy, mesh, 500.0, ElasticAreaMaterial(6e7), {})


def forbidden(*a, **kw):
    raise AssertionError("Forbidden native solve")


@pytest.mark.parametrize(
    "method", ["analytic_force", "gaussian_geometry", "area_geometry", "sparse_geometry"]
)
def test_repaired_prediction_has_no_native_force_or_future_inputs(audit_scene, method):
    mj, _, mesh, _, legacy, model = audit_scene
    conditions = make_conditions(legacy, mesh)
    pipe = RepairedJointPipeline(legacy, model, conditions, method)
    state = legacy.gaussian_state(
        mesh["rest_positions_world"], np.zeros_like(mesh["rest_positions_world"])
    )
    assert set(inspect.signature(pipe.current_inputs).parameters) == {"state", "robot"}
    with (
        patch.object(mj, "mj_forward", forbidden),
        patch.object(mj, "mj_contactForce", forbidden),
        patch.object(mj, "mj_step", forbidden),
    ):
        result = pipe.step(state, touching_robot())
        if method != "analytic_force":
            with (
                patch.object(legacy.native, "query", forbidden),
                patch.object(mj, "mj_fwdPosition", forbidden),
            ):
                pipe.step(state, touching_robot())
    assert result.contacts.num_contacts > 0
    assert (
        torch.isfinite(result.state.centers).all() and torch.isfinite(result.state.velocities).all()
    )
    assert not torch.any(result.delta[legacy.pinned])


def test_native_route_preserves_historical_predictions_and_F_geometry(audit_scene):
    _, _, mesh, _, legacy, model = audit_scene
    conditions = make_conditions(legacy, mesh)
    state = legacy.gaussian_state(
        mesh["rest_positions_world"], np.zeros_like(mesh["rest_positions_world"])
    )
    old = JointStatePipeline(legacy, model).step(state, touching_robot())
    new = RepairedJointPipeline(legacy, model, conditions, "mujoco_force").step(
        state, touching_robot()
    )
    assert torch.equal(old.state.centers, new.state.centers)
    assert torch.equal(old.state.velocities, new.state.velocities)
    a, _ = conditions.condition(state, touching_robot(), "mujoco_force")
    b, _ = conditions.condition(state, touching_robot(), "analytic_force")
    for field in a.__dataclass_fields__:
        assert torch.equal(getattr(a, field), getattr(b, field))


def test_face_area_loads_preserve_net_force_and_torque_and_fixed_nodes(audit_scene):
    _, _, mesh, _, legacy, model = audit_scene
    conditions = make_conditions(legacy, mesh)
    state = legacy.gaussian_state(
        mesh["rest_positions_world"], np.zeros_like(mesh["rest_positions_world"])
    )
    dirty_x, dirty_v = state.centers.clone(), state.velocities.clone()
    dirty_x[legacy.pinned] += 0.1
    dirty_v[legacy.pinned] += 3.0
    dirty = replace(state, centers=dirty_x, velocities=dirty_v)
    for method in ["gaussian_geometry", "area_geometry"]:
        pipe = RepairedJointPipeline(legacy, model, conditions, method)
        a, b = pipe.step(state, touching_robot()), pipe.step(dirty, touching_robot())
        assert torch.equal(a.state.centers, b.state.centers)
        assert torch.equal(a.state.velocities, b.state.velocities)
        assert torch.equal(a.state.centers[legacy.pinned], legacy.rest[legacy.pinned])
        assert not torch.any(a.state.velocities[legacy.pinned])
        if method == "area_geometry":
            p = torch.tensor(touching_robot().position)
            origin = legacy.center_of_mass(state)
            torch.testing.assert_close(
                a.forces.torques.sum(0),
                torch.cross(p - origin, a.forces.forces.sum(0), dim=-1),
                rtol=1e-10,
                atol=1e-14,
            )
            assert a.contacts.object_indices.max() < len(mesh["node_ids"])


def test_face_loads_not_individual_integration_samples_and_reload(audit_scene, tmp_path):
    _, _, mesh, _, legacy, model = audit_scene
    state = legacy.gaussian_state(
        mesh["rest_positions_world"], np.zeros_like(mesh["rest_positions_world"])
    )
    a = make_conditions(legacy, mesh)
    b = RepairedContactConditions(
        legacy, mesh, 500.0, ElasticAreaMaterial(6e7), {"orders": [8, 16, 32]}
    )
    ca, fa = a.condition(state, touching_robot(), "area_geometry")
    cb, fb = b.condition(state, touching_robot(), "area_geometry")
    assert ca.num_contacts == cb.num_contacts
    assert ca.num_contacts < a.integration_diagnostics[-1]["evaluated_samples"]
    torch.testing.assert_close(fa.forces, fb.forces, rtol=2e-4, atol=1e-12)
    saved = tmp_path / "weights.pt"
    torch.save(model.state_dict(), saved)
    reloaded = copy.deepcopy(model)
    reloaded.load_state_dict(torch.load(saved, weights_only=True))
    for method in ["gaussian_geometry", "area_geometry"]:
        p = RepairedJointPipeline(legacy, model, a, method)
        q = RepairedJointPipeline(legacy, reloaded, a, method)
        x, y = p.rollout(state, [touching_robot()] * 3), q.rollout(state, [touching_robot()] * 3)
        assert all(
            torch.equal(i.state.centers, j.state.centers)
            and torch.equal(i.state.velocities, j.state.velocities)
            for i, j in zip(x, y, strict=True)
        )
