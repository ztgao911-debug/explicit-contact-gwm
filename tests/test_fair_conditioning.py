from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_contact_mechanism_audit import audit_scene as audit_scene
from test_contact_mechanism_audit import touching_robot

from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.fair_conditioning import RawJointPipeline, training_path
from contact_gwm.repaired_contact_pipeline import RepairedContactConditions, RepairedJointPipeline


def forbidden(*args, **kwargs):
    raise AssertionError("Forward leaked a native query or forbidden contact condition")


def current(legacy):
    x = legacy.rest.clone()
    x[~legacy.pinned, 2] -= 0.0001
    v = torch.zeros_like(x)
    v[~legacy.pinned, 2] = 0.0003
    return x, v


def test_raw_sees_actual_tool_and_command_without_native_or_contact_queries(audit_scene):
    _, _, _, _, legacy, model = audit_scene
    pipeline = RawJointPipeline(legacy, model)
    x, v = current(legacy)
    state, robot = legacy.gaussian_state(x.numpy(), v.numpy()), touching_robot()
    with patch.object(legacy.native, "query", forbidden):
        base = pipeline.current_inputs(state, robot)
        changed_actual = pipeline.current_inputs(
            state, replace(robot, velocity=np.asarray(robot.velocity) + [0, 0, 0.01])
        )
        changed_command = pipeline.current_inputs(
            state, replace(robot, target_position=np.asarray(robot.target_position) + [0.001, 0, 0])
        )
    assert base.context.shape == (len(x), 36)
    assert torch.all(base.gates == 1)
    assert not torch.equal(base.context, changed_actual.context)
    assert not torch.equal(base.context, changed_command.context)
    assert base.contacts.num_contacts == 0 and len(base.forces.forces) == 0


@pytest.mark.parametrize("variant", ["RAW", "G"])
def test_both_paths_keep_three_step_gradient_and_match_inference(audit_scene, variant):
    _, _, mesh, _, legacy, model = audit_scene
    conditions = RepairedContactConditions(legacy, mesh, 500.0, ElasticAreaMaterial(6e7), {})
    pipeline = (
        RawJointPipeline(legacy, model)
        if variant == "RAW"
        else RepairedJointPipeline(legacy, model, conditions, "gaussian_geometry")
    )
    path = training_path(variant, legacy, model, conditions, pipeline.encoder)
    x, v = current(legacy)
    robot = touching_robot()
    with patch.object(legacy.native, "query", forbidden):
        states = path.rollout(
            x[None].requires_grad_(True), v[None].requires_grad_(True), [[robot]] * 3
        )
        result = pipeline.step(legacy.gaussian_state(x.numpy(), v.numpy()), robot)
    assert torch.equal(states[1][0][0].detach(), result.state.centers)
    assert torch.equal(states[1][1][0].detach(), result.state.velocities)
    loss = (states[-1][0] - legacy.rest[None]).square().sum() + states[-1][1].square().sum()
    gx, gv = torch.autograd.grad(loss, states[1])
    assert torch.isfinite(gx).all() and torch.isfinite(gv).all()
    assert gx.norm() > 0 and gv.norm() > 0
    for px, pv in states:
        assert torch.equal(px[0, legacy.pinned], legacy.rest[legacy.pinned])
        assert not pv[:, legacy.pinned].any()
    with pytest.raises(TypeError):
        path.rollout(x[None], v[None], [[robot]], future_forces=np.zeros((20, 3)))


def test_raw_encoder_does_not_inspect_force_or_geometry_objects(audit_scene):
    _, _, _, _, legacy, model = audit_scene
    pipeline = RawJointPipeline(legacy, model)
    x, v = current(legacy)
    state = legacy.gaussian_state(x.numpy(), v.numpy())
    action = torch.zeros(12, dtype=state.dtype)

    class Inaccessible:
        def __getattribute__(self, name):
            raise AssertionError("Privileged condition read by raw encoder")

    a, gate = pipeline.encoder.encode_context(state, Inaccessible(), Inaccessible(), action, None)
    b, _ = pipeline.encoder.encode_context(state, None, None, action, None)
    assert torch.equal(a, b) and torch.all(gate == 1)


def test_raw_saved_weights_reload_twenty_steps_without_boundary_drift(audit_scene, tmp_path):
    _, _, _, _, legacy, model = audit_scene
    checkpoint = tmp_path / "raw.pt"
    torch.save(model.state_dict(), checkpoint)
    loaded = copy.deepcopy(model)
    loaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    pipes = [RawJointPipeline(legacy, m) for m in (model, loaded)]
    x, v = current(legacy)
    states = [legacy.gaussian_state(x.numpy(), v.numpy()) for _ in pipes]
    with patch.object(legacy.native, "query", forbidden):
        for _ in range(20):
            states = [p.step(s, touching_robot()).state for p, s in zip(pipes, states, strict=True)]
            assert torch.equal(states[0].centers, states[1].centers)
            assert torch.equal(states[0].velocities, states[1].velocities)
            assert torch.equal(states[0].centers[legacy.pinned], legacy.rest[legacy.pinned])
            assert not states[0].velocities[legacy.pinned].any()
