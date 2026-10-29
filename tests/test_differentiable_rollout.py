from __future__ import annotations

import copy
import inspect
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_contact_mechanism_audit import audit_scene as audit_scene
from test_contact_mechanism_audit import touching_robot

from contact_gwm.differentiable_rollout import (
    DifferentiableContactRollout,
    mesh_validity_penalty,
    signed_volume_ratios,
)
from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.repaired_contact_pipeline import (
    RepairedContactConditions,
    RepairedJointPipeline,
)


def make_rollout(audit_scene, *, model=None):
    mj, _, mesh, _, legacy, original_model = audit_scene
    model = original_model if model is None else model
    conditions = RepairedContactConditions(legacy, mesh, 500.0, ElasticAreaMaterial(6e7), {})
    old = RepairedJointPipeline(legacy, model, conditions, "mujoco_force")
    new = DifferentiableContactRollout(legacy, model, conditions, old.encoder)
    return mj, mesh, legacy, model, conditions, old, new


def inputs(legacy, *, requires_grad=False):
    x = legacy.rest.clone()
    x[~legacy.pinned, 2] -= 0.0001
    v = torch.zeros_like(x)
    v[~legacy.pinned, 2] = 0.0003
    return (
        x[None].requires_grad_(requires_grad),
        v[None].requires_grad_(requires_grad),
    )


def forbidden(*args, **kwargs):
    raise AssertionError("Training condition queried native time advancement")


def test_terminal_five_step_loss_reaches_first_position_and_velocity(audit_scene):
    mj, _, legacy, _, conditions, _, rollout = make_rollout(audit_scene)
    x, v = inputs(legacy, requires_grad=True)
    query = conditions.condition
    observations = []

    def observed(state, robot, method):
        assert not state.centers.requires_grad
        assert not state.velocities.requires_grad
        assert not state.features.requires_grad
        contacts, forces = query(state, robot, method)
        assert not contacts.points.requires_grad
        assert not forces.forces.requires_grad
        observations.append((state.centers.clone(), state.velocities.clone()))
        return contacts, forces

    with (
        patch.object(conditions, "condition", side_effect=observed),
        patch.object(mj, "mj_step", forbidden),
    ):
        states = rollout.rollout(x, v, [[touching_robot()]] * 5)
    assert len(observations) == rollout.query_count == 5
    first_x, first_v = states[1]
    end_x, end_v = states[-1]
    loss = ((end_x - legacy.rest[None]) / 0.001).square().mean()
    loss = loss + ((end_v - 0.002) / 0.1).square().mean()
    gx, gv = torch.autograd.grad(loss, (first_x, first_v), retain_graph=True)
    active = ~legacy.pinned
    assert torch.isfinite(gx).all() and torch.isfinite(gv).all()
    assert gx[:, active].abs().sum() > 0 and gv[:, active].abs().sum() > 0
    assert not gx[:, legacy.pinned].any() and not gv[:, legacy.pinned].any()
    # A velocity-only terminal loss cannot reach earlier x through the direct
    # position identity. This catches detached neural state/context features.
    velocity_to_x = torch.autograd.grad(end_v.square().sum(), first_x)[0]
    assert torch.isfinite(velocity_to_x).all()
    assert velocity_to_x[:, active].abs().sum() > 0
    for step, (next_x, next_v) in enumerate(states[1:], start=1):
        assert torch.equal(next_x[0, legacy.pinned], legacy.rest[legacy.pinned])
        assert not next_v[:, legacy.pinned].any()
        if step < 5:
            assert torch.equal(observations[step][0], next_x[0].detach())
            assert torch.equal(observations[step][1], next_v[0].detach())


def test_recomputed_features_and_context_keep_state_gradients(audit_scene):
    _, _, legacy, _, _, _, rollout = make_rollout(audit_scene)
    x, v = inputs(legacy, requires_grad=True)
    nx, nv = rollout.step(x, v, [touching_robot()])
    assert rollout.last_features.grad_fn is not None
    assert rollout.last_context.grad_fn is not None
    probe = rollout.last_features[..., 3:9].square().sum()
    gx, gv = torch.autograd.grad(probe, (x, v), retain_graph=True)
    assert gx[:, ~legacy.pinned].abs().sum() > 0
    assert gv[:, ~legacy.pinned].abs().sum() > 0
    # The velocity increment has no direct x identity path.
    neural_gradient = torch.autograd.grad((nv - v).square().sum(), x)[0]
    assert torch.isfinite(neural_gradient).all() and neural_gradient.abs().sum() > 0
    assert nx.grad_fn is not None and nv.grad_fn is not None


def test_attached_one_step_matches_existing_inference_bitwise(audit_scene):
    _, _, legacy, _, _, old, rollout = make_rollout(audit_scene)
    x, v = inputs(legacy)
    state = legacy.gaussian_state(x[0].numpy(), v[0].numpy())
    expected = old.step(state, touching_robot())
    actual_x, actual_v = rollout.step(x, v, [touching_robot()])
    assert torch.equal(actual_x[0], expected.state.centers)
    assert torch.equal(actual_v[0], expected.state.velocities)


def test_instantaneous_velocity_is_an_independent_output_not_a_secant(audit_scene):
    _, _, legacy, model, _, _, rollout = make_rollout(audit_scene)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.contact_network[4].bias[3:] = 0.2
    x = legacy.rest[None].clone()
    v = torch.zeros_like(x)
    nx, nv = rollout.step(x, v, [touching_robot()])
    assert torch.equal(nx, x)
    assert torch.any(nv[:, ~legacy.pinned] != 0)
    assert not torch.equal(nv, (nx - x) / 0.01)


def test_reload_twenty_steps_and_explicit_autoregressive_interface(audit_scene, tmp_path):
    _, _, legacy, model, _, _, rollout = make_rollout(audit_scene)
    # Small nonzero heads keep this structural replay test inside a mild state
    # region; the formal runtime preflight uses the actual fixed search caps.
    with torch.no_grad():
        model.internal_network[4].weight.mul_(0.05)
        model.contact_network[4].weight.mul_(0.05)
    artifact = tmp_path / "weights.pt"
    torch.save(model.state_dict(), artifact)
    reloaded = copy.deepcopy(model)
    reloaded.load_state_dict(torch.load(artifact, weights_only=True))
    *_, another = make_rollout(audit_scene, model=reloaded)
    x, v = inputs(legacy)
    external = [[touching_robot()]] * 20
    with torch.no_grad():
        a = rollout.rollout(x, v, external)
        b = another.rollout(x, v, external, teacher_states=None)
    assert all(
        torch.equal(ax, bx) and torch.equal(av, bv) for (ax, av), (bx, bv) in zip(a, b, strict=True)
    )
    assert set(inspect.signature(rollout.rollout).parameters) == {
        "initial_x",
        "initial_v",
        "external_steps",
        "teacher_states",
    }
    with pytest.raises(TypeError):
        rollout.rollout(x, v, external, future_forces=np.full((20, 3), np.nan))
    with pytest.raises(ValueError):
        rollout.rollout(x, v, external, teacher_states=[(x, v)])


def test_teacher_forcing_is_explicit_and_uses_the_five_prescribed_current_states(audit_scene):
    _, _, legacy, _, conditions, _, rollout = make_rollout(audit_scene)
    x, v = inputs(legacy)
    truth = [(x + i * 1e-5, v) for i in range(5)]
    observed = []
    original = conditions.condition

    def query(state, robot, method):
        observed.append(state.centers.clone())
        return original(state, robot, method)

    with patch.object(conditions, "condition", side_effect=query):
        rollout.rollout(x, v, [[touching_robot()]] * 5, teacher_states=truth)
    for source, actual in zip(truth, observed, strict=True):
        expected, _ = rollout.constrain(*source)
        assert torch.equal(expected[0], actual)


def test_signed_volume_penalty_has_correct_orientation_and_gradient():
    rest = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    tetrahedra = torch.tensor([[0, 1, 2, 3]])
    rest_volumes = torch.tensor([1 / 6], dtype=torch.float64)
    torch.testing.assert_close(
        signed_volume_ratios(rest, tetrahedra, rest_volumes), torch.ones(1).double()
    )
    assert mesh_validity_penalty(rest, tetrahedra, rest_volumes) == 0
    inverted = rest.clone()
    inverted[3, 2] = -0.2
    inverted.requires_grad_()
    penalty = mesh_validity_penalty(inverted, tetrahedra, rest_volumes)
    assert float(penalty.detach()) == pytest.approx(0.09)
    gradient = torch.autograd.grad(penalty, inverted)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert float(gradient[3, 2]) == pytest.approx(-0.6)
