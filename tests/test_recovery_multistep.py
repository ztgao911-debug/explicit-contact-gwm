"""Differentiable recurrence, matched supervision and independent velocity."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
import torch

from contact_gwm.neighbor_dynamics import make_neighbor_model
from contact_gwm.recovery_multistep import (
    autoregressive_recovery,
    joint_endpoint_loss,
    no_contact_step,
    overlapping_indices,
    teacher_forced_recovery,
)

NORM = {"rest_position": 0.03, "displacement": 0.01, "velocity": 0.05}
TRAINING = {
    "position_loss_unit_m": 0.001,
    "velocity_loss_unit_m_s": 0.1,
    "neighborhood_weight": 0.5,
    "velocity_weight": 0.5,
}


def scene(nonzero_head=False):
    torch.manual_seed(7)
    mesh = {
        "rest_positions_world": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.01, 0.0, 0.01]]),
        "pinned_mask": np.array([True, False, False]),
        "edges": np.array([[0, 1], [1, 2]]),
    }
    config = {
        "dt": 0.01,
        "model": {
            "hidden_dim": 8,
            "contact_sigma": 0.02,
            "max_position_residual": 0.001,
            "max_velocity_increment": 0.1,
        },
    }
    model = make_neighbor_model("B", mesh, config)
    if nonzero_head:
        with torch.no_grad():
            model.internal_network[4].weight.normal_(0, 0.01)
    rest = torch.tensor(mesh["rest_positions_world"], dtype=torch.float64)
    pinned = torch.tensor(mesh["pinned_mask"])
    x, v = rest.clone(), torch.zeros_like(rest)
    x[1, 2] -= 0.0001
    v[1:, 2] = torch.tensor([0.003, -0.002], dtype=v.dtype)
    return model, rest, pinned, x, v, mesh, config


def test_overlapping_target_occurrences_match_both_training_arms():
    current, target = overlapping_indices(21, 5)
    assert current.shape == (16, 5)
    assert torch.equal(target, current + 1)
    assert current[0].tolist() == [0, 1, 2, 3, 4]
    assert target[-1].tolist() == [16, 17, 18, 19, 20]
    assert target.numel() == 80 and len(target.unique()) == 20
    with pytest.raises(ValueError):
        overlapping_indices(5, 5)


def test_zero_head_velocity_and_fixed_boundary_survive_long_recurrence():
    model, rest, pin, x, v, *_ = scene()
    px, pv = autoregressive_recovery(model, x, v, rest, pin, NORM, 20)
    expected = x[None] + torch.arange(1, 21)[:, None, None] * 0.01 * v[None]
    torch.testing.assert_close(px[:, ~pin], expected[:, ~pin], atol=5e-10, rtol=0)
    torch.testing.assert_close(pv, v.expand_as(pv), atol=0, rtol=0)
    assert torch.equal(px[:, pin], rest[pin].expand(20, -1, -1))
    assert torch.count_nonzero(pv[:, pin]) == 0


def test_teacher_forcing_and_autoregression_are_distinct_training_conditions():
    model, rest, pin, x, v, *_ = scene()
    cx = x.expand(2, 5, -1, -1).clone()
    cv = v.expand_as(cx).clone()
    cx[:, 1:, 1, 2] += 0.0005
    teacher_x, teacher_v = teacher_forced_recovery(model, cx, cv, rest, pin, NORM)
    free_x, free_v = autoregressive_recovery(model, cx[:, 0], cv[:, 0], rest, pin, NORM, 5)
    assert torch.equal(teacher_x[:, 0], free_x[:, 0])
    assert not torch.equal(teacher_x[:, 1:], free_x[:, 1:])
    assert torch.equal(teacher_v, free_v)


def test_last_step_gradient_reaches_initial_state_and_model_parameters():
    model, rest, pin, x, v, *_ = scene(nonzero_head=True)
    x.requires_grad_()
    v.requires_grad_()
    px, pv = autoregressive_recovery(model, x, v, rest, pin, NORM, 5)
    loss = px[-1, ~pin].square().sum() + pv[-1, ~pin].square().sum()
    loss.backward()
    assert x.grad is not None and torch.count_nonzero(x.grad[~pin]) > 0
    assert v.grad is not None and torch.count_nonzero(v.grad[~pin]) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.internal_network[0].weight.grad[:, 28:]) > 0
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.contact_network.parameters())


def test_batched_and_individual_recurrence_agree_and_reload_is_exact():
    model, rest, pin, x, v, *_ = scene(nonzero_head=True)
    bx, bv = x.expand(2, -1, -1).clone(), v.expand(2, -1, -1).clone()
    bx[1, 2, 2] += 0.0002
    full = autoregressive_recovery(model, bx, bv, rest, pin, NORM, 5)
    for i in range(2):
        single = autoregressive_recovery(model, bx[i], bv[i], rest, pin, NORM, 5)
        for batched, individual in zip(full, single, strict=True):
            torch.testing.assert_close(batched[i], individual)
    restored = copy.deepcopy(model)
    actual = autoregressive_recovery(restored, bx, bv, rest, pin, NORM, 5)
    assert all(torch.equal(a, b) for a, b in zip(actual, full, strict=True))


def test_endpoint_loss_units_and_region_denominators():
    _, rest, pin, *_ = scene()
    true_x, true_v = torch.zeros((2, 5, len(rest), 3)), torch.zeros((2, 5, len(rest), 3))
    near = torch.tensor([False, True, False])
    total, components = joint_endpoint_loss(
        true_x + 0.001, true_v + 0.1, true_x, true_v, pin, near, TRAINING
    )
    torch.testing.assert_close(total, torch.tensor(1.0))
    assert all(torch.isclose(c, torch.tensor(1.0)) for c in components)


def test_inference_contract_never_accepts_future_or_supervision_arrays():
    model, rest, pin, x, v, *_ = scene(nonzero_head=True)
    assert set(inspect.signature(autoregressive_recovery).parameters) == {
        "model",
        "initial_positions",
        "initial_velocities",
        "rest",
        "pinned",
        "normalization",
        "steps",
    }
    labels_x, labels_v = x.expand(21, -1, -1).clone(), v.expand(21, -1, -1).clone()
    before = autoregressive_recovery(model, labels_x[0], labels_v[0], rest, pin, NORM, 5)
    labels_x[1:] = float("nan")
    labels_v[1:] = float("nan")
    after = autoregressive_recovery(model, labels_x[0], labels_v[0], rest, pin, NORM, 5)
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_nonfinite_predicted_state_is_rejected():
    model, rest, pin, x, v, *_ = scene()
    x[1, 2] = float("nan")
    with pytest.raises(FloatingPointError):
        no_contact_step(model, x, v, rest, pin, NORM)
