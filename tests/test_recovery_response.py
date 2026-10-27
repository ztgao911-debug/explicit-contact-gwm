"""Signed physical response supervision without future inputs or pin motion."""

from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from contact_gwm.neighbor_dynamics import make_neighbor_model
from contact_gwm.recovery_multistep import autoregressive_recovery
from contact_gwm.recovery_response import (
    central_response,
    paired_current_step,
    response_matching_loss,
)
from contact_gwm.recovery_stability import make_perturbations

NORM = {"rest_position": 0.03, "displacement": 0.01, "velocity": 0.05}
TRAINING = {
    "position_loss_unit_m": 0.001,
    "velocity_loss_unit_m_s": 0.1,
    "neighborhood_weight": 0.5,
    "velocity_weight": 0.5,
}


@pytest.fixture
def scene():
    torch.manual_seed(7)
    mesh = {
        "rest_positions_world": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.01, 0.0, 0.01]]),
        "pinned_mask": np.array([True, False, False]),
        "edges": np.array([[0, 1], [1, 2]]),
    }
    config = {"dt": 0.01, "model": {"hidden_dim": 8}}
    model = make_neighbor_model("B", mesh, config)
    rest = torch.tensor(mesh["rest_positions_world"])
    pin = torch.tensor(mesh["pinned_mask"])
    x = rest.expand(2, 2, -1, -1).clone()
    v = torch.zeros_like(x)
    # Pair 0 perturbs position; pair 1 velocity, always signs [-,+].
    x[:, :, 1:, 2] -= 0.0001
    x[0, 0, 1:, 2] -= 1e-6
    x[0, 1, 1:, 2] += 1e-6
    v[1, 0, 1:, 2] = -1e-4
    v[1, 1, 1:, 2] = 1e-4
    return model, rest, pin, x, v, mesh, config


def test_zero_initialized_model_matches_cv_signed_response_and_keeps_pins(scene):
    model, rest, pin, x, v, *_ = scene
    px, pv = paired_current_step(model, x, v, rest, pin, NORM)
    tx, tv = central_response(x + 0.01 * v), central_response(v)
    loss, _ = response_matching_loss(
        px,
        pv,
        tx,
        tv,
        torch.full((2,), 0.001),
        pin,
        ~pin,
        TRAINING,
    )
    assert float(loss.detach()) < 1e-12
    assert torch.equal(px[..., pin, :], rest[pin].expand_as(px[..., pin, :]))
    assert not torch.any(pv[..., pin, :])
    # The velocity perturbation has gain sqrt(2)>1 after one step. An exactly
    # matched physical response must still have ZERO matching loss.
    assert torch.linalg.vector_norm(central_response(px)[1]) > 0


def test_response_loss_preserves_direction_instead_of_matching_scalar_gain():
    paired = torch.zeros((1, 2, 2, 3), dtype=torch.float64)
    paired[0, 0, 1, 2], paired[0, 1, 1, 2] = -1e-6, 1e-6
    response = central_response(paired)
    zero = torch.zeros_like(paired)
    pin = torch.tensor([True, False])
    same, _ = response_matching_loss(
        paired,
        zero,
        response,
        central_response(zero),
        torch.tensor([0.001]),
        pin,
        ~pin,
        TRAINING,
    )
    opposite, _ = response_matching_loss(
        paired,
        zero,
        -response,
        central_response(zero),
        torch.tensor([0.001]),
        pin,
        ~pin,
        TRAINING,
    )
    assert same == 0 and opposite > 0


def test_response_normalization_is_invariant_to_linear_amplitude_rescaling():
    pin = torch.tensor([True, False])
    pair = torch.zeros((1, 2, 2, 3), dtype=torch.float64)
    pair[0, 0, 1, 2], pair[0, 1, 1, 2] = -1e-6, 1e-6
    zeros = torch.zeros_like(pair)

    def loss(factor):
        return response_matching_loss(
            pair * factor,
            zeros,
            torch.zeros_like(central_response(pair)),
            torch.zeros_like(central_response(pair)),
            torch.tensor([0.001 * factor]),
            pin,
            ~pin,
            TRAINING,
        )[0]

    torch.testing.assert_close(loss(1), loss(10))


def test_native_response_targets_produce_finite_nonzero_parameter_gradient(scene):
    model, rest, pin, x, v, *_ = scene
    px, pv = paired_current_step(model, x, v, rest, pin, NORM)
    tx, tv = central_response(x + 0.01 * v).detach(), central_response(v).detach()
    # Independent synthetic restoring response: a positive position perturbation
    # causes a negative velocity response. This label must drive parameter change.
    tv[0, 1:, 2] = -0.0001
    loss, _ = response_matching_loss(
        px,
        pv,
        tx,
        tv,
        torch.full((2,), 0.001),
        pin,
        ~pin,
        TRAINING,
    )
    before = copy.deepcopy(model.state_dict())
    loss.backward()
    gradients = [p.grad for p in model.internal_network.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert any(not torch.equal(value, before[k]) for k, value in model.state_dict().items())


def test_endpoint_label_changes_cannot_change_prediction_inputs(scene):
    model, rest, pin, x, v, *_ = scene
    assert set(inspect.signature(paired_current_step).parameters) == {
        "model",
        "current_x",
        "current_v",
        "rest",
        "pinned",
        "normalization",
    }
    a = paired_current_step(model, x, v, rest, pin, NORM)
    reference = central_response(a[0]).detach()
    loss_a, _ = response_matching_loss(
        *a,
        reference,
        central_response(a[1]).detach(),
        torch.full((2,), 0.001),
        pin,
        ~pin,
        TRAINING,
    )
    poisoned = reference + 0.01
    loss_b, _ = response_matching_loss(
        *a,
        poisoned,
        central_response(a[1]).detach(),
        torch.full((2,), 0.001),
        pin,
        ~pin,
        TRAINING,
    )
    b = paired_current_step(model, x, v, rest, pin, NORM)
    assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))
    assert loss_b > loss_a


def test_response_trained_checkpoint_reload_preserves_twenty_step_prediction(scene, tmp_path):
    model, rest, pin, x, v, mesh, config = scene
    with torch.no_grad():
        model.internal_network[4].weight.normal_(0, 0.001)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    reloaded = make_neighbor_model("B", mesh, config)
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    a = autoregressive_recovery(model, x[0, 0], v[0, 0], rest, pin, NORM, 20)
    b = autoregressive_recovery(reloaded, x[0, 0], v[0, 0], rest, pin, NORM, 20)
    assert all(
        torch.equal(left, right) and torch.isfinite(left).all()
        for left, right in zip(a, b, strict=True)
    )
    assert torch.equal(a[0][:, pin], rest[pin].expand(20, -1, -1))
    assert not torch.any(a[1][:, pin])


def test_response_loss_rejects_degenerate_pair_or_initial_norm(scene):
    model, rest, pin, x, v, *_ = scene
    with pytest.raises(ValueError, match="Paired"):
        paired_current_step(model, x[:, :1], v[:, :1], rest, pin, NORM)
    px, pv = paired_current_step(model, x, v, rest, pin, NORM)
    with pytest.raises(ValueError, match="positive"):
        response_matching_loss(
            px,
            pv,
            central_response(px),
            central_response(pv),
            torch.zeros(2),
            pin,
            ~pin,
            TRAINING,
        )


def test_stability_exports_accept_new_names_and_preserve_original_defaults(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    entry = importlib.import_module("diagnose_recovery_stability")
    original_methods = entry.METHODS
    methods = ("mujoco", "constant_velocity", "five_step_control", "five_step_response")
    settings = {
        "direction_count": 3,
        "signs": [-1, 1],
        "position_rms_m": [1e-7, 1e-6],
        "velocity_rms_m_s": [1e-5, 1e-4],
    }
    pin = np.array([True, False, False])
    cases, dx, dv, _ = make_perturbations(pin, settings, 7)
    t = np.arange(21)[:, None, None] * 0.01
    x = np.asarray([a + t * b for a, b in zip(dx, dv, strict=True)])
    v = np.asarray([np.broadcast_to(b, (21, 3, 3)) for b in dv])
    saved = {
        f"{method}__{field}": value
        for method in methods
        for field, value in (("position", x), ("velocity", v))
    }
    config = {
        "state_norm": {"position_unit_m": 0.001, "velocity_unit_m_s": 0.1},
        "perturbations": settings,
        "evaluation": {"horizons": [1, 5, 10, 20]},
    }
    _, rows, _ = entry.metrics(
        config, cases, saved, {"pinned": pin, "neighborhood": ~pin}, tmp_path, methods=methods
    )
    assert {row["method"] for row in rows} == set(methods)
    assert all(row["gain_ratio_of_means_vs_mujoco"] == 1 for row in rows)
    assert entry.METHODS == original_methods
