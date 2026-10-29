from __future__ import annotations

import inspect

import pytest
import torch

from contact_gwm.joint_state_dynamics import JointStateDynamics
from contact_gwm.paired_contact_response import (
    absolute_loss,
    central_derived_caps,
    output_bound_audit,
    paired_loss,
    predict_cache,
)


def example():
    torch.manual_seed(7)
    rest = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.01, 0.0, 0.01]])
    pinned = torch.tensor([True, False, False])
    model = JointStateDynamics(rest, pinned, hidden_dim=8)
    features = model.rest_features[None].repeat(2, 1, 1)
    features[:, 1:, 3:6] = 0.1
    cache = {
        "context": torch.zeros(2, 3, 36),
        "features": features,
        "gates": torch.ones(2, 3),
        "velocity": torch.zeros(2, 3, 3),
        "target_dx": torch.zeros(2, 3, 3),
        "target_dv": torch.zeros(2, 3, 3),
        "neighborhood": (~pinned)[None].repeat(2, 1),
        "pinned": pinned,
    }
    cache["context"][1, :, 11] = 0.5
    pairs = {
        "base_index": torch.tensor([0]),
        "variant_index": torch.tensor([1]),
        "target_dx": torch.zeros(1, 3, 3),
        "target_dv": torch.zeros(1, 3, 3),
        "neighborhood": (~pinned)[None],
    }
    settings = {
        "position_loss_unit_m": 0.001,
        "velocity_loss_unit_m_s": 0.1,
        "neighborhood_weight": 0.5,
        "velocity_weight": 0.5,
    }
    return model, cache, pairs, settings, rest, pinned


def test_same_initial_outputs_for_original_and_expanded_caps():
    model, cache, _, _, rest, pinned = example()
    expanded = JointStateDynamics(
        rest, pinned, hidden_dim=8, max_position_residual=0.032, max_velocity_increment=3.2
    )
    expanded.load_state_dict(model.state_dict())
    a, b = (
        predict_cache(model, cache, torch.arange(2)),
        predict_cache(expanded, cache, torch.arange(2)),
    )
    assert torch.equal(a.delta_position, b.delta_position)
    assert torch.equal(a.delta_velocity, b.delta_velocity)
    assert not torch.any(a.delta_position[pinned[None].expand(2, -1)])


def test_pair_internal_and_constant_velocity_cancel_and_contact_gradient_is_finite():
    model, cache, pairs, settings, _, _ = example()
    with torch.no_grad():
        model.internal_network[4].weight.normal_(0, 0.01)
        model.contact_network[4].weight.normal_(0, 0.01)
    a, b = (
        predict_cache(model, cache, torch.tensor([0])),
        predict_cache(model, cache, torch.tensor([1])),
    )
    torch.testing.assert_close(
        b.delta_position - a.delta_position, b.contact_position - a.contact_position
    )
    pairs["target_dx"][:, 1:, 2] = 0.0002
    loss = paired_loss(model, cache, pairs, torch.tensor([0]), settings)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(float(p.grad.abs().sum()) for p in model.contact_network.parameters()) > 0
    assert sum(
        float(p.grad.abs().sum()) for p in model.internal_network.parameters()
    ) == pytest.approx(0, abs=1e-10)


def test_targets_do_not_enter_forward_and_pair_labels_only_change_loss():
    model, cache, pairs, settings, _, _ = example()
    before = predict_cache(model, cache, torch.arange(2))
    cache["target_dx"].fill_(float("nan"))
    cache["target_dv"].fill_(float("nan"))
    after = predict_cache(model, cache, torch.arange(2))
    assert torch.equal(before.delta_position, after.delta_position)
    assert torch.equal(before.delta_velocity, after.delta_velocity)
    assert paired_loss(model, cache, pairs, torch.tensor([0]), settings) == 0
    pairs["target_dx"][:, 1:, 2] = 0.002
    assert paired_loss(model, cache, pairs, torch.tensor([0]), settings) > 0
    assert "validation" not in inspect.signature(central_derived_caps).parameters


def test_output_bound_diagnoses_impossible_pairs_without_removing_them():
    _, cache, pairs, _, _, _ = example()
    cache["target_dx"][1, 1, 2] = 0.008
    cache["target_dv"][1, 1, 2] = 0.6
    pairs["target_dx"][0, 1, 2] = 0.008
    pairs["target_dv"][0, 1, 2] = 0.6
    audit = output_bound_audit(cache, pairs, 0.001, 0.1)
    assert audit["position"]["absolute_violating_components"] == 1
    assert audit["position"]["pair_violating_components"] == 1
    caps = central_derived_caps(audit)
    assert caps["max_position_residual"] == 0.008
    assert caps["max_velocity_increment"] == 0.4
    expanded = output_bound_audit(
        cache, pairs, caps["max_position_residual"], caps["max_velocity_increment"]
    )
    assert expanded["position"]["pair_violating_components"] == 0
    assert expanded["velocity"]["absolute_violating_components"] == 0
    cache["gates"].zero_()
    zero = output_bound_audit(cache, pairs, 0.1, 1.0)
    assert zero["position"]["zero_gate_pair_nonzero_target_components"] == 1
    assert len(pairs["base_index"]) == 1


def test_fixed_nodes_never_contribute_to_losses():
    model, cache, pairs, settings, _, _ = example()
    cache["target_dx"][:, 0] = 123
    cache["target_dv"][:, 0] = 456
    pairs["target_dx"][:, 0] = 123
    pairs["target_dv"][:, 0] = 456
    assert absolute_loss(model, cache, torch.arange(2), settings) == 0
    assert paired_loss(model, cache, pairs, torch.tensor([0]), settings) == 0
