"""Rest-mesh neighborhood contracts for the internal recovery ablation."""

from __future__ import annotations

import inspect
from io import BytesIO

import pytest
import torch
from torch import nn

from contact_gwm.joint_state_dynamics import JointIncrement
from contact_gwm.neighbor_dynamics import (
    NeighborJointStateDynamics,
    closest_internal_width,
    make_neighbor_model,
    model_specification,
    neighbor_mean_matrix,
    neighbor_relative_features,
)


def small_scene():
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.01, 0.0, 0.01], [0.02, 0.0, 0.01]]
    )
    pinned = torch.tensor([True, False, False, False])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    velocity = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.02], [0.01, 0.0, 0.01], [0.0, -0.01, 0.0]]
    )
    displacement = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.002], [0.0, 0.001, -0.001], [0.001, 0.0, 0.0]]
    )
    features = torch.cat(
        (rest / 0.03, displacement / 0.01, velocity / 0.05, pinned[:, None].float()), dim=-1
    )
    mesh = {"rest_positions_world": rest, "pinned_mask": pinned, "edges": edges}
    return mesh, features, velocity


def model_and_inputs():
    torch.manual_seed(7)
    mesh, features, velocity = small_scene()
    model = NeighborJointStateDynamics(
        mesh["rest_positions_world"], mesh["pinned_mask"], mesh["edges"], hidden_dim=12
    )
    context = torch.randn(len(features), 36)
    gates = torch.tensor([1.0, 0.8, 0.0, 0.4])
    return model, mesh, context, gates, features, velocity


def enable_heads(model):
    with torch.no_grad():
        for network in (model.internal_network, model.contact_network):
            head = [module for module in network.modules() if isinstance(module, nn.Linear)][-1]
            head.weight.copy_(
                torch.linspace(-0.1, 0.1, head.weight.numel()).reshape_as(head.weight)
            )
            head.bias.fill_(0.04)


def test_mean_matrix_uses_unique_undirected_edges_and_keeps_isolated_nodes_zero():
    edges = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 2], [1, 1]])
    weights = neighbor_mean_matrix(4, edges)
    expected = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0],
         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    assert torch.equal(weights, expected)
    assert torch.count_nonzero(torch.diagonal(weights)) == 0
    assert torch.count_nonzero(neighbor_mean_matrix(3, torch.empty(0, 2))) == 0


@pytest.mark.parametrize("edges", [torch.tensor([[-1, 1]]), torch.tensor([[0, 4]])])
def test_graph_rejects_node_indices_outside_stable_range(edges):
    with pytest.raises(ValueError, match="stable node range"):
        neighbor_mean_matrix(4, edges)


def test_relative_features_match_hand_computed_means_in_both_normalized_units():
    weights = neighbor_mean_matrix(4, torch.tensor([[0, 1], [1, 2]]))
    features = torch.zeros(4, 10)
    features[:, 3] = torch.tensor([1.0, 3.0, 9.0, 20.0])
    features[:, 8] = torch.tensor([2.0, 5.0, 10.0, 30.0])
    original = features.clone()
    relative = neighbor_relative_features(features, weights, torch.zeros(4, dtype=torch.bool))
    assert torch.equal(relative[:, 0], torch.tensor([2.0, 2.0, -6.0, 0.0]))
    assert torch.equal(relative[:, 5], torch.tensor([3.0, 1.0, -5.0, 0.0]))
    assert torch.count_nonzero(relative[:, 1:5]) == 0
    assert torch.equal(features, original)


def test_rigid_same_displacement_and_velocity_have_zero_relative_features_without_pins():
    mesh, features, _ = small_scene()
    features[:, 3:9] = torch.tensor([0.2, -0.1, 0.4, 0.1, 0.6, -0.3])
    weights = neighbor_mean_matrix(len(features), mesh["edges"])
    result = neighbor_relative_features(features, weights, torch.zeros(4, dtype=torch.bool))
    assert torch.count_nonzero(result) == 0


def test_fixed_nodes_contribute_as_static_neighbors_even_if_input_features_are_polluted():
    weights = neighbor_mean_matrix(2, torch.tensor([[0, 1]]))
    features = torch.zeros(2, 10)
    features[0, 3:9] = float("nan")
    features[1, 3:9] = torch.tensor([0.1, 0.2, 0.3, -0.4, -0.5, -0.6])
    result = neighbor_relative_features(features, weights, torch.tensor([True, False]))
    assert torch.isfinite(result).all()
    assert torch.equal(result[1], -features[1, 3:9])
    assert torch.equal(result[0], features[1, 3:9])


def test_changing_only_a_neighbor_changes_appended_inputs_of_current_node():
    model, _, context, gates, features, velocity = model_and_inputs()
    captured = []
    hook = model.internal_network[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        model(context, gates, features, velocity)
        before = captured[0]
        captured.clear()
        altered = features.clone()
        altered[2, 3] += 1.0
        model(context, gates, altered, velocity)
        after = captured[0]
    finally:
        hook.remove()
    assert before.shape == (4, 34)
    assert torch.equal(before[1, :28], after[1, :28])
    assert not torch.equal(before[1, 28:], after[1, 28:])
    assert torch.equal(after[1, 28:] - before[1, 28:], torch.tensor([0.5, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize("batched", [False, True])
def test_zero_heads_equal_constant_velocity_and_zero_instantaneous_velocity_increment(batched):
    model, mesh, context, gates, features, velocity = model_and_inputs()
    if batched:
        context, gates, features, velocity = [
            value.unsqueeze(0).expand(2, *value.shape).clone()
            for value in (context, gates, features, velocity)
        ]
    result = model(context, gates, features, velocity)
    expected = velocity * 0.01
    expected[..., mesh["pinned_mask"], :] = 0
    assert torch.equal(result.delta_position, expected)
    for field in JointIncrement.__dataclass_fields__:
        if field != "delta_position":
            assert torch.count_nonzero(getattr(result, field)) == 0


def test_internal_rest_equilibrium_and_static_reference_neighbor_features():
    model, _, context, gates, features, velocity = model_and_inputs()
    enable_heads(model)
    features[:, 3:9] = 0
    velocity.zero_()
    captured = []
    hook = model.internal_network[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        result = model(context, torch.zeros_like(gates), features, velocity)
    finally:
        hook.remove()
    assert len(captured) == 2
    assert torch.count_nonzero(captured[1][..., 28:]) == 0
    assert torch.equal(captured[0], captured[1])
    assert torch.count_nonzero(result.delta_position) == 0
    assert torch.count_nonzero(result.delta_velocity) == 0


def test_batch_matches_individual_predictions_and_fixed_nodes_remain_zero():
    model, mesh, context, gates, features, velocity = model_and_inputs()
    enable_heads(model)
    contexts = torch.stack((context, context * 2))
    gates_batch = torch.stack((gates, gates * 0.5))
    features_batch = torch.stack((features, features * 1.1))
    velocities = torch.stack((velocity, velocity * 2))
    result = model(contexts, gates_batch, features_batch, velocities)
    for i in range(2):
        single = model(contexts[i], gates_batch[i], features_batch[i], velocities[i])
        for field in JointIncrement.__dataclass_fields__:
            torch.testing.assert_close(getattr(result, field)[i], getattr(single, field))
    for field in JointIncrement.__dataclass_fields__:
        assert torch.count_nonzero(getattr(result, field)[:, mesh["pinned_mask"]]) == 0


def test_finite_gradients_reach_added_neighbor_columns_without_future_inputs():
    model, _, context, gates, features, velocity = model_and_inputs()
    enable_heads(model)
    assert set(inspect.signature(model.forward).parameters) == {
        "context", "gates", "features", "current_velocity"
    }
    result = model(context, gates, features, velocity)
    assert all(
        torch.isfinite(getattr(result, field)).all()
        for field in JointIncrement.__dataclass_fields__
    )
    loss = result.delta_position.square().sum() + result.delta_velocity.square().sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.count_nonzero(model.internal_network[0].weight.grad[:, 28:]) > 0


def test_reload_preserves_graph_and_predictions_exactly():
    model, mesh, context, gates, features, velocity = model_and_inputs()
    enable_heads(model)
    expected = model(context, gates, features, velocity)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded = NeighborJointStateDynamics(
        mesh["rest_positions_world"], mesh["pinned_mask"], mesh["edges"], hidden_dim=12
    )
    loaded.load_state_dict(torch.load(buffer, weights_only=True))
    actual = loaded(context, gates, features, velocity)
    assert torch.equal(
        model.internal_network.neighbor_weights, loaded.internal_network.neighbor_weights
    )
    for field in JointIncrement.__dataclass_fields__:
        assert torch.equal(getattr(expected, field), getattr(actual, field))


def test_capacity_control_keeps_contact_weights_identical_and_matches_total_parameters():
    mesh, _, _ = small_scene()
    config = {"model": {"hidden_dim": 128}, "dt": 0.01}
    models = {}
    for group in ("A", "B", "C"):
        torch.manual_seed(7)
        models[group] = make_neighbor_model(group, mesh, config)
    specifications = {group: model_specification(model) for group, model in models.items()}
    assert [specifications[group]["parameter_count"] for group in ("A", "B", "C")] == [
        45324, 46092, 46209
    ]
    assert [specifications[group]["internal_input_dim"] for group in ("A", "B", "C")] == [
        28, 34, 28
    ]
    assert [specifications[group]["internal_hidden_dim"] for group in ("A", "B", "C")] == [
        128, 128, 131
    ]
    assert closest_internal_width(46092) == 131
    for group in ("B", "C"):
        for name, value in models["A"].contact_network.state_dict().items():
            assert torch.equal(value, models[group].contact_network.state_dict()[name])
        assert specifications[group]["contact_parameters"] == 24326
        assert specifications[group]["contact_input_dim"] == 54
        assert specifications[group]["contact_hidden_dim"] == 128
    assert torch.equal(
        models["A"].internal_network[0].weight, models["B"].internal_network[0].weight[:, :28]
    )
    assert torch.count_nonzero(models["B"].internal_network[0].weight[:, 28:]) == 0
