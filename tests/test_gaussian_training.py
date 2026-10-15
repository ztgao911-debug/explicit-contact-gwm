"""Check stable node mapping, causal frame inputs and pinned prediction behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from contact_gwm.comparison import fit_model, make_model, prepare_pairs, validate_splits
from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import (
    ConstantVelocityResidualDeformation,
    PinnedGaussianDeformation,
    constant_velocity_increment,
)
from contact_gwm.structures import ContactForces, ContactGeometry


@pytest.fixture
def source(tmp_path):
    """Small deterministic loader-format fixture; not a simulated training dataset."""
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/train_soft_local.yaml").read_text()
    )
    rest = np.array(
        [[x, y, z] for x in (-0.005, 0.005) for y in (-0.005, 0.005) for z in (0, 0.01)]
    )
    pinned = rest[:, 2] == 0
    np.savez_compressed(
        tmp_path / "mesh.npz", node_ids=np.arange(8), rest_positions_world=rest, pinned_mask=pinned
    )
    position = np.repeat(rest[None], 5, axis=0)
    position[:, ~pinned, 2] -= np.arange(5)[:, None] * 0.00002
    velocity = np.zeros_like(position)
    velocity[:, ~pinned, 2] = -0.002
    frame = np.array([0, 0, -1, 1, 0, 0, 0, -1, 0])
    arrays = {
        "node_ids": np.arange(8),
        "time": np.arange(5) * 0.01,
        "node_position_world": position,
        "node_velocity_world": velocity,
        "phase_id": np.array([1, 1, 2, 2, 3]),
        "contact_count": np.ones(5, dtype=int),
        "contact_offsets": np.arange(6),
        "contact_position_world": np.tile([0, 0, 0.01], (5, 1)),
        "contact_frame_world": np.tile(frame, (5, 1)),
        "contact_signed_distance": np.full((5, 1), -0.0001),
        "contact_force_on_block_world": np.tile([0, 0, -0.1], (5, 1)),
        "contact_flex_ids": np.tile([-1, 0], (5, 1)),
        "action_target_position": np.tile([0, 0, 0.02], (5, 1)),
        "action_target_velocity": np.tile([0, 0, -0.002], (5, 1)),
        "indenter_position_world": np.tile([0, 0, 0.02], (5, 1)),
        "indenter_velocity_world": np.tile([0, 0, -0.002], (5, 1)),
    }
    manifest = {
        "shared_sha256": {"mesh.npz": file_sha256(tmp_path / "mesh.npz")},
        "config": {
            "block": {"young": 5000.0, "poisson": 0.3, "rayleigh_damping": 0.0002},
            "contact": {"friction": [0.3, 0.005, 0.0001]},
        },
        "trajectories": [],
    }
    for name in config["trajectories"]:
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        manifest["trajectories"].append({"file": path.name, "sha256": file_sha256(path)})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, config


def test_node_gaussians_and_increment_target(source):
    directory, config = source
    dataset = GaussianPairDataset(directory, config)
    pair = dataset[1]
    assert pair.frame == 1
    assert pair.next_time - pair.time == pytest.approx(0.01)
    torch.testing.assert_close(pair.inputs.gaussians.scales, torch.full((8, 3), 0.005))
    torch.testing.assert_close(
        pair.inputs.gaussians.rotations, torch.tensor([[1.0, 0, 0, 0]]).expand(8, -1)
    )
    assert torch.equal(pair.inputs.node_ids, torch.arange(8))
    assert pair.inputs.gaussians.features.shape == (8, 10)
    torch.testing.assert_close(pair.target_delta[~dataset.pinned, 2], torch.full((4,), -0.00002))
    assert dataset.selection_counts["center_shallow"]["pairs"] == 3


@pytest.mark.parametrize(
    "model_class", [PinnedGaussianDeformation, ConstantVelocityResidualDeformation]
)
def test_future_state_action_and_force_do_not_enter_inputs(source, model_class):
    directory, config = source
    dataset = GaussianPairDataset(directory, config)
    pair = dataset[1]
    model = model_class(hidden_dim=16)
    with torch.no_grad():
        model.deformation.network[-1].weight.fill_(0.02)
    before = model.encode(pair.inputs)
    prediction_before = model(pair.inputs)
    arrays = dataset.trajectories[pair.trajectory]
    future = pair.frame + 1
    arrays["node_position_world"][future] += 10
    arrays["node_velocity_world"][future] += 100
    arrays["action_target_position"][future] += 100
    arrays["action_target_velocity"][future] += 100
    arrays["indenter_position_world"][future] += 100
    arrays["indenter_velocity_world"][future] += 100
    start, end = arrays["contact_offsets"][future : future + 2]
    arrays["contact_force_on_block_world"][start:end] *= 1000
    arrays["contact_position_world"][start:end] += 100
    after_pair = dataset[1]
    after = model.encode(after_pair.inputs)
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(prediction_before, model(after_pair.inputs))
    assert not torch.equal(pair.target_delta, after_pair.target_delta)


def test_pinned_geometry_and_cached_batch_agree_after_nonzero_weights(source):
    directory, config = source
    dataset = GaussianPairDataset(directory, config)
    model = PinnedGaussianDeformation(hidden_dim=16)
    with torch.no_grad():
        model.deformation.network[-1].weight.fill_(0.02)
        model.deformation.network[-1].bias.fill_(0.03)
    inputs = [dataset[i].inputs for i in range(2)]
    encoded = [model.encode(item) for item in inputs]
    cached = model.predict_encoded(
        torch.stack([x[0] for x in encoded]), torch.stack([x[1] for x in encoded]), dataset.pinned
    )
    direct = torch.stack([model(item) for item in inputs])
    torch.testing.assert_close(cached, direct)
    assert torch.count_nonzero(direct[:, dataset.pinned]) == 0
    assert torch.count_nonzero(direct[:, ~dataset.pinned]) > 0
    result = model.next_gaussians(inputs[0])
    assert torch.equal(result.centers[dataset.pinned], dataset.rest[dataset.pinned])
    assert torch.equal(result.log_scales, inputs[0].gaussians.log_scales)
    assert torch.equal(result.rotations, inputs[0].gaussians.rotations)
    model.zero_grad()
    direct.square().mean().backward()
    assert model.deformation.network[-1].weight.grad.abs().sum() > 0


def test_no_contact_stays_zero_after_learning_a_bias(source):
    directory, config = source
    pair = GaussianPairDataset(directory, config)[0]
    inputs = replace(
        pair.inputs,
        contacts=ContactGeometry.empty(torch.device("cpu"), torch.float32),
        contact_forces=ContactForces.empty(torch.device("cpu"), torch.float32),
    )
    model = PinnedGaussianDeformation(hidden_dim=16)
    with torch.no_grad():
        model.deformation.network[-1].bias.fill_(1.0)
    assert torch.count_nonzero(model(inputs)) == 0
    delta, scales, gates = model.deformation(
        inputs.gaussians,
        inputs.contacts,
        inputs.contact_forces,
        inputs.action,
        inputs.material_features,
    )
    assert all(torch.count_nonzero(value) == 0 for value in (delta, scales, gates))


@pytest.mark.parametrize("field", ["time", "node_ids"])
def test_rejects_wrong_dt_or_node_order(source, field):
    directory, config = source
    path = directory / "center_shallow.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if field == "time":
        arrays[field][1] += 0.001
    else:
        arrays[field] = arrays[field][::-1]
    np.savez_compressed(path, **arrays)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["trajectories"][0]["sha256"] = file_sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="times|IDs"):
        GaussianPairDataset(directory, config)


def test_zero_residual_is_exactly_constant_velocity(source):
    directory, config = source
    inputs = GaussianPairDataset(directory, config)[1].inputs
    model = ConstantVelocityResidualDeformation(hidden_dim=16)
    expected = constant_velocity_increment(inputs.gaussians.velocities, inputs.pinned_mask)
    assert torch.equal(model(inputs), expected)
    context, gates = model.encode(inputs)
    assert torch.count_nonzero(model.gated_residual(context, gates, inputs.pinned_mask)) == 0
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in PinnedGaussianDeformation(hidden_dim=16).parameters()
    )


def test_total_residual_increment_keeps_fixed_nodes_even_with_nonzero_velocity(source):
    directory, config = source
    inputs = GaussianPairDataset(directory, config)[1].inputs
    inputs = replace(inputs, gaussians=replace(inputs.gaussians, velocities=torch.ones((8, 3))))
    model = ConstantVelocityResidualDeformation(hidden_dim=16)
    with torch.no_grad():
        model.deformation.network[-1].bias.fill_(0.5)
    prediction = model(inputs)
    assert torch.count_nonzero(prediction[inputs.pinned_mask]) == 0
    assert torch.any(prediction[~inputs.pinned_mask] != 0.01)
    after = model.next_gaussians(inputs)
    assert torch.equal(after.centers[inputs.pinned_mask], inputs.rest_positions[inputs.pinned_mask])
    assert torch.equal(after.log_scales, inputs.gaussians.log_scales)
    assert torch.equal(after.rotations, inputs.gaussians.rotations)


def test_no_contact_zeroes_only_learned_residual(source):
    directory, config = source
    inputs = GaussianPairDataset(directory, config)[1].inputs
    inputs = replace(
        inputs,
        contacts=ContactGeometry.empty(torch.device("cpu"), torch.float32),
        contact_forces=ContactForces.empty(torch.device("cpu"), torch.float32),
    )
    model = ConstantVelocityResidualDeformation(hidden_dim=16)
    with torch.no_grad():
        model.deformation.network[-1].bias.fill_(0.5)
    context, gates = model.encode(inputs)
    assert torch.count_nonzero(model.gated_residual(context, gates, inputs.pinned_mask)) == 0
    assert torch.equal(
        model(inputs), constant_velocity_increment(inputs.gaussians.velocities, inputs.pinned_mask)
    )
    assert torch.count_nonzero(model(inputs)[~inputs.pinned_mask]) > 0


def test_whole_trajectory_split_rejects_overlap():
    valid = {
        "train": ["center_shallow", "center_deep"],
        "validation": ["left_shallow", "left_deep"],
        "test": ["right_shallow", "right_deep"],
    }
    validate_splits(valid)
    valid["test"][0] = "center_shallow"
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(valid)


def test_checkpoint_selection_uses_validation_not_training_loss(source, tmp_path):
    directory, config = source
    config["model"]["hidden_dim"] = 16
    config["training"]["epochs"] = 3
    config["training"]["log_interval"] = 3
    dataset = GaussianPairDataset(directory, config)
    pack = prepare_pairs(dataset, make_model("direct", config))
    # Deliberately conflicting validation targets: learning the train direction worsens validation.
    validation = replace(pack, target=-100 * pack.target)
    result = fit_model("direct", pack, validation, config, tmp_path / "selection_test")
    assert result["selected_epoch"] == 0
    assert (
        result["selected_epoch"]
        == min(result["history"], key=lambda row: row["validation_loss"])["epoch"]
    )
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
