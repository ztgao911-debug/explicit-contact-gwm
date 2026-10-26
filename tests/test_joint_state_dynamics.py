"""Current-state elastic dynamics and endpoint velocity regression contracts."""

from __future__ import annotations

import inspect
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from contact_gwm.joint_state_dynamics import (
    JointIncrement,
    JointStateDynamics,
    JointStatePipeline,
    joint_state_update,
)
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline
from contact_gwm.spatial_ablation import SpatialEncodedResidual
from contact_gwm.structures import GaussianSet


def small_scene():
    rest = torch.tensor(
        [[-0.01, 0.0, 0.0], [0.0, 0.0, 0.02], [0.01, 0.0, 0.02], [0.0, 0.01, 0.02]]
    )
    pinned = torch.tensor([True, False, False, False])
    velocity = torch.tensor(
        [[0.01, 0.0, -0.02], [0.0, 0.0, -0.02], [0.01, 0.0, 0.01], [0.0, -0.01, 0.0]]
    )
    displacement = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.002], [0.0, 0.001, -0.001], [0.001, 0.0, 0.0]]
    )
    features = torch.cat(
        (rest / 0.03, displacement / 0.01, velocity / 0.05, pinned[:, None].float()), dim=-1
    )
    return rest, pinned, velocity, displacement, features


def model_and_inputs(*, autonomous=True):
    torch.manual_seed(7)
    rest, pinned, velocity, displacement, features = small_scene()
    model = JointStateDynamics(rest, pinned, hidden_dim=12, autonomous=autonomous)
    context = torch.randn(len(rest), 36)
    gates = torch.tensor([1.0, 0.8, 0.0, 0.4])
    return model, rest, pinned, velocity, displacement, features, context, gates


def enable_heads(model):
    """Perturb only zero-initialized heads, preserving the hidden representation."""
    with torch.no_grad():
        for network in (model.internal_network, model.contact_network):
            head = [module for module in network.modules() if isinstance(module, nn.Linear)][-1]
            values = torch.linspace(-0.1, 0.1, head.weight.numel()).reshape_as(head.weight)
            head.weight.copy_(values)
            head.bias.fill_(0.04)


@pytest.mark.parametrize("batched", [False, True])
def test_zero_initialization_is_constant_velocity_with_zero_endpoint_velocity_change(batched):
    model, _, pinned, velocity, _, features, context, gates = model_and_inputs()
    if batched:
        context, gates, features, velocity = [
            tensor.unsqueeze(0).expand(2, *tensor.shape).clone()
            for tensor in (context, gates, features, velocity)
        ]
    result = model(context, gates, features, velocity)
    expected = velocity * 0.01
    expected[..., pinned, :] = 0
    assert torch.equal(result.delta_position, expected)
    for name in (
        "delta_velocity",
        "internal_position",
        "internal_velocity",
        "contact_position",
        "contact_velocity",
    ):
        assert torch.count_nonzero(getattr(result, name)) == 0


def test_no_contact_retains_autonomous_state_response_and_suppresses_contact_branch():
    model, _, pinned, velocity, _, features, context, gates = model_and_inputs()
    enable_heads(model)
    gates = torch.zeros_like(gates)
    result = model(context, gates, features, velocity)
    assert torch.count_nonzero(result.internal_position[~pinned]) > 0
    assert torch.count_nonzero(result.internal_velocity[~pinned]) > 0
    assert torch.count_nonzero(result.contact_position) == 0
    assert torch.count_nonzero(result.contact_velocity) == 0
    changed_context = model(context * 100 + 7, gates, features, velocity)
    assert torch.equal(result.delta_position, changed_context.delta_position)
    assert torch.equal(result.delta_velocity, changed_context.delta_velocity)


def test_same_capacity_gated_control_cannot_restore_without_contact():
    autonomous, _, pinned, velocity, _, features, context, gates = model_and_inputs()
    gated, *_ = model_and_inputs(autonomous=False)
    enable_heads(autonomous)
    gated.load_state_dict(autonomous.state_dict())
    assert sum(p.numel() for p in gated.parameters()) == sum(
        p.numel() for p in autonomous.parameters()
    )
    result = gated(context, torch.zeros_like(gates), features, velocity)
    expected = velocity * 0.01
    expected[pinned] = 0
    assert torch.equal(result.delta_position, expected)
    assert torch.count_nonzero(result.delta_velocity) == 0


def test_rest_shape_with_zero_velocity_is_exact_internal_equilibrium_after_head_change():
    model, rest, pinned, velocity, _, features, context, gates = model_and_inputs()
    enable_heads(model)
    features[:, 3:9] = 0
    features[:, :3] = rest / 0.03
    velocity.zero_()
    result = model(context, torch.zeros_like(gates), features, velocity)
    assert torch.count_nonzero(result.delta_position) == 0
    assert torch.count_nonzero(result.delta_velocity) == 0
    assert torch.count_nonzero(result.internal_position) == 0
    assert torch.count_nonzero(result.internal_velocity) == 0
    assert torch.count_nonzero(result.delta_position[pinned]) == 0


def test_batch_matches_individual_predictions_and_pins_survive_nonzero_heads():
    model, _, pinned, velocity, _, features, context, gates = model_and_inputs()
    enable_heads(model)
    contexts = torch.stack((context, context * 2))
    features_batch = torch.stack((features, features * 1.1))
    gates_batch = torch.stack((gates, gates * 0.5))
    velocities = torch.stack((velocity, velocity * 2))
    batch = model(contexts, gates_batch, features_batch, velocities)
    for i in range(2):
        individual = model(contexts[i], gates_batch[i], features_batch[i], velocities[i])
        for name in JointIncrement.__dataclass_fields__:
            torch.testing.assert_close(getattr(batch, name)[i], getattr(individual, name))
    assert torch.count_nonzero(batch.delta_position[:, pinned]) == 0
    assert torch.count_nonzero(batch.delta_velocity[:, pinned]) == 0


def gaussian_state(rest, velocity, displacement):
    count = len(rest)
    return GaussianSet(
        centers=rest + displacement,
        log_scales=torch.full_like(rest, -5.3),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(count, -1).clone(),
        opacity_logits=torch.zeros(count, 1),
        features=torch.zeros(count, 10),
        velocities=velocity.clone(),
    )


def test_endpoint_velocity_update_uses_its_own_increment_not_displacement_over_dt():
    _, rest, pinned, velocity, displacement, _, _, _ = model_and_inputs()
    state = gaussian_state(rest, velocity, displacement)
    dx = torch.full_like(rest, 0.0003)
    dv = torch.full_like(rest, -0.007)
    zeros = torch.zeros_like(rest)
    increment = JointIncrement(dx, dv, zeros, zeros, zeros, zeros)
    updated = joint_state_update(state, increment, rest, pinned)
    torch.testing.assert_close(updated.centers[~pinned], state.centers[~pinned] + dx[~pinned])
    torch.testing.assert_close(updated.velocities[~pinned], velocity[~pinned] + dv[~pinned])
    assert not torch.allclose(updated.velocities[~pinned], dx[~pinned] / 0.01)
    assert torch.equal(updated.centers[pinned], rest[pinned])
    assert torch.count_nonzero(updated.velocities[pinned]) == 0
    assert torch.equal(state.velocities, velocity)
    assert torch.equal(updated.log_scales, state.log_scales)
    assert torch.equal(updated.rotations, state.rotations)


def test_current_input_contract_finite_predictions_and_gradients():
    model, _, _, velocity, _, features, context, gates = model_and_inputs()
    assert set(inspect.signature(model.forward).parameters) == {
        "context", "gates", "features", "current_velocity"
    }
    enable_heads(model)
    result = model(context, gates, features, velocity)
    assert all(
        torch.isfinite(getattr(result, name)).all() for name in JointIncrement.__dataclass_fields__
    )
    (result.delta_position.square().sum() + result.delta_velocity.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.internal_network.parameters())
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.contact_network.parameters())


def test_saved_model_reloads_bitwise_identical_predictions():
    model, rest, pinned, velocity, _, features, context, gates = model_and_inputs()
    enable_heads(model)
    expected = model(context, gates, features, velocity)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded = JointStateDynamics(rest, pinned, hidden_dim=12)
    loaded.load_state_dict(torch.load(buffer, weights_only=True))
    actual = loaded(context, gates, features, velocity)
    for name in JointIncrement.__dataclass_fields__:
        assert torch.equal(getattr(expected, name), getattr(actual, name))


@pytest.mark.parametrize("bad_field", ["delta_position", "delta_velocity"])
def test_state_update_rejects_nonfinite_predictions(bad_field):
    _, rest, pinned, velocity, displacement, _, _, _ = model_and_inputs()
    state = gaussian_state(rest, velocity, displacement)
    values = {name: torch.zeros_like(rest) for name in JointIncrement.__dataclass_fields__}
    values[bad_field][1, 2] = float("nan")
    with pytest.raises(FloatingPointError, match="finite"):
        joint_state_update(state, JointIncrement(**values), rest, pinned)


@pytest.fixture
def joint_pipeline(tmp_path):
    """Build a fresh tiny pilot scene, without reading recorded trajectories."""
    mujoco = pytest.importorskip("mujoco")
    from contact_gwm.data.mujoco_soft import build_xml, mesh_from_model

    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    torch.manual_seed(7)
    scene_config = yaml.safe_load((root / "configs/mujoco_soft_pilot.yaml").read_text())
    scene = tmp_path / "scene.xml"
    scene.write_text(build_xml(scene_config))
    native_model = mujoco.MjModel.from_xml_path(str(scene))
    native_data = mujoco.MjData(native_model)
    native_data.qpos[-3:] = [0, 0, 0.068]
    mujoco.mj_forward(native_model, native_data)
    mesh = mesh_from_model(native_model, native_data)
    rest = torch.tensor(mesh["rest_positions_world"]).float()
    pinned = torch.tensor(mesh["pinned_mask"])
    training = yaml.safe_load((root / "configs/train_soft_local.yaml").read_text())
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"model_state": SpatialEncodedResidual(rest).state_dict(), "config": training}, checkpoint
    )
    config = yaml.safe_load((root / "configs/validate_contact_pipeline.yaml").read_text())
    legacy = PinnedContactPipeline(mesh, scene, checkpoint, config)
    model = JointStateDynamics(rest, pinned, hidden_dim=12)
    enable_heads(model)
    pipeline = JointStatePipeline(legacy, model)
    position = mesh["rest_positions_world"].copy()
    position[~mesh["pinned_mask"], 2] -= 0.0005
    initial = legacy.gaussian_state(position, np.zeros_like(position))
    robot = IndenterInput(
        position=np.array([0.0, 0.0, 0.2]),
        velocity=np.zeros(3),
        target_position=np.array([0.0, 0.0, 0.2]),
        target_velocity=np.zeros(3),
    )
    return pipeline, initial, robot


def test_detached_pipeline_retains_soft_state_and_hard_pins(joint_pipeline):
    pipeline, initial, robot = joint_pipeline
    inputs = pipeline.current_inputs(initial, robot)
    assert torch.count_nonzero(inputs.gates) == 0
    assert inputs.contacts.num_contacts == 0
    assert torch.count_nonzero(inputs.features[:, 3:6]) > 0
    result = pipeline.step(initial, robot)
    free = ~pipeline.legacy.pinned
    assert torch.count_nonzero(result.residual[free]) > 0
    assert torch.count_nonzero(result.state.velocities[free]) > 0
    for result in pipeline.rollout(initial, [robot] * 5):
        assert torch.isfinite(result.state.centers).all()
        assert torch.isfinite(result.state.velocities).all()
        assert torch.equal(result.state.centers[~free], pipeline.legacy.rest[~free])
        assert torch.count_nonzero(result.state.velocities[~free]) == 0


def test_joint_rollout_ignores_all_external_soft_and_contact_labels(joint_pipeline):
    pipeline, initial, robot = joint_pipeline
    count = initial.num_gaussians
    arrays = {
        "indenter_position_world": np.tile(robot.position, (6, 1)),
        "indenter_velocity_world": np.tile(robot.velocity, (6, 1)),
        "action_target_position": np.tile(robot.target_position, (6, 1)),
        "action_target_velocity": np.tile(robot.target_velocity, (6, 1)),
        "node_position_world": np.ones((6, count, 3)),
        "node_velocity_world": np.ones((6, count, 3)),
        "contact_force_on_block_world": np.ones((6, 3)),
        "contact_position_world": np.ones((6, 3)),
        "contact_frame_world": np.ones((6, 9)),
    }
    before = pipeline.rollout(
        initial, [IndenterInput.from_arrays(arrays, t) for t in range(5)]
    )
    for key, values in arrays.items():
        if key.startswith(("node_", "contact_")):
            values[:] = np.nan
    after = pipeline.rollout(
        initial, [IndenterInput.from_arrays(arrays, t) for t in range(5)]
    )
    assert set(inspect.signature(pipeline.current_inputs).parameters) == {"state", "robot"}
    assert set(inspect.signature(pipeline.rollout).parameters) == {"initial", "robot_inputs"}
    for actual, expected in zip(after, before, strict=True):
        assert torch.equal(actual.state.centers, expected.state.centers)
        assert torch.equal(actual.state.velocities, expected.state.velocities)
