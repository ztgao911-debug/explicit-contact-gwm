"""Boundary, velocity semantics and oracle-isolation checks for the full path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from contact_gwm.pinned_contact_pipeline import (
    METHODS,
    IndenterInput,
    PinnedContactPipeline,
    constant_acceleration_update,
    nonnegative_two_column_fit,
)
from contact_gwm.spatial_ablation import SpatialEncodedResidual

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from contact_gwm.data.mujoco_soft import build_xml, mesh_from_model

    torch.set_num_threads(1)
    torch.manual_seed(7)
    scene_config = yaml.safe_load((ROOT / "configs/mujoco_soft_pilot.yaml").read_text())
    xml = build_xml(scene_config)
    scene = tmp_path / "scene.xml"
    scene.write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[-3:] = [0, 0, 0.068]
    mujoco.mj_forward(model, data)
    mesh = mesh_from_model(model, data)
    training = yaml.safe_load((ROOT / "configs/train_soft_local.yaml").read_text())
    net = SpatialEncodedResidual(torch.tensor(mesh["rest_positions_world"]).float())
    with torch.no_grad():
        net.deformation.network[-1].weight.normal_(0, 0.001)
        net.deformation.network[-1].bias[:3].fill_(0.001)
    checkpoint = tmp_path / "network.pt"
    torch.save({"model_state": net.state_dict(), "config": training}, checkpoint)
    config = yaml.safe_load((ROOT / "configs/validate_contact_pipeline.yaml").read_text())
    result = PinnedContactPipeline(mesh, scene, checkpoint, config)
    state = result.gaussian_state(mesh["rest_positions_world"], np.zeros((140, 3)))
    robot = IndenterInput(
        np.array([0.0, 0.0, 0.041]),
        np.array([0.0, 0.0, -0.01]),
        np.array([0.0, 0.0, 0.041]),
        np.array([0.0, 0.0, -0.01]),
    )
    return result, state, robot, (mesh, scene, checkpoint, config)


def test_constant_acceleration_endpoint_velocity_and_hard_boundary(pipeline):
    predictor, state, _, _ = pipeline
    velocity = torch.ones_like(state.velocities) * 0.01
    state = replace(state, velocities=velocity)
    residual = torch.full_like(state.centers, 1e-6)
    next_state, delta = constant_acceleration_update(
        state, residual, predictor.rest, predictor.pinned, 0.01
    )
    free = ~predictor.pinned
    torch.testing.assert_close(delta[free], velocity[free] * 0.01 + residual[free])
    torch.testing.assert_close(
        next_state.velocities[free], velocity[free] + 2 * residual[free] / 0.01
    )
    assert not torch.equal(next_state.velocities[free], delta[free] / 0.01)
    assert torch.count_nonzero(delta[predictor.pinned]) == 0
    assert torch.count_nonzero(next_state.velocities[predictor.pinned]) == 0
    assert torch.equal(next_state.centers[predictor.pinned], predictor.rest[predictor.pinned])
    with pytest.raises(FloatingPointError):
        constant_acceleration_update(
            state, residual * float("nan"), predictor.rest, predictor.pinned, 0.01
        )


@pytest.mark.parametrize("method", METHODS)
def test_all_branches_pinned_finite_and_reload_exact(pipeline, method):
    predictor, state, robot, constructor = pipeline
    steps = predictor.rollout(state, [robot] * 20, method)
    other = PinnedContactPipeline(*constructor)
    reloaded = other.rollout(state, [robot] * 20, method)
    for actual, expected in zip(steps, reloaded, strict=True):
        assert torch.equal(actual.state.centers, expected.state.centers)
        assert torch.equal(actual.state.velocities, expected.state.velocities)
        assert torch.equal(actual.state.log_scales, state.log_scales)
        assert torch.equal(actual.state.rotations, state.rotations)
        assert torch.isfinite(actual.state.centers).all()
        assert torch.isfinite(actual.state.velocities).all()
        assert torch.count_nonzero(actual.delta[predictor.pinned]) == 0
        assert torch.equal(actual.state.centers[predictor.pinned], predictor.rest[predictor.pinned])


@pytest.mark.parametrize("method", ("analytic_force", "gaussian_geometry"))
def test_analytic_forward_never_calls_true_force_solver(pipeline, monkeypatch, method):
    import mujoco

    predictor, state, robot, _ = pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("True force solver/label used in an analytic forward path")

    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    monkeypatch.setattr(mujoco, "mj_contactForce", forbidden)
    if method == "gaussian_geometry":
        monkeypatch.setattr(predictor.native, "query", forbidden)
    result = predictor.step(state, robot, method)
    assert result.contacts.num_contacts > 0
    assert result.forces.forces[:, 2].sum() < 0


def test_native_reconstruction_frame_sign_velocity_and_com_torque(pipeline):
    predictor, state, robot, _ = pipeline
    contacts, native_forces = predictor.condition(state, robot, "mujoco_force")
    np.testing.assert_allclose(predictor.native.data.flexvert_xpos, state.centers, atol=1e-15)
    assert contacts.num_contacts > 0
    assert torch.all(contacts.normals[:, 2] < 0)
    assert torch.all((contacts.relative_velocities * contacts.normals).sum(-1) > 0)
    assert native_forces.forces[:, 2].sum() < 0
    assert torch.all((native_forces.forces * contacts.normals).sum(-1) >= -1e-12)
    com = predictor.center_of_mass(state)
    expected = torch.cross(contacts.points - com, native_forces.forces, dim=-1)
    torch.testing.assert_close(native_forces.torques, expected, atol=0, rtol=0)
    analytic, _ = predictor.condition(state, robot, "analytic_force")
    torch.testing.assert_close(contacts.points, analytic.points, atol=0, rtol=0)
    torch.testing.assert_close(contacts.normals, analytic.normals, atol=0, rtol=0)


@pytest.mark.parametrize("method", METHODS[:-1])
def test_rollout_has_no_future_soft_or_contact_inputs(pipeline, method):
    predictor, initial, robot, _ = pipeline
    arrays = {
        "indenter_position_world": np.tile(robot.position, (6, 1)),
        "indenter_velocity_world": np.tile(robot.velocity, (6, 1)),
        "action_target_position": np.tile(robot.target_position, (6, 1)),
        "action_target_velocity": np.tile(robot.target_velocity, (6, 1)),
        "node_position_world": np.ones((6, 140, 3)),
        "node_velocity_world": np.ones((6, 140, 3)),
        "contact_force_on_block_world": np.ones((6, 3)),
        "contact_position_world": np.ones((6, 3)),
        "contact_frame_world": np.ones((6, 9)),
    }
    before = predictor.rollout(
        initial, [IndenterInput.from_arrays(arrays, t) for t in range(5)], method
    )
    for key in arrays:
        if key.startswith(("node_", "contact_")):
            arrays[key][:] = np.nan  # Poison *all* soft/contact labels, including current forces.
    after = predictor.rollout(
        initial, [IndenterInput.from_arrays(arrays, t) for t in range(5)], method
    )
    assert all(
        torch.equal(a.state.centers, b.state.centers)
        and torch.equal(a.state.velocities, b.state.velocities)
        for a, b in zip(before, after, strict=True)
    )


def test_detached_gate_exposes_constant_velocity_recovery_limit(pipeline):
    predictor, state, robot, _ = pipeline
    velocity = torch.zeros_like(state.velocities)
    velocity[~predictor.pinned, 2] = -0.001
    displaced = state.centers.clone()
    displaced[~predictor.pinned, 2] -= 0.001
    initial = predictor.gaussian_state(displaced, velocity)
    robot = replace(robot, position=np.array([0.0, 0.0, 0.2]), velocity=np.zeros(3))
    for method in METHODS[:-1]:
        result = predictor.step(initial, robot, method)
        assert result.contacts.num_contacts == 0
        assert torch.count_nonzero(result.residual) == 0
        torch.testing.assert_close(result.delta, velocity * 0.01, atol=0, rtol=0)
        assert torch.equal(result.state.velocities, velocity)


def test_two_parameter_nonnegative_calibration_handles_boundary_optimum():
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(nonnegative_two_column_fit(design, design @ [2.0, 3.0]), [2.0, 3.0])
    np.testing.assert_allclose(nonnegative_two_column_fit(np.eye(2), [4.0, -1.0]), [4.0, 0.0])
