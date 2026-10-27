"""Force-free analytic prediction, independent interventions and signed responses."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from contact_gwm.contact_audit import (
    NativeContactIntervention,
    intervention_robot,
    signed_response_metrics,
)
from contact_gwm.joint_state_dynamics import JointStateDynamics, JointStatePipeline
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline
from contact_gwm.spatial_ablation import SpatialEncodedResidual


@pytest.fixture
def audit_scene(tmp_path):
    mj = pytest.importorskip("mujoco")
    from contact_gwm.data.mujoco_soft import build_xml, mesh_from_model

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/mujoco_soft_pilot.yaml").read_text())
    scene = tmp_path / "scene.xml"
    scene.write_text(build_xml(config))
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    data.qpos[-3:] = [0, 0, 0.068]
    data.ctrl[:] = [0, 0, 0.068]
    mj.mj_forward(model, data)
    mesh = mesh_from_model(model, data)
    initial = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, initial, mj.mjtState.mjSTATE_INTEGRATION)
    torch.manual_seed(7)
    rest = torch.tensor(mesh["rest_positions_world"]).float()
    pin = torch.tensor(mesh["pinned_mask"])
    training = yaml.safe_load((root / "configs/train_soft_local.yaml").read_text())
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"model_state": SpatialEncodedResidual(rest).state_dict(), "config": training}, checkpoint
    )
    legacy = PinnedContactPipeline(
        mesh,
        scene,
        checkpoint,
        yaml.safe_load((root / "configs/validate_contact_pipeline.yaml").read_text()),
    )
    network = JointStateDynamics(rest, pin, hidden_dim=8)
    with torch.no_grad():
        network.internal_network[4].weight.normal_(0, 0.01)
        network.contact_network[4].weight.normal_(0, 0.01)
    return mj, scene, mesh, initial, legacy, network


def touching_robot():
    return IndenterInput(
        np.array([0.0, 0.0, 0.042]), np.zeros(3), np.array([0.0, 0.0, 0.042]), np.zeros(3)
    )


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden native query")


@pytest.mark.parametrize("method", ["analytic_force", "gaussian_geometry"])
def test_analytic_joint_conditions_never_call_native_force_solver(audit_scene, method):
    mj, _, mesh, _, legacy, network = audit_scene
    state = legacy.gaussian_state(
        mesh["rest_positions_world"], np.zeros_like(mesh["rest_positions_world"])
    )
    pipe = JointStatePipeline(legacy, network, contact_method=method)
    with patch.object(mj, "mj_contactForce", forbidden), patch.object(mj, "mj_forward", forbidden):
        result = pipe.step(state, touching_robot())
    assert torch.isfinite(result.state.centers).all() and result.contacts.num_contacts > 0
    assert not torch.any(result.delta[legacy.pinned])
    if method == "gaussian_geometry":
        with patch.object(legacy.native, "query", forbidden):
            pipe.step(state, touching_robot())


def test_default_native_condition_is_backward_compatible_and_common_geometry_matches(audit_scene):
    _, _, mesh, _, legacy, network = audit_scene
    state = legacy.gaussian_state(
        mesh["rest_positions_world"], np.zeros_like(mesh["rest_positions_world"])
    )
    original = JointStatePipeline(legacy, network)
    explicit = JointStatePipeline(legacy, network, contact_method="mujoco_force")
    a, b = original.step(state, touching_robot()), explicit.step(state, touching_robot())
    assert torch.equal(a.state.centers, b.state.centers) and torch.equal(
        a.state.velocities, b.state.velocities
    )
    c = (
        JointStatePipeline(legacy, network, contact_method="analytic_force")
        .current_inputs(state, touching_robot())
        .contacts
    )
    assert torch.equal(a.contacts.points, c.points) and torch.equal(a.contacts.normals, c.normals)
    with pytest.raises(ValueError):
        JointStatePipeline(legacy, network, contact_method="future_force")


def test_internal_only_is_robot_invariant_but_retains_soft_dynamics(audit_scene):
    _, _, mesh, _, legacy, network = audit_scene
    x = mesh["rest_positions_world"].copy()
    x[~mesh["pinned_mask"], 2] -= 0.0003
    state = legacy.gaussian_state(x, np.zeros_like(x))
    pipe = JointStatePipeline(legacy, network, contact_method="none")
    distant = intervention_robot(
        touching_robot(), {"position_offset_m": [0, 0, 0.02], "velocity_offset_m_s": [0, 0, 0]}
    )
    with patch.object(legacy.native, "query", forbidden):
        a, b = pipe.step(state, touching_robot()), pipe.step(state, distant)
    assert torch.equal(a.state.centers, b.state.centers) and torch.equal(
        a.state.velocities, b.state.velocities
    )
    assert not a.contacts.num_contacts and torch.any(a.residual[~legacy.pinned])
    assert not torch.any(a.state.velocities[legacy.pinned])


def test_native_intervention_changes_only_robot_initial_state_and_produces_contact_response(
    audit_scene,
):
    _, scene, mesh, initial, _, _ = audit_scene
    solver = NativeContactIntervention(scene, mesh)
    robot = touching_robot()
    a, ac = solver.rollout(initial, robot, 2, 0.01)
    lifted = intervention_robot(
        robot, {"position_offset_m": [0, 0, 0.006], "velocity_offset_m_s": [0, 0, 0]}
    )
    b, bc = solver.rollout(initial, lifted, 2, 0.01)
    assert np.array_equal(a["position"][0], b["position"][0])
    assert np.array_equal(a["velocity"][0], b["velocity"][0])
    assert a["net_force"][0, 2] < 0 and np.array_equal(b["net_force"][0], np.zeros(3))
    assert not np.array_equal(a["position"][1], b["position"][1])
    assert (
        ac["instant_velocity_vs_jacobian_max_error"]
        == bc["instant_velocity_vs_jacobian_max_error"]
        == 0
    )
    assert (
        ac["max_fixed_position_or_velocity_error"]
        == bc["max_fixed_position_or_velocity_error"]
        == 0
    )


def test_native_intervention_restores_once_and_accepts_no_future_labels(audit_scene, monkeypatch):
    mj, scene, mesh, initial, _, _ = audit_scene
    solver = NativeContactIntervention(scene, mesh)
    assert set(inspect.signature(solver.rollout).parameters) == {
        "initial_integration_state",
        "robot",
        "steps",
        "record_dt",
    }
    calls = []
    original = mj.mj_setState

    def observe(*args):
        calls.append(args[2].copy())
        return original(*args)

    monkeypatch.setattr(mj, "mj_setState", observe)
    saved = initial.copy()
    a, checks = solver.rollout(initial, touching_robot(), 2, 0.01)
    assert len(calls) == 1 and np.array_equal(initial, saved)
    b, _ = solver.rollout(initial, touching_robot(), 2, 0.01)
    assert all(np.array_equal(value, b[key]) for key, value in a.items())
    assert checks["initial_warmstart_reset_to_zero"]


def test_signed_response_distinguishes_correct_opposite_and_zero_direction():
    mask = np.array([False, True])
    truth = np.array([[9.0, 9.0, 9.0], [0.0, 0.0, -1e-6]])
    same = signed_response_metrics(truth, truth, mask, 1e6)
    opposite = signed_response_metrics(-truth, truth, mask, 1e6)
    zero = signed_response_metrics(np.zeros_like(truth), truth, mask, 1e6)
    assert same["response_rmse"] == 0 and same["projection_gain"] == same["direction_cosine"] == 1
    assert (
        opposite["response_rmse"] == pytest.approx(2)
        and opposite["projection_gain"] == opposite["direction_cosine"] == -1
    )
    assert zero["response_rmse"] == zero["reference_response_rms"] == 1
    assert (
        zero["amplitude_ratio"] == zero["projection_gain"] == 0 and zero["direction_cosine"] is None
    )
