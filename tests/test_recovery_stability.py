"""Finite perturbation units, fixed boundary and actual independent replay."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from contact_gwm.recovery_stability import (
    NativeRecoveryReplay,
    make_perturbations,
    normalized_state_norm,
)

SETTINGS = {
    "direction_count": 3,
    "signs": [-1, 1],
    "position_rms_m": [1e-7, 1e-6],
    "velocity_rms_m_s": [1e-5, 1e-4],
}


def test_paired_perturbations_seed_units_and_pins():
    pinned = np.array([True, False, False, False])
    cases, dx, dv, directions = make_perturbations(pinned, SETTINGS, 7)
    other = make_perturbations(pinned, SETTINGS, 7)
    assert len(cases) == 25
    assert all(np.array_equal(a, b) for a, b in zip((dx, dv, directions), other[1:], strict=True))
    assert not np.any(dx[:, pinned]) and not np.any(dv[:, pinned])
    np.testing.assert_allclose(np.square(directions[:, ~pinned]).sum(-1).mean(-1), 1.0)
    for i in range(1, len(cases), 2):
        np.testing.assert_array_equal(dx[i], -dx[i + 1])
        np.testing.assert_array_equal(dv[i], -dv[i + 1])
        field = dx[i] if cases[i]["kind"] == "position" else dv[i]
        assert np.sqrt(np.square(field[~pinned]).sum(-1).mean()) == pytest.approx(
            cases[i]["amplitude"]
        )


def test_constant_velocity_gain_is_not_an_energy_or_instability_test():
    mask = np.array([False, True, True])
    direction = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    eps, xu, vu = 1e-4, 1e-3, 0.1
    t = np.arange(21)[:, None, None] * 0.01
    dv = np.broadcast_to(eps * direction, (21, 3, 3))
    dx = t * dv
    gains = normalized_state_norm(dx, dv, mask, xu, vu) / (eps / vu)
    np.testing.assert_allclose(gains, np.sqrt(1 + np.arange(21) ** 2))
    assert gains[-1] > 20
    pos_only = normalized_state_norm(
        np.broadcast_to(eps * direction, dx.shape), np.zeros_like(dx), mask, xu, vu
    )
    np.testing.assert_allclose(pos_only / (eps / xu), 1.0)


def test_normalized_norm_masks_units_and_sign_symmetry():
    mask = np.array([False, True])
    dx = np.array([[100.0, 100.0, 100.0], [0.001, 0.0, 0.0]])
    dv = np.array([[100.0, 100.0, 100.0], [0.0, 0.1, 0.0]])
    assert normalized_state_norm(dx, dv, mask, 0.001, 0.1) == pytest.approx(np.sqrt(2))
    assert normalized_state_norm(-dx, -dv, mask, 0.001, 0.1) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        normalized_state_norm(dx, dv, mask, 0.0, 0.1)
    with pytest.raises(ValueError):
        normalized_state_norm(dx, dv, np.zeros(2, dtype=bool), 0.001, 0.1)


@pytest.fixture
def native_scene(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from contact_gwm.data.mujoco_soft import build_xml, mesh_from_model

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/mujoco_soft_pilot.yaml").read_text()
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(build_xml(config))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[-3:] = [0.0, 0.0, 0.068]
    data.ctrl[:] = [0.0, 0.0, 0.068]
    mujoco.mj_forward(model, data)
    mesh = mesh_from_model(model, data)
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
    return NativeRecoveryReplay(scene, mesh), mesh, initial, data.ctrl.copy()


def test_native_zero_replay_is_deterministic_and_preserves_rest(native_scene):
    replay, mesh, initial, command = native_scene
    zero = np.zeros_like(mesh["rest_positions_world"])
    a = replay.rollout(initial, command, zero, zero, 2, 0.01)
    b = replay.rollout(initial, command, zero, zero, 2, 0.01)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    np.testing.assert_allclose(
        a[0], np.broadcast_to(mesh["rest_positions_world"], a[0].shape), atol=1e-12
    )
    np.testing.assert_allclose(a[1], 0.0, atol=1e-12)
    assert a[2]["nodal_velocity_reader_vs_jacobian_error"] == 0
    assert a[2]["max_fixed_position_or_velocity_error"] == 0


@pytest.mark.parametrize("kind", ["position", "velocity"])
def test_native_injects_only_selected_initial_state_field(native_scene, kind):
    replay, mesh, initial, command = native_scene
    delta = np.zeros_like(mesh["rest_positions_world"])
    free = np.flatnonzero(~mesh["pinned_mask"])
    delta[free[0], 2] = 1e-7 if kind == "position" else 1e-5
    zero = np.zeros_like(delta)
    dx, dv = (delta, zero) if kind == "position" else (zero, delta)
    x, v, stats = replay.rollout(initial, command, dx, dv, 2, 0.01)
    np.testing.assert_allclose(x[0], mesh["rest_positions_world"] + dx, atol=1e-15)
    np.testing.assert_allclose(v[0], dv, atol=1e-15)
    assert stats["warnings"] == [] and stats["minimum_tetra_volume_ratio"] > 0
    assert stats["max_contact_count_including_internal_steps"] == 0
    assert np.isfinite(x).all() and np.isfinite(v).all()


def test_native_restores_state_once_and_accepts_no_future_arrays(native_scene, monkeypatch):
    import mujoco

    replay, mesh, initial, command = native_scene
    assert set(inspect.signature(replay.rollout).parameters) == {
        "initial_integration_state",
        "command",
        "delta_x",
        "delta_v",
        "steps",
        "record_dt",
    }
    original, calls = mujoco.mj_setState, []

    def observed(*args):
        calls.append(args[2].copy())
        return original(*args)

    monkeypatch.setattr(mujoco, "mj_setState", observed)
    zero = np.zeros_like(mesh["rest_positions_world"])
    replay.rollout(initial, command, zero, zero, 3, 0.01)
    assert len(calls) == 1 and np.array_equal(calls[0], initial)
    zero[mesh["pinned_mask"], 2] = 1e-9
    with pytest.raises(ValueError, match="fixed"):
        replay.rollout(initial, command, zero, np.zeros_like(zero), 1, 0.01)


def test_exported_central_metrics_match_exact_linear_shear(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    entry = importlib.import_module("diagnose_recovery_stability")
    pinned = np.array([True, False, False])
    cases, dx, dv, _ = make_perturbations(pinned, SETTINGS, 7)
    x0, v0 = np.zeros((3, 3)), np.zeros((3, 3))
    v0[1:, 2] = 0.001
    t = np.arange(21)[:, None, None] * 0.01
    x = np.asarray(
        [x0 + delta_x + t * (v0 + delta_v) for delta_x, delta_v in zip(dx, dv, strict=True)]
    )
    v = np.asarray([np.broadcast_to(v0 + delta_v, (21, 3, 3)) for delta_v in dv])
    saved = {
        f"{method}__{field}": value.copy()
        for method in entry.METHODS
        for field, value in (("position", x), ("velocity", v))
    }
    config = {
        "state_norm": {"position_unit_m": 0.001, "velocity_unit_m_s": 0.1},
        "perturbations": SETTINGS,
        "evaluation": {"horizons": [1, 5, 10, 20]},
    }
    baseline, summary, linearity = entry.metrics(
        config,
        cases,
        saved,
        {"pinned": pinned, "neighborhood": np.array([False, True, False])},
        tmp_path,
    )
    assert all(row["position_rmse_um"] == row["velocity_rmse_mm_s"] == 0 for row in baseline)
    for row in summary:
        expected = 1.0 if row["kind"] == "position" else np.sqrt(1 + row["horizon"] ** 2)
        assert row["mean_gain"] == pytest.approx(expected, rel=1e-8)
        assert row["gain_ratio_of_means_vs_mujoco"] == pytest.approx(1.0)
        assert row["mean_response_mismatch_vs_mujoco"] == 0
    assert max(row["relative_response_difference_between_amplitudes"] for row in linearity) < 1e-8
