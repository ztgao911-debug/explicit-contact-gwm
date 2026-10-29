from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from contact_gwm.budget_evaluation import (
    assess_existing_evaluation,
    physical_archive_checks,
    physical_state_checks,
    recovery_waveform_metrics,
    signed_waveform_summary,
)
from contact_gwm.data.mujoco_soft import tetra_volumes


def tetra_mesh():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tetrahedra = np.array([[0, 1, 2, 3]])
    return {
        "rest_positions_world": rest,
        "node_ids": np.arange(4),
        "pinned_mask": np.array([True, False, False, False]),
        "tetrahedra": tetrahedra,
        "rest_signed_tetra_volumes": tetra_volumes(rest, tetrahedra),
    }


def test_physics_gate_retains_inverted_and_fixed_boundary_states():
    mesh = tetra_mesh()
    x = np.repeat(mesh["rest_positions_world"][None], 3, axis=0)
    v = np.zeros_like(x)
    good = physical_state_checks(x, v, mesh, expected_steps=2)
    assert good["eligible"] and good["minimum_volume_ratio"] == pytest.approx(1)
    x[1, 3, 2] = -1
    v[2, 0, 2] = 1e-8
    bad = physical_state_checks(x, v, mesh, expected_steps=2)
    assert not bad["eligible"]
    assert bad["inverted_states"] == 1 and bad["inverted_horizons"] == [1]
    assert bad["fixed_boundary_violating_states"] == 1
    assert bad["minimum_volume_ratio"] == pytest.approx(-1)
    x[2, 2, 1] = np.nan
    assert physical_state_checks(x, v, mesh)["nonfinite_states"] == 1


def test_archive_methods_are_separate_and_missing_windows_cannot_disappear(tmp_path):
    mesh = tetra_mesh()
    x = np.repeat(mesh["rest_positions_world"][None], 3, axis=0)
    bad = x.copy()
    bad[2, 3, 2] = -1
    path = tmp_path / "predictions.npz"
    np.savez(
        path,
        node_ids=mesh["node_ids"],
        **{
            "left_deep__70__M__position": x,
            "left_deep__70__M__velocity": np.zeros_like(x),
            "left_deep__70__G__position": bad,
            "left_deep__70__G__velocity": np.zeros_like(x),
        },
    )
    both = physical_archive_checks(
        path, mesh, expected_windows=[("left_deep", 70)], expected_steps=2
    )
    assert not both["eligible"]
    assert both["methods"]["M"]["eligible"]
    assert both["methods"]["G"]["inverted_windows"] == 1
    only_m = physical_archive_checks(
        path, mesh, method="M", expected_windows=[("left_deep", 70)], expected_steps=2
    )
    assert only_m["eligible"]
    missing = physical_archive_checks(
        path,
        mesh,
        method="M",
        expected_windows=[("left_deep", 70), ("left_deep", 110)],
        expected_steps=2,
    )
    assert not missing["eligible"] and missing["methods"]["M"]["missing_windows"] == 1
    assert any(row.get("error") == "missing_window" for row in missing["windows"])


def test_fixed_ranking_never_uses_a_candidate_training_velocity_weight():
    rows = [
        {
            "method": "M",
            "split": "validation",
            "phase": phase,
            "region": region,
            "horizon": h,
            "contact_group": "all",
            "position_rmse_um": 1000.0,
            "velocity_rmse_mm_s": 100.0,
        }
        for phase in ("press", "hold", "release", "recover")
        for region in ("active", "contact_neighborhood")
        for h in (5, 10, 20)
    ]
    result = assess_existing_evaluation({"rollout": rows, "checks": {}}, method="M")
    assert result["eligible"]
    assert result["scores"]["joint_score"] == pytest.approx(1)
    del rows[-1]
    missing = assess_existing_evaluation({"rollout": rows, "checks": {}}, method="M")
    assert (
        not missing["eligible"] and "ranking_cells_incomplete" in missing["ineligibility_reasons"]
    )


def test_phase_is_undefined_for_short_monotone_recovery():
    reference = np.linspace(1, 0, 21)
    prediction = reference * 0.5
    result = signed_waveform_summary(prediction, reference, 0.01)
    assert result["amplitude_ratio"] == pytest.approx(0.5)
    assert result["phase_status"] == "insufficient_cycles"
    assert result["phase_error_degrees"] is None and result["lag_s"] is None


def test_phase_proxy_has_correct_lag_sign_for_repeated_oscillation():
    dt = 0.001
    t = np.arange(501) * dt
    reference = np.sin(2 * np.pi * 10 * t)
    delayed = np.sin(2 * np.pi * 10 * (t - 0.02))
    result = signed_waveform_summary(delayed, reference, dt)
    assert result["phase_status"] == "lag_proxy_at_least_two_cycles"
    assert result["lag_s"] == pytest.approx(0.02, abs=0.001)
    assert result["phase_error_degrees"] == pytest.approx(72, abs=4)
    json.dumps(result, allow_nan=False)


def test_signed_recovery_export_preserves_material_node_and_velocity(tmp_path):
    mesh = tetra_mesh()
    n = 31
    rest = mesh["rest_positions_world"]
    x = np.repeat(rest[None], n, axis=0)
    v = np.zeros_like(x)
    x[:, 3, 2] -= np.linspace(0, 0.01, n)
    v[:, 3, 2] = -0.2
    arrays = {
        "time": np.arange(n) * 0.01,
        "node_position_world": x,
        "node_velocity_world": v,
        "indenter_position_world": np.tile([0.0, 0.0, 2.0], (n, 1)),
    }
    pred_x, pred_v = x[10:31].copy(), v[10:31].copy()
    pred_x[:, 3, 2] += 0.001
    pred_v[:, 3, 2] = -0.3
    path = tmp_path / "rollout.npz"
    np.savez(
        path,
        node_ids=mesh["node_ids"],
        **{
            "left_deep__10__M__position": pred_x,
            "left_deep__10__M__velocity": pred_v,
        },
    )
    rois = {"left_deep": np.tile(~mesh["pinned_mask"], (n, 1))}
    result = recovery_waveform_metrics(
        path, mesh, {"left_deep": arrays}, tmp_path / "curves", recover_starts=(10,), roi_masks=rois
    )
    assert result["waveforms"] == 3 and result["samples"] == 63
    with (tmp_path / "curves/recovery_signed_waveforms.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["channel"] == "representative_node"]
    assert len(rows) == 21 and {int(r["representative_node"]) for r in rows} == {3}
    assert float(rows[0]["predicted_z_displacement_um"]) < 0
    assert float(rows[0]["predicted_vz_mm_s"]) == pytest.approx(-300)
    assert float(rows[0]["reference_vz_mm_s"]) == pytest.approx(-200)
