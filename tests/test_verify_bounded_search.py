from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_bounded_search import independent_physics, independent_scores  # noqa: E402


def mesh_fixture():
    return {
        "rest_positions_world": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "pinned_mask": np.array([True, False, False, False]),
        "tetrahedra": np.array([[0, 1, 2, 3]]),
        "rest_signed_tetra_volumes": np.array([1 / 6]),
    }


def test_independent_determinant_gate_retains_flip_and_fixed_violation():
    mesh = mesh_fixture()
    x = np.repeat(mesh["rest_positions_world"][None], 21, axis=0)
    v = np.zeros_like(x)
    assert independent_physics(x, v, mesh)["eligible"]
    x[5, 3, 2] = -0.2
    v[7, 0, 2] = 1e-10
    result = independent_physics(x, v, mesh)
    assert not result["eligible"]
    assert result["minimum_volume_ratio"] == pytest.approx(-0.2)
    assert result["inverted_states"] == 1
    assert result["fixed_boundary_violating_states"] == 1
    assert not independent_physics(x[:20], v[:20], mesh)["eligible"]


def test_independent_score_requires_all_24_cells_and_uses_node_norm(tmp_path):
    folder = tmp_path / "evaluation"
    (folder / "ordinary").mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    mesh = mesh_fixture()
    predictions, masks = {}, {}
    for name in ("left_shallow", "left_deep"):
        phases = np.zeros(121, dtype=int)
        truth = np.zeros((121, 4, 3))
        for phase, start in enumerate((0, 30, 60, 90), start=1):
            phases[start] = phase
            prefix = f"{name}__{start}__M__"
            predictions[prefix + "position"] = np.ones((21, 4, 3)) * 0.001
            predictions[prefix + "velocity"] = np.ones((21, 4, 3)) * 0.1
        masks[f"{name}__contact_neighborhood_mask"] = np.ones((121, 4), dtype=bool)
        np.savez(
            data / f"{name}.npz",
            node_position_world=truth,
            node_velocity_world=truth,
            phase_id=phases,
        )
    np.savez(folder / "ordinary/rollout_predictions.npz", **predictions)
    np.savez(folder / "ordinary/one_step_predictions.npz", **masks)
    result = independent_scores(folder, data, mesh)
    assert result["coverage_complete"] and len(result["cells"]) == 24
    # Three coordinate squares per node: these are vector RMSE, not per-coordinate RMSE.
    assert result["joint_score"] == pytest.approx(3)
    for key in list(predictions):
        if "__90__" in key:
            del predictions[key]
    np.savez(folder / "ordinary/rollout_predictions.npz", **predictions)
    assert independent_scores(folder, data, mesh)["joint_score"] is None
