from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bounded_search_evaluation as evaluation  # noqa: E402

from contact_gwm.data.mujoco_soft import tetra_volumes  # noqa: E402


def mesh_and_arrays():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tet = np.array([[0, 1, 2, 3]])
    mesh = {
        "rest_positions_world": rest,
        "node_ids": np.arange(4),
        "pinned_mask": np.array([True, False, False, False]),
        "tetrahedra": tet,
        "rest_signed_tetra_volumes": tetra_volumes(rest, tet),
    }
    arrays = {
        "node_position_world": np.repeat(rest[None], 301, axis=0),
        "node_velocity_world": np.zeros((301, 4, 3)),
        "time": np.arange(301) * 0.01,
        "indenter_position_world": np.repeat([[0.0, 0.0, 2.0]], 301, axis=0),
        "contact_offsets": np.zeros(302, dtype=int),
        "contact_position_world": np.empty((0, 3)),
        "contact_signed_distance": np.empty((0, 1)),
        "contact_force_on_block_world": np.empty((0, 3)),
    }
    return mesh, arrays


@pytest.mark.parametrize("exception_type", [FloatingPointError, TimeoutError])
def test_failed_forward_keeps_exact_current_input_and_bubbles_algorithm_failure(
    tmp_path, monkeypatch, exception_type
):
    mesh, arrays = mesh_and_arrays()
    state = SimpleNamespace(
        centers=torch.tensor(mesh["rest_positions_world"]),
        velocities=torch.zeros(4, 3),
        features=torch.zeros(4, 10),
    )
    robot = SimpleNamespace(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.zeros(3),
        target_position=np.array([0.0, 0.0, 2.0]),
        target_velocity=np.zeros(3),
    )

    original_exception = exception_type("candidate forward stopped")

    def fail(state, robot):
        raise original_exception

    def ordinary(predictors, *a, **kw):
        predictors["M"](state, robot)

    monkeypatch.setattr(evaluation, "evaluate_models", ordinary)
    with pytest.raises(exception_type, match="candidate forward stopped") as caught:
        evaluation.evaluate_trial(
            {"M": fail},
            SimpleNamespace(gaussian_state=None),
            {"left_deep": arrays},
            {"left_deep": "validation"},
            mesh,
            {},
            tmp_path,
        )
    assert caught.value is original_exception
    with np.load(tmp_path / "failed_forward/M.npz") as saved:
        assert np.array_equal(saved["input_position"], mesh["rest_positions_world"])
        assert saved["input_features"].shape == (4, 10)
        assert not any("reference" in k or "future" in k for k in saved.files)
    failure = json.loads((tmp_path / "failed_forward/failure.json").read_text())
    assert failure["last_started_method"] == "M"
    assert failure["last_forward_evidence"][0]["matching_true_current_frames"]


def test_invalid_method_cannot_enter_valid_ranking_and_does_not_mask_other_method(
    tmp_path, monkeypatch
):
    mesh, arrays = mesh_and_arrays()
    x = np.repeat(mesh["rest_positions_world"][None], 21, axis=0)
    v = np.zeros_like(x)
    inverted = x.copy()
    inverted[10, 3, 2] = -1
    score_rows = [
        {
            "method": method,
            "split": "validation",
            "phase": phase,
            "region": region,
            "horizon": h,
            "contact_group": "all",
            "position_rmse_um": 100.0,
            "velocity_rmse_mm_s": 10.0,
        }
        for method in ("M", "G")
        for phase in ("press", "hold", "release", "recover")
        for region in ("active", "contact_neighborhood")
        for h in (5, 10, 20)
    ]

    def ordinary(predictors, state_factory, trajectories, splits, mesh, config, folder):
        folder.mkdir(parents=True)
        np.savez(
            folder / "rollout_predictions.npz",
            node_ids=mesh["node_ids"],
            **{
                "left_deep__20__M__position": x,
                "left_deep__20__M__velocity": v,
                "left_deep__20__G__position": inverted,
                "left_deep__20__G__velocity": v,
            },
        )
        np.savez(folder / "one_step_predictions.npz", node_ids=mesh["node_ids"])
        return {
            "rollout": score_rows,
            "one_step": [],
            "checks": {"rollout_failures": [], "nonfinite_result_count": 0},
        }

    def responses(predictors, legacy, trajectories, mesh, config, output, root):
        folder = output / "response_evaluation"
        folder.mkdir()
        for frame in config["collection"]["validation_frames"]:
            for variant in config["_variants"]:
                np.savez(
                    folder / f"{frame}_{variant['name']}_predictions.npz",
                    **{
                        f"{method}__{field}": value
                        for method in predictors
                        for field, value in [("position", x), ("velocity", v)]
                    },
                )
        return {"summary": [], "failures": []}

    monkeypatch.setattr(evaluation, "evaluate_models", ordinary)
    monkeypatch.setattr(evaluation, "evaluate_responses", responses)
    config = {
        "rollout": {"start_frames": [20], "horizons": [5, 10, 20]},
        "collection": {"validation_frames": [70]},
        "_variants": [{"name": "base"}, {"name": "detach"}],
    }
    predictors = {"M": None, "G": None}
    result = evaluation.evaluate_trial(
        predictors,
        SimpleNamespace(gaussian_state=None),
        {"left_deep": arrays},
        {"left_deep": "validation"},
        mesh,
        config,
        tmp_path,
    )
    assert result["per_method"]["M"]["eligible"]
    assert not result["per_method"]["G"]["eligible"]
    assert result["per_method"]["G"]["inverted_windows"] == 1
    assert result["per_method"]["G"]["minimum_volume_ratio"] == pytest.approx(-1)
    assert (tmp_path / "ordinary_physics_windows.csv").exists()
    assert (tmp_path / "recovery/recovery_checks.json").exists()
