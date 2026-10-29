from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bounded_search_training import (  # noqa: E402
    SearchDeadline,
    branch_window_loss,
    fixed_validation,
)


def fixture_data():
    rest = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).double()
    cache = {
        "context": torch.zeros((20, 4, 36)),
        "gates": torch.ones((20, 4)),
        "features": torch.zeros((20, 4, 10)),
        "velocity": torch.zeros((20, 4, 3)),
        "pinned": torch.zeros(4, dtype=torch.bool),
    }
    cache["features"][:, :, 0] = torch.arange(20)[:, None]
    data = {
        "branch_positions": rest[None, None].repeat(1, 21, 1, 1),
        "branch_velocities": torch.zeros((1, 21, 4, 3)).double(),
        "branch_roi": torch.ones((1, 4), dtype=torch.bool),
        "branch_cache": cache,
        "branch_robots": [[None] * 20],
    }
    mesh = {
        "tetra_tensor": torch.tensor([[0, 1, 2, 3]]),
        "rest_volume_tensor": torch.tensor([1 / 6]).double(),
        "pinned_mask": cache["pinned"].numpy(),
    }
    settings = {
        "position_loss_unit_m": 0.001,
        "velocity_loss_unit_m_s": 0.1,
        "velocity_weight": 0.5,
        "neighborhood_weight": 0.5,
    }
    return rest, data, mesh, settings


class ConstantEndpointPath:
    def __init__(self, rest, offset, velocity_offset):
        self.rest, self.offset, self.velocity_offset = rest, offset, velocity_offset
        self.calls = 0

    def step(self, x, v, robots):
        self.calls += 1
        return self.rest[None].expand_as(x) + self.offset, torch.zeros_like(
            v
        ) + self.velocity_offset


class ConstantIncrementModel:
    def __init__(self, offset, velocity_offset):
        self.offset, self.velocity_offset = offset, velocity_offset
        self.observed_indices = None

    def __call__(self, context, gates, features, velocity):
        self.observed_indices = features[:, 0, 0].clone()
        return SimpleNamespace(
            delta_position=torch.zeros_like(velocity) + self.offset,
            delta_velocity=torch.zeros_like(velocity) + self.velocity_offset,
        )


@pytest.mark.parametrize("horizon", [3, 5, 10])
@pytest.mark.parametrize("mode", ["teacher_forcing", "autoregressive"])
def test_window_loss_averages_the_same_endpoint_error_over_time(horizon, mode):
    rest, data, mesh, settings = fixture_data()
    offset = torch.tensor(0.001, dtype=torch.float64, requires_grad=True)
    model = ConstantIncrementModel(offset, 0.02)
    path = ConstantEndpointPath(rest, offset, 0.02)
    loss, penalty, _ = branch_window_loss(
        model,
        path,
        data,
        torch.tensor([0, 0]),
        torch.tensor([0.25, 0.75]),
        {"horizon": horizon, "mode": mode, "mesh_weight": 0.0},
        settings,
        mesh,
        float("inf"),
    )
    # Three-coordinate MSE: position=1, velocity=.04, fixed .5/.5 -> .52.
    # A sum instead of a temporal mean would scale this by horizon.
    assert float(loss.detach()) == pytest.approx(0.52)
    assert float(penalty.detach()) == 0
    gradient = torch.autograd.grad(loss, offset)[0]
    assert torch.isfinite(gradient) and float(gradient) == pytest.approx(1000.0)
    if mode == "teacher_forcing":
        assert path.calls == 0
        starts = (torch.tensor([0.25, 0.75]) * (21 - horizon)).long()
        expected = (starts[:, None] + torch.arange(horizon)).flatten().float()
        assert torch.equal(model.observed_indices, expected)
    else:
        assert path.calls == horizon


def test_mesh_penalty_is_unweighted_and_averaged_over_endpoints():
    rest, data, mesh, settings = fixture_data()
    inverted = rest.clone()
    inverted[3, 2] = -0.2
    path = ConstantEndpointPath(inverted, torch.tensor(0.0).double(), 0.0)
    penalties = []
    for horizon, weight in [(3, 0.001), (10, 0.1)]:
        _, penalty, info = branch_window_loss(
            None,
            path,
            data,
            torch.tensor([0]),
            torch.tensor([0.0]),
            {"horizon": horizon, "mode": "autoregressive", "mesh_weight": weight},
            settings,
            mesh,
            float("inf"),
        )
        penalties.append(float(penalty))
        assert info["training_inverted_state_observations"] == horizon
        assert info["training_window_minimum_volume_ratio"] == pytest.approx(-0.2)
    assert penalties == pytest.approx([0.09, 0.09])


class ValidationPath:
    def rollout(self, x, v, external):
        if external[0][0] == "fail":
            raise FloatingPointError("Deliberate invalid algorithmic window")
        return [(x, v), *[(x + 0.001, v + 0.02) for _ in range(5)]]


def windows(rest, *, fail_one=False):
    result = []
    for phase in ("press", "hold", "release", "recover"):
        for repeat in range(2):
            result.append(
                {
                    "initial_x": rest,
                    "initial_v": torch.zeros_like(rest),
                    "targets_x": rest[None].repeat(5, 1, 1),
                    "targets_v": torch.zeros((5, 4, 3)).double(),
                    "roi": torch.tensor([False, True, True, False]),
                    "phase": phase,
                    "sourcecase": f"{phase}/{repeat}",
                    "external": [
                        "fail" if fail_one and phase == "recover" and repeat == 1 else phase
                    ]
                    * 5,
                }
            )
    return result


def test_fixed_validation_has_exact_fixed_position_velocity_weights_and_units():
    rest, _, mesh, _ = fixture_data()
    scores, rows, failures = fixed_validation(ValidationPath(), windows(rest), mesh, float("inf"))
    assert not failures and scores["coverage_complete"] and len(rows) == 8
    # Evaluation uses VECTOR RMSE (three times coordinate MSE), independent of
    # all candidate training weights: normalized position=3, velocity=.12.
    assert scores["position_score"] == pytest.approx(3.0)
    assert scores["velocity_score"] == pytest.approx(0.12)
    assert scores["joint_score"] == pytest.approx(1.56)


def test_any_failed_validation_window_disqualifies_even_with_all_eight_cells_present():
    rest, _, mesh, _ = fixture_data()
    scores, rows, failures = fixed_validation(
        ValidationPath(),
        windows(rest, fail_one=True),
        mesh,
        float("inf"),
    )
    assert len(rows) == 8 and len(failures) == 1
    assert not scores["coverage_complete"]
    assert scores["joint_score"] is None


def test_deadline_is_not_converted_to_an_algorithm_failure():
    rest, _, mesh, _ = fixture_data()
    with pytest.raises(SearchDeadline):
        fixed_validation(ValidationPath(), windows(rest), mesh, 0.0)
