"""Protocol isolation, across-seed statistics and validation-only custom-model selection."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from contact_gwm.comparison import PreparedPairs, fit_model
from contact_gwm.spatial_ablation import make_ablation_model
from contact_gwm.trajectory_spatial import aggregate_seed_metrics, validate_protocol


def config():
    root = Path(__file__).resolve().parents[1]
    result = yaml.safe_load((root / "configs/train_soft_local.yaml").read_text())
    extra = yaml.safe_load((root / "configs/compare_soft_spatial_trajectories.yaml").read_text())
    extra.pop("base_config")
    result.update(extra)
    return result


def test_protocol_rejects_test_arrays_and_duplicate_seeds():
    settings = config()
    validate_protocol(settings)
    wrong = copy.deepcopy(settings)
    wrong["splits"]["validation"] = ["right_shallow", "right_deep"]
    with pytest.raises(ValueError, match="no test split"):
        validate_protocol(wrong)
    wrong = copy.deepcopy(settings)
    wrong["seeds"] = [7, 7, 17]
    with pytest.raises(ValueError, match="distinct seeds"):
        validate_protocol(wrong)


def test_statistics_use_sample_sd_across_seeds_and_do_not_replicate_cv():
    shared = dict(
        split="validation",
        trajectory="all",
        phase="all",
        region="active_nodes",
        node_frame_count=315,
    )
    rows = [
        {**shared, "method": "B", "seed": seed, "vector_rmse_um": value}
        for seed, value in zip((7, 17, 27), (2.0, 3.0, 4.0), strict=True)
    ]
    rows.append({**shared, "method": "constant_velocity", "seed": None, "vector_rmse_um": 5.0})
    grouped = {row["method"]: row for row in aggregate_seed_metrics(rows)}
    assert grouped["B"]["mean_rmse_um"] == 3.0
    assert grouped["B"]["sample_sd_rmse_um"] == 1.0
    assert grouped["B"]["seed_count"] == 3
    assert grouped["constant_velocity"]["seed_count"] == 0
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate_seed_metrics([*rows, rows[0]])


def small_pack(target):
    generator = torch.Generator().manual_seed(5)
    pinned = torch.tensor([True, False, False, False])
    dataset = SimpleNamespace(pinned=pinned, rest=torch.zeros(4, 3), source_hashes={})
    return PreparedPairs(
        dataset=dataset,
        context=torch.randn(3, 4, 36, generator=generator),
        gates=torch.full((3, 4), 0.8),
        velocity=torch.zeros(3, 4, 3),
        target=target,
        neighborhood=(~pinned)[None].expand(3, -1),
        names=["center_deep"] * 3,
        frames=[0, 1, 2],
        phases=[1, 1, 1],
        times=[0.0, 0.01, 0.02],
    )


def test_factory_and_scaled_training_keep_validation_epoch_zero_candidate(tmp_path):
    settings = config()
    settings["model"]["hidden_dim"] = 16
    settings["training"].update(epochs=2, batch_size=2, log_interval=2)
    train_target = torch.zeros(3, 4, 3)
    train_target[:, 1:, 2] = -3e-6
    train = small_pack(train_target)
    validation = small_pack(torch.zeros_like(train_target))
    before = validation.target.clone()
    result = fit_model(
        "velocity_residual",
        train,
        validation,
        settings,
        tmp_path / "B",
        model_factory=lambda: make_ablation_model("B", train.dataset.rest, settings),
        loss_multiplier=100.0,
    )
    assert result["selected_epoch"] == 0
    checkpoint = torch.load(tmp_path / "B/checkpoint.pt", weights_only=True)
    assert checkpoint["model_state"]["deformation.network.0.weight"].shape == (16, 54)
    assert torch.count_nonzero(checkpoint["model_state"]["deformation.network.4.weight"]) == 0
    assert torch.equal(validation.target, before)
    assert len(result["history"]) == 3
    assert result["history"][0]["train_loss"] > 0
    assert result["history"][0]["validation_loss"] == 0


def test_invalid_loss_multiplier_is_rejected_before_training(tmp_path):
    pack = small_pack(torch.zeros(3, 4, 3))
    with pytest.raises(ValueError, match="finite and positive"):
        fit_model(
            "velocity_residual",
            pack,
            pack,
            config(),
            tmp_path / "invalid",
            loss_multiplier=float("nan"),
        )
    assert not (tmp_path / "invalid").exists()
