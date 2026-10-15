"""Loss-only scaling and non-invasive update measurement on a small artificial batch."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from contact_gwm.loss_scale_diagnostics import fit_loss_scale, grouped_probe_metrics
from contact_gwm.optimization_diagnostics import ProbeBatch, fit_probe


@pytest.fixture
def probe_case():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train_soft_local.yaml").read_text())
    config["probe"] = yaml.safe_load(
        (root / "configs/diagnose_soft_optimization.yaml").read_text()
    )["probe"]
    config["probe"].update(adam_steps=2, gradient_steps=[0, 1, 2], log_interval=2)
    config["model"]["hidden_dim"] = 16
    generator = torch.Generator().manual_seed(5)
    batch = ProbeBatch(
        context=torch.randn(1, 4, 36, generator=generator),
        gates=torch.full((1, 4), 0.7),
        velocity=torch.zeros(1, 4, 3),
        target=torch.tensor(
            [[[0.0, 0.0, 0.0], [1e-6, 2e-6, -3e-6], [2e-6, -1e-6, -4e-6], [3e-6, 1e-6, -2e-6]]]
        ),
        pinned=torch.tensor([True, False, False, False]),
        neighborhood=torch.tensor([[False, True, True, False]]),
        fit_mask=torch.tensor([False, True, True, True]),
    )
    return batch, config


def test_scaling_changes_initial_gradient_but_not_data_or_physical_score(probe_case, tmp_path):
    batch, config = probe_case
    saved = copy.deepcopy(config)
    first = fit_loss_scale(batch, config, 1.0, [1, 2], tmp_path / "original")
    amplified = fit_loss_scale(batch, config, 100.0, [1, 2], tmp_path / "amplified")
    assert first["initial_parameter_sha256"] == amplified["initial_parameter_sha256"]
    assert first["baseline_metrics"] == amplified["baseline_metrics"]
    assert first["history"][0]["canonical_loss"] == amplified["history"][0]["canonical_loss"]
    assert amplified["history"][0]["scaled_loss"] == pytest.approx(
        100 * first["history"][0]["scaled_loss"]
    )
    rows = [
        [
            r
            for r in x["parameter_updates"]
            if r["step"] == 1 and r["tensor"] == "deformation.network.4.weight"
        ][0]
        for x in (first, amplified)
    ]
    assert rows[1]["raw_gradient_l2"] == pytest.approx(100 * rows[0]["raw_gradient_l2"], rel=1e-5)
    assert config == saved
    assert torch.count_nonzero(amplified["prediction"][:, batch.pinned]) == 0


def test_measurement_preserves_original_updates_and_records_zero_head_delay(probe_case, tmp_path):
    batch, config = probe_case
    fit_probe("mlp", "adam", batch, config, tmp_path / "reference")
    measured = fit_loss_scale(batch, config, 1.0, [1, 2], tmp_path / "measured")
    a = torch.load(tmp_path / "reference/last_checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "measured/last_checkpoint.pt", weights_only=True)
    assert all(torch.equal(v, b["model_state"][key]) for key, v in a["model_state"].items())
    hidden = [
        r for r in measured["parameter_updates"] if r["tensor"] == "deformation.network.0.weight"
    ]
    assert hidden[0]["raw_gradient_l2"] == 0
    assert hidden[0]["actual_update_l2"] == 0
    assert hidden[1]["raw_gradient_l2"] > 0
    assert hidden[1]["actual_update_l2"] > 0


def test_multi_pair_scoring_weights_node_frames_and_keeps_phase_masks(probe_case, tmp_path):
    batch, config = probe_case
    multi = replace(
        batch,
        context=batch.context.repeat(2, 1, 1),
        gates=batch.gates.repeat(2, 1),
        velocity=batch.velocity.repeat(2, 1, 1),
        target=torch.cat((batch.target, batch.target * 10)),
        neighborhood=torch.tensor([[False, True, True, False], [False, False, False, True]]),
    )
    rows = grouped_probe_metrics(
        multi, torch.zeros_like(multi.target), ["center_shallow", "center_deep"], [1, 2], "static"
    )
    overall = {r["region"]: r for r in rows if r["trajectory"] == "all" and r["phase"] == "all"}
    assert overall["active_nodes"]["node_frames"] == 6
    assert overall["contact_neighborhood"]["node_frames"] == 3
    for region, value in overall.items():
        phases = [
            r
            for r in rows
            if r["trajectory"] == "all" and r["phase"] != "all" and r["region"] == region
        ]
        expected_square = (
            sum(r["vector_rmse_um"] ** 2 * r["node_frames"] for r in phases) / value["node_frames"]
        )
        assert value["vector_rmse_um"] ** 2 == pytest.approx(expected_square, rel=1e-6)
    fitted = fit_loss_scale(multi, config, 1.0, [1, 2], tmp_path / "multi")
    assert fitted["fitted_pair_count"] == 2
    assert fitted["prediction"].shape == multi.target.shape
    assert torch.count_nonzero(fitted["prediction"][:, multi.pinned]) == 0
    saved = torch.load(tmp_path / "multi/checkpoint.pt", weights_only=True)
    assert "fixed_pairs" in saved["selection"]
