"""Optimization diagnostic controls and numerical gradients, without pilot-data access."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from contact_gwm.optimization_diagnostics import (
    IndependentNodeResidual,
    ProbeBatch,
    finite_difference_audit,
)


def test_zero_head_gradient_flow_and_numerical_derivatives():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train_soft_local.yaml").read_text())
    config["probe"] = yaml.safe_load(
        (root / "configs/diagnose_soft_optimization.yaml").read_text()
    )["probe"]
    config["model"]["hidden_dim"] = 16
    generator = torch.Generator().manual_seed(5)
    batch = ProbeBatch(
        context=torch.randn(1, 4, 36, generator=generator),
        gates=torch.ones(1, 4) * 0.7,
        velocity=torch.zeros(1, 4, 3),
        target=torch.tensor(
            [[[0.0, 0.0, 0.0], [1e-6, 2e-6, -3e-6], [2e-6, -1e-6, -4e-6], [3e-6, 1e-6, -2e-6]]]
        ),
        pinned=torch.tensor([True, False, False, False]),
        neighborhood=torch.tensor([[False, True, True, False]]),
        fit_mask=torch.tensor([False, True, True, True]),
    )
    result = finite_difference_audit(batch, config)
    assert result["initial_hidden_gradients_zero"]
    assert result["initial_translation_head_gradient_nonzero"]
    assert result["hidden_gradients_nonzero_after_one_update"]
    assert result["all_finite_difference_checks_pass"]


def test_independent_node_control_can_represent_each_residual_and_keeps_pins():
    model = IndependentNodeResidual(nodes=3, max_translation=0.001)
    gates = torch.tensor([[1.0, 0.5, 0.2]])
    velocity = torch.full((1, 3, 3), 0.02)  # Nonzero fixed-node velocity must still be masked.
    pinned = torch.tensor([True, False, False])
    wanted = torch.tensor([[[0.0, 0.0, 0.0], [1e-6, -2e-6, 3e-6], [-4e-6, 2e-6, 1e-6]]])
    with torch.no_grad():
        model.raw.copy_(torch.atanh(wanted[0] / (gates[0, :, None] * 0.001)))
        model.raw[0].fill_(1.0)
    result = model.predict_encoded(None, gates, pinned, velocity)
    assert torch.count_nonzero(result[:, pinned]) == 0
    torch.testing.assert_close(
        result[:, ~pinned] - velocity[:, ~pinned] * 0.01, wanted[:, ~pinned], atol=1e-10, rtol=1e-5
    )
