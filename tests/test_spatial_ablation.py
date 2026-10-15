"""Fixed Fourier features, capacity matching and the shared training engine."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
import yaml

from contact_gwm.local_training import constant_velocity_increment
from contact_gwm.loss_scale_diagnostics import fit_loss_scale
from contact_gwm.optimization_diagnostics import ProbeBatch, probe_predict
from contact_gwm.spatial_ablation import (
    SpatialEncodedResidual,
    closest_original_width,
    fixed_rest_encoding,
    make_ablation_model,
    model_specifications,
    original_parameter_count,
)


def base_config():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/train_soft_local.yaml").read_text()
    )
    config["probe"] = {"adam_steps": 2, "log_interval": 2}
    return config


def test_encoding_formula_order_and_original_context_preserved():
    rest = torch.tensor([[0.0, 0.03, -0.03]])
    expected = torch.tensor(
        [
            [
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ]
        ]
    )
    torch.testing.assert_close(fixed_rest_encoding(rest), expected, atol=5e-7, rtol=0)
    model = SpatialEncodedResidual(rest)
    context = torch.randn(2, 1, 36)
    augmented = model.augment_context(context)
    assert augmented.shape == (2, 1, 54)
    assert torch.equal(augmented[..., :36], context)
    assert torch.equal(augmented[0, :, 36:], augmented[1, :, 36:])
    assert not model.rest_encoding.requires_grad
    assert all(name.startswith("deformation.network.") for name, _ in model.named_parameters())


def test_actual_counts_and_globally_closest_integer_capacity_width():
    specs = model_specifications(torch.zeros(4, 3), base_config())
    assert [specs[g]["parameter_count"] for g in ("A", "B", "C")] == [22022, 24326, 24171]
    assert specs["B"]["input_dim"] == 54
    assert specs["C"]["hidden_dim"] == closest_original_width(24326) == 135
    assert original_parameter_count(135) == 24171
    # Counts grow strictly with positive width, so the bracketing pair gives the global optimum.
    assert original_parameter_count(135) < 24326 < original_parameter_count(136)
    assert 24326 - original_parameter_count(135) < original_parameter_count(136) - 24326


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_all_models_zero_init_pins_zero_gates_and_finite_gradients(group):
    config = base_config()
    rest = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.03], [0.01, 0.0, 0.03], [-0.01, 0.0, 0.03]])
    model = make_ablation_model(group, rest, config)
    context = torch.randn(2, 4, 36)
    gates = torch.tensor([[1.0, 0.8, 0.0, 0.5]]).expand(2, -1)
    pinned = torch.tensor([True, False, False, False])
    velocity = torch.full((2, 4, 3), 0.02)
    baseline = constant_velocity_increment(velocity, pinned)
    assert torch.equal(model.predict_encoded(context, gates, pinned, velocity), baseline)
    with torch.no_grad():
        model.deformation.network[-1].weight.fill_(0.02)
        model.deformation.network[-1].bias.fill_(0.03)
    prediction = model.predict_encoded(context, gates, pinned, velocity)
    assert torch.count_nonzero(prediction[:, pinned]) == 0
    assert torch.equal(prediction[:, 2], baseline[:, 2])
    assert torch.any(prediction[:, 1] != baseline[:, 1])
    prediction.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.deformation.network[-1].weight.grad[:3]) > 0


def test_shared_trainer_uses_spatial_model_on_creation_and_reload(tmp_path):
    config = base_config()
    rest = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.03], [0.01, 0.0, 0.03], [-0.01, 0.0, 0.03]])
    pinned = torch.tensor([True, False, False, False])
    batch = ProbeBatch(
        context=torch.randn(2, 4, 36),
        gates=torch.full((2, 4), 0.8),
        velocity=torch.zeros(2, 4, 3),
        target=torch.zeros(2, 4, 3),
        pinned=pinned,
        neighborhood=(~pinned)[None].expand(2, -1),
        fit_mask=~pinned,
    )
    batch.target[:, ~pinned, 2] = -3e-6
    factory = partial(make_ablation_model, "B", rest, config)
    result = fit_loss_scale(
        batch, config, 100.0, [1, 2], tmp_path / "spatial", model_factory=factory
    )
    checkpoint = torch.load(tmp_path / "spatial/checkpoint.pt", weights_only=True)
    assert checkpoint["model_state"]["deformation.network.0.weight"].shape == (128, 54)
    assert torch.equal(checkpoint["model_state"]["rest_encoding"], fixed_rest_encoding(rest))
    model = factory()
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        assert torch.equal(probe_predict(model, batch), result["prediction"])
    assert result["checkpoint_reload_exact"]
