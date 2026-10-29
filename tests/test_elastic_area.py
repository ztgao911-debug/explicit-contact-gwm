from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from contact_gwm.elastic_area import ElasticAreaMaterial, fit_elastic_area_stiffness


def test_exact_stiffness_recovery_with_zero_contact_rows_and_numerical_rescaling():
    basis = np.array([[0, 0, 0], [1e-10, -2e-10, -8e-10], [-2e-10, 1e-10, -2e-9]])
    stiffness = 2.7e8
    force = stiffness * basis
    for scale in [1e-12, 1e-9, 1.0]:
        model, info = fit_elastic_area_stiffness(basis, force, numerical_basis_scale_m3=scale)
        assert model.stiffness_N_m3 == pytest.approx(stiffness, rel=1e-14)
        assert info["fit_frames"] == 3 and info["nonzero_basis_frames"] == 2


def test_nonnegative_constraint_and_no_hidden_intercept():
    basis = np.array([[0, 0, 0], [0, 0, -1e-9]])
    model, info = fit_elastic_area_stiffness(basis, -basis * 1e8)
    assert model.stiffness_N_m3 == 0 and info["constraint_active"]
    material = ElasticAreaMaterial(1e8)
    force, torque = material.predict(torch.zeros(3).double(), torch.zeros(3).double())
    assert torch.count_nonzero(force) == torch.count_nonzero(torque) == 0


def test_torque_origin_translation_and_downward_normal_force():
    model = ElasticAreaMaterial(2e8)
    basis = torch.tensor([0, 0, -1e-9]).double()
    torque = torch.tensor([1e-11, 2e-11, 0]).double()
    shift = torch.tensor([0.01, 0.02, 0.03]).double()
    force, old = model.predict(basis, torque)
    _, new = model.predict(basis, torque - torch.cross(shift, basis, dim=-1))
    torch.testing.assert_close(new, old - torch.cross(shift, force, dim=-1))
    assert force[2] < 0


def test_reload_bitwise_and_prediction_has_no_force_or_future_inputs(tmp_path):
    model = ElasticAreaMaterial(2.3456789e8)
    path = tmp_path / "material.json"
    model.save(path)
    restored = ElasticAreaMaterial.load(path)
    basis = torch.tensor([[1e-11, 0, -2e-9]]).double()
    torque = torch.tensor([[1e-11, -3e-11, 0]]).double()
    a, b = model.predict(basis, torque), restored.predict(basis, torque)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert set(inspect.signature(model.predict).parameters) == {
        "elastic_vector_m3",
        "elastic_torque_m4",
    }
    data = json.loads(path.read_text())
    data["stiffness_unit"] = "N/m"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ElasticAreaMaterial.load(path)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_bad_scale_and_unidentifiable_data_rejected(bad):
    with pytest.raises(ValueError):
        fit_elastic_area_stiffness(np.ones((2, 3)), np.ones((2, 3)), numerical_basis_scale_m3=bad)
    with pytest.raises(ValueError):
        fit_elastic_area_stiffness(np.zeros((2, 3)), np.ones((2, 3)))


def test_nonfinite_labels_and_parameters_rejected():
    with pytest.raises(FloatingPointError):
        fit_elastic_area_stiffness(np.ones((2, 3)), np.full((2, 3), np.nan))
    with pytest.raises(ValueError):
        ElasticAreaMaterial(float("inf"))
    with pytest.raises(ValueError):
        ElasticAreaMaterial(-1)
