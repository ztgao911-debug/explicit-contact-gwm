"""Exact cached supervision and inclusive three-model evaluation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from contact_gwm.recovery_response import central_response
from contact_gwm.recovery_stability import make_perturbations, normalized_state_norm


@pytest.fixture
def entry(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("ablate_recovery_response_weight")


@pytest.fixture
def cache(tmp_path):
    rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.01, 0.0, 0.01]])
    mesh = {
        "rest_positions_world": rest,
        "pinned_mask": np.array([True, False, False]),
        "node_ids": np.arange(3),
    }
    batch = {
        "x": torch.tensor(np.broadcast_to(rest, (21, 3, 3)).copy()),
        "v": torch.zeros((21, 3, 3), dtype=torch.float64),
    }
    settings = {
        "direction_count": 3,
        "signs": [-1, 1],
        "position_rms_m": [1e-6],
        "velocity_rms_m_s": [1e-4],
    }
    cases, dx, dv, directions = make_perturbations(mesh["pinned_mask"], settings, 7)
    x = batch["x"][:-1, None].numpy() + dx[None, 1:]
    v = batch["v"][:-1, None].numpy() + dv[None, 1:]
    raw = {
        "node_ids": mesh["node_ids"],
        "current_frames": np.arange(210, 230),
        "directions": directions,
        "input_x": x.reshape(20, 6, 2, 3, 3),
        "input_v": v.reshape(20, 6, 2, 3, 3),
        "target_x": (x + 0.01 * v).reshape(20, 6, 2, 3, 3),
        "target_v": v.reshape(20, 6, 2, 3, 3),
        "initial_norm": normalized_state_norm(dx[2::2], dv[2::2], ~mesh["pinned_mask"], 0.001, 0.1),
    }
    raw["native_rx"] = central_response(torch.tensor(raw["target_x"])).numpy()
    raw["native_rv"] = central_response(torch.tensor(raw["target_v"])).numpy()
    np.savez_compressed(tmp_path / "response_labels.npz", **raw)
    (tmp_path / "response_cases.json").write_text(json.dumps(cases))
    return tmp_path, mesh, batch, {"response": settings}, raw


def test_reused_labels_preserve_signed_endpoints_and_exact_current_inputs(entry, cache):
    source, mesh, batch, config, raw = cache
    labels = entry.cached_labels(source, mesh, batch, config)
    assert all(torch.equal(value, torch.tensor(raw[key])) for key, value in labels.items())


@pytest.mark.parametrize("field", ["input_x", "native_rv", "node_ids"])
def test_reused_labels_reject_wrong_current_state_response_or_node_order(entry, cache, field):
    source, mesh, batch, config, raw = cache
    raw[field] = raw[field].copy()
    if field == "node_ids":
        raw[field] = raw[field][::-1]
    else:
        raw[field].flat[-1] += 1e-5
    np.savez_compressed(source / "response_labels.npz", **raw)
    with pytest.raises(ValueError):
        entry.cached_labels(source, mesh, batch, config)


def test_three_arm_probe_and_rollout_evaluation_keeps_every_requested_arm(entry, tmp_path):
    count = 72
    x, v = (
        torch.zeros((21, count, 3), dtype=torch.float64),
        torch.zeros((21, count, 3), dtype=torch.float64),
    )
    v[:, 71, 2] = torch.sin(torch.arange(21, dtype=torch.float64) * 1.3) * 0.002
    x[:, 71, 2] = 0.01 * torch.cumsum(v[:, 71, 2], 0)
    pin = torch.zeros(count, dtype=torch.bool)
    pin[0] = True
    near = torch.zeros_like(pin)
    near[71] = True
    batch = {
        "x": x,
        "v": v,
        "rest": torch.zeros((count, 3), dtype=torch.float64),
        "pinned": pin,
        "neighborhood": near,
    }
    groups = ("control_0", "response_3e5", "response_1e4")
    saved = {
        f"{method}__{field}": value.numpy()[None]
        for method in ("constant_velocity", *groups)
        for field, value in (("position", x), ("velocity", v))
    }
    config = {
        "evaluation": {"horizons": [1, 5, 10, 20]},
        "probe": {"velocity_sign_threshold_mm_s": 0.01},
    }
    state_metrics = []
    rows, probes = entry.probes_and_rollout_metrics(
        saved, batch, config, state_metrics, tmp_path, groups=groups
    )
    assert {row["method"] for row in rows} == {"constant_velocity", *groups}
    assert len(probes) == 12 and len(state_metrics) == 32
    assert all(row["signed_vz_rmse_mm_s"] == 0 for row in probes)
    assert all(row["vz_std_ratio"] == pytest.approx(1) for row in probes)
    assert all(row["position_rmse_um"] == row["velocity_rmse_mm_s"] == 0 for row in state_metrics)
