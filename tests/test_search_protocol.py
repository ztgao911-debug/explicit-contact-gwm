from __future__ import annotations

import copy
import json
from itertools import product

import pytest

from contact_gwm.search_protocol import (
    LEARNING_RATES,
    MESH_WEIGHTS,
    PHASES,
    RANKING_HORIZONS,
    REGIONS,
    TRAINING_HORIZONS,
    TRAINING_VELOCITY_WEIGHTS,
    fixed_five_step_scores,
    fixed_metric_scores,
    frozen_protocol,
    generate_candidates,
    protocol_digest,
)


def rows(position_um=1000.0, velocity_mm_s=100.0):
    return [
        {
            "phase": phase,
            "region": region,
            "horizon": horizon,
            "position_rmse_um": position_um,
            "velocity_rmse_mm_s": velocity_mm_s,
            "split": "validation",
            "contact_group": "all",
            "node_count": 10,
        }
        for phase, region, horizon in product(PHASES, REGIONS, RANKING_HORIZONS)
    ]


def test_candidates_have_exact_controls_unique_in_pool_and_reproduce():
    a = generate_candidates(7)
    assert a == generate_candidates(7)
    assert len(a) == 12 and [r["id"] for r in a[:3]] == ["A", "B", "C"]
    assert a[0]["mode"] == "teacher_forcing" and a[0]["horizon"] == 5
    assert a[0]["learning_rate"] == a[1]["learning_rate"] == a[2]["learning_rate"] == 0.001
    assert a[0]["velocity_weight"] == a[1]["velocity_weight"] == a[2]["velocity_weight"] == 0.5
    assert [r["mesh_weight"] for r in a[:3]] == [0.0, 0.0, 0.01]
    keys = [
        tuple(r[k] for k in ("mode", "learning_rate", "horizon", "velocity_weight", "mesh_weight"))
        for r in a
    ]
    assert len(set(keys)) == 12
    for r in a[3:]:
        assert r["mode"] == "autoregressive"
        assert r["learning_rate"] in LEARNING_RATES and r["horizon"] in TRAINING_HORIZONS
        assert r["velocity_weight"] in TRAINING_VELOCITY_WEIGHTS
        assert r["mesh_weight"] in MESH_WEIGHTS
    json.dumps(a, allow_nan=False)


def test_protocol_digest_covers_candidates_weights_budget_and_is_order_independent():
    protocol = frozen_protocol()
    original = protocol_digest(protocol)
    reordered = dict(reversed(list(protocol.items())))
    assert protocol_digest(reordered) == original
    changed = copy.deepcopy(protocol)
    changed["candidates"][3]["learning_rate"] = 0.9
    assert protocol_digest(changed) != original
    assert protocol["max_search_configurations"] == 12
    assert protocol["max_replication_trainings"] == 4
    assert protocol["wall_clock_budget_seconds"] == 21600
    assert protocol["fixed"]["position_cap_per_branch_m"] == 0.016
    assert protocol["fixed"]["instantaneous_velocity_increment_cap_per_branch_m_s"] == 1.6
    assert protocol["data_roles"]["independent_test"] == "not_verified"


def test_ranking_balances_phase_region_horizon_and_not_frame_counts():
    values = rows(0.0, 0.0)
    for r in values:
        r["frame_count"] = 100000 if r["phase"] == "press" else 1
        if r["phase"] == "press":
            r["position_rmse_um"] = 2000.0
    scores = fixed_metric_scores(values)
    assert scores["coverage_complete"] and scores["expected_cells"] == 24
    assert scores["position_score"] == pytest.approx(1.0)
    assert scores["velocity_score"] == 0.0
    assert scores["joint_score"] == pytest.approx(0.5)


def test_fixed_units_and_velocity_weight_ignore_candidate_training_settings():
    values = rows()
    for r in values:
        r["velocity_weight"] = 0.75
        r["training_velocity_weight"] = 0.25
    assert fixed_metric_scores(values)["joint_score"] == pytest.approx(1.0)
    si = [
        {
            "phase": r["phase"],
            "region": r["region"],
            "horizon": r["horizon"],
            "position_rmse_m": 0.001,
            "velocity_rmse_m_s": 0.1,
        }
        for r in values
    ]
    assert fixed_metric_scores(si)["joint_score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "nan", "negative", "empty_region", "overflow"]
)
def test_incomplete_or_invalid_evaluation_is_unranked_not_silently_filtered(kind):
    values = rows()
    if kind == "missing":
        values.pop()
    elif kind == "duplicate":
        values.append(values[0].copy())
    elif kind == "nan":
        values[0]["velocity_rmse_mm_s"] = float("nan")
    elif kind == "negative":
        values[0]["position_rmse_um"] = -1.0
    elif kind == "overflow":
        values[0]["position_rmse_um"] = 1e308
    else:
        values[0]["node_count"] = 0
    result = fixed_metric_scores(values)
    assert not result["coverage_complete"]
    assert result["joint_score"] is result["position_score"] is result["velocity_score"] is None
    json.dumps(result, allow_nan=False)


def test_approach_control_and_contact_subgroups_do_not_change_fixed_score():
    values = rows()
    extra = copy.deepcopy(values)
    for r in extra:
        r["split"] = "control"
        r["position_rmse_um"] = 1e12
    values.extend(extra)
    values.append({**values[0], "phase": "approach", "position_rmse_um": 1e12})
    values.append({**values[0], "contact_group": "no_contact", "position_rmse_um": 1e12})
    assert fixed_metric_scores(values)["joint_score"] == pytest.approx(1.0)
    assert fixed_five_step_scores(values)["coverage_complete"]
    assert fixed_five_step_scores(values)["expected_cells"] == 8
