"""Frozen candidates and fixed, complete-coverage development-validation scores.

Candidate training weights may vary. Evaluation weights, units, phase/region
balance and horizons here never vary with a candidate. Missing or duplicated
cells make a score unavailable; this module never silently drops a window.
Physical eligibility must be checked separately before ranking these scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import product

import numpy as np

PHASES = ("press", "hold", "release", "recover")
REGIONS = ("active", "contact_neighborhood")
RANKING_HORIZONS = (5, 10, 20)
POSITION_UNIT_M = 0.001
VELOCITY_UNIT_M_S = 0.1
POSITION_CAP_M = 0.016
VELOCITY_CAP_M_S = 1.6
MESH_VOLUME_RATIO_THRESHOLD = 0.1

LEARNING_RATES = (0.0001, 0.0003, 0.001)
TRAINING_HORIZONS = (3, 5, 10)
TRAINING_VELOCITY_WEIGHTS = (0.25, 0.5, 0.75)
MESH_WEIGHTS = (0.0, 0.001, 0.01, 0.1)


def generate_candidates(search_seed: int = 7) -> list[dict]:
    """Three controls and nine without-replacement samples from the fixed pool.

    Pool order and RNG are explicit. B/C are excluded from the random pool so
    there are exactly twelve distinct experimental configurations. A shares
    B's numeric settings deliberately, differing only in teacher forcing.
    No result, score, label, wall-clock value or model enters this function.
    """
    controls = [
        {
            "id": "A",
            "mode": "teacher_forcing",
            "horizon": 5,
            "learning_rate": 0.001,
            "velocity_weight": 0.5,
            "mesh_weight": 0.0,
        },
        {
            "id": "B",
            "mode": "autoregressive",
            "horizon": 5,
            "learning_rate": 0.001,
            "velocity_weight": 0.5,
            "mesh_weight": 0.0,
        },
        {
            "id": "C",
            "mode": "autoregressive",
            "horizon": 5,
            "learning_rate": 0.001,
            "velocity_weight": 0.5,
            "mesh_weight": 0.01,
        },
    ]
    excluded = {(0.001, 5, 0.5, 0.0), (0.001, 5, 0.5, 0.01)}
    pool = [
        value
        for value in product(
            LEARNING_RATES, TRAINING_HORIZONS, TRAINING_VELOCITY_WEIGHTS, MESH_WEIGHTS
        )
        if value not in excluded
    ]
    selected = np.random.default_rng(search_seed).choice(len(pool), size=9, replace=False)
    candidates = list(controls)
    for number, index in enumerate(selected, start=1):
        learning_rate, horizon, velocity_weight, mesh_weight = pool[int(index)]
        candidates.append(
            {
                "id": f"S{number:02d}",
                "mode": "autoregressive",
                "horizon": horizon,
                "learning_rate": learning_rate,
                "velocity_weight": velocity_weight,
                "mesh_weight": mesh_weight,
            }
        )
    return candidates


def frozen_protocol(search_seed: int = 7) -> dict:
    """Machine-readable core; caller attaches source/data digests BEFORE fitting."""
    return {
        "schema_version": 1,
        "search_seed": search_seed,
        "search_training_seed": 7,
        "max_search_configurations": 12,
        "max_replication_candidates": 2,
        "replication_seeds": [17, 27],
        "max_replication_trainings": 4,
        "wall_clock_budget_seconds": 21600,
        "max_updates_per_training": 2000,
        "batch_size": 16,
        "candidates": generate_candidates(search_seed),
        "fixed": {
            "dt_s": 0.01,
            "position_cap_per_branch_m": POSITION_CAP_M,
            "instantaneous_velocity_increment_cap_per_branch_m_s": VELOCITY_CAP_M_S,
            "training_condition": "M_current_native_geometry_and_force",
            "solver_gradient": "stop_at_current_contact_condition_only",
            "state_and_state_feature_gradient": "retained_across_unroll",
            "network_structure_and_spatial_encoding": "unchanged",
            "normalization_geometry_force_parameters_data_splits_masks": "unchanged",
            "unroll_reduction": "mean_over_supervised_endpoints",
            "mesh_penalty": "mean(relu(0.1-signed_volume/rest_signed_volume)^2)",
            "mesh_volume_ratio_threshold": MESH_VOLUME_RATIO_THRESHOLD,
            "central_native_threshold_conflict_action": "stop_and_report",
        },
        "checkpoint_selection": {
            "horizon": 5,
            "include_update_zero": True,
            "evaluation_position_weight": 0.5,
            "evaluation_velocity_weight": 0.5,
            "position_unit_m": POSITION_UNIT_M,
            "velocity_unit_m_s": VELOCITY_UNIT_M_S,
            "phase_weights": "equal_press_hold_release_recover",
            "region_weights": "equal_active_contact_neighborhood",
        },
        "ranking": {
            "horizons": list(RANKING_HORIZONS),
            "phase_weights": "equal_press_hold_release_recover",
            "region_weights": "equal_active_contact_neighborhood",
            "horizon_weights": "equal_5_10_20",
            "position_velocity_weights": [0.5, 0.5],
            "position_unit_m": POSITION_UNIT_M,
            "velocity_unit_m_s": VELOCITY_UNIT_M_S,
            "joint_score": "0.5*mean((position_rmse_m/.001)^2)+0.5*mean((velocity_rmse_m_s/.1)^2)",
            "incomplete_or_duplicate_cells": "unavailable_not_filtered",
            "eligibility": "all_predictions_finite_fixed_boundary_exact_no_inverted_windows",
        },
        "separate_acceptance": [
            "faster_down_direction",
            "response_amplitude",
            "signed_recovery_waveform",
            "recovery_amplitude",
            "recovery_phase",
        ],
        "data_roles": {
            "training": "central_ordinary_and_existing_108_central_branches",
            "development_validation": "left_ordinary_and_existing_18_left_branches",
            "static_control": "existing_no_contact_trajectory",
            "right_used_for_search": False,
            "independent_test": "not_verified",
        },
    }


def protocol_digest(protocol: dict) -> str:
    """Stable digest of the exact JSON content, without mutating the protocol."""
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _metric_scores(rows: list[dict], horizons: tuple[int, ...]) -> dict:
    expected = set(product(PHASES, REGIONS, horizons))
    cells, duplicates, invalid = {}, set(), set()
    for row in rows:
        if row.get("split", "validation") != "validation":
            continue
        if row.get("contact_group", "all") != "all":
            continue
        try:
            key = (row["phase"], row["region"], int(row["horizon"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key not in expected:
            continue
        if key in cells:
            duplicates.add(key)
        try:
            position = (
                float(row["position_rmse_m"])
                if "position_rmse_m" in row
                else (float(row["position_rmse_um"]) * 1e-6)
            )
            velocity = (
                float(row["velocity_rmse_m_s"])
                if "velocity_rmse_m_s" in row
                else (float(row["velocity_rmse_mm_s"]) * 1e-3)
            )
            if not all(math.isfinite(v) and v >= 0 for v in (position, velocity)):
                raise ValueError("Invalid RMSE")
            if not all(
                math.isfinite(v)
                for v in ((position / POSITION_UNIT_M) ** 2, (velocity / VELOCITY_UNIT_M_S) ** 2)
            ):
                raise ValueError("Overflow in normalized evaluation error")
            if "node_count" in row and int(row["node_count"]) <= 0:
                raise ValueError("Empty evaluation region")
            cells[key] = (position, velocity)
        except (KeyError, TypeError, ValueError, OverflowError):
            cells[key] = None
            invalid.add(key)
    missing = expected - cells.keys()
    complete = not missing and not duplicates and not invalid
    result = {
        "joint_score": None,
        "position_score": None,
        "velocity_score": None,
        "coverage_complete": complete,
        "expected_cells": len(expected),
        "observed_cells": len(cells),
        "missing_cells": [list(key) for key in sorted(missing)],
        "duplicate_cells": [list(key) for key in sorted(duplicates)],
        "invalid_cells": [list(key) for key in sorted(invalid)],
        "position_unit_m": POSITION_UNIT_M,
        "velocity_unit_m_s": VELOCITY_UNIT_M_S,
    }
    if complete:
        position_score = sum((v[0] / POSITION_UNIT_M) ** 2 / len(cells) for v in cells.values())
        velocity_score = sum((v[1] / VELOCITY_UNIT_M_S) ** 2 / len(cells) for v in cells.values())
        result.update(
            {
                "joint_score": 0.5 * position_score + 0.5 * velocity_score,
                "position_score": position_score,
                "velocity_score": velocity_score,
            }
        )
    return result


def fixed_metric_scores(rows: list[dict]) -> dict:
    """Rank ONE method on fixed 4 phases x 2 regions x 3 horizons.

    Caller must filter the method before calling. Input can use SI RMSE fields
    or the existing CSV's position_rmse_um / velocity_rmse_mm_s fields. Pooled
    cell RMSE is squared before averaging equally over cells, not over frames.
    This routine supplies numerical scores only; physical screening is separate.
    """
    return _metric_scores(rows, RANKING_HORIZONS)


def fixed_five_step_scores(rows: list[dict]) -> dict:
    """Same fixed weights for checkpoint selection, exactly eight 5-step cells."""
    return _metric_scores(rows, (5,))
