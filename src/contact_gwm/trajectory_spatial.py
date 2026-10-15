"""Validation-only protocol and seed aggregation for the spatial trajectory study."""

from __future__ import annotations

import math

import numpy as np


def validate_protocol(config: dict) -> None:
    expected = {
        "train": ["center_shallow", "center_deep"],
        "validation": ["left_shallow", "left_deep"],
    }
    if config["splits"] != expected:
        raise ValueError("Use full center/train and left/validation only; no test split")
    if config["groups"] != ["A", "B", "C"]:
        raise ValueError("Compare the declared A/B/C groups")
    seeds = config["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least three distinct seeds")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers")
    if config["device"] != "cpu" or config["dt"] != 0.01:
        raise ValueError("Use CPU and dt=0.01 s")
    if config["loss_multiplier"] != 100.0:
        raise ValueError("Keep the preselected loss multiplier of 100")
    if (
        not config["include_epoch_zero"]
        or config["selection_metric"] != "validation_balanced_coordinate_mse"
    ):
        raise ValueError("Select by validation canonical loss, including epoch zero")


def aggregate_seed_metrics(rows: list[dict]) -> list[dict]:
    """Mean and sample SD across optimizer seeds, never across time-correlated frames."""
    keys = ("split", "method", "trajectory", "phase", "region")
    groups = {}
    for row in rows:
        group = tuple(row[key] for key in keys)
        groups.setdefault(group, []).append(row)
    result = []
    for key, members in sorted(groups.items()):
        seeds = [row["seed"] for row in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed in one metric group")
        counts = {row["node_frame_count"] for row in members}
        if len(counts) != 1:
            raise ValueError("Different node masks/counts between seeds")
        values = np.array([row["vector_rmse_um"] for row in members], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite metric")
        deterministic = members[0]["method"] == "constant_velocity"
        if deterministic and (len(members) != 1 or seeds != [None]):
            raise ValueError("Record the deterministic baseline once")
        if not deterministic and len(members) < 3:
            raise ValueError("At least three seeds required per learned method")
        sd = float(values.std(ddof=1)) if len(members) > 1 else 0.0
        if not math.isfinite(sd):
            raise ValueError("Non-finite standard deviation")
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "seed_count": 0 if deterministic else len(members),
                "node_frame_count": counts.pop(),
                "mean_rmse_um": float(values.mean()),
                "sample_sd_rmse_um": sd,
            }
        )
    return result
