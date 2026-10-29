from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contact_response_evaluation import summarize_response_rows  # noqa: E402


def test_response_summary_is_node_weighted_and_json_serializable():
    rows = []
    for count, error, cosine in [(2, 1.0, np.float64(-0.5)), (6, 3.0, np.float64(0.5))]:
        row = dict(method="model", horizon=1, region="active", node_count=count)
        for field in ("position", "velocity"):
            row.update(
                {
                    f"{field}_response_rmse": error,
                    f"{field}_reference_response_rms": 2.0,
                    f"{field}_direction_cosine": cosine,
                    f"{field}_amplitude_ratio": 0.5,
                }
            )
        rows.append(row)
    summary = summarize_response_rows(rows)
    json.dumps(summary, allow_nan=False)
    assert summary[0]["position_response_rmse"] == pytest.approx(np.sqrt(7))
    assert summary[0]["position_negative_cosines"] == 1
    assert type(summary[0]["position_negative_cosines"]) is int


def test_zero_response_direction_remains_undefined_in_summary():
    row = dict(method="zero", horizon=1, region="active", node_count=3)
    for field in ("position", "velocity"):
        row.update(
            {
                f"{field}_response_rmse": 2.0,
                f"{field}_reference_response_rms": 2.0,
                f"{field}_direction_cosine": None,
                f"{field}_amplitude_ratio": 0.0,
            }
        )
    result = summarize_response_rows([row])[0]
    assert result["position_defined_cosines"] == 0 and result["position_negative_cosines"] == 0
    assert result["position_median_amplitude_ratio"] == 0
    json.dumps(result, allow_nan=False)
