from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fair_replication_statistics import summarize_groups  # noqa: E402
from replicate_fair_mainline import verify_configuration  # noqa: E402


def records(values):
    return [
        {"seed": s, "method": "G", "error": v} for s, v in zip((7, 17, 27), values, strict=True)
    ]


def test_three_seed_summary_uses_sample_standard_deviation():
    result = summarize_groups(records([1, 2, 3]), ["method"], ["error"])[0]
    assert result["mean"] == 2 and result["sample_std"] == 1
    assert result["all_three_finite"] and result["n_available"] == 3


def test_missing_or_failed_seed_is_not_removed_to_produce_an_official_mean():
    result = summarize_groups(records([1, 2, None]), ["method"], ["error"])[0]
    assert result["recorded_seeds"] == "7,17,27"
    assert result["n_available"] == 2 and not result["all_three_finite"]
    assert result["missing_metric_seeds"] == "27"
    assert result["mean"] is None and result["sample_std"] is None


def test_invalid_but_finite_seed_remains_in_descriptive_statistics():
    rows = records([1, 2, 30])
    rows[-1]["physically_valid"] = False
    result = summarize_groups(rows, ["method"], ["error"])[0]
    assert result["mean"] == 11
    assert result["n_available"] == 3


def test_duplicate_or_unexpected_seed_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_groups(records([1, 2, 3]) + records([1, 2, 3])[:1], ["method"], ["error"])
    with pytest.raises(ValueError, match="Unexpected"):
        summarize_groups([{"seed": 99, "method": "G", "error": 1}], ["method"], ["error"])


@pytest.mark.parametrize("section", ["model", "training", "validation", "evaluation", "comparison"])
def test_replication_rejects_changed_scientific_settings(section):
    config = yaml.safe_load((ROOT / "configs/fair_mainline_multiseed.yaml").read_text())
    parent = json.loads((ROOT / "experiments/fair_mainline/run_001/protocol.json").read_text())
    verify_configuration(config, parent)
    changed = copy.deepcopy(config)
    changed[section]["unregistered_change"] = True
    with pytest.raises(ValueError, match="Frozen scientific"):
        verify_configuration(changed, parent)
