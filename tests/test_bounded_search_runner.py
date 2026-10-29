from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bounded_search_training import AlgorithmFailure, SearchDeadline  # noqa: E402
from run_bounded_search import (  # noqa: E402
    failure_status,
    latest_results,
    new_attempt_directory,
    rankings,
    seed_summary,
)


def test_zero_trials_produce_readable_empty_reports(tmp_path):
    assert rankings([], tmp_path) == []
    seed_summary([], [], tmp_path)
    for filename in ("candidate_rankings.csv", "multiseed_summary.csv"):
        with (tmp_path / filename).open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames and reader.fieldnames[0] == "candidate_id"
            assert list(reader) == []


def test_failure_categories_preserve_algorithm_and_budget_distinction():
    assert failure_status(SearchDeadline("budget")) == "timeout"
    assert failure_status(AlgorithmFailure("bad state")) == "algorithm_failure"
    assert failure_status(RuntimeError("non-finite state")) == "algorithm_failure"
    assert failure_status(RuntimeError("coding fault")) == "technical_failure"


def test_technical_retry_preserves_original_and_is_limited(tmp_path):
    candidate = {"id": "A"}
    config = {"technical_fault_policy": {"maximum_retries_per_configuration_seed": 1}}
    directory, number = new_attempt_directory(candidate, 7, config, tmp_path, None)
    assert number == 1
    original = {"status": "technical_failure", "reason": "original trace"}
    (directory / "result.json").write_text(json.dumps(original))
    with pytest.raises(RuntimeError, match="requires"):
        new_attempt_directory(candidate, 7, config, tmp_path, None)
    second, number = new_attempt_directory(candidate, 7, config, tmp_path, "fixed coding fault")
    assert number == 2 and second != directory
    assert json.loads((directory / "result.json").read_text()) == original
    (second / "result.json").write_text(json.dumps({"status": "technical_failure"}))
    with pytest.raises(RuntimeError, match="exhausted"):
        new_attempt_directory(candidate, 7, config, tmp_path, "another fix")


@pytest.mark.parametrize("status", ["algorithm_failure", "timeout", "complete_invalid"])
def test_non_technical_attempts_cannot_be_retried(tmp_path, status):
    config = {"technical_fault_policy": {"maximum_retries_per_configuration_seed": 1}}
    directory, _ = new_attempt_directory({"id": "B"}, 7, config, tmp_path, None)
    (directory / "result.json").write_text(json.dumps({"status": status}))
    with pytest.raises(RuntimeError, match="cannot be retried"):
        new_attempt_directory({"id": "B"}, 7, config, tmp_path, "reason")


def test_latest_attempts_do_not_inflate_distinct_configuration_counts():
    records = [
        {"candidate_id": "A", "seed": 7, "attempt_number": 1},
        {"candidate_id": "A", "seed": 7, "attempt_number": 2},
        {"candidate_id": "B", "seed": 7, "attempt_number": 1},
    ]
    latest = latest_results(records)
    assert len(latest) == 2
    assert latest[0]["attempt_number"] == 2
