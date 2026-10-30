"""Bootstrap A+B report directories; create a budget only for a new run."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = yaml.safe_load((ROOT / "configs/ab_training_force.yaml").read_text())
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    budget_file = report / "budget.json"
    if not budget_file.exists():
        if (report / "protocol.json").exists():
            raise RuntimeError("An existing protocol cannot receive a replacement budget")
        start = time.time()
        budget_file.write_text(
            json.dumps(
                {
                    "wall_start_epoch": start,
                    "deadline_epoch": start + config["wall_clock_budget_seconds"],
                    "budget_seconds": config["wall_clock_budget_seconds"],
                    "maximum_trainings": 9,
                    "updates_each": 2000,
                    "maximum_parallel_workers": 2,
                },
                indent=2,
            )
            + "\n"
        )
    for seed in config["experiments"]["seeds"]:
        for stage in config["experiments"]["evaluation_stages"]:
            (report / "evaluation" / f"seed_{seed}" / stage).mkdir(parents=True, exist_ok=True)
    print("Report directories ready; existing budget, protocol and results were preserved")


if __name__ == "__main__":
    main()
