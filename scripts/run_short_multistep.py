"""Run one previously frozen three-step candidate for a short pipeline diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch
import yaml
from bounded_search_data import load_search_data, validation_windows
from bounded_search_training import check_deadline, gradient_preflight
from run_bounded_search import (
    evaluate_model,
    failure_status,
    foundation,
    publish_metrics,
    run_trial,
    source_files,
)
from train_contact_response import model_for
from validate_contact_pipeline import hashes, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256
from contact_gwm.pinned_contact_pipeline import IndenterInput
from contact_gwm.repaired_contact_pipeline import RepairedJointPipeline
from contact_gwm.search_protocol import frozen_protocol, protocol_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/short_multistep.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report, output = ROOT / config["report_directory"], ROOT / config["output_directory"]
    if (report / "trial_ledger.json").exists():
        raise RuntimeError("This short experiment already ran; preserve its evidence")
    budget = json.loads((report / "budget.json").read_text())
    deadline = budget["deadline_epoch"]
    check_deadline(deadline)
    torch.set_num_threads(config["cpu_threads"])
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    parent_path = ROOT / config["short_experiment"]["parent_protocol"]
    parent = json.loads(parent_path.read_text())
    candidate = next(
        c for c in parent["candidates"] if c["id"] == config["short_experiment"]["candidate_id"]
    )
    assert candidate["horizon"] == 3 and candidate["mode"] == "autoregressive"
    assert config["training"]["updates"] == 200 and config["training"]["batch_size"] == 16
    for name, expected in parent["data_hashes"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen data changed: {name}")
    native = json.loads((parent_path.parent / "preflight_native_data.json").read_text())
    if native["native_data_threshold_conflict"]:
        raise RuntimeError("Native data conflicts with fixed mesh threshold")
    objects = foundation(config)
    base, mesh, trajectories, splits, legacy, conditions, encoder, _ = objects
    data = load_search_data(ROOT, trajectories, mesh)
    windows = validation_windows(trajectories, mesh, ROOT)
    probe = model_for(mesh, config["model"], {}, 7)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for network in (probe.internal_network, probe.contact_network):
            network[4].weight.copy_(
                torch.randn(network[4].weight.shape, generator=generator) * 1e-5
            )
    write_json(
        report / "gradient_preflight.json",
        gradient_preflight(probe, legacy, conditions, encoder, data, mesh, deadline),
    )
    protocol = frozen_protocol(7)
    protocol.update(
        experiment_kind="single_frozen_candidate_short_diagnostic",
        candidates=[candidate],
        max_search_configurations=1,
        max_replication_candidates=0,
        max_replication_trainings=0,
        replication_seeds=[],
        max_updates_per_training=200,
        wall_clock_budget_seconds=900,
        parent_protocol_sha256=file_sha256(parent_path),
        configuration=config,
        source_hashes=hashes([*source_files(), args.config.resolve()]),
        data_hashes=parent["data_hashes"],
        reference_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        validation_windows=parent["validation_windows"],
        shortened_schedule="200-update cosine schedule; not equivalent to first 200 of 2000",
        independent_test="not_verified_right_arrays_never_loaded",
    )
    protocol["checkpoint_selection"]["interval_updates"] = 50
    write_json(report / "protocol.json", protocol)
    write_json(report / "protocol_digest.json", {"sha256": protocol_digest(protocol)})
    write_json(report / "data_loader_checks.json", data["checks"])
    write_json(report / "candidates.json", [candidate])
    write_csv(report / "candidates.csv", [candidate])
    print(f"Frozen short experiment: {candidate}, updates=200, budget=900s", flush=True)
    result = run_trial(
        candidate,
        7,
        "short_diagnostic",
        config,
        protocol,
        data,
        windows,
        objects,
        report,
        output,
        deadline,
    )
    write_json(report / "trial_ledger.json", [result])
    status = {"training_status": result["status"], "routes_status": "not_run"}
    if result["status"].startswith("complete"):
        saved = torch.load(
            ROOT / result["artifact_directory"] / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        models = [model_for(mesh, config["model"], {}, 7) for _ in range(2)]
        predictions = []
        for model in models:
            model.load_state_dict(saved["model_state"], strict=True)
            model.eval().requires_grad_(False)
            arrays = trajectories["center_deep"]
            state = legacy.gaussian_state(
                arrays["node_position_world"][75], arrays["node_velocity_world"][75]
            )
            predictions.append(
                RepairedJointPipeline(legacy, model, conditions, "mujoco_force").step(
                    state, IndenterInput.from_arrays(arrays, 75)
                )
            )
        same = all(
            torch.equal(getattr(predictions[0].state, key), getattr(predictions[1].state, key))
            for key in ("centers", "velocities")
        )
        write_json(report / "reload_checks.json", {"bitwise_equal_position_velocity": same})
        assert same
        try:
            evaluation = evaluate_model(
                models[0],
                legacy,
                conditions,
                mesh,
                trajectories,
                splits,
                base,
                output / "final_routes",
                deadline,
                routes=True,
            )
            write_json(report / "final_routes_summary.json", evaluation["per_method"])
            publish_metrics(output / "final_routes", report / "final_routes")
            status["routes_status"] = "complete"
        except Exception as error:
            status.update(
                routes_status=failure_status(error),
                error=str(error),
                traceback=traceback.format_exc(),
            )
            write_json(report / "route_failure.json", status)
            publish_metrics(output / "final_routes", report / "final_routes")
    changed = [
        name
        for name, expected in parent["data_hashes"].items()
        if file_sha256(ROOT / name) != expected
    ]
    write_json(
        report / "preservation_checks.json",
        {
            "frozen_input_files": len(parent["data_hashes"]),
            "changed": changed,
            "right_arrays_loaded": False,
            "independent_test_verified": False,
        },
    )
    status.update(
        wall_elapsed_seconds=time.time() - budget["wall_start_epoch"],
        completed_epoch=time.time(),
        search_configurations_started=1,
        replication_trainings_started=0,
        independent_test_verified=False,
    )
    write_json(report / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
