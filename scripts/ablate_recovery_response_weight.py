"""One predeclared smaller response weight; reuse frozen controls and native labels."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from ablate_recovery_response import (
    collect_rollouts,
    evaluate_responses,
    evaluate_state_fits,
    fit,
    probes_and_rollout_metrics,
    response_plot,
    waveform_detail,
)
from ablate_recovery_response import (
    load_config as load_response_config,
)
from diagnose_recovery_multistep import plots, prepare
from diagnose_recovery_stability import metrics as stability_metrics
from diagnose_recovery_stability import plots as stability_plots
from diagnose_recovery_stability import representative_curve
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.neighbor_dynamics import make_neighbor_model, model_specification
from contact_gwm.recovery_multistep import autoregressive_recovery
from contact_gwm.recovery_response import central_response
from contact_gwm.recovery_stability import make_perturbations, normalized_state_norm

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("control_0", "response_3e5", "response_1e4")
METHODS = ("mujoco", "constant_velocity", *GROUPS)


def load_config(path):
    extra = yaml.safe_load(path.read_text())
    config = load_response_config(path.parent / extra.pop("base_config"))
    config.update(extra)
    if config["smaller_response_coefficient"] != 3e-5 or config["protocol"][
        "comparison_coefficients"
    ] != [0, 3e-5, 1e-4]:
        raise ValueError("Run only the predeclared smaller coefficient, not a weight search")
    config["response"]["coefficient"] = config["smaller_response_coefficient"]
    return config


def cached_labels(source, mesh, batch, config):
    """Validate exact CURRENT inputs and independent cached endpoint supervision."""
    raw = read_npz(source / "response_labels.npz")
    if not np.array_equal(raw["node_ids"], mesh["node_ids"]) or not np.array_equal(
        raw["current_frames"], np.arange(210, 230)
    ):
        raise ValueError("Keep stable node IDs and the original twenty current frames")
    cases, dx, dv, directions = make_perturbations(mesh["pinned_mask"], config["response"], 7)
    shape = (12, len(mesh["node_ids"]), 3)
    expected_x = (batch["x"][:-1, None].numpy() + dx[1:].reshape(shape)).reshape(
        20, 6, 2, len(mesh["node_ids"]), 3
    )
    expected_v = (batch["v"][:-1, None].numpy() + dv[1:].reshape(shape)).reshape(expected_x.shape)
    if (
        not np.array_equal(raw["input_x"], expected_x)
        or not np.array_equal(raw["input_v"], expected_v)
        or not np.array_equal(raw["directions"], directions)
    ):
        raise ValueError(
            "Cached labels do not correspond to the unchanged current-state perturbations"
        )
    eta = normalized_state_norm(dx[2::2], dv[2::2], ~mesh["pinned_mask"], 0.001, 0.1)
    if not np.array_equal(raw["initial_norm"], eta) or cases != json.loads(
        (source / "response_cases.json").read_text()
    ):
        raise ValueError("Keep original perturbation normalization and signs")
    labels = {
        key: torch.tensor(raw[key])
        for key in (
            "input_x",
            "input_v",
            "target_x",
            "target_v",
            "native_rx",
            "native_rv",
            "initial_norm",
        )
    }
    if not all(torch.isfinite(value).all() for value in labels.values()):
        raise ValueError("Non-finite cached supervision")
    for field, target in (("native_rx", "target_x"), ("native_rv", "target_v")):
        if not torch.equal(labels[field], central_response(labels[target])):
            raise ValueError("Cached signed response differs from independent endpoint labels")
    pin = mesh["pinned_mask"]
    if not np.array_equal(
        raw["target_x"][..., pin, :],
        np.broadcast_to(mesh["rest_positions_world"][pin], raw["target_x"][..., pin, :].shape),
    ) or np.any(raw["target_v"][..., pin, :]):
        raise ValueError("Cached physical endpoints violate the fixed boundary")
    return labels


def history(path):
    with path.open() as stream:
        return [
            {key: int(value) if key == "step" else float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


@torch.no_grad()
def load_model(checkpoint, mesh, config):
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_neighbor_model("B", mesh, config)
    model.load_state_dict(artifact["model_state"], strict=True)
    model.eval()
    return model, artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ablate_recovery_response_weight.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    data = ROOT / config["data_directory"]
    source = ROOT / config["reference_response_run"]
    reference_config = yaml.safe_load((source / "config.yaml").read_text())
    for key in ("model", "training", "window", "dt", "seed", "cpu_threads"):
        if config[key] != reference_config[key]:
            raise ValueError(f"Keep the previous training condition: {key}")
    expected_response = copy.deepcopy(reference_config["response"])
    expected_response["coefficient"] = 3e-5
    if config["response"] != expected_response:
        raise ValueError("Only the response coefficient may change")
    before = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
    output = args.output
    if output is None:
        parent, index = ROOT / config["output_root"], 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        output = parent / f"run_{index:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve previous results: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for name in (
        "scripts/ablate_recovery_response_weight.py",
        "scripts/ablate_recovery_response.py",
        "scripts/diagnose_recovery_multistep.py",
        "scripts/diagnose_recovery_stability.py",
        "src/contact_gwm/recovery_response.py",
        "src/contact_gwm/recovery_multistep.py",
        "src/contact_gwm/neighbor_dynamics.py",
        "src/contact_gwm/joint_state_dynamics.py",
        "configs/ablate_recovery_response_weight.yaml",
    ):
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    mesh, _, batch, checks = prepare(data, config)
    original_window = read_npz(source / "fixed_window.npz")
    assert all(np.array_equal(value.numpy(), original_window[key]) for key, value in batch.items())
    labels = cached_labels(source, mesh, batch, config)
    for name in (
        "response_labels.npz",
        "response_cases.json",
        "fixed_window.npz",
        "sample_list.json",
    ):
        shutil.copy2(source / name, output / name)
    label_hash = file_sha256(source / "response_labels.npz")
    assert file_sha256(output / "response_labels.npz") == label_hash
    source_checks = json.loads((source / "checks.json").read_text())
    checks["native_response_labels"] = source_checks["native_response_labels"]
    checks["response_training_extra_bound_m"] = source_checks["response_training_extra_bound_m"]
    checks["cached_label_source_sha256"] = label_hash
    normalization = yaml.safe_load((ROOT / "configs/train_soft_local.yaml").read_text())[
        "normalization"
    ]
    torch.manual_seed(7)
    template = make_neighbor_model("B", mesh, config)
    initial = copy.deepcopy(template.state_dict())
    assert (
        tensor_digest(initial)
        == json.loads((source / "initialization.json").read_text())["state_sha256"]
    )
    write_json(
        output / "initialization.json",
        {"state_sha256": tensor_digest(initial), "model": model_specification(template)},
    )
    trained = make_neighbor_model("B", mesh, config)
    fit_summary, new_history = fit(
        trained, initial, batch, labels, normalization, config, "five_step_response", output
    )
    checkpoint_map = {
        "control_0": source / "five_step_control/checkpoint.pt",
        "response_3e5": output / "five_step_response/checkpoint.pt",
        "response_1e4": source / "five_step_response/checkpoint.pt",
    }
    coefficients = {"control_0": 0.0, "response_3e5": 3e-5, "response_1e4": 1e-4}
    models, model_metadata = {}, {}
    for method, checkpoint in checkpoint_map.items():
        model, artifact = load_model(checkpoint, mesh, config)
        models[method] = model
        model_metadata[method] = {
            "coefficient": coefficients[method],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "sha256": file_sha256(checkpoint),
            "selected_step": artifact["selected_step"],
            "trained_this_run": method == "response_3e5",
            "updates_in_original_fit": 2000,
        }
    with torch.no_grad():
        a = autoregressive_recovery(
            trained, batch["x"][0], batch["v"][0], batch["rest"], batch["pinned"], normalization, 20
        )
        b = autoregressive_recovery(
            models["response_3e5"],
            batch["x"][0],
            batch["v"][0],
            batch["rest"],
            batch["pinned"],
            normalization,
            20,
        )
    assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))
    checks["new_checkpoint_reload_predictions_bitwise_equal"] = True
    histories = {
        "control_0": history(source / "five_step_control/training_curve.csv"),
        "response_3e5": new_history,
        "response_1e4": history(source / "five_step_response/training_curve.csv"),
    }
    state_metrics = evaluate_state_fits(models, batch, normalization, config)
    evaluate_responses(models, batch, labels, normalization, config, output)
    cases, saved, window, rollout_checks = collect_rollouts(
        models, mesh, batch, normalization, config, output, control_method="control_0"
    )
    checks.update(rollout_checks)
    old_predictions = read_npz(source / "trajectories.npz")
    for method, old_name in (
        ("control_0", "five_step_control"),
        ("response_1e4", "five_step_response"),
    ):
        for field in ("position", "velocity"):
            assert np.array_equal(
                saved[f"{method}__{field}"], old_predictions[f"{old_name}__{field}"]
            )
    checks["both_frozen_reference_rollouts_bitwise_equal_previous_run"] = True
    probe_rows, probe_summary = probes_and_rollout_metrics(
        saved, batch, config, state_metrics, output, groups=GROUPS
    )
    stability_config = yaml.safe_load(
        (ROOT / config["reference_stability_run"] / "config.yaml").read_text()
    )
    baseline, gains, _ = stability_metrics(
        stability_config, cases, saved, window, output, methods=METHODS
    )
    stability_plots(gains, baseline, output, methods=METHODS)
    representative_curve(cases, saved, window, output, methods=METHODS)
    plots(histories, state_metrics, probe_rows, output)
    response_plot(histories, output)
    waveform_detail(saved, batch, output, groups=GROUPS)
    assert before == {name: file_sha256(ROOT / name) for name in before}
    checks["historical_file_count_unchanged_sha256"] = len(before)
    checks["only_center_deep_source_array_loaded"] = True
    checks["new_optimizer_updates"] = 2000
    checks["new_native_solver_steps"] = 0
    write_json(output / "checks.json", checks)
    write_json(output / "model_sources.json", model_metadata)
    write_json(output / "historical_hashes.json", before)
    environment = {
        "experiment_date": config["experiment_date"],
        "host_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": sum(f.stat().st_size for f in output.rglob("*") if f.is_file()),
        "seed": 7,
        "cpu_threads": 1,
    }
    write_json(output / "environment.json", environment)
    result = {
        "environment": environment,
        "checks": checks,
        "training": fit_summary,
        "models": model_metadata,
        "state_metrics": state_metrics,
        "probe_metrics": probe_summary,
        "gain_at_twenty_steps": [r for r in gains if r["horizon"] == 20],
        "training_segment_diagnostic_only": True,
        "coefficient_predeclared_not_selected_from_rollout": True,
    }
    write_json(output / "summary.json", result)
    print(
        json.dumps(
            {"output": str(output), "environment": environment, "training": fit_summary}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
