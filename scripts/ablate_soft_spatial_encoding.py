"""Fixed eight-pair rest-coordinate encoding ablation with a capacity control."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from diagnose_soft_loss_scale import (
    directory_hashes,
    make_eight_pair_batch,
    read_batch,
    write_csv,
    write_json,
)
from diagnose_soft_loss_scale import (
    load_config as load_loss_config,
)
from diagnose_soft_loss_scale import (
    make_plots as make_training_plots,
)

from contact_gwm.data.gaussian_pairs import file_sha256
from contact_gwm.local_training import (
    ConstantVelocityResidualDeformation,
    constant_velocity_increment,
)
from contact_gwm.loss_scale_diagnostics import fit_loss_scale, grouped_probe_metrics
from contact_gwm.optimization_diagnostics import probe_metrics, probe_predict
from contact_gwm.spatial_ablation import (
    FREQUENCIES,
    fixed_rest_encoding,
    make_ablation_model,
    model_specifications,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("A", "B", "C")


def load_config(path):
    extra = yaml.safe_load(path.read_text())
    protocol = load_loss_config(path.parent / extra.pop("base_config"))
    setting = extra["spatial_ablation"]
    base = protocol["fit_config"]
    if setting != {
        "frequencies": list(FREQUENCIES),
        "coordinate_scale_m": 0.03,
        "hidden_dim": 128,
        "loss_multiplier": 100.0,
    }:
        raise ValueError("Use the requested fixed encoding and loss x100; no frequency search")
    if (
        base["seed"] != 7
        or base["probe"]["adam_steps"] != 2000
        or base["model"] != {"hidden_dim": 128, "contact_sigma": 0.02, "max_translation": 0.001}
        or base["overfit"]["required_rmse_reduction"] != 0.5
    ):
        raise ValueError("Keep the original eight-pair protocol and model limits")
    return {"protocol": protocol, **extra}


def group_config(base, group, specs):
    result = copy.deepcopy(base)
    result["model"]["hidden_dim"] = specs[group]["hidden_dim"]
    result["spatial_encoding_ablation"] = {
        "group": group,
        **specs[group],
        "frequencies": list(FREQUENCIES) if group == "B" else [],
        "coordinate_scale_m": 0.03,
    }
    return result


def per_pair_rows(batch, prediction, samples, method):
    rows = []
    squared = (prediction - batch.target).square().sum(-1)
    for index, sample in enumerate(samples):
        for region, mask in (
            ("active_nodes", batch.fit_mask),
            ("contact_neighborhood", batch.neighborhood[index] & batch.fit_mask),
        ):
            rows.append(
                {
                    "method": method,
                    "trajectory": sample["trajectory"],
                    "frame": sample["frame"],
                    "time": sample["time"],
                    "phase": sample["phase"],
                    "region": region,
                    "node_count": int(mask.sum()),
                    "vector_rmse_um": float(squared[index, mask].mean().sqrt()) * 1e6,
                }
            )
    return rows


def audit_current_inputs(dataset, samples, base):
    """Poison future arrays only in memory and restore them, checking nonzero B predictions."""
    pair = dataset[samples[0]["index"]]
    encoder = ConstantVelocityResidualDeformation(**base["model"], dt=base["dt"])
    model = make_ablation_model("B", dataset.rest, base)
    with torch.no_grad():
        model.deformation.network[-1].weight.fill_(0.02)
        model.deformation.network[-1].bias.fill_(0.03)
        context, gate = encoder.encode(pair.inputs)
        before = model.predict_encoded(
            context, gate, dataset.pinned, pair.inputs.gaussians.velocities
        )
    arrays = dataset.trajectories[pair.trajectory]
    t = pair.frame + 1
    fields = (
        "node_position_world",
        "node_velocity_world",
        "action_target_position",
        "action_target_velocity",
        "indenter_position_world",
        "indenter_velocity_world",
    )
    saved = {key: arrays[key][t].copy() for key in fields}
    begin, end = arrays["contact_offsets"][t : t + 2]
    contact_fields = ("contact_force_on_block_world", "contact_position_world")
    saved_contacts = {key: arrays[key][begin:end].copy() for key in contact_fields}
    try:
        for key in fields:
            arrays[key][t] += 100
        for key in contact_fields:
            arrays[key][begin:end] += 100
        changed = dataset[samples[0]["index"]]
        with torch.no_grad():
            changed_context, changed_gate = encoder.encode(changed.inputs)
            after = model.predict_encoded(
                changed_context, changed_gate, dataset.pinned, changed.inputs.gaussians.velocities
            )
        if (
            not torch.equal(context, changed_context)
            or not torch.equal(gate, changed_gate)
            or not torch.equal(before, after)
            or torch.equal(pair.target_delta, changed.target_delta)
        ):
            raise ValueError(
                "Future-state poisoning changed current inputs or failed to change labels"
            )
    finally:
        for key, value in saved.items():
            arrays[key][t] = value
        for key, value in saved_contacts.items():
            arrays[key][begin:end] = value
    return {
        "future_state_action_indenter_and_contact_poisoned_in_memory": True,
        "nonzero_B_prediction_and_current_context_unchanged": True,
        "target_changed": True,
        "source_arrays_restored": True,
    }


def check_a_reference(output, reference):
    current = torch.load(output / "A/checkpoint.pt", weights_only=True)
    previous = torch.load(reference / "loss_x100/checkpoint.pt", weights_only=True)
    if current["step"] != previous["step"] or any(
        not torch.equal(value, previous["model_state"][key])
        for key, value in current["model_state"].items()
    ):
        raise ValueError("A did not reproduce the original x100 eight-pair weights")
    histories = []
    for path in (output / "A/history.csv", reference / "loss_x100/history.csv"):
        with path.open() as handle:
            histories.append([float(row["canonical_loss"]) for row in csv.DictReader(handle)])
    if histories[0] != histories[1]:
        raise ValueError("A did not reproduce all original x100 canonical losses")
    return {"selected_parameter_tensors_exact": True, "all_2001_canonical_losses_exact": True}


def make_comparison_plots(output, rows):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("constant_velocity", *GROUPS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for column, phase in enumerate(("all", "press", "hold")):
        for row, region in enumerate(("active_nodes", "contact_neighborhood")):
            values = [
                next(
                    item["vector_rmse_um"]
                    for item in rows
                    if item["method"] == method
                    and item["trajectory"] == "all"
                    and item["phase"] == phase
                    and item["region"] == region
                )
                for method in methods
            ]
            ax = axes[row, column]
            ax.bar(
                ("CV", "A: original", "B: spatial", "C: capacity"),
                values,
                color=("gray", "tab:blue", "tab:orange", "tab:green"),
            )
            ax.axhline(values[0] * 0.5, color="black", linestyle=":", label="50% below CV")
            ax.set(title=f"{phase}: {region}", ylabel="Increment vector RMSE (um)")
            ax.tick_params(axis="x", rotation=15)
            ax.legend(fontsize=7)
    fig.suptitle("Same eight FITTED pairs; current solver states/contacts; seed 7")
    fig.savefig(output / "group_comparison.png", dpi=160)
    plt.close(fig)


def comparison_claims(rows):
    claims = {}
    for other in ("A", "C"):
        regions = {}
        for region in ("active_nodes", "contact_neighborhood"):
            values = {
                row["method"]: row["vector_rmse_um"]
                for row in rows
                if row["trajectory"] == "all" and row["phase"] == "all" and row["region"] == region
            }
            regions[region] = {
                "B_rmse_um": values["B"],
                f"{other}_rmse_um": values[other],
                "B_relative_rmse_reduction": 1 - values["B"] / values[other],
            }
        claims[f"B_vs_{other}"] = {
            "regions": regions,
            "B_better_in_both_regions": all(
                value["B_relative_rmse_reduction"] > 0 for value in regions.values()
            ),
        }
    return claims


@torch.no_grad()
def verify_run(output):
    summary = json.loads((output / "summary.json").read_text())
    config = yaml.safe_load((output / "config.yaml").read_text())
    for relative, expected in summary["artifact_sha256"].items():
        if file_sha256(output / relative) != expected:
            raise ValueError(f"Changed artifact: {relative}")
    batch = read_batch(output / "sample.npz")
    with np.load(output / "sample.npz", allow_pickle=False) as archive:
        rest = torch.from_numpy(archive["rest_positions"])
        np.testing.assert_array_equal(
            archive["fixed_rest_encoding"], fixed_rest_encoding(rest).numpy()
        )
    base = config["protocol"]["fit_config"]
    specs = model_specifications(rest, base)
    if specs != summary["model_specifications"]:
        raise ValueError("Actual model dimensions/parameter counts changed")
    samples = summary["fixed_samples"]
    names = [item["trajectory"] for item in samples]
    phases = [1 if item["phase"] == "press" else 2 for item in samples]
    baseline = constant_velocity_increment(batch.velocity, batch.pinned)
    rows = grouped_probe_metrics(batch, baseline, names, phases, "constant_velocity")
    pairs = per_pair_rows(batch, baseline, samples, "constant_velocity")
    for group in GROUPS:
        checkpoint = torch.load(output / group / "checkpoint.pt", weights_only=True)
        if (
            checkpoint["config"] != group_config(base, group, specs)
            or checkpoint["loss_multiplier"] != 100.0
        ):
            raise ValueError("Saved ablation configuration changed")
        model = make_ablation_model(group, rest, base)
        model.load_state_dict(checkpoint["model_state"])
        if group == "B" and not torch.equal(model.rest_encoding, fixed_rest_encoding(rest)):
            raise ValueError("Static encoding changed during training")
        prediction = probe_predict(model, batch)
        with np.load(output / group / "predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["prediction"], prediction.numpy())
            np.testing.assert_array_equal(archive["target"], batch.target.numpy())
        if probe_metrics(prediction, batch) != summary["experiments"][group]["selected_metrics"]:
            raise ValueError("Reloaded physical metrics differ")
        with (output / group / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        if (
            int(min(history, key=lambda row: float(row["canonical_loss"]))["step"])
            != checkpoint["step"]
        ):
            raise ValueError("Selected weights differ from lowest common fitting MSE")
        rows.extend(grouped_probe_metrics(batch, prediction, names, phases, group))
        pairs.extend(per_pair_rows(batch, prediction, samples, group))
    if rows != summary["grouped_metrics"] or pairs != summary["per_pair_metrics"]:
        raise ValueError("Reloaded group or per-pair metrics differ")
    if comparison_claims(rows) != summary["comparisons"]:
        raise ValueError("Reloaded A/B/C comparison differs")
    return {
        "checkpoints_reloaded": 3,
        "npz_files_reloaded": 4,
        "predictions_and_metrics_exact": True,
        "grouped_metric_rows_exact": len(rows),
        "per_pair_metric_rows_exact": len(pairs),
        "selection_argmin_matches": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ablate_soft_spatial_encoding.yaml"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_spatial_encoding_ablation/run_001"
    )
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        config = yaml.safe_load((args.verify_only / "config.yaml").read_text())
        torch.set_num_threads(config["protocol"]["base"]["cpu_threads"])
        print(verify_run(args.verify_only.resolve()))
        return
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output exists; choose another --output-dir: {output}")
    config = load_config(args.config)
    protocol = config["protocol"]
    base = protocol["fit_config"]
    reference_a = ROOT / config["reference_a_run"]
    if yaml.safe_load((reference_a / "config.yaml").read_text()) != protocol:
        raise ValueError("The fixed settings differ from the previous eight-pair run")
    torch.set_num_threads(base["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    protected = {
        "data": directory_hashes(args.data_dir),
        "existing_outputs": directory_hashes(ROOT / "outputs", exclude=output),
    }
    batch, dataset, samples, phases = make_eight_pair_batch(
        args.data_dir, protocol["base"], ROOT / protocol["reference_run"]
    )
    previous = read_batch(reference_a / "sample.npz")
    if any(
        not torch.equal(getattr(batch, key), getattr(previous, key))
        for key in batch.__dataclass_fields__
    ):
        raise ValueError("Current batch differs from the previous x100 eight-pair batch")
    specs = model_specifications(dataset.rest, base)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / "fixed_samples.json", samples)
    write_json(output / "model_specifications.json", specs)
    np.savez_compressed(
        output / "sample.npz",
        **{key: getattr(batch, key).numpy() for key in batch.__dataclass_fields__},
        rest_positions=dataset.rest.numpy(),
        node_ids=dataset.node_ids.numpy(),
        fixed_rest_encoding=fixed_rest_encoding(dataset.rest).numpy(),
    )
    causal = audit_current_inputs(dataset, samples, base)
    write_json(output / "causality_audit.json", causal)
    print("Model specifications:", specs, flush=True)
    baseline = constant_velocity_increment(batch.velocity, batch.pinned)
    names = [item["trajectory"] for item in samples]
    rows = grouped_probe_metrics(batch, baseline, names, phases, "constant_velocity")
    pairs = per_pair_rows(batch, baseline, samples, "constant_velocity")
    results, histories, updates = {}, {}, {}
    encoder = ConstantVelocityResidualDeformation(**base["model"], dt=base["dt"])
    for group in GROUPS:
        result = fit_loss_scale(
            batch,
            group_config(base, group, specs),
            100.0,
            protocol["loss_scale"]["update_steps"],
            output / group,
            model_factory=partial(make_ablation_model, group, dataset.rest, base),
        )
        history, update_rows, prediction = (
            result.pop("history"),
            result.pop("parameter_updates"),
            result.pop("prediction"),
        )
        histories[group], updates[group] = history, update_rows
        write_csv(output / group / "history.csv", history)
        write_csv(output / group / "parameter_updates.csv", update_rows)
        np.savez_compressed(
            output / group / "predictions.npz",
            prediction=prediction.numpy(),
            target=batch.target.numpy(),
            node_ids=dataset.node_ids.numpy(),
        )
        checkpoint = torch.load(output / group / "checkpoint.pt", weights_only=True)
        model = make_ablation_model(group, dataset.rest, base)
        model.load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            live = []
            for item in samples:
                inputs = dataset[item["index"]].inputs
                context, gates = encoder.encode(inputs)
                live.append(
                    model.predict_encoded(
                        context, gates, dataset.pinned, inputs.gaussians.velocities
                    )
                )
            if not torch.allclose(torch.stack(live), prediction, rtol=1e-5, atol=1e-10):
                raise ValueError("Live current-frame predictions differ from the batch cache")
        reductions = {
            key: 1 - result["selected_metrics"][key] / result["baseline_metrics"][key]
            for key in ("fit_rmse_um", "fit_neighborhood_rmse_um")
        }
        result.update(
            model_specification=specs[group],
            live_cache_agree=True,
            region_rmse_reductions=reductions,
            meets_predeclared_fitting_criterion=all(value >= 0.5 for value in reductions.values()),
        )
        results[group] = result
        rows.extend(grouped_probe_metrics(batch, prediction, names, phases, group))
        pairs.extend(per_pair_rows(batch, prediction, samples, group))
    reference_check = check_a_reference(output, reference_a)
    write_csv(output / "grouped_metrics.csv", rows)
    write_csv(output / "per_pair_metrics.csv", pairs)
    # C's additional count-matching details remain in the JSON model specifications.
    write_csv(
        output / "comparison.csv",
        [
            {
                "group": group,
                "input_dim": specs[group]["input_dim"],
                "hidden_dim": specs[group]["hidden_dim"],
                "parameter_count": specs[group]["parameter_count"],
                "selected_step": result["selected_step"],
                **result["selected_metrics"],
                "canonical_selected_loss": result["canonical_selected_loss"],
                "clipped_steps": result["clipped_steps"],
            }
            for group, result in results.items()
        ],
    )
    make_training_plots(output, histories, updates, required_reduction=0.5)
    make_comparison_plots(output, rows)
    sources = {
        "spatial_ablation_entrypoint.py": Path(__file__),
        "loss_scale_entrypoint.py": ROOT / "scripts/diagnose_soft_loss_scale.py",
        "gaussian_pairs.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
        **{
            f"{name}.py": ROOT / f"src/contact_gwm/{name}.py"
            for name in (
                "spatial_ablation",
                "loss_scale_diagnostics",
                "optimization_diagnostics",
                "local_training",
                "deformation",
                "comparison",
                "residual_diagnostics",
            )
        },
    }
    for name, source in sources.items():
        shutil.copy2(source, output / name)
    unchanged = {
        "data": directory_hashes(args.data_dir) == protected["data"],
        "existing_outputs": directory_hashes(ROOT / "outputs", exclude=output)
        == protected["existing_outputs"],
    }
    if not all(unchanged.values()):
        raise ValueError("Existing data or outputs changed")
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": "same_eight_fitted_pairs_spatial_encoding_capacity_ablation",
        "model_specifications": specs,
        "experiments": results,
        "fixed_samples": samples,
        "grouped_metrics": rows,
        "per_pair_metrics": pairs,
        "comparisons": comparison_claims(rows),
        "A_reference_check": reference_check,
        "causality_audit": causal,
        "same_seed": 7,
        "all_initial_parameter_tensors_identical": False,
        "validation_arrays_loaded": False,
        "test_arrays_loaded": False,
        "free_rollout": False,
        "future_inputs_used": False,
        "source_hashes": dataset.source_hashes,
        "protected_unchanged": unchanged,
        "protected_sha256": protected,
        "artifact_sha256": directory_hashes(output),
        "total_wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
    }
    write_json(output / "summary.json", summary)
    verification = verify_run(output)
    write_json(output / "verification.json", verification)
    print("A reference check:", reference_check, flush=True)
    print("Independent verification:", verification, flush=True)
    print("Comparisons:", summary["comparisons"], flush=True)
    print("Output:", output, flush=True)


if __name__ == "__main__":
    main()
