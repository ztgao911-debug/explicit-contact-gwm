"""Full center/left trajectory study with three optimizer seeds and no test evaluation."""

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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from ablate_soft_spatial_encoding import audit_current_inputs
from diagnose_soft_loss_scale import directory_hashes, write_csv, write_json

from contact_gwm.comparison import PreparedPairs, fit_model, metric_rows, predict, prepare_pairs
from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import ConstantVelocityResidualDeformation
from contact_gwm.loss_scale_diagnostics import parameter_fingerprint
from contact_gwm.spatial_ablation import (
    fixed_rest_encoding,
    make_ablation_model,
    model_specifications,
)
from contact_gwm.trajectory_spatial import aggregate_seed_metrics, validate_protocol

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    addition = yaml.safe_load(path.read_text())
    config = yaml.safe_load((path.parent / addition.pop("base_config")).read_text())
    config.update(addition)
    validate_protocol(config)
    config["evaluation"]["protocol"] = (
        "full_center_train_left_validation_three_seeds_oracle_one_step"
    )
    return config


def save_pack(path, pack):
    np.savez_compressed(
        path,
        **{
            key: getattr(pack, key).numpy()
            for key in ("context", "gates", "velocity", "target", "neighborhood")
        },
        names=np.array(pack.names),
        frames=pack.frames,
        phases=pack.phases,
        times=pack.times,
        pinned=pack.dataset.pinned.numpy(),
        rest=pack.dataset.rest.numpy(),
        node_ids=pack.dataset.node_ids.numpy(),
    )


def read_pack(path):
    with np.load(path, allow_pickle=False) as arrays:
        dataset = SimpleNamespace(
            **{key: torch.from_numpy(arrays[key]) for key in ("pinned", "rest", "node_ids")}
        )
        return PreparedPairs(
            dataset=dataset,
            **{
                key: torch.from_numpy(arrays[key])
                for key in ("context", "gates", "velocity", "target", "neighborhood")
            },
            **{key: arrays[key].tolist() for key in ("names", "frames", "phases", "times")},
        )


def score(split, method, seed, prediction, pack):
    if not torch.isfinite(prediction).all() or torch.count_nonzero(
        prediction[:, pack.dataset.pinned]
    ):
        raise ValueError("Non-finite prediction or movement of fixed nodes")
    result = metric_rows(split, method, prediction, pack)
    for row in result:
        row["seed"] = seed
        row["vector_rmse_um"] = row.pop("vector_rmse_mm") * 1000
        row["vector_mae_um"] = row.pop("vector_mae_mm") * 1000
    return result


def plot_results(output, aggregate):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("constant_velocity", "A", "B", "C")
    for split in ("train", "validation"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for row, region in enumerate(("active_nodes", "contact_neighborhood")):
            for column, phase in enumerate(("all", "press", "hold")):
                selected = {
                    item["method"]: item
                    for item in aggregate
                    if item["split"] == split
                    and item["trajectory"] == "all"
                    and item["phase"] == phase
                    and item["region"] == region
                }
                axes[row, column].bar(
                    ("CV", "A", "B", "C"),
                    [selected[m]["mean_rmse_um"] for m in methods],
                    yerr=[selected[m]["sample_sd_rmse_um"] for m in methods],
                    capsize=4,
                    color=("gray", "tab:blue", "tab:orange", "tab:green"),
                )
                axes[row, column].set(
                    title=f"{phase}: {region}", ylabel="Increment vector RMSE (um)"
                )
        fig.suptitle(f"{split}: selected by LEFT validation; mean +/- sample SD over seeds")
        fig.savefig(output / f"{split}_comparison.png", dpi=160)
        plt.close(fig)


@torch.no_grad()
def verify_run(output):
    summary = json.loads((output / "summary.json").read_text())
    config = yaml.safe_load((output / "config.yaml").read_text())
    validate_protocol(config)
    torch.set_num_threads(config["cpu_threads"])
    for relative, expected in summary["artifact_sha256"].items():
        if file_sha256(output / relative) != expected:
            raise ValueError(f"Changed artifact: {relative}")
    packs = {split: read_pack(output / f"{split}_inputs.npz") for split in config["splits"]}
    rows, last_train_rows = [], []
    for split, pack in packs.items():
        rows.extend(
            score(split, "constant_velocity", None, predict("constant_velocity", None, pack), pack)
        )
    lock = json.loads((output / "selection_locked.json").read_text())
    for run in summary["runs"]:
        path = output / run["directory"]
        checkpoint = torch.load(path / "checkpoint.pt", weights_only=True, map_location="cpu")
        if file_sha256(path / "checkpoint.pt") != lock["checkpoint_sha256"][run["directory"]]:
            raise ValueError("Checkpoint changed after validation selection")
        local = copy.deepcopy(config)
        local.update(seed=run["seed"], ablation_group=run["group"])
        if checkpoint["config"] != local:
            raise ValueError("Checkpoint settings differ from the saved protocol")
        model = make_ablation_model(run["group"], packs["train"].dataset.rest, config)
        model.load_state_dict(checkpoint["model_state"])
        if run["group"] == "B" and not torch.equal(
            model.rest_encoding, fixed_rest_encoding(packs["train"].dataset.rest)
        ):
            raise ValueError("Static encoding changed during training")
        with (path / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        best = min(history, key=lambda row: float(row["validation_loss"]))
        if int(best["epoch"]) != checkpoint["epoch"]:
            raise ValueError("Selection is not argmin of validation canonical loss")
        for split, pack in packs.items():
            prediction = predict("velocity_residual", model, pack)
            with np.load(path / f"{split}_predictions.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(prediction.numpy(), arrays["prediction"])
            rows.extend(score(split, run["group"], run["seed"], prediction, pack))
        last = torch.load(path / "last_checkpoint.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(last["model_state"])
        last_train_rows.extend(
            score(
                "train",
                run["group"],
                run["seed"],
                predict("velocity_residual", model, packs["train"]),
                packs["train"],
            )
        )
    if rows != summary["metrics"] or aggregate_seed_metrics(rows) != summary["aggregate_metrics"]:
        raise ValueError("Reloaded metrics differ from the saved report")
    if last_train_rows != summary["last_checkpoint_train_diagnostic"]:
        raise ValueError("Last-checkpoint training diagnostics differ after reloading")
    return {
        "checkpoints_reloaded": len(summary["runs"]),
        "metric_rows_exact": len(rows),
        "aggregate_rows_exact": len(summary["aggregate_metrics"]),
        "validation_argmin_checked": True,
        "test_arrays_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/compare_soft_spatial_trajectories.yaml"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_spatial_trajectories/run_001"
    )
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        print(verify_run(args.verify_only.resolve()))
        return
    config = load_config(args.config)
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output exists; choose a new directory: {output}")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    source_hashes_before = directory_hashes(args.data_dir)
    encoder = ConstantVelocityResidualDeformation(**config["model"], dt=config["dt"])
    packs = {}
    for split, names in config["splits"].items():
        local = copy.deepcopy(config)
        local["trajectories"] = names
        packs[split] = prepare_pairs(GaussianPairDataset(args.data_dir, local), encoder)
    if not torch.equal(packs["train"].dataset.rest, packs["validation"].dataset.rest):
        raise ValueError("The node geometry differs between train and validation")
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for split, pack in packs.items():
        save_pack(output / f"{split}_inputs.npz", pack)
    specs = model_specifications(packs["train"].dataset.rest, config)
    write_json(output / "model_specifications.json", specs)
    causal = audit_current_inputs(packs["train"].dataset, [{"index": 0}], config)
    write_json(output / "causality_audit.json", causal)
    rows, runs, last_train_rows = [], [], []
    for split, pack in packs.items():
        rows.extend(
            score(split, "constant_velocity", None, predict("constant_velocity", None, pack), pack)
        )
    for seed in config["seeds"]:
        for group in config["groups"]:
            local = copy.deepcopy(config)
            local.update(seed=seed, ablation_group=group)
            directory = f"seed_{seed}/{group}"
            path = output / directory
            initial = []

            def factory(group=group, initial=initial):
                model = make_ablation_model(group, packs["train"].dataset.rest, config)
                initial.append(parameter_fingerprint(model))
                return model

            result = fit_model(
                "velocity_residual",
                packs["train"],
                packs["validation"],
                local,
                path,
                model_factory=factory,
                loss_multiplier=config["loss_multiplier"],
            )
            history = result.pop("history")
            write_csv(path / "history.csv", history)
            checkpoint = torch.load(path / "checkpoint.pt", weights_only=True)
            model = make_ablation_model(group, packs["train"].dataset.rest, config)
            model.load_state_dict(checkpoint["model_state"])
            with torch.no_grad():
                for split, pack in packs.items():
                    prediction = predict("velocity_residual", model, pack)
                    live = []
                    for i in range(len(pack.target)):
                        inputs = pack.dataset[i].inputs
                        context, gates = encoder.encode(inputs)
                        live.append(
                            model.predict_encoded(
                                context, gates, pack.dataset.pinned, inputs.gaussians.velocities
                            )
                        )
                    if not torch.allclose(torch.stack(live), prediction, atol=1e-10, rtol=1e-5):
                        raise ValueError("Live current-frame inference differs from cached inputs")
                    np.savez_compressed(
                        path / f"{split}_predictions.npz", prediction=prediction.numpy()
                    )
                    rows.extend(score(split, group, seed, prediction, pack))
                last = torch.load(path / "last_checkpoint.pt", weights_only=True)
                model.load_state_dict(last["model_state"])
                last_train_rows.extend(
                    score(
                        "train",
                        group,
                        seed,
                        predict("velocity_residual", model, packs["train"]),
                        packs["train"],
                    )
                )
            runs.append(
                {
                    "seed": seed,
                    "group": group,
                    "directory": directory,
                    "initial_parameter_sha256": initial[0],
                    **result,
                }
            )
            print(
                f"FINISHED seed={seed} group={group} selected_epoch={result['selected_epoch']}",
                flush=True,
            )
            write_json(
                output / "progress.json",
                {
                    "completed_runs": runs,
                    "total_runs": len(config["seeds"]) * len(config["groups"]),
                },
            )
    lock = {
        "config_sha256": file_sha256(output / "config.yaml"),
        "checkpoint_sha256": {
            r["directory"]: file_sha256(output / r["directory"] / "checkpoint.pt") for r in runs
        },
        "selection": config["selection_metric"],
        "test_arrays_loaded": False,
    }
    write_json(output / "selection_locked.json", lock)
    aggregate = aggregate_seed_metrics(rows)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "aggregate_metrics.csv", aggregate)
    write_csv(output / "last_checkpoint_train_diagnostic.csv", last_train_rows)
    plot_results(output, aggregate)
    sources = {
        "entrypoint.py": Path(__file__),
        "gaussian_pairs.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
    }
    for name in (
        "comparison",
        "spatial_ablation",
        "trajectory_spatial",
        "deformation",
        "local_training",
    ):
        sources[f"{name}.py"] = ROOT / f"src/contact_gwm/{name}.py"
    for name, path in sources.items():
        shutil.copy2(path, output / name)
    if directory_hashes(args.data_dir) != source_hashes_before:
        raise ValueError("Source data changed")
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "runs": runs,
        "model_specifications": specs,
        "metrics": rows,
        "aggregate_metrics": aggregate,
        "last_checkpoint_train_diagnostic": last_train_rows,
        "causality_audit": causal,
        "source_hashes": {split: pack.dataset.source_hashes for split, pack in packs.items()},
        "pair_counts": {split: len(pack.target) for split, pack in packs.items()},
        "data_unchanged": True,
        "test_arrays_loaded": False,
        "free_rollout": False,
        "oracle_current_contacts_and_forces": True,
        "sd_scope": "optimizer seeds on the same two train and two validation trajectories",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "wall_seconds": time.perf_counter() - started,
        "artifact_sha256": directory_hashes(output),
    }
    write_json(output / "summary.json", summary)
    verification = verify_run(output)
    write_json(output / "verification.json", verification)
    print("Independent verification:", verification, flush=True)
    for row in aggregate:
        if row["split"] == "validation" and row["trajectory"] == "all":
            print(row, flush=True)
    print("Output:", output, flush=True)


if __name__ == "__main__":
    main()
