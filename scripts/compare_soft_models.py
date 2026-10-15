"""CPU trajectory-split comparison with validation-only checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from contact_gwm.comparison import (
    METHODS,
    balanced_loss,
    fit_model,
    make_model,
    metric_rows,
    predict,
    prepare_pairs,
    validate_splits,
)
from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import PinnedGaussianDeformation

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def directory_hashes(directory: Path) -> dict:
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def load_config(path: Path) -> dict:
    addition = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = yaml.safe_load((path.parent / addition.pop("base_config")).read_text(encoding="utf-8"))
    config.update(addition)
    validate_splits(config["splits"])
    if config["device"] != "cpu" or config["dt"] != 0.01:
        raise ValueError("This comparison is restricted to CPU and dt=0.01 s")
    if (
        config["selection_metric"] != "validation_balanced_coordinate_mse"
        or not config["include_epoch_zero"]
    ):
        raise ValueError("Use the declared validation loss and include epoch zero")
    config["evaluation"]["protocol"] = config.pop("evaluation_protocol")
    return config


def load_split(data_dir: Path, config: dict, split: str, encoder: PinnedGaussianDeformation):
    local_config = copy.deepcopy(config)
    local_config["trajectories"] = config["splits"][split]
    return prepare_pairs(GaussianPairDataset(data_dir, local_config), encoder)


def load_selected(directory: Path, config: dict) -> dict:
    lock = json.loads((directory / "selection_locked.json").read_text())
    if file_sha256(directory / "config.yaml") != lock["resolved_config_sha256"]:
        raise ValueError("Resolved configuration changed after selection")
    models = {}
    for kind in ("direct", "velocity_residual"):
        path = directory / kind / "checkpoint.pt"
        if file_sha256(path) != lock["checkpoint_sha256"][kind]:
            raise ValueError("Selected checkpoint changed after validation selection")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["kind"] != kind or checkpoint["config"] != config:
            raise ValueError("Checkpoint kind/config differs from the locked protocol")
        model = make_model(kind, config)
        model.load_state_dict(checkpoint["model_state"])
        models[kind] = model.eval()
    return models


def make_plots(directory: Path, rows: list[dict], selections: dict, predictions: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("Static", "Constant velocity", "Direct network", "Velocity residual")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for column, split in enumerate(("train", "validation", "test")):
        for row, region in enumerate(("active_nodes", "contact_neighborhood")):
            values = [
                next(
                    item["vector_rmse_mm"]
                    for item in rows
                    if item["split"] == split
                    and item["method"] == method
                    and item["trajectory"] == "all"
                    and item["phase"] == "all"
                    and item["region"] == region
                )
                for method in METHODS
            ]
            axes[row, column].bar(
                labels, values, color=("gray", "tab:blue", "tab:orange", "tab:green")
            )
            axes[row, column].set(
                title=f"{split}: {region}", ylabel="Increment vector RMSE (mm)", yscale="log"
            )
            axes[row, column].tick_params(axis="x", labelrotation=20)
            axes[row, column].grid(axis="y", alpha=0.25)
    fig.suptitle("Whole-trajectory split; true current state/contact conditioning; one step")
    fig.savefig(directory / "method_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, phase in enumerate(("press", "hold")):
        for column, region in enumerate(("active_nodes", "contact_neighborhood")):
            values = [
                next(
                    item["vector_rmse_mm"]
                    for item in rows
                    if item["split"] == "test"
                    and item["method"] == method
                    and item["trajectory"] == "all"
                    and item["phase"] == phase
                    and item["region"] == region
                )
                for method in METHODS
            ]
            axes[row, column].bar(
                labels, values, color=("gray", "tab:blue", "tab:orange", "tab:green")
            )
            axes[row, column].set(
                title=f"Test {phase}: {region}", ylabel="Increment vector RMSE (mm)", yscale="log"
            )
            axes[row, column].tick_params(axis="x", labelrotation=15)
    fig.savefig(directory / "test_phase_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, kind in zip(axes, ("direct", "velocity_residual"), strict=True):
        with (directory / kind / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        epochs = [int(item["epoch"]) for item in history]
        ax.semilogy(epochs, [float(item["train_loss"]) for item in history], label="Train")
        ax.semilogy(
            epochs, [float(item["validation_loss"]) for item in history], label="Validation"
        )
        ax.axvline(
            selections[kind]["selected_epoch"], linestyle=":", color="black", label="Selected epoch"
        )
        ax.set(title=kind, xlabel="Epoch", ylabel="Balanced scaled coordinate MSE")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.savefig(directory / "selection_curves.png", dpi=160)
    plt.close(fig)

    test = predictions["test"]
    node = int(np.argmin(np.linalg.norm(test["rest_positions"] - [0.02, 0, 0.03], axis=1)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for ax, name in zip(axes, ("right_shallow", "right_deep"), strict=True):
        mask = test["trajectory"] == name
        times = test["time"][mask] + 0.01
        ax.plot(
            times,
            -test["target_delta"][mask, node, 2] * 1000,
            color="black",
            label="True increment",
            linewidth=2,
        )
        for method, label in zip(METHODS, labels, strict=True):
            ax.plot(times, -test[method][mask, node, 2] * 1000, label=label, alpha=0.8)
        ax.set(
            title=name,
            xlabel="Next record time (s)",
            ylabel="Downward increment at right-top node (mm)",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(directory / "test_increment_curves.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_selected(
    directory: Path, data_dir: Path, config: dict, packs: dict | None = None
) -> tuple[list[dict], dict, dict]:
    # This function is called only after both checkpoints have been selected and hashed.
    models = load_selected(directory, config)
    encoder = models["direct"]
    packs = {} if packs is None else packs
    for split in ("train", "validation", "test"):
        if split not in packs:
            packs[split] = load_split(data_dir, config, split, encoder)
    rows, predictions, checks = [], {}, {}
    for split, pack in packs.items():
        values = {}
        for method in METHODS:
            model = models.get(method)
            result = predict(method, model, pack)
            if not torch.isfinite(result).all() or torch.count_nonzero(
                result[:, pack.dataset.pinned]
            ):
                raise ValueError("Non-finite or nonzero fixed-node increment")
            if model is not None:
                live = torch.stack(
                    [model(pack.dataset[i].inputs) for i in range(len(pack.dataset))]
                )
                if not torch.allclose(result, live, rtol=1e-5, atol=1e-10):
                    raise ValueError("Live current-frame predictions differ from the batch cache")
            values[method] = result.numpy()
            rows.extend(metric_rows(split, method, result, pack))
        values.update(
            {
                "trajectory": np.asarray(pack.names),
                "frame": np.asarray(pack.frames),
                "phase_id": np.asarray(pack.phases),
                "time": np.asarray(pack.times),
                "target_delta": pack.target.numpy(),
                "current_velocity": pack.velocity.numpy(),
                "node_ids": pack.dataset.node_ids.numpy(),
                "pinned_mask": pack.dataset.pinned.numpy(),
                "contact_neighborhood_mask": pack.neighborhood.numpy(),
                "rest_positions": pack.dataset.rest.numpy(),
            }
        )
        predictions[split] = values
        checks[split] = {
            "pairs": len(pack.target),
            "selection_counts": pack.dataset.selection_counts,
            "source_hashes": pack.dataset.source_hashes,
            "live_cache_agree": True,
            "pinned_increment_max_m": 0.0,
        }
    return rows, predictions, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/compare_soft_residual.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_residual_comparison/run_001"
    )
    parser.add_argument("--evaluate-only", type=Path, help="Re-evaluate already locked checkpoints")
    args = parser.parse_args()
    if args.evaluate_only:
        directory = args.evaluate_only.resolve()
        config = yaml.safe_load((directory / "config.yaml").read_text())
    else:
        directory = args.output_dir.resolve()
        if directory.exists():
            parser.error(f"Output exists; select a new --output-dir: {directory}")
        config = load_config(args.config)
        directory.mkdir(parents=True)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    original_results = ROOT / "outputs/soft_local_training/run_001"
    protected_before = directory_hashes(original_results)
    data_before = directory_hashes(args.data_dir)
    if args.evaluate_only:
        rows, predictions, checks = evaluate_selected(directory, args.data_dir, config)
        previous = json.loads((directory / "summary.json").read_text())
        if rows != previous["metrics"]:
            raise ValueError("Re-evaluated metrics differ from the saved results")
        print(
            "Locked checkpoints re-evaluated: exact saved metrics match; source results preserved."
        )
        if (
            directory_hashes(original_results) != protected_before
            or directory_hashes(args.data_dir) != data_before
        ):
            raise ValueError("Evaluation changed original data/results")
        return
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    encoder = PinnedGaussianDeformation(**config["model"])
    packs = {
        split: load_split(args.data_dir, config, split, encoder)
        for split in ("train", "validation")
    }
    baseline_validation_losses = {
        method: float(
            balanced_loss(
                predict(method, None, packs["validation"]),
                packs["validation"].target,
                packs["validation"].active,
                packs["validation"].neighborhood,
                config["training"],
            )
        )
        for method in ("static", "constant_velocity")
    }
    print(
        "Loaded only train and validation:",
        {key: value.dataset.selection_counts for key, value in packs.items()},
        flush=True,
    )
    selections = {}
    for kind in ("direct", "velocity_residual"):
        result = fit_model(kind, packs["train"], packs["validation"], config, directory / kind)
        write_csv(directory / kind / "history.csv", result.pop("history"))
        selections[kind] = result
    lock = {
        "locked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "metric": config["selection_metric"],
        "include_epoch_zero": True,
        "resolved_config_sha256": file_sha256(directory / "config.yaml"),
        "checkpoint_sha256": {
            kind: file_sha256(directory / kind / "checkpoint.pt") for kind in selections
        },
        "selections": selections,
        "test_arrays_loaded": False,
    }
    write_json(directory / "selection_locked.json", lock)
    print("Both checkpoints locked; now loading test trajectories.", flush=True)
    rows, predictions, checks = evaluate_selected(directory, args.data_dir, config, packs)
    for split, values in predictions.items():
        np.savez_compressed(directory / f"{split}_predictions.npz", **values)
    write_csv(directory / "metrics.csv", rows)
    write_csv(
        directory / "per_trajectory_metrics.csv",
        [row for row in rows if row["trajectory"] != "all"],
    )
    protected_after = directory_hashes(original_results)
    data_after = directory_hashes(args.data_dir)
    if protected_after != protected_before or data_after != data_before:
        raise ValueError("Original dataset or direct-model results were changed")
    source_files = {
        "comparison_entrypoint.py": Path(__file__),
        "comparison_backend.py": ROOT / "src/contact_gwm/comparison.py",
        "local_models.py": ROOT / "src/contact_gwm/local_training.py",
        "gaussian_pairs.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
        "deformation.py": ROOT / "src/contact_gwm/deformation.py",
    }
    for name, source in source_files.items():
        shutil.copy2(source, directory / name)
    summary = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": config["evaluation"]["protocol"],
        "splits": config["splits"],
        "selection": selections,
        "selection_lock": lock,
        "split_checks": checks,
        "metrics": rows,
        "baseline_validation_losses": baseline_validation_losses,
        "oracle_current_contact_conditioning": True,
        "future_inputs_used": False,
        "test_loaded_after_both_selections_locked": True,
        "free_rollout": False,
        "original_data_unchanged": data_after == data_before,
        "original_direct_results_unchanged": protected_after == protected_before,
        "original_direct_result_sha256": protected_before,
        "source_data_sha256": data_before,
        "code_snapshot_sha256": {name: file_sha256(directory / name) for name in source_files},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
    }
    write_json(directory / "summary.json", summary)
    make_plots(directory, rows, selections, predictions)
    print("\nTest increment vector RMSE (mm):")
    for phase in ("all", "press", "hold"):
        for method in METHODS:
            values = {
                row["region"]: row["vector_rmse_mm"]
                for row in rows
                if row["split"] == "test"
                and row["trajectory"] == "all"
                and row["phase"] == phase
                and row["method"] == method
            }
            print(phase, method, values)
    print("Selections:", selections)
    print("Output:", directory)


if __name__ == "__main__":
    main()
