"""Inspect true velocity residuals and overfit eight fixed center pairs on CPU."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from contact_gwm.comparison import metric_rows, predict, prepare_pairs
from contact_gwm.data.gaussian_pairs import GaussianPairDataset, file_sha256
from contact_gwm.local_training import ConstantVelocityResidualDeformation
from contact_gwm.residual_diagnostics import (
    fit_fixed_subset,
    fixed_subset_indices,
    gated_output_floor,
    velocity_residual,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict | list) -> None:
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
    config = yaml.safe_load((path.parent / addition.pop("base_config")).read_text())
    config.update(addition)
    overrides = config.get("model_overrides", {})
    if set(overrides) - {"max_translation"}:
        raise ValueError("Only output amplitude may change in this controlled diagnostic")
    config["model"].update(overrides)
    if config["diagnostic_splits"] != {
        "train": ["center_shallow", "center_deep"],
        "validation": ["left_shallow", "left_deep"],
    }:
        raise ValueError("Use only center/train and left/validation in this diagnostic")
    if config["device"] != "cpu" or config["dt"] != 0.01:
        raise ValueError("CPU and dt=0.01 s are required")
    config["evaluation"]["protocol"] = "fixed_subset_optimization_diagnostic_not_generalization"
    return config


def compare_runs(output, previous, config, samples, metrics, history):
    """Require one output-amplitude difference before drawing a controlled comparison."""
    old_config = yaml.safe_load((previous / "config.yaml").read_text())
    comparable = []
    for value in (old_config, config):
        value = copy.deepcopy(value)
        value.pop("model_directory", None)
        value.pop("model_overrides", None)
        value["model"].pop("max_translation")
        comparable.append(value)
    old_summary = json.loads((previous / "summary.json").read_text())
    if comparable[0] != comparable[1] or samples != old_summary["fixed_samples"]:
        raise ValueError("Controlled comparison changed settings or fixed samples")
    rows = []
    histories = {}
    for label, directory, run_metrics, run_history, run_config in (
        ("Original amplitude", previous, old_summary["metrics"], None, old_config),
        ("Smaller amplitude", output, metrics, history, config),
    ):
        if run_history is None:
            model_directory = run_config.get("model_directory", "unchanged_model")
            with (directory / model_directory / "history.csv").open() as handle:
                run_history = list(csv.DictReader(handle))
        histories[label] = run_history
        for row in run_metrics:
            if row["trajectory"] == "all" and row["phase"] == "all":
                # Include the common CV baseline only once.
                if row["method"] == "constant_velocity" and directory == previous:
                    continue
                rows.append(
                    {
                        "run": label if row["method"] == "velocity_residual" else "CV baseline",
                        "max_translation_m": run_config["model"]["max_translation"],
                        **row,
                    }
                )
    write_csv(output / "controlled_comparison.csv", rows)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, key, label in zip(
        axes,
        ("active_rmse_mm", "neighborhood_rmse_mm"),
        ("All active nodes", "Contact neighborhood"),
        strict=True,
    ):
        for run, values in histories.items():
            ax.semilogy(
                [int(row["epoch"]) for row in values],
                [float(row[key]) * 1000 for row in values],
                label=run,
            )
        ax.axhline(float(history[0][key]) * 1000, color="black", linestyle=":", label="CV baseline")
        ax.set(title=label, xlabel="Epoch", ylabel="Same-eight-pair vector RMSE (um)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "controlled_comparison.png", dpi=160)
    plt.close(fig)
    return {
        "previous_run": str(previous),
        "only_model_change": "max_translation",
        "previous_max_translation_m": old_config["model"]["max_translation"],
        "current_max_translation_m": config["model"]["max_translation"],
        "fixed_samples_and_other_settings_match": True,
    }


def slice_pack(pack, indices):
    return replace(
        pack,
        context=pack.context[indices],
        gates=pack.gates[indices],
        velocity=pack.velocity[indices],
        target=pack.target[indices],
        neighborhood=pack.neighborhood[indices],
        names=[pack.names[i] for i in indices],
        frames=[pack.frames[i] for i in indices],
        phases=[pack.phases[i] for i in indices],
        times=[pack.times[i] for i in indices],
    )


def residual_statistics(split, pack, config):
    residual = velocity_residual(pack).double()
    magnitude = residual.norm(dim=-1)
    floor = gated_output_floor(
        residual, pack.gates.double(), pack.dataset.pinned, config["model"]["max_translation"]
    )
    rows = []
    for name in ("all", *sorted(set(pack.names))):
        for phase_id, phase in ((None, "all"), (1, "press"), (2, "hold")):
            chosen = torch.tensor(
                [
                    (name == "all" or name == n) and (phase_id is None or p == phase_id)
                    for n, p in zip(pack.names, pack.phases, strict=True)
                ]
            )
            if not chosen.any():
                continue
            for region, region_mask in (
                ("active_nodes", pack.active[None].expand(len(pack.target), -1)),
                ("contact_neighborhood", pack.neighborhood),
            ):
                mask = chosen[:, None] & region_mask
                values = magnitude[mask]
                delta_rms = pack.target.double().norm(dim=-1)[mask].square().mean().sqrt()
                residual_rms = values.square().mean().sqrt()
                needed = residual.abs()[mask] / pack.gates.double()[mask, None].clamp_min(1e-30)
                percentiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 1.0]).double())
                rows.append(
                    {
                        "split": split,
                        "trajectory": name,
                        "phase": phase,
                        "region": region,
                        "pairs": int(chosen.sum()),
                        "node_frames": int(mask.sum()),
                        "increment_vector_rms_um": float(delta_rms * 1e6),
                        "residual_vector_rms_um": float(residual_rms * 1e6),
                        "residual_to_increment_rms_ratio": float(residual_rms / delta_rms),
                        **{
                            f"residual_vector_{label}_um": float(value * 1e6)
                            for label, value in zip(
                                ("median", "p90", "p99", "max"), percentiles, strict=True
                            )
                        },
                        "gate_min": float(pack.gates[mask].min()),
                        "required_ungated_axis_p99_um": float(torch.quantile(needed, 0.99) * 1e6),
                        "required_ungated_axis_max_um": float(needed.max() * 1e6),
                        "bound_limited_node_fraction": float(
                            (needed.max(dim=-1).values > config["model"]["max_translation"])
                            .double()
                            .mean()
                        ),
                        "optimistic_bound_floor_vector_rmse_um": float(
                            floor.norm(dim=-1)[mask].square().mean().sqrt() * 1e6
                        ),
                    }
                )
    return rows


def save_diagnostic_arrays(path, pack):
    values = {
        "trajectory": np.asarray(pack.names),
        "frame": np.asarray(pack.frames),
        "phase_id": np.asarray(pack.phases),
        "time": np.asarray(pack.times),
        "node_ids": pack.dataset.node_ids.numpy(),
        "pinned_mask": pack.dataset.pinned.numpy(),
        "rest_positions": pack.dataset.rest.numpy(),
        "contact_neighborhood_mask": pack.neighborhood.numpy(),
        "gate": pack.gates.numpy(),
        "current_velocity": pack.velocity.numpy(),
        "target_delta": pack.target.numpy(),
        "true_velocity_residual": velocity_residual(pack).numpy(),
    }
    np.savez_compressed(path, **values)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(values) or any(
            not np.array_equal(archive[key], value) for key, value in values.items()
        ):
            raise RuntimeError("Diagnostic NPZ reload mismatch")


def make_plots(output, packs, subset, history, predictions):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, region in zip(axes, ("active_nodes", "contact_neighborhood"), strict=True):
        for split, pack in packs.items():
            r = velocity_residual(pack).norm(dim=-1).numpy() * 1e6
            for phase, label in ((1, "press"), (2, "hold")):
                mask = np.asarray(pack.phases) == phase
                nodes = np.broadcast_to(pack.active.numpy(), r.shape).copy()
                if region == "contact_neighborhood":
                    nodes &= pack.neighborhood.numpy()
                values = np.sort(r[mask][nodes[mask]])
                ax.plot(
                    values, np.arange(1, len(values) + 1) / len(values), label=f"{split} {label}"
                )
        ax.set(xscale="log", xlabel="True residual vector norm (um)", ylabel="CDF", title=region)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "residual_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for column, (split, pack) in enumerate(packs.items()):
        residual = velocity_residual(pack)
        for row, name in enumerate(sorted(set(pack.names))):
            mask = np.asarray(pack.names) == name
            ax = axes[row, column]
            for label, values, nodes in (
                ("True increment: active", pack.target, pack.active[None].expand(len(mask), -1)),
                ("CV residual: active", residual, pack.active[None].expand(len(mask), -1)),
                ("CV residual: near", residual, pack.neighborhood),
            ):
                rms = [
                    float(values[i, nodes[i]].square().sum(-1).mean().sqrt()) * 1e6
                    for i in np.flatnonzero(mask)
                ]
                ax.plot(np.asarray(pack.times)[mask], rms, label=label)
            ax.axvline(1.0, color="black", linestyle=":", label="Press -> hold")
            ax.set(title=f"{split}: {name}", xlabel="Current time (s)", ylabel="Vector RMS (um)")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    fig.savefig(output / "residual_time_curves.png", dpi=160)
    plt.close(fig)

    spatial = {}
    for split, pack in packs.items():
        for phase in (1, 2):
            r = velocity_residual(pack)[np.asarray(pack.phases) == phase]
            spatial[split, phase] = r.square().sum(-1).mean(0).sqrt()[pack.active].numpy() * 1e6
    upper = max(float(value.max()) for value in spatial.values())
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    for column, (split, pack) in enumerate(packs.items()):
        xyz = pack.dataset.rest[pack.active].numpy() * 1000
        for row, phase in enumerate((1, 2)):
            ax = fig.add_subplot(2, 2, row * 2 + column + 1, projection="3d")
            points = ax.scatter(*xyz.T, c=spatial[split, phase], norm=Normalize(0, upper), s=24)
            ax.set(
                title=f"{split}: {'press' if phase == 1 else 'hold'}",
                xlabel="Rest x (mm)",
                ylabel="Rest y (mm)",
                zlabel="Rest z (mm)",
            )
    fig.colorbar(points, ax=fig.axes, label="Node residual vector RMS (um)", shrink=0.7)
    fig.savefig(output / "residual_spatial.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, key, label in zip(
        axes,
        ("active_rmse_mm", "neighborhood_rmse_mm"),
        ("All active nodes", "Contact neighborhood"),
        strict=True,
    ):
        ax.semilogy([r["epoch"] for r in history], [r[key] * 1000 for r in history])
        ax.axhline(history[0][key] * 1000, color="black", linestyle=":", label="CV baseline")
        ax.set(title=label, xlabel="Epoch", ylabel="Same-subset increment vector RMSE (um)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "overfit_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    truth = velocity_residual(subset)[:, subset.active].numpy() * 1e6
    correction = (
        predictions["fitted_subset"]["velocity_residual"]
        - predictions["fitted_subset"]["constant_velocity"]
    )[:, subset.active].numpy() * 1e6
    for axis, label in ((0, "x"), (2, "z")):
        ax = axes[axis // 2]
        ax.scatter(truth[..., axis], correction[..., axis], s=5, alpha=0.4)
        bound = max(abs(truth[..., axis]).max(), abs(correction[..., axis]).max())
        ax.plot([-bound, bound], [-bound, bound], color="black", label="Exact correction")
        ax.axhline(0, color="gray", linestyle=":", label="CV: zero correction")
        ax.set(
            title=f"Fitted eight pairs: {label} correction",
            xlabel="True residual (um)",
            ylabel="Learned gated correction (um)",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "overfit_corrections.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnose_soft_residual.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mujoco_soft_pilot/pilot_001")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/soft_residual_diagnostics/run_001"
    )
    parser.add_argument("--compare-to", type=Path, help="Compare to the unchanged-model run")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output exists; select another --output-dir: {output}")
    config = load_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    protected_dirs = {
        "data": args.data_dir.resolve(),
        "first_direct_fit": ROOT / "outputs/soft_local_training/run_001",
        "trajectory_comparison": ROOT / "outputs/soft_residual_comparison/run_001",
    }
    if args.compare_to:
        protected_dirs["previous_diagnostic"] = args.compare_to.resolve()
    before = {key: directory_hashes(path) for key, path in protected_dirs.items()}
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    packs = {}
    encoder = ConstantVelocityResidualDeformation(**config["model"], dt=config["dt"])
    for split, names in config["diagnostic_splits"].items():
        local_config = copy.deepcopy(config)
        local_config["trajectories"] = names
        packs[split] = prepare_pairs(GaussianPairDataset(args.data_dir, local_config), encoder)
        save_diagnostic_arrays(output / f"{split}_residuals.npz", packs[split])
    stats = [
        row for split, pack in packs.items() for row in residual_statistics(split, pack, config)
    ]
    write_csv(output / "residual_statistics.csv", stats)
    train = packs["train"]
    indices = fixed_subset_indices(train, config["subset_quantiles"])
    samples = [
        {
            "index": i,
            "trajectory": train.names[i],
            "frame": train.frames[i],
            "time": train.times[i],
            "phase": "press" if train.phases[i] == 1 else "hold",
        }
        for i in indices.tolist()
    ]
    write_json(output / "fixed_samples.json", samples)
    print("Fixed label-independent center samples:", samples, flush=True)
    model_directory = output / config.get("model_directory", "unchanged_model")
    result = fit_fixed_subset(train, indices, config, model_directory)
    history = result.pop("history")
    write_csv(model_directory / "history.csv", history)
    checkpoint = torch.load(model_directory / "checkpoint.pt", weights_only=True)
    model = ConstantVelocityResidualDeformation(**config["model"], dt=config["dt"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # Validation is inspected only after fixing the same-subset-selected checkpoint.
    write_json(
        output / "overfit_selection_locked.json",
        {
            **result,
            "selection_metric": checkpoint["selection_metric"],
            "checkpoint_sha256": file_sha256(model_directory / "checkpoint.pt"),
            "validation_used_for_selection": False,
            "test_arrays_loaded": False,
        },
    )
    evaluation_packs = {"fitted_subset": slice_pack(train, indices), **packs}
    predictions, metrics, checks = {}, [], {}
    with torch.no_grad():
        for split, pack in evaluation_packs.items():
            predictions[split] = {}
            for kind in ("constant_velocity", "velocity_residual"):
                value = predict(kind, model if kind == "velocity_residual" else None, pack)
                if not torch.isfinite(value).all() or torch.count_nonzero(
                    value[:, pack.dataset.pinned]
                ):
                    raise RuntimeError("Non-finite prediction or fixed-node movement")
                predictions[split][kind] = value
                metrics.extend(metric_rows(split, kind, value, pack))
            if split != "fitted_subset":
                live = torch.stack([model(pack.dataset[i].inputs) for i in range(len(pack.target))])
                if not torch.allclose(
                    live, predictions[split]["velocity_residual"], atol=1e-10, rtol=1e-5
                ):
                    raise RuntimeError("Current-input and cached predictions differ")
                checks[split] = {
                    "live_cache_agree": True,
                    "pairs": len(pack.target),
                    "source_hashes": pack.dataset.source_hashes,
                }
            np.savez_compressed(
                output / f"{split}_predictions.npz",
                **{kind: value.numpy() for kind, value in predictions[split].items()},
                target_delta=pack.target.numpy(),
                trajectory=np.asarray(pack.names),
                frame=np.asarray(pack.frames),
                node_ids=pack.dataset.node_ids.numpy(),
                pinned_mask=pack.dataset.pinned.numpy(),
                contact_neighborhood_mask=pack.neighborhood.numpy(),
            )
    write_csv(output / "increment_metrics.csv", metrics)
    reductions = {}
    for region in ("active_nodes", "contact_neighborhood"):
        values = {
            row["method"]: row["vector_rmse_mm"]
            for row in metrics
            if row["split"] == "fitted_subset"
            and row["trajectory"] == "all"
            and row["phase"] == "all"
            and row["region"] == region
        }
        reductions[region] = 1 - values["velocity_residual"] / values["constant_velocity"]
    make_plots(output, packs, evaluation_packs["fitted_subset"], history, predictions)
    controlled_comparison = None
    if args.compare_to:
        controlled_comparison = compare_runs(
            output, args.compare_to.resolve(), config, samples, metrics, history
        )
    source_files = {
        "diagnostic_entrypoint.py": Path(__file__),
        "diagnostic_backend.py": ROOT / "src/contact_gwm/residual_diagnostics.py",
        "comparison_backend.py": ROOT / "src/contact_gwm/comparison.py",
        "local_models.py": ROOT / "src/contact_gwm/local_training.py",
        "gaussian_pairs.py": ROOT / "src/contact_gwm/data/gaussian_pairs.py",
        "deformation.py": ROOT / "src/contact_gwm/deformation.py",
    }
    for name, source in source_files.items():
        shutil.copy2(source, output / name)
    unchanged = {key: directory_hashes(path) == before[key] for key, path in protected_dirs.items()}
    if not all(unchanged.values()):
        raise RuntimeError("Existing data/results were modified")
    write_json(
        output / "summary.json",
        {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "protocol": config["evaluation"]["protocol"],
            "status": "completed",
            "overfit": result,
            "controlled_comparison": controlled_comparison,
            "fixed_samples": samples,
            "statistics": stats,
            "metrics": metrics,
            "fitted_subset_rmse_reductions": reductions,
            "meets_predeclared_fitting_criterion": all(
                value >= config["overfit"]["required_rmse_reduction"]
                for value in reductions.values()
            ),
            "split_checks": checks,
            "test_arrays_loaded": False,
            "validation_used_for_selection": False,
            "oracle_current_contact_conditioning": True,
            "future_inputs_used": False,
            "free_rollout": False,
            "npz_reload_exact": True,
            "protected_unchanged": unchanged,
            "protected_sha256": before,
            "code_snapshot_sha256": {name: file_sha256(output / name) for name in source_files},
            "total_wall_seconds": time.perf_counter() - started,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "platform": platform.platform(),
            },
        },
    )
    print("Same-subset RMSE reductions:", reductions, flush=True)
    print("Protected data/results unchanged:", unchanged, flush=True)
    print("Output:", output, flush=True)


if __name__ == "__main__":
    main()
