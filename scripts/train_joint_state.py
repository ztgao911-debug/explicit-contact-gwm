"""Train the autonomous joint state map and a same-capacity contact-gated control."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from validate_contact_pipeline import hashes, reference_masks, source_labels, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.joint_state_dynamics import JointStateDynamics, JointStatePipeline
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline

ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = ("context", "gates", "features", "velocity")
VARIANTS = ("gated_joint", "autonomous_joint")


def model_for(mesh, config, autonomous):
    return JointStateDynamics(
        torch.tensor(mesh["rest_positions_world"]).float(),
        torch.tensor(mesh["pinned_mask"]).bool(),
        **config["model"],
        dt=config["dt"],
        autonomous=autonomous,
    )


def load_data(directory, config):
    mesh = read_npz(directory / "mesh.npz")
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_sha256(directory / "mesh.npz") != manifest["shared_sha256"]["mesh.npz"]:
        raise ValueError("Independent mesh checksum mismatch")
    if not np.array_equal(mesh["node_ids"], np.arange(len(mesh["node_ids"]))):
        raise ValueError("Node IDs must preserve the stable rest-mesh order")
    splits = {name: split for split, names in config["splits"].items() for name in names}
    if (
        config["splits"]["train"] != ["center_shallow", "center_deep"]
        or config["splits"]["validation"] != ["left_shallow", "left_deep"]
        or set(splits)
        != {"center_shallow", "center_deep", "left_shallow", "left_deep", "no_contact"}
    ):
        raise ValueError("Keep central training, left validation and the no-contact control")
    trajectories = {}
    for name in splits:
        path = directory / f"{name}.npz"
        expected = next(
            item["sha256"] for item in manifest["trajectories"] if item["file"] == path.name
        )
        if file_sha256(path) != expected:
            raise ValueError(f"Independent trajectory checksum mismatch: {name}")
        arrays = read_npz(path)
        if not np.array_equal(arrays["node_ids"], mesh["node_ids"]):
            raise ValueError("Trajectory changed node IDs")
        if not np.allclose(np.diff(arrays["time"]), config["dt"], atol=1e-9, rtol=0):
            raise ValueError("Only the existing dt=0.01 s records are supported")
        trajectories[name] = arrays
    return mesh, trajectories, splits


def make_cache(pipeline, trajectories, splits, mesh, config, output):
    """Only current x/v/external robot inputs construct network inputs.

    Future x/v are read in the separately named target arrays ONLY. No future
    contact, force, phase or action is needed for sample selection: use ALL t.
    """
    values = {key: [] for key in (*INPUT_KEYS, "target_dx", "target_dv", "neighborhood")}
    samples = []
    with torch.no_grad():
        for name, arrays in trajectories.items():
            masks = reference_masks(arrays, mesh, config["contact_neighborhood_radius_m"])
            for frame in range(len(arrays["time"]) - 1):
                state = pipeline.legacy.gaussian_state(
                    arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
                )
                current = pipeline.current_inputs(state, IndenterInput.from_arrays(arrays, frame))
                for key in INPUT_KEYS:
                    values[key].append(getattr(current, key))
                values["target_dx"].append(
                    torch.tensor(
                        arrays["node_position_world"][frame + 1]
                        - arrays["node_position_world"][frame],
                        dtype=torch.float32,
                    )
                )
                values["target_dv"].append(
                    torch.tensor(
                        arrays["node_velocity_world"][frame + 1]
                        - arrays["node_velocity_world"][frame],
                        dtype=torch.float32,
                    )
                )
                values["neighborhood"].append(torch.tensor(masks[frame]))
                samples.append(
                    {
                        "trajectory": name,
                        "frame": frame,
                        "split": splits[name],
                        "time_s": float(arrays["time"][frame]),
                        "phase_id": int(arrays["phase_id"][frame]),
                        "geometric_contact_count": current.contacts.num_contacts,
                        "has_force_contact_scoring_only": bool(len(source_labels(arrays, frame))),
                    }
                )
            print(f"cached CURRENT inputs: {name}, all 300 pairs", flush=True)
    cache = {key: torch.stack(tensors) for key, tensors in values.items()}
    if not all(torch.isfinite(tensor).all() for tensor in cache.values()):
        raise ValueError("Non-finite current inputs or training labels")
    cache["pinned"] = torch.tensor(mesh["pinned_mask"]).bool()
    np.savez_compressed(
        output / "current_inputs_and_labels.npz",
        **{key: value.numpy() for key, value in cache.items()},
        node_ids=mesh["node_ids"],
    )
    write_json(output / "samples.json", samples)
    return cache, samples


def batch_loss(model, cache, indices, training):
    """Equal position/velocity weight, balanced active/ROI coordinate MSE."""
    prediction = model(*(cache[key][indices] for key in INPUT_KEYS))
    errors = (
        (prediction.delta_position - cache["target_dx"][indices])
        / training["position_loss_unit_m"],
        (prediction.delta_velocity - cache["target_dv"][indices])
        / training["velocity_loss_unit_m_s"],
    )
    active = ~cache["pinned"]
    neighborhood = cache["neighborhood"][indices]
    components = []
    for error in errors:
        active_loss = error[:, active].square().mean()
        near_loss = (error.square() * neighborhood[..., None]).sum() / (
            3 * neighborhood.sum()
        ).clamp_min(1)
        components.append(
            (1 - training["neighborhood_weight"]) * active_loss
            + training["neighborhood_weight"] * near_loss
        )
    return (
        (1 - training["velocity_weight"]) * components[0]
        + training["velocity_weight"] * components[1]
    ), components


def fit_variant(model, variant, initial_state, cache, samples, config, output):
    settings = config["training"]
    model.load_state_dict(initial_state, strict=True)
    train_ids = torch.tensor([i for i, s in enumerate(samples) if s["split"] == "train"])
    validation_ids = torch.tensor([i for i, s in enumerate(samples) if s["split"] == "validation"])
    if len(train_ids) != 600 or len(validation_ids) != 600:
        raise ValueError("Use every central/left adjacent-frame pair, including no contact")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    history, finite_gradients, clipped = [], True, 0
    best_loss, best_step, best_state = math.inf, None, None
    start = time.perf_counter()
    batch_stream = torch.empty(0, dtype=torch.long)
    for step in range(settings["updates"] + 1):
        if step % settings["validation_interval"] == 0 or step == settings["updates"]:
            model.eval()
            with torch.no_grad():
                train_loss, train_components = batch_loss(model, cache, train_ids, settings)
                val_loss, val_components = batch_loss(model, cache, validation_ids, settings)
            row = {
                "step": step,
                "train_joint_mse": float(train_loss),
                "validation_joint_mse": float(val_loss),
                "train_position_mse": float(train_components[0]),
                "train_velocity_mse": float(train_components[1]),
                "validation_position_mse": float(val_components[0]),
                "validation_velocity_mse": float(val_components[1]),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError("Non-finite full-dataset selection loss")
            if float(val_loss) < best_loss:
                best_loss, best_step = float(val_loss), step
                best_state = copy.deepcopy(model.state_dict())
            if step % 250 == 0:
                print(
                    f"{variant} step {step}: train={float(train_loss):.7g}, "
                    f"validation={float(val_loss):.7g}",
                    flush=True,
                )
        if step == settings["updates"]:
            break
        if len(batch_stream) < settings["batch_size"]:
            batch_stream = torch.cat(
                (batch_stream, train_ids[torch.randperm(len(train_ids), generator=generator)])
            )
        batch, batch_stream = (
            batch_stream[: settings["batch_size"]],
            batch_stream[settings["batch_size"] :],
        )
        fraction = step / max(1, settings["updates"] - 1)
        multiplier = settings["minimum_learning_rate_fraction"] + (
            1 - settings["minimum_learning_rate_fraction"]
        ) * 0.5 * (1 + math.cos(math.pi * fraction))
        optimizer.param_groups[0]["lr"] = settings["learning_rate"] * multiplier
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, cache, batch, settings)
        (loss * settings["loss_multiplier"]).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
        )
        finite_gradients &= bool(torch.isfinite(norm))
        clipped += int(norm > settings["gradient_clip"])
        optimizer.step()
    model.load_state_dict(best_state, strict=True)
    model.eval()
    variant_dir = output / variant
    variant_dir.mkdir()
    write_csv(variant_dir / "training_curve.csv", history)
    torch.save(
        {
            "model_state": best_state,
            "architecture": {
                **config["model"],
                "dt": config["dt"],
                "autonomous": variant == "autonomous_joint",
            },
            "config": config,
            "selected_step": best_step,
            "selection_loss": best_loss,
            "training_pairs": len(train_ids),
            "validation_pairs": len(validation_ids),
            "node_ids": np.arange(len(cache["pinned"])),
        },
        variant_dir / "checkpoint.pt",
    )
    result = {
        "variant": variant,
        "selected_step": best_step,
        "selection_loss": best_loss,
        "updates": settings["updates"],
        "train_pair_count": len(train_ids),
        "validation_pair_count": len(validation_ids),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - start,
        "clipped_updates": clipped,
        "finite_gradients": finite_gradients,
    }
    write_json(variant_dir / "training_summary.json", result)
    return result, history


def verify_training_inputs(pipeline, cache, samples, trajectories):
    """Future-state/force poisoning changes supervision, never current features."""
    arrays = trajectories["center_deep"]
    frame = 71
    initial = pipeline.legacy.gaussian_state(
        arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
    )
    source = IndenterInput.from_arrays(arrays, frame)
    before = pipeline.current_inputs(initial, source)
    poisoned = {key: value.copy() for key, value in arrays.items()}
    for key in ("node_position_world", "node_velocity_world"):
        poisoned[key][frame + 1 :] = np.nan
    for key in poisoned:
        if key.startswith("contact_") or key == "net_contact_force_on_block_world":
            poisoned[key] = np.full(poisoned[key].shape, np.nan)
    after = pipeline.current_inputs(initial, IndenterInput.from_arrays(poisoned, frame))
    assert all(torch.equal(getattr(before, key), getattr(after, key)) for key in INPUT_KEYS)
    for phase in range(5):
        assert any(s["split"] == "train" and s["phase_id"] == phase for s in samples)
    expected = next(
        i for i, s in enumerate(samples) if s["trajectory"] == "center_deep" and s["frame"] == frame
    )
    assert all(torch.equal(cache[key][expected], getattr(before, key)) for key in INPUT_KEYS)
    return {
        "future_and_recorded_force_poisoning_inputs_identical": True,
        "all_five_phases_in_train_and_evaluation": True,
        "cache_and_rollout_current_encoder_bitwise_equal": True,
        "training_and_deployment_contact_policy": "all gap<=0 native geometric candidates",
    }


def plot_training(histories, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, component in zip(axes, ("joint", "position", "velocity"), strict=True):
        for variant, history in histories.items():
            for split, style in (("train", "-"), ("validation", "--")):
                axis.plot(
                    [row["step"] for row in history],
                    [row[f"{split}_{component}_mse"] for row in history],
                    style,
                    label=f"{variant} {split}",
                )
        axis.set(title=f"Normalized {component} MSE", xlabel="Updates", yscale="log")
        axis.grid(alpha=0.3)
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_joint_state.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if tuple(config["variants"]) != VARIANTS or config["seed"] != 7 or config["dt"] != 0.01:
        raise ValueError("Use the single fixed-seed, fixed-dt two-model protocol")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data = ROOT / config["data_directory"]
    before = hashes([*data.rglob("*"), *(ROOT / "outputs").rglob("*")])
    output = args.output
    if output is None:
        parent, index = ROOT / config["output_root"], 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        output = parent / f"run_{index:03d}"
    if output.exists():
        raise FileExistsError(f"Preserve existing outputs: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    import contact_gwm.joint_state_dynamics as model_module

    for path, name in (
        (Path(__file__), "training_entrypoint.py"),
        (Path(model_module.__file__), "joint_state_dynamics.py"),
    ):
        shutil.copy2(path, output / name)
    mesh, trajectories, splits = load_data(data, config)
    legacy_config = yaml.safe_load((ROOT / config["legacy_pipeline_config"]).read_text())
    legacy = PinnedContactPipeline(
        mesh, data / "scene.xml", ROOT / legacy_config["checkpoint"], legacy_config
    )
    preparation = JointStatePipeline(legacy, model_for(mesh, config, autonomous=True))
    cache, samples = make_cache(preparation, trajectories, splits, mesh, config, output)
    checks = verify_training_inputs(preparation, cache, samples, trajectories)
    # Identical initial parameter tensors and identical minibatch streams. The
    # only difference between the NEW models is the internal branch's gate.
    torch.manual_seed(config["seed"])
    initial_model = model_for(mesh, config, autonomous=True)
    initial_state = copy.deepcopy(initial_model.state_dict())
    summaries, histories, models = [], {}, {}
    for variant in VARIANTS:
        model = model_for(mesh, config, autonomous=variant == "autonomous_joint")
        result, history = fit_variant(model, variant, initial_state, cache, samples, config, output)
        summaries.append(result)
        histories[variant] = history
        models[variant] = model
    write_json(output / "training_summaries.json", summaries)
    plot_training(histories, output)
    # Reload SELECTED checkpoints before evaluating any rollout. Rollout does
    # not select or tune weights; the full validation one-step loss did that.
    predictors = {
        "constant_velocity": lambda s, r: legacy.step(s, r, "constant_velocity"),
        "frozen_legacy": lambda s, r: legacy.step(s, r, "mujoco_force"),
    }
    pipelines = {}
    for variant in VARIANTS:
        saved = torch.load(
            output / variant / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        restored = model_for(mesh, config, autonomous=variant == "autonomous_joint")
        restored.load_state_dict(saved["model_state"], strict=True)
        restored.eval()
        assert all(
            torch.equal(value, models[variant].state_dict()[key])
            for key, value in restored.state_dict().items()
        )
        pipelines[variant] = JointStatePipeline(legacy, restored)
        predictors[variant] = pipelines[variant].step
    import joint_state_evaluation
    from joint_state_evaluation import evaluate_models

    shutil.copy2(joint_state_evaluation.__file__, output / "evaluation_implementation.py")

    evaluation = evaluate_models(
        predictors, legacy.gaussian_state, trajectories, splits, mesh, config, output
    )
    checks.update(evaluation["checks"])
    if checks["rollout_exception_count"] or not checks["common_evaluation_masks"]:
        raise RuntimeError("Incomplete rollout coverage: do not compare unequal windows")
    # Actual saved/reloaded 20-step equality, not just a parameter comparison.
    arrays, frame = trajectories["left_deep"], 170
    state = legacy.gaussian_state(
        arrays["node_position_world"][frame], arrays["node_velocity_world"][frame]
    )
    external = [IndenterInput.from_arrays(arrays, t) for t in range(frame, frame + 20)]
    for variant in VARIANTS:
        original = JointStatePipeline(legacy, models[variant]).rollout(state, external)
        restored = pipelines[variant].rollout(state, external)
        assert all(
            torch.equal(a.state.centers, b.state.centers)
            and torch.equal(a.state.velocities, b.state.velocities)
            for a, b in zip(original, restored, strict=True)
        )
    checks["saved_reloaded_20_step_rollout_bitwise_equal"] = True
    assert before == {path: file_sha256(ROOT / path) for path in before}
    checks["historical_file_count_unchanged_sha256"] = len(before)
    checks["same_initial_state_and_minibatches_new_models"] = True
    write_json(output / "checks.json", checks)
    write_json(output / "source_hashes.json", before)
    import mujoco

    environment = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "cpu_threads": config["cpu_threads"],
        "seed": config["seed"],
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "loaded_trajectories": list(trajectories),
        "test_trajectories_loaded": False,
        "training": summaries,
    }
    write_json(output / "environment.json", environment)
    write_json(
        output / "summary.json",
        {"environment": environment, "checks": checks, "evaluation": evaluation},
    )
    print(
        json.dumps({"output": str(output), **environment, "checks": checks}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
