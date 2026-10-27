"""Fixed central detached-window fit: one-step vs differentiable five-step loss."""

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
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import load_config as load_neighbor_config
from ablate_neighbor_recovery import tensor_digest
from validate_contact_pipeline import hashes, reference_masks, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.joint_state_dynamics import JointStatePipeline
from contact_gwm.neighbor_dynamics import make_neighbor_model, model_specification
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline
from contact_gwm.recovery_multistep import (
    autoregressive_recovery,
    joint_endpoint_loss,
    overlapping_indices,
    teacher_forced_recovery,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("one_step", "five_step")


def load_config(path):
    extra = yaml.safe_load(path.read_text())
    config = load_neighbor_config(path.parent / extra.pop("base_config"))
    config.update(extra)
    if (
        tuple(config["groups"]) != GROUPS
        or config["window"]["trajectory"] != "center_deep"
        or config["window"]["first_frame"] != 210
        or config["window"]["last_frame"] != 230
        or config["window"]["training_horizon"] != 5
        or config["window"]["selection"]
        != "lowest_common_five_step_training_loss_including_step_zero"
    ):
        raise ValueError("Keep the predeclared central recovery window and common selection")
    return config


def prepare(data, config):
    mesh = read_npz(data / "mesh.npz")
    arrays = read_npz(data / "center_deep.npz")
    manifest = json.loads((data / "manifest.json").read_text())
    expected = next(t["sha256"] for t in manifest["trajectories"] if t["file"] == "center_deep.npz")
    if (
        file_sha256(data / "center_deep.npz") != expected
        or file_sha256(data / "mesh.npz") != manifest["shared_sha256"]["mesh.npz"]
    ):
        raise ValueError("Independent source checksums differ")
    if not np.array_equal(mesh["node_ids"], arrays["node_ids"]):
        raise ValueError("Stable node IDs differ")
    first, last = config["window"]["first_frame"], config["window"]["last_frame"]
    if np.any(arrays["contact_count"][first : last + 1]):
        raise ValueError("Predeclared window is not a detached recovery segment")
    if not np.allclose(np.diff(arrays["time"][first : last + 1]), config["dt"], atol=1e-9, rtol=0):
        raise ValueError("Record cadence differs")
    rest = torch.tensor(mesh["rest_positions_world"], dtype=torch.float64)
    pinned = torch.tensor(mesh["pinned_mask"], dtype=torch.bool)
    x = torch.tensor(arrays["node_position_world"][first : last + 1], dtype=torch.float64)
    v = torch.tensor(arrays["node_velocity_world"][first : last + 1], dtype=torch.float64)
    current, target = overlapping_indices(len(x), config["window"]["training_horizon"])
    near = torch.tensor(
        reference_masks(arrays, mesh, config["contact_neighborhood_radius_m"])[first]
    )
    if not torch.equal(x[:, pinned], rest[pinned].expand(len(x), -1, -1)):
        raise ValueError("Recorded fixed boundary changed")
    batch = {
        "rest": rest,
        "pinned": pinned,
        "x": x,
        "v": v,
        "current": current,
        "target": target,
        "neighborhood": near,
    }
    # Conservative upper bound valid for EVERY training iterate: each detached
    # internal rx<=1mm, dv<=0.1m/s; gate=0 removes the external branch.
    horizon = config["window"]["training_horizon"]
    upper_z = (
        float(x[:-1, :, 2].max())
        + horizon * config["model"]["max_position_residual"]
        + config["dt"] * horizon * max(0.0, float(v[:-1, :, 2].max()))
        + config["dt"] * config["model"]["max_velocity_increment"] * horizon * (horizon - 1) / 2
        + 1e-7
    )
    material = manifest["config"]
    safe_z = (
        float(arrays["indenter_position_world"][first : last + 1, 2].min())
        - material["indenter"]["radius"]
        - material["block"]["collision_radius"]
    )
    if upper_z >= safe_z:
        raise ValueError("Cannot certify no contact for all five-step training predictions")
    certification = {
        "source_contact_count": 0,
        "all_iterates_node_z_upper_bound_m": upper_z,
        "sphere_bottom_minus_flex_skin_m": safe_z,
        "guaranteed_vertical_separation_m": safe_z - upper_z,
        "no_mujoco_derivative_or_hidden_force_used": True,
    }
    return mesh, arrays, batch, certification


def predict_training(model, batch, normalization, arm):
    rest, pin = batch["rest"], batch["pinned"]
    current = batch["current"]
    if arm == "one_step":
        return teacher_forced_recovery(
            model, batch["x"][current], batch["v"][current], rest, pin, normalization
        )
    if arm == "five_step":
        return autoregressive_recovery(
            model,
            batch["x"][current[:, 0]],
            batch["v"][current[:, 0]],
            rest,
            pin,
            normalization,
            current.shape[1],
        )
    raise ValueError(arm)


def objective(model, batch, normalization, config, arm):
    x, v = predict_training(model, batch, normalization, arm)
    return joint_endpoint_loss(
        x,
        v,
        batch["x"][batch["target"]],
        batch["v"][batch["target"]],
        batch["pinned"],
        batch["neighborhood"],
        config["training"],
    )


def fit(model, initial, batch, normalization, config, arm, output):
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    settings = config["training"]
    best, selected_step, selected = math.inf, None, None
    curve, clipped, max_grad = [], 0, 0.0
    started = time.perf_counter()
    for step in range(settings["updates"] + 1):
        if step % settings["validation_interval"] == 0 or step == settings["updates"]:
            model.eval()
            with torch.no_grad():
                one, one_parts = objective(model, batch, normalization, config, "one_step")
                five, five_parts = objective(model, batch, normalization, config, "five_step")
            row = {
                "step": step,
                "one_step_joint": float(one),
                "five_step_joint": float(five),
                "one_step_position": float(one_parts[0]),
                "one_step_velocity": float(one_parts[1]),
                "five_step_position": float(five_parts[0]),
                "five_step_velocity": float(five_parts[1]),
            }
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError("Non-finite common fitting objective")
            curve.append(row)
            if float(five) < best:
                best, selected_step = float(five), step
                selected = copy.deepcopy(model.state_dict())
            if step % 250 == 0:
                print(
                    f"{arm} update {step}: teacher={float(one):.7g}, common5={float(five):.7g}",
                    flush=True,
                )
        if step == settings["updates"]:
            break
        model.train()
        fraction = step / max(1, settings["updates"] - 1)
        multiplier = settings["minimum_learning_rate_fraction"] + (
            1 - settings["minimum_learning_rate_fraction"]
        ) * 0.5 * (1 + math.cos(math.pi * fraction))
        optimizer.param_groups[0]["lr"] = settings["learning_rate"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, batch, normalization, config, arm)
        (loss * settings["loss_multiplier"]).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
        )
        clipped += int(norm > settings["gradient_clip"])
        max_grad = max(max_grad, float(norm))
        optimizer.step()
    arm_dir = output / arm
    arm_dir.mkdir()
    torch.save(
        {"model_state": copy.deepcopy(model.state_dict()), "step": settings["updates"]},
        arm_dir / "last.pt",
    )
    model.load_state_dict(selected, strict=True)
    model.eval()
    assert all(
        torch.equal(value, initial[key])
        for key, value in model.state_dict().items()
        if key.startswith("contact_network.")
    )
    torch.save(
        {
            "model_state": selected,
            "config": config,
            "architecture": "unchanged_B",
            "selected_step": selected_step,
            "common_five_step_loss": best,
            "selection_uses_training_window_only": True,
        },
        arm_dir / "checkpoint.pt",
    )
    write_csv(arm_dir / "training_curve.csv", curve)
    summary = {
        "arm": arm,
        "updates": settings["updates"],
        "selected_step": selected_step,
        "selected_common_five_step_loss": best,
        "clipped_updates": clipped,
        "max_gradient_norm": max_grad,
        "finite_gradients": True,
        "contact_parameters_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(arm_dir / "training_summary.json", summary)
    return summary, curve


def rmse_rows(pred_x, pred_v, true_x, true_v, batch, **metadata):
    rows = []
    for region, mask in (
        ("active", ~batch["pinned"]),
        ("contact_neighborhood", batch["neighborhood"]),
    ):
        error_x, error_v = pred_x - true_x, pred_v - true_v
        rows.append(
            {
                **metadata,
                "region": region,
                "node_frames": int(mask.sum()) * int(error_x[..., 0, 0].numel()),
                "position_rmse_um": float(
                    error_x[..., mask, :].square().sum(-1).mean().sqrt() * 1e6
                ),
                "velocity_rmse_mm_s": float(
                    error_v[..., mask, :].square().sum(-1).mean().sqrt() * 1e3
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate(models, legacy, mesh, arrays, batch, normalization, config, output):
    metrics, probe_rows, probe_metrics, saved = [], [], [], {"node_ids": mesh["node_ids"]}
    x, v = batch["x"], batch["v"]
    first, last = config["window"]["first_frame"], config["window"]["last_frame"]
    top = np.flatnonzero(
        np.isclose(mesh["rest_positions_world"][:, 2], mesh["rest_positions_world"][:, 2].max())
    )
    node = int(
        top[
            np.argmin(
                np.linalg.norm(
                    mesh["rest_positions_world"][top, :2]
                    - arrays["indenter_position_world"][first, :2],
                    axis=1,
                )
            )
        ]
    )
    max_pin, native_counts, reload_errors = 0.0, {}, {}
    for method in ("constant_velocity", *models):
        if method == "constant_velocity":
            one_x, one_v = x[:-1] + v[:-1] * config["dt"], v[:-1]
            starts = batch["current"][:, 0]
            h = torch.arange(1, 6, dtype=x.dtype)[None, :, None, None]
            five_x = x[starts, None] + h * config["dt"] * v[starts, None]
            five_v = v[starts, None].expand_as(five_x)
        else:
            model = models[method]
            one_x, one_v = autoregressive_recovery(
                model, x[:-1], v[:-1], batch["rest"], batch["pinned"], normalization, 1
            )
            one_x, one_v = one_x[:, 0], one_v[:, 0]
            five_x, five_v = predict_training(model, batch, normalization, "five_step")
        metrics.extend(
            rmse_rows(
                one_x,
                one_v,
                x[1:],
                v[1:],
                batch,
                method=method,
                evaluation="one_step_unique_20",
                horizon=1,
            )
        )
        for k in range(5):
            metrics.extend(
                rmse_rows(
                    five_x[:, k],
                    five_v[:, k],
                    x[batch["target"][:, k]],
                    v[batch["target"][:, k]],
                    batch,
                    method=method,
                    evaluation="overlapping_five_step_fit",
                    horizon=k + 1,
                )
            )
        state = legacy.gaussian_state(x[0].numpy(), v[0].numpy())
        pipeline = (
            None if method == "constant_velocity" else JointStatePipeline(legacy, models[method])
        )
        long_x, long_v, contact_counts = [], [], []
        for frame in range(first, last):
            external = IndenterInput.from_arrays(arrays, frame)
            if pipeline is None:
                result = legacy.step(state, external, "constant_velocity")
            else:
                result = pipeline.step(state, external)
            state = result.state
            long_x.append(state.centers)
            long_v.append(state.velocities)
            contact_counts.append(result.contacts.num_contacts)
            max_pin = max(
                max_pin,
                float(
                    (state.centers[batch["pinned"]] - batch["rest"][batch["pinned"]]).abs().max()
                ),
                float(state.velocities[batch["pinned"]].abs().max()),
            )
        lx, lv = torch.stack(long_x), torch.stack(long_v)
        native_counts[method] = contact_counts
        saved[f"{method}__position"] = torch.cat((x[:1], lx)).numpy()
        saved[f"{method}__velocity"] = torch.cat((v[:1], lv)).numpy()
        if pipeline is not None and not any(contact_counts):
            functional_x, functional_v = autoregressive_recovery(
                models[method],
                x[0],
                v[0],
                batch["rest"],
                batch["pinned"],
                normalization,
                last - first,
            )
            error = max(
                float((lx - functional_x).abs().max()), float((lv - functional_v).abs().max())
            )
            reload_errors[method] = error
            if error != 0:
                raise AssertionError(
                    "Differentiable no-contact and original native pipeline disagree"
                )
        for horizon in config["evaluation"]["horizons"]:
            metrics.extend(
                rmse_rows(
                    lx[horizon - 1 : horizon],
                    lv[horizon - 1 : horizon],
                    x[horizon : horizon + 1],
                    v[horizon : horizon + 1],
                    batch,
                    method=method,
                    evaluation="free_rollout_in_training_segment",
                    horizon=horizon,
                )
            )
        true_z = (x[:, node, 2] - batch["rest"][node, 2]).numpy() * 1e6
        pred_z = (saved[f"{method}__position"][:, node, 2] - batch["rest"][node, 2].item()) * 1e6
        true_v = v[:, node, 2].numpy() * 1e3
        pred_v = saved[f"{method}__velocity"][:, node, 2] * 1e3
        for offset in range(len(x)):
            probe_rows.append(
                {
                    "method": method,
                    "node_id": node,
                    "frame": first + offset,
                    "true_z_um": float(true_z[offset]),
                    "predicted_z_um": float(pred_z[offset]),
                    "true_vz_mm_s": float(true_v[offset]),
                    "predicted_vz_mm_s": float(pred_v[offset]),
                }
            )
        threshold = config["probe"]["velocity_sign_threshold_mm_s"]

        def changes(value, threshold=threshold):
            return int(np.count_nonzero(np.diff(np.sign(value[np.abs(value) > threshold]))))

        tv, pv = true_v[1:], pred_v[1:]
        correlation = None if np.ptp(pv) == 0 else float(np.corrcoef(tv, pv)[0, 1])
        probe_metrics.append(
            {
                "method": method,
                "node_id": node,
                "signed_z_rmse_um": float(np.sqrt(np.square(pred_z[1:] - true_z[1:]).mean())),
                "signed_vz_rmse_mm_s": float(np.sqrt(np.square(pv - tv).mean())),
                "vz_std_ratio": float(pv.std() / tv.std()),
                "vz_correlation": correlation,
                "true_sign_changes_including_initial": changes(true_v),
                "predicted_sign_changes_including_initial": changes(pred_v),
            }
        )
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "probe_series.csv", probe_rows)
    write_csv(output / "probe_metrics.csv", probe_metrics)
    saved.update(true_position=x.numpy(), true_velocity=v.numpy())
    np.savez_compressed(output / "predictions.npz", **saved)
    loaded = read_npz(output / "predictions.npz")
    assert all(np.array_equal(value, loaded[key]) for key, value in saved.items())
    return (
        metrics,
        probe_rows,
        probe_metrics,
        {
            "max_fixed_position_or_velocity_error": max_pin,
            "native_contact_count_per_predicted_frame": native_counts,
            "functional_vs_native_pipeline_max_error": reload_errors,
            "saved_prediction_npz_bitwise_equal": True,
            "selected_probe_node": node,
        },
    )


def plots(histories, metrics, probe, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key in zip(axes, ("one_step_joint", "five_step_joint"), strict=True):
        for group, history in histories.items():
            ax.plot([r["step"] for r in history], [r[key] for r in history], label=group)
        ax.set(title=key, xlabel="Updates", ylabel="Normalized fitting MSE", yscale="log")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    methods = list(dict.fromkeys(r["method"] for r in probe))
    for i, method in enumerate(methods):
        rows = [r for r in probe if r["method"] == method]
        t = [(r["frame"] - rows[0]["frame"]) * 0.01 for r in rows]
        if i == 0:
            axes[0].plot(t, [r["true_z_um"] for r in rows], "k--", label="MuJoCo")
            axes[1].plot(t, [r["true_vz_mm_s"] for r in rows], "k--", label="MuJoCo")
            axes[2].plot(
                [r["true_z_um"] for r in rows],
                [r["true_vz_mm_s"] for r in rows],
                "k--",
                label="MuJoCo",
            )
        axes[0].plot(t, [r["predicted_z_um"] for r in rows], label=method)
        axes[1].plot(t, [r["predicted_vz_mm_s"] for r in rows], label=method)
        axes[2].plot(
            [r["predicted_z_um"] for r in rows],
            [r["predicted_vz_mm_s"] for r in rows],
            label=method,
        )
    for ax, title in zip(
        axes,
        ("Signed z displacement (um)", "Instantaneous vz (mm/s)", "Signed vz vs z"),
        strict=True,
    ):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(output / "signed_probe_recovery.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key in zip(axes, ("position_rmse_um", "velocity_rmse_mm_s"), strict=True):
        for method in methods:
            rows = [
                r
                for r in metrics
                if r["method"] == method
                and r["region"] == "active"
                and r["evaluation"] == "free_rollout_in_training_segment"
            ]
            ax.plot([r["horizon"] for r in rows], [r[key] for r in rows], marker="o", label=method)
        ax.set(xlabel="Free prediction horizon", ylabel=key)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.savefig(output / "free_rollout_errors.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/diagnose_recovery_multistep.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
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
        raise FileExistsError(f"Preserve existing results: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    sources = (
        "scripts/diagnose_recovery_multistep.py",
        "src/contact_gwm/recovery_multistep.py",
        "src/contact_gwm/neighbor_dynamics.py",
        "src/contact_gwm/joint_state_dynamics.py",
    )
    for path in sources:
        shutil.copy2(ROOT / path, output / Path(path).name)
    mesh, arrays, batch, checks = prepare(data, config)
    legacy_config = yaml.safe_load((ROOT / config["legacy_pipeline_config"]).read_text())
    legacy = PinnedContactPipeline(
        mesh, data / "scene.xml", ROOT / legacy_config["checkpoint"], legacy_config
    )
    normalization = legacy.training_config["normalization"]
    torch.manual_seed(config["seed"])
    template = make_neighbor_model("B", mesh, config)
    initial = copy.deepcopy(template.state_dict())
    write_json(
        output / "sample_list.json",
        {
            "trajectory": "center_deep",
            "source_frames": list(range(210, 231)),
            "initial_frames": (batch["current"][:, 0] + 210).tolist(),
            "current_occurrences": (batch["current"] + 210).tolist(),
            "target_occurrences": (batch["target"] + 210).tolist(),
            "unique_pair_count": 20,
            "supervised_endpoint_occurrences_per_update": 80,
            "same_target_weighting_both_arms": True,
            "selection": config["window"]["selection"],
        },
    )
    np.savez_compressed(
        output / "fixed_window.npz",
        node_ids=mesh["node_ids"],
        **{k: value.numpy() for k, value in batch.items()},
    )
    write_json(
        output / "initialization.json",
        {"state_sha256": tensor_digest(initial), "model": model_specification(template)},
    )
    models, summaries, histories = {}, [], {}
    for arm in GROUPS:
        model = make_neighbor_model("B", mesh, config)
        summary, curve = fit(model, initial, batch, normalization, config, arm, output)
        summaries.append(summary)
        histories[arm] = curve
        saved = torch.load(output / arm / "checkpoint.pt", map_location="cpu", weights_only=False)
        reloaded = make_neighbor_model("B", mesh, config)
        reloaded.load_state_dict(saved["model_state"], strict=True)
        reloaded.eval()
        a = autoregressive_recovery(
            model, batch["x"][0], batch["v"][0], batch["rest"], batch["pinned"], normalization, 20
        )
        b = autoregressive_recovery(
            reloaded,
            batch["x"][0],
            batch["v"][0],
            batch["rest"],
            batch["pinned"],
            normalization,
            20,
        )
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        models[arm] = reloaded
    historical = torch.load(
        ROOT / "outputs/neighbor_recovery_ablation/run_001/B/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    old_model = make_neighbor_model("B", mesh, config)
    old_model.load_state_dict(historical["model_state"], strict=True)
    old_model.eval()
    models = {"historical_B": old_model, **models}
    metrics, probe, probe_metrics, evaluation_checks = evaluate(
        models, legacy, mesh, arrays, batch, normalization, config, output
    )
    plots(histories, metrics, probe, output)
    checks.update(evaluation_checks)
    checks.update(
        saved_reload_20_steps_bitwise_equal=True,
        identical_initial_parameters_both_arms=True,
        only_mesh_and_center_deep_arrays_loaded=True,
        historical_file_count_unchanged=len(before),
    )
    assert before == {path: file_sha256(ROOT / path) for path in before}
    assert checks["max_fixed_position_or_velocity_error"] == 0
    write_json(output / "checks.json", checks)
    write_json(output / "historical_hashes.json", before)
    environment = {
        "client_experiment_date": config["experiment_date"],
        "host_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "loaded_trajectory_arrays": ["center_deep"],
        "seed": config["seed"],
        "cpu_threads": config["cpu_threads"],
        "fit_only_not_generalization": True,
    }
    write_json(output / "environment.json", environment)
    write_json(
        output / "summary.json",
        {
            "environment": environment,
            "training": summaries,
            "checks": checks,
            "probe_metrics": probe_metrics,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "environment": environment,
                "training": summaries,
                "checks": checks,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
