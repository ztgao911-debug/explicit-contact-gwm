"""Fixed-window five-step fits with/without native local-response supervision."""

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

import mujoco
import numpy as np
import torch
import yaml
from ablate_neighbor_recovery import tensor_digest
from diagnose_recovery_multistep import load_config as load_multistep_config
from diagnose_recovery_multistep import objective, plots, predict_training, prepare, rmse_rows
from diagnose_recovery_stability import metrics as stability_metrics
from diagnose_recovery_stability import plots as stability_plots
from diagnose_recovery_stability import representative_curve
from validate_contact_pipeline import hashes, write_csv, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.data.mujoco_soft import tetra_volumes
from contact_gwm.neighbor_dynamics import make_neighbor_model, model_specification
from contact_gwm.recovery_multistep import autoregressive_recovery
from contact_gwm.recovery_response import (
    central_response,
    paired_current_step,
    response_matching_loss,
)
from contact_gwm.recovery_stability import (
    NativeRecoveryReplay,
    make_perturbations,
    normalized_state_norm,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("five_step_control", "five_step_response")
METHODS = ("mujoco", "constant_velocity", *GROUPS)


def load_config(path):
    extra = yaml.safe_load(path.read_text())
    config = load_multistep_config(path.parent / extra.pop("base_config"))
    config.update(extra)
    if (
        tuple(config["groups"]) != GROUPS
        or config["response"]["coefficient"] != 1e-4
        or config["response"]["position_rms_m"] != [1e-6]
        or config["response"]["velocity_rms_m_s"] != [1e-4]
        or config["response"]["direction_count"] != 3
        or config["response"]["seed"] != 7
    ):
        raise ValueError("Keep the predeclared response experiment; no weight/amplitude search")
    return config


def prepare_responses(data, mesh, arrays, batch, config, output):
    """Native one-step endpoint LABELS; each rollout restores only its current state."""
    settings = config["response"]
    cases, dx, dv, directions = make_perturbations(mesh["pinned_mask"], settings, settings["seed"])
    assert len(cases) == 13
    first = config["window"]["first_frame"]
    frame_count = len(batch["x"]) - 1
    command = arrays["action_target_position"][first].copy()
    if not np.array_equal(
        arrays["action_target_position"][first : first + frame_count + 1],
        np.broadcast_to(command, (frame_count + 1, 3)),
    ):
        raise ValueError("This experiment requires the original constant known robot command")
    replay = NativeRecoveryReplay(data / "scene.xml", mesh)
    input_x, input_v, target_x, target_v, replay_checks = [], [], [], [], []
    max_base_x, max_base_v, max_initial = 0.0, 0.0, 0.0
    for offset in range(frame_count):
        frame = first + offset
        native_x, native_v = [], []
        for i, case in enumerate(cases):
            x, v, checks = replay.rollout(
                arrays["integration_state"][frame].copy(), command, dx[i], dv[i], 1, config["dt"]
            )
            checks.pop("robot_positions")
            checks.pop("robot_velocities")
            if checks["max_contact_count_including_internal_steps"]:
                raise AssertionError("Native local supervision left the no-contact regime")
            replay_checks.append({"current_frame": frame, "case": case["case"], **checks})
            max_initial = max(
                max_initial,
                float(np.abs(x[0] - batch["x"][offset].numpy() - dx[i]).max()),
                float(np.abs(v[0] - batch["v"][offset].numpy() - dv[i]).max()),
            )
            if i == 0:
                max_base_x = max(
                    max_base_x, float(np.abs(x[1] - batch["x"][offset + 1].numpy()).max())
                )
                max_base_v = max(
                    max_base_v, float(np.abs(v[1] - batch["v"][offset + 1].numpy()).max())
                )
            else:
                native_x.append(x[1])
                native_v.append(v[1])
        shape = (6, 2, len(mesh["node_ids"]), 3)
        input_x.append((batch["x"][offset].numpy() + dx[1:]).reshape(shape))
        input_v.append((batch["v"][offset].numpy() + dv[1:]).reshape(shape))
        target_x.append(np.asarray(native_x).reshape(shape))
        target_v.append(np.asarray(native_v).reshape(shape))
    if max_base_x > 1e-10 or max_base_v > 1e-8 or max_initial > 1e-12:
        raise AssertionError("Local native replay or perturbation injection mismatch")
    initial_norm = normalized_state_norm(
        dx[2::2],
        dv[2::2],
        ~mesh["pinned_mask"],
        settings["position_unit_m"],
        settings["velocity_unit_m_s"],
    )
    labels = {
        "input_x": torch.tensor(np.asarray(input_x)),
        "input_v": torch.tensor(np.asarray(input_v)),
        "target_x": torch.tensor(np.asarray(target_x)),
        "target_v": torch.tensor(np.asarray(target_v)),
        "initial_norm": torch.tensor(initial_norm),
    }
    labels["native_rx"] = central_response(labels["target_x"])
    labels["native_rv"] = central_response(labels["target_v"])
    np.savez_compressed(
        output / "response_labels.npz",
        node_ids=mesh["node_ids"],
        directions=directions,
        current_frames=np.arange(first, first + frame_count),
        **{k: v.numpy() for k, v in labels.items()},
    )
    write_json(output / "response_cases.json", cases)
    write_json(output / "native_response_checks.json", replay_checks)
    checks = {
        "native_current_frames": frame_count,
        "perturbed_native_one_step_labels": frame_count * (len(cases) - 1),
        "extra_native_unperturbed_replays": frame_count,
        "native_internal_steps": sum(row["native_internal_steps"] for row in replay_checks),
        "unperturbed_replay_max_position_error_m": max_base_x,
        "unperturbed_replay_max_velocity_error_m_s": max_base_v,
        "matched_initial_perturbations_max_error": max_initial,
        "source": "independent_native_integrator_no_future_state_refill",
    }
    print(f"Native response labels completed: {checks}", flush=True)
    return labels, checks


def response_objective(model, labels, batch, normalization, config, frame=None):
    selected = slice(None) if frame is None else frame
    px, pv = paired_current_step(
        model,
        labels["input_x"][selected],
        labels["input_v"][selected],
        batch["rest"],
        batch["pinned"],
        normalization,
    )
    return response_matching_loss(
        px,
        pv,
        labels["native_rx"][selected],
        labels["native_rv"][selected],
        labels["initial_norm"],
        batch["pinned"],
        batch["neighborhood"],
        config["training"],
    )


def fit(model, initial, batch, labels, normalization, config, arm, output):
    """Only the new arm backpropagates the added finite-response objective."""
    model.load_state_dict(initial, strict=True)
    settings = config["training"]
    coefficient = config["response"]["coefficient"] if arm == GROUPS[1] else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    best, selected_step, selected = math.inf, None, None
    curve, clipped, max_grad = [], 0, 0.0
    started = time.perf_counter()
    for step in range(settings["updates"] + 1):
        if step % settings["validation_interval"] == 0 or step == settings["updates"]:
            model.eval()
            with torch.no_grad():
                one, _ = objective(model, batch, normalization, config, "one_step")
                five, parts = objective(model, batch, normalization, config, "five_step")
                response, rparts = response_objective(model, labels, batch, normalization, config)
            row = {
                "step": step,
                "one_step_joint": float(one),
                "five_step_joint": float(five),
                "five_step_position": float(parts[0]),
                "five_step_velocity": float(parts[1]),
                "response_joint": float(response),
                "response_position": float(rparts[0]),
                "response_velocity": float(rparts[1]),
                "weighted_response_joint": coefficient * float(response),
                "full_objective": float(five) + coefficient * float(response),
            }
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError("Non-finite common state or response objective")
            curve.append(row)
            # Identical PREDECLARED selection for both arms: state fitting only.
            if float(five) < best:
                best, selected_step, selected = float(five), step, copy.deepcopy(model.state_dict())
            if step % 250 == 0:
                print(
                    f"{arm} update {step}: common5={float(five):.7g}, "
                    f"response={float(response):.7g}",
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
        state_loss, _ = objective(model, batch, normalization, config, "five_step")
        loss = state_loss
        if coefficient:
            response_loss, _ = response_objective(
                model, labels, batch, normalization, config, frame=step % len(labels["input_x"])
            )
            loss = loss + coefficient * response_loss
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
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert all(
        torch.equal(value, initial[key])
        for key, value in model.state_dict().items()
        if key.startswith("contact_network.")
    )
    with torch.no_grad():
        response, _ = response_objective(model, labels, batch, normalization, config)
    torch.save(
        {
            "model_state": selected,
            "config": config,
            "architecture": "unchanged_B",
            "selected_step": selected_step,
            "common_five_step_loss": best,
            "response_loss": float(response),
            "selection_uses_training_state_loss_only": True,
        },
        arm_dir / "checkpoint.pt",
    )
    write_csv(arm_dir / "training_curve.csv", curve)
    summary = {
        "arm": arm,
        "updates": settings["updates"],
        "response_coefficient": coefficient,
        "selected_step": selected_step,
        "selected_common_five_step_loss": best,
        "selected_response_loss": float(response),
        "clipped_updates": clipped,
        "max_gradient_norm": max_grad,
        "finite_gradients": True,
        "contact_parameters_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
        "response_frame_passes": settings["updates"] // len(labels["input_x"])
        if coefficient
        else 0,
    }
    write_json(arm_dir / "training_summary.json", summary)
    return summary, curve


@torch.no_grad()
def evaluate_state_fits(models, batch, normalization, config):
    rows = []
    for method in ("constant_velocity", *models):
        if method == "constant_velocity":
            one_x, one_v = batch["x"][:-1] + config["dt"] * batch["v"][:-1], batch["v"][:-1]
            starts = batch["current"][:, 0]
            h = torch.arange(1, 6, dtype=batch["x"].dtype)[None, :, None, None]
            five_x = batch["x"][starts, None] + h * config["dt"] * batch["v"][starts, None]
            five_v = batch["v"][starts, None].expand_as(five_x)
        else:
            ox, ov = autoregressive_recovery(
                models[method],
                batch["x"][:-1],
                batch["v"][:-1],
                batch["rest"],
                batch["pinned"],
                normalization,
                1,
            )
            one_x, one_v = ox[:, 0], ov[:, 0]
            five_x, five_v = predict_training(models[method], batch, normalization, "five_step")
        rows.extend(
            rmse_rows(
                one_x,
                one_v,
                batch["x"][1:],
                batch["v"][1:],
                batch,
                method=method,
                evaluation="one_step_unique_20",
                horizon=1,
            )
        )
        for k in range(5):
            rows.extend(
                rmse_rows(
                    five_x[:, k],
                    five_v[:, k],
                    batch["x"][batch["target"][:, k]],
                    batch["v"][batch["target"][:, k]],
                    batch,
                    method=method,
                    evaluation="overlapping_five_step_fit",
                    horizon=k + 1,
                )
            )
    return rows


@torch.no_grad()
def evaluate_responses(models, batch, labels, normalization, config, output):
    rows = []
    cases = json.loads((output / "response_cases.json").read_text())[2::2]
    for method, model in models.items():
        px, pv = paired_current_step(
            model,
            labels["input_x"],
            labels["input_v"],
            batch["rest"],
            batch["pinned"],
            normalization,
        )
        rx, rv = central_response(px).numpy(), central_response(pv).numpy()
        tx, tv = labels["native_rx"].numpy(), labels["native_rv"].numpy()
        for region, mask in (
            ("active", ~batch["pinned"].numpy()),
            ("contact_neighborhood", batch["neighborhood"].numpy()),
        ):
            error_x = np.sqrt(np.square(rx[..., mask, :] - tx[..., mask, :]).sum(-1).mean(-1)) * 1e6
            error_v = np.sqrt(np.square(rv[..., mask, :] - tv[..., mask, :]).sum(-1).mean(-1)) * 1e3
            for frame in range(len(rx)):
                for i, case in enumerate(cases):
                    rows.append(
                        {
                            "method": method,
                            "region": region,
                            "current_frame": frame + 210,
                            "kind": case["kind"],
                            "direction": case["direction"],
                            "amplitude": case["amplitude"],
                            "position_response_rmse_um": float(error_x[frame, i]),
                            "velocity_response_rmse_mm_s": float(error_v[frame, i]),
                        }
                    )
    write_csv(output / "local_response_metrics.csv", rows)


@torch.no_grad()
def collect_rollouts(
    models, mesh, batch, normalization, config, output, *, control_method=GROUPS[0]
):
    """Reuse independently simulated references; only initial soft x/v enter prediction."""
    source = ROOT / config["reference_stability_run"]
    cases = json.loads((source / "cases.json").read_text())
    perturbations = read_npz(source / "perturbations.npz")
    reference = read_npz(source / "trajectories.npz")
    window = {key: value.numpy() for key, value in batch.items()}
    saved = {
        key: reference[key].copy()
        for key in (
            "mujoco__position",
            "mujoco__velocity",
            "constant_velocity__position",
            "constant_velocity__velocity",
            "node_ids",
            "robot_position",
            "robot_velocity",
        )
    }
    assert np.array_equal(reference["node_ids"], mesh["node_ids"])
    assert np.array_equal(saved["mujoco__position"][0], window["x"])
    assert np.array_equal(saved["mujoco__velocity"][0], window["v"])
    assert np.array_equal(
        perturbations["directions"], read_npz(output / "response_labels.npz")["directions"]
    )
    max_pin, min_separation, min_volume = 0.0, {}, {}
    material = json.loads((ROOT / config["data_directory"] / "manifest.json").read_text())["config"]
    for method, model in models.items():
        positions, velocities = [], []
        for i in range(len(cases)):
            initial_x = batch["x"][0] + torch.tensor(perturbations["delta_position"][i])
            initial_v = batch["v"][0] + torch.tensor(perturbations["delta_velocity"][i])
            x, v = autoregressive_recovery(
                model,
                initial_x,
                initial_v,
                batch["rest"],
                batch["pinned"],
                normalization,
                20,
            )
            positions.append(torch.cat((initial_x[None], x)).numpy())
            velocities.append(torch.cat((initial_v[None], v)).numpy())
        saved[f"{method}__position"], saved[f"{method}__velocity"] = (
            np.asarray(positions),
            np.asarray(velocities),
        )
    for field in ("position", "velocity"):
        assert np.array_equal(saved[f"{control_method}__{field}"], reference[f"five_step__{field}"])
    for method in ("mujoco", "constant_velocity", *models):
        x, v = saved[f"{method}__position"], saved[f"{method}__velocity"]
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise FloatingPointError("Non-finite free rollout")
        max_pin = max(
            max_pin,
            float(np.abs(x[..., window["pinned"], :] - window["rest"][window["pinned"]]).max()),
            float(np.abs(v[..., window["pinned"], :]).max()),
        )
        min_separation[method] = float(
            (
                saved["robot_position"][..., 2]
                - material["indenter"]["radius"]
                - material["block"]["collision_radius"]
                - x[..., 2].max(-1)
            ).min()
        )
        min_volume[method] = float(
            (tetra_volumes(x, mesh["tetrahedra"]) / mesh["rest_signed_tetra_volumes"]).min()
        )
    if max_pin != 0 or min(min_separation.values()) <= 0 or min(min_volume.values()) <= 0:
        raise AssertionError("Fixed boundary, no-contact or unflipped-mesh check failed")
    np.savez_compressed(output / "trajectories.npz", **saved)
    reloaded = read_npz(output / "trajectories.npz")
    assert all(np.array_equal(value, reloaded[key]) for key, value in saved.items())
    write_json(output / "cases.json", cases)
    shutil.copy2(source / "perturbations.npz", output / "perturbations.npz")
    return (
        cases,
        saved,
        window,
        {
            "max_fixed_position_or_velocity_error": max_pin,
            "minimum_robot_separation_margin_m": min_separation,
            "minimum_tetra_volume_ratio": min_volume,
            "prediction_npz_reload_bitwise_equal": True,
            "physical_reference_reused_sha256": file_sha256(source / "trajectories.npz"),
            "same_directions_as_previous_stability_diagnostic": True,
            "control_all_perturbed_rollouts_bitwise_equal_previous_five_step": True,
        },
    )


def probes_and_rollout_metrics(saved, batch, config, state_metrics, output, *, groups=GROUPS):
    node = 71  # Original static top-center selection, retained before seeing results.
    x, v = batch["x"], batch["v"]
    rows, summaries = [], []
    for method in ("constant_velocity", *groups):
        px, pv = saved[f"{method}__position"][0], saved[f"{method}__velocity"][0]
        for h in config["evaluation"]["horizons"]:
            state_metrics.extend(
                rmse_rows(
                    torch.tensor(px[h : h + 1]),
                    torch.tensor(pv[h : h + 1]),
                    x[h : h + 1],
                    v[h : h + 1],
                    batch,
                    method=method,
                    evaluation="free_rollout_in_training_segment",
                    horizon=h,
                )
            )
        for h in range(21):
            rows.append(
                {
                    "method": method,
                    "node_id": node,
                    "frame": 210 + h,
                    "true_z_um": float((x[h, node, 2] - batch["rest"][node, 2]) * 1e6),
                    "predicted_z_um": float((px[h, node, 2] - batch["rest"][node, 2].item()) * 1e6),
                    "true_vz_mm_s": float(v[h, node, 2] * 1e3),
                    "predicted_vz_mm_s": float(pv[h, node, 2] * 1e3),
                }
            )
        for last in (5, 10, 20):
            true = v[1 : last + 1, node, 2].numpy() * 1e3
            pred = pv[1 : last + 1, node, 2] * 1e3

            def sign_changes(values):
                active = values[np.abs(values) > config["probe"]["velocity_sign_threshold_mm_s"]]
                return int(np.count_nonzero(np.diff(np.sign(active))))

            summaries.append(
                {
                    "method": method,
                    "node_id": node,
                    "horizon": last,
                    "signed_z_rmse_um": float(
                        np.sqrt(
                            np.square(
                                px[1 : last + 1, node, 2] - x[1 : last + 1, node, 2].numpy()
                            ).mean()
                        )
                        * 1e6
                    ),
                    "signed_vz_rmse_mm_s": float(np.sqrt(np.square(pred - true).mean())),
                    "vz_std_ratio": float(pred.std() / true.std()),
                    "vz_correlation": None
                    if np.ptp(pred) == 0
                    else float(np.corrcoef(true, pred)[0, 1]),
                    "true_sign_changes_including_initial": sign_changes(
                        v[: last + 1, node, 2].numpy() * 1e3
                    ),
                    "predicted_sign_changes_including_initial": sign_changes(
                        pv[: last + 1, node, 2] * 1e3
                    ),
                }
            )
    write_csv(output / "metrics.csv", state_metrics)
    write_csv(output / "probe_series.csv", rows)
    write_csv(output / "probe_metrics.csv", summaries)
    return rows, summaries


def response_plot(histories, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for method, rows in histories.items():
        ax.plot([r["step"] for r in rows], [r["response_joint"] for r in rows], label=method)
    ax.set(xlabel="Updates", ylabel="Normalized signed response MSE", yscale="log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output / "response_training_curve.png", dpi=150)
    plt.close(fig)


def waveform_detail(saved, batch, output, *, groups=GROUPS):
    """Keep both time ranges; separate axes reveal amplitude loss without clipping."""
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(groups), 2, figsize=(12, 3.5 * len(groups)), squeeze=False, layout="constrained"
    )
    for row, method in enumerate(groups):
        for column, last in enumerate((10, 20)):
            ax = axes[row, column]
            t = np.arange(last + 1) * 0.01
            true = batch["v"][: last + 1, 71, 2].numpy() * 1e3
            pred = saved[f"{method}__velocity"][0, : last + 1, 71, 2] * 1e3
            ax.plot(t, true, "k--", label="MuJoCo")
            ax.plot(t, pred, label=method)
            ax.set(
                title=f"{method}, first {last} steps",
                xlabel="Time (s)",
                ylabel="Node 71 instantaneous vz (mm/s)",
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.savefig(output / "probe_waveform_detail.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ablate_recovery_response.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    data = ROOT / config["data_directory"]
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
    snapshot = output / "source_snapshot"
    for name in (
        "scripts/ablate_recovery_response.py",
        "src/contact_gwm/recovery_response.py",
        "scripts/diagnose_recovery_multistep.py",
        "scripts/diagnose_recovery_stability.py",
        "src/contact_gwm/recovery_multistep.py",
        "src/contact_gwm/recovery_stability.py",
        "src/contact_gwm/neighbor_dynamics.py",
        "src/contact_gwm/joint_state_dynamics.py",
        "configs/ablate_recovery_response.yaml",
    ):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    mesh, arrays, batch, checks = prepare(data, config)
    reference = ROOT / config["reference_multistep_run"]
    old_window = read_npz(reference / "fixed_window.npz")
    assert all(np.array_equal(value.numpy(), old_window[key]) for key, value in batch.items())
    normalization = yaml.safe_load((ROOT / "configs/train_soft_local.yaml").read_text())[
        "normalization"
    ]
    labels, native_checks = prepare_responses(data, mesh, arrays, batch, config, output)
    checks["native_response_labels"] = native_checks
    # Original proof applies to all five-step iterates; allow also the largest
    # current micro-perturbations in the one-step response branch.
    extra_z = float(np.abs(labels["input_x"].numpy() - batch["x"][:-1, None, None].numpy()).max())
    extra_vz = float(np.abs(labels["input_v"].numpy() - batch["v"][:-1, None, None].numpy()).max())
    if extra_z + config["dt"] * extra_vz >= checks["guaranteed_vertical_separation_m"]:
        raise AssertionError("Cannot certify no contact for response training")
    checks["response_training_extra_bound_m"] = extra_z + config["dt"] * extra_vz
    np.savez_compressed(
        output / "fixed_window.npz",
        node_ids=mesh["node_ids"],
        **{k: v.numpy() for k, v in batch.items()},
    )
    write_json(
        output / "sample_list.json",
        {
            "trajectory": "center_deep",
            "source_frames": list(range(210, 231)),
            "current_occurrences": (batch["current"] + 210).tolist(),
            "target_occurrences": (batch["target"] + 210).tolist(),
            "response_current_frames": list(range(210, 230)),
            "response_frame_schedule": [210 + i % 20 for i in range(config["training"]["updates"])],
            "only_response_arm_has_additional_solver_supervision": True,
        },
    )
    torch.manual_seed(config["seed"])
    template = make_neighbor_model("B", mesh, config)
    initial = copy.deepcopy(template.state_dict())
    assert (
        tensor_digest(initial)
        == json.loads((reference / "initialization.json").read_text())["state_sha256"]
    )
    write_json(
        output / "initialization.json",
        {"state_sha256": tensor_digest(initial), "model": model_specification(template)},
    )
    models, histories, fit_summaries = {}, {}, []
    for arm in GROUPS:
        model = make_neighbor_model("B", mesh, config)
        summary, curve = fit(model, initial, batch, labels, normalization, config, arm, output)
        histories[arm] = curve
        fit_summaries.append(summary)
        reloaded = make_neighbor_model("B", mesh, config)
        artifact = torch.load(
            output / arm / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        reloaded.load_state_dict(artifact["model_state"], strict=True)
        reloaded.eval()
        with torch.no_grad():
            a = autoregressive_recovery(
                model,
                batch["x"][0],
                batch["v"][0],
                batch["rest"],
                batch["pinned"],
                normalization,
                20,
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
        assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))
        models[arm] = reloaded
    old_control = torch.load(
        reference / "five_step/checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert all(
        torch.equal(value, old_control["model_state"][key])
        for key, value in models[GROUPS[0]].state_dict().items()
    )
    checks["control_selected_weights_bitwise_equal_previous_five_step"] = True
    checks["checkpoint_reload_predictions_bitwise_equal"] = True
    state_metrics = evaluate_state_fits(models, batch, normalization, config)
    evaluate_responses(models, batch, labels, normalization, config, output)
    cases, saved, window, rollout_checks = collect_rollouts(
        models, mesh, batch, normalization, config, output
    )
    checks.update(rollout_checks)
    probe_rows, probe_summary = probes_and_rollout_metrics(
        saved, batch, config, state_metrics, output
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
    waveform_detail(saved, batch, output)
    assert before == {name: file_sha256(ROOT / name) for name in before}
    checks["historical_file_count_unchanged_sha256"] = len(before)
    checks["only_center_deep_source_array_loaded"] = True
    write_json(output / "checks.json", checks)
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
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "same_optimizer_update_count_not_same_wall_time_or_supervision_volume": True,
    }
    write_json(output / "environment.json", environment)
    result = {
        "environment": environment,
        "checks": checks,
        "training": fit_summaries,
        "state_metrics": state_metrics,
        "probe_metrics": probe_summary,
        "gain_at_twenty_steps": [r for r in gains if r["horizon"] == 20],
        "training_segment_diagnostic_only": True,
    }
    write_json(output / "summary.json", result)
    print(
        json.dumps(
            {"output": str(output), "environment": environment, "training": fit_summaries}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
