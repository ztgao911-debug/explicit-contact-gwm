"""Dedicated attached multi-step training; frozen stop-gradient native contacts."""

from __future__ import annotations

import copy
import math
import time

import torch
from validate_contact_pipeline import write_csv, write_json

from contact_gwm.differentiable_rollout import (
    DifferentiableContactRollout,
    signed_volume_ratios,
)
from contact_gwm.paired_contact_response import absolute_loss, normalized_state_loss, paired_loss
from contact_gwm.search_protocol import fixed_five_step_scores


class SearchDeadline(TimeoutError):
    pass


class AlgorithmFailure(RuntimeError):
    pass


def check_deadline(deadline):
    if time.time() >= deadline:
        raise SearchDeadline("Six-hour total wall-clock deadline reached")


def make_schedules(data, seed, updates, batch_size):
    generator = torch.Generator().manual_seed(seed)
    return {
        "ordinary": torch.randint(
            len(data["ordinary_train"]["context"]), (updates, batch_size), generator=generator
        ),
        "branch": torch.randint(
            len(data["branch_positions"]), (updates, batch_size), generator=generator
        ),
        "window_uniform": torch.rand(updates, batch_size, generator=generator),
        "pair": torch.randint(
            len(data["pairs"]["base_index"]), (updates, batch_size), generator=generator
        ),
    }


def branch_window_loss(model, path, data, branch_ids, uniform, candidate, settings, mesh, deadline):
    horizon = candidate["horizon"]
    starts = (uniform * (21 - horizon)).long()
    times = starts[:, None] + torch.arange(horizon + 1)
    true_x = data["branch_positions"][branch_ids[:, None], times]
    true_v = data["branch_velocities"][branch_ids[:, None], times]
    roi = data["branch_roi"][branch_ids]
    pinned = data["branch_cache"]["pinned"]
    if candidate["mode"] == "teacher_forcing":
        indices = (branch_ids[:, None] * 20 + starts[:, None] + torch.arange(horizon)).flatten()
        # Every factual current-state cache is from the SAME fixed window. A
        # resets at every endpoint; B/C instead use the attached rollout below.
        cache = data["branch_cache"]
        prediction = model(
            *(cache[k][indices] for k in ("context", "gates", "features", "velocity"))
        )
        px = true_x[:, :-1].reshape(-1, len(pinned), 3) + prediction.delta_position.double()
        pv = true_v[:, :-1].reshape(-1, len(pinned), 3) + prediction.delta_velocity.double()
        predicted = list(
            zip(
                px.reshape(len(branch_ids), horizon, len(pinned), 3).unbind(1),
                pv.reshape(len(branch_ids), horizon, len(pinned), 3).unbind(1),
                strict=True,
            )
        )
    else:
        x, v = true_x[:, 0], true_v[:, 0]
        predicted = []
        for t in range(horizon):
            check_deadline(deadline)
            robots = [
                data["branch_robots"][int(b)][int(s) + t]
                for b, s in zip(branch_ids, starts, strict=True)
            ]
            x, v = path.step(x, v, robots)
            predicted.append((x, v))
    loss = true_x.new_zeros(())
    grid = true_x.new_zeros(())
    min_ratio, inverted_observations = math.inf, 0
    for t, (x, v) in enumerate(predicted):
        # Absolute endpoint supervision; instantaneous v is independent of dx/dt.
        loss = loss + normalized_state_loss(
            x, v, true_x[:, t + 1], true_v[:, t + 1], roi, pinned, settings
        )
        ratio = signed_volume_ratios(x, mesh["tetra_tensor"], mesh["rest_volume_tensor"])
        min_ratio = min(min_ratio, float(ratio.detach().min()))
        inverted_observations += int(torch.any(ratio.detach() <= 0, dim=-1).sum())
        if candidate["mesh_weight"]:
            grid = grid + torch.relu(0.1 - ratio).square().mean()
    return (
        loss / horizon,
        grid / horizon,
        {
            "training_window_minimum_volume_ratio": min_ratio,
            "training_inverted_state_observations": inverted_observations,
        },
    )


@torch.no_grad()
def fixed_validation(path, windows, mesh, deadline):
    """Same 30 windows, five AR endpoints, fixed weights and eight cells.

    Pool node SSE across the five endpoints within each initial phase/region.
    Candidate training horizon, velocity weight and grid weight never enter.
    """
    pooled = {}
    failures = []
    for window in windows:
        check_deadline(deadline)
        try:
            states = path.rollout(
                window["initial_x"][None],
                window["initial_v"][None],
                [[r] for r in window["external"]],
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            failures.append({"sourcecase": window["sourcecase"], "error": str(error)})
            continue
        for t, (x, v) in enumerate(states[1:]):
            for region, mask in (
                ("active", ~mesh["pinned_mask"]),
                ("contact_neighborhood", window["roi"]),
            ):
                mask = torch.as_tensor(mask, dtype=torch.bool)
                key = window["phase"], region
                values = pooled.setdefault(key, [0.0, 0.0, 0])
                values[0] += float((x[0, mask] - window["targets_x"][t, mask]).square().sum())
                values[1] += float((v[0, mask] - window["targets_v"][t, mask]).square().sum())
                values[2] += int(mask.sum())
    rows = [
        {
            "phase": phase,
            "region": region,
            "horizon": 5,
            "node_count": count,
            "position_rmse_m": math.sqrt(psse / count),
            "velocity_rmse_m_s": math.sqrt(vsse / count),
        }
        for (phase, region), (psse, vsse, count) in sorted(pooled.items())
        if count
    ]
    scores = fixed_five_step_scores(rows)
    # A partial window set never rescues a checkpoint, even if every phase has data.
    if failures:
        scores.update(
            joint_score=None, position_score=None, velocity_score=None, coverage_complete=False
        )
    return scores, rows, failures


def train_candidate(
    model,
    legacy,
    conditions,
    encoder,
    data,
    windows,
    mesh,
    candidate,
    settings,
    schedules,
    output,
    deadline,
):
    path = DifferentiableContactRollout(legacy, model, conditions, encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"])
    training_settings = {**settings, "velocity_weight": candidate["velocity_weight"]}
    best_score, selected_step, best_state = math.inf, None, None
    history, validation_history, update_history = [], [], []
    clipped = 0
    start_time = time.perf_counter()
    last_losses = {}
    initial_queries = path.query_count
    try:
        for step in range(settings["updates"] + 1):
            check_deadline(deadline)
            if step % settings["validation_interval"] == 0 or step == settings["updates"]:
                model.eval()
                scores, rows, failures = fixed_validation(path, windows, mesh, deadline)
                validation_history.append(
                    {"update": step, "scores": scores, "rows": rows, "failures": failures}
                )
                score = scores["joint_score"]
                if score is not None and score < best_score:
                    best_score, selected_step, best_state = (
                        score,
                        step,
                        copy.deepcopy(model.state_dict()),
                    )
                    torch.save(
                        {
                            "model_state": best_state,
                            "selected_step": selected_step,
                            "selection_score": best_score,
                        },
                        output / "selected_weights.pt",
                    )
                row = {
                    "update": step,
                    "validation_joint_score": score,
                    "validation_position_score": scores["position_score"],
                    "validation_velocity_score": scores["velocity_score"],
                    "validation_failed_windows": len(failures),
                    "selected_step_so_far": selected_step,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "native_queries": path.query_count - initial_queries,
                    **last_losses,
                }
                history.append(row)
                # JSON records have stable fields; detailed scalar CSV is written at end.
                write_json(output / "training_history.json", history)
                write_json(output / "checkpoint_selection.json", validation_history)
                if update_history:
                    write_csv(output / "training_updates.csv", update_history)
                print(
                    f"{candidate['id']} seed {settings['seed']} update {step}: "
                    f"fixed val={score}, selected={selected_step}",
                    flush=True,
                )
            if step == settings["updates"]:
                break
            model.train()
            fraction = step / max(1, settings["updates"] - 1)
            rate = settings["minimum_learning_rate_fraction"] + (
                1 - settings["minimum_learning_rate_fraction"]
            ) * 0.5 * (1 + math.cos(math.pi * fraction))
            optimizer.param_groups[0]["lr"] = candidate["learning_rate"] * rate
            optimizer.zero_grad(set_to_none=True)
            window_loss, grid_loss, grid_info = branch_window_loss(
                model,
                path,
                data,
                schedules["branch"][step],
                schedules["window_uniform"][step],
                candidate,
                training_settings,
                mesh,
                deadline,
            )
            ordinary = absolute_loss(
                model, data["ordinary_train"], schedules["ordinary"][step], training_settings
            )
            pair = paired_loss(
                model,
                data["branch_cache"],
                data["pairs"],
                schedules["pair"][step],
                training_settings,
            )
            loss = ordinary + window_loss + pair + candidate["mesh_weight"] * grid_loss
            if not torch.isfinite(loss):
                raise AlgorithmFailure("Non-finite training loss")
            (settings["loss_multiplier"] * loss).backward()
            try:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
                )
            except RuntimeError as error:
                raise AlgorithmFailure(f"Non-finite training gradient: {error}") from error
            clipped += int(norm > settings["gradient_clip"])
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise AlgorithmFailure("Non-finite updated weights")
            last_losses = {
                "last_ordinary_loss": float(ordinary.detach()),
                "last_window_loss": float(window_loss.detach()),
                "last_initial_pair_loss": float(pair.detach()),
                "last_mesh_penalty": float(grid_loss.detach()),
                "last_gradient_norm": float(norm),
                **grid_info,
            }
            update_history.append(
                {
                    "update": step + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "native_queries": path.query_count - initial_queries,
                    **last_losses,
                }
            )
    except BaseException:
        torch.save(
            {"model_state": model.state_dict(), "update": step}, output / "interrupted_weights.pt"
        )
        write_json(output / "training_history.json", history)
        write_json(output / "checkpoint_selection.json", validation_history)
        if update_history:
            write_csv(output / "training_updates.csv", update_history)
        raise
    if best_state is None:
        raise AlgorithmFailure("No complete finite common five-step validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval().requires_grad_(False)
    # Fill optional diagnostics at update zero instead of changing the CSV schema.
    fields = list(dict.fromkeys(key for row in history for key in row))
    write_csv(
        output / "training_curve.csv", [{key: row.get(key) for key in fields} for row in history]
    )
    write_csv(output / "training_updates.csv", update_history)
    result = {
        "candidate": candidate,
        "seed": settings["seed"],
        "updates_completed": settings["updates"],
        "selected_step": selected_step,
        "selection_score": best_score,
        "training_elapsed_seconds": time.perf_counter() - start_time,
        "native_queries_training_and_selection": path.query_count - initial_queries,
        "clipped_updates": clipped,
        "finite_gradients": True,
        "conditional_gradient": (
            "native current contacts stop-gradient; x/v/features/kernel attached"
        ),
    }
    write_json(output / "training_summary.json", result)
    return result


def gradient_preflight(model, legacy, conditions, encoder, data, mesh, deadline):
    path = DifferentiableContactRollout(legacy, model, conditions, encoder)
    # A fixed central ordinary branch/current-state probe; not a search or optimizer update.
    i, begin = 6, 0
    x = data["branch_positions"][i, begin][None].clone().requires_grad_(True)
    v = data["branch_velocities"][i, begin][None].clone().requires_grad_(True)
    external = [[data["branch_robots"][i][begin + t]] for t in range(5)]
    states = [(x, v)]
    context_gradient_flags = []
    for robots in external:
        x, v = path.step(x, v, robots)
        states.append((x, v))
        context_gradient_flags.append(path.last_context.grad_fn is not None)
    first_x, first_v = states[1]
    loss = (states[-1][0] - data["branch_positions"][i, 5][None]).square().mean() / 0.001**2
    loss = loss + (states[-1][1] - data["branch_velocities"][i, 5][None]).square().mean() / 0.1**2
    gx, gv = torch.autograd.grad(loss, (first_x, first_v), retain_graph=True)
    checks = {
        "terminal_loss": float(loss.detach()),
        "terminal_to_first_x_gradient_norm": float(gx.norm()),
        "terminal_to_first_v_gradient_norm": float(gv.norm()),
        "state_features_have_grad_fn": path.last_features.grad_fn is not None,
        "context_has_grad_fn": any(context_gradient_flags),
        "context_gradient_flags_by_step": context_gradient_flags,
        "native_solver_gradient": False,
        "no_optimizer_updates": True,
    }
    assert torch.isfinite(gx).all() and torch.isfinite(gv).all() and gx.norm() > 0 and gv.norm() > 0
    assert checks["state_features_have_grad_fn"]
    assert checks["context_has_grad_fn"]
    check_deadline(deadline)
    return checks
