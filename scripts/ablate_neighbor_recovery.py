"""Fixed-seed internal-neighbor feature ablation with a capacity control."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from analyze_joint_recovery import analyze
from joint_state_evaluation import evaluate_models
from train_joint_state import (
    fit_variant,
    load_data,
    make_cache,
    plot_training,
    verify_training_inputs,
)
from validate_contact_pipeline import hashes, write_json

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.joint_state_dynamics import JointStatePipeline
from contact_gwm.neighbor_dynamics import (
    make_neighbor_model,
    model_specification,
    neighbor_relative_features,
)
from contact_gwm.pinned_contact_pipeline import IndenterInput, PinnedContactPipeline

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("A", "B", "C")


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_config(path):
    extra = yaml.safe_load(path.read_text())
    base = yaml.safe_load((path.parent / extra.pop("base_config")).read_text())
    base.pop("variants")
    config = {**base, **extra}
    if (
        tuple(config["groups"]) != GROUPS
        or config["seed"] != 7
        or config["dt"] != 0.01
        or config["training"]["updates"] != 2000
        or config["neighbor"]
        != {
            "topology": "mesh_edges_undirected_unique",
            "aggregation": "uniform_neighbor_mean_minus_self",
            "channels": ["relative_displacement_xyz", "relative_velocity_xyz"],
            "displacement_normalization_m": 0.01,
            "velocity_normalization_m_s": 0.05,
            "pinned_neighbors": "keep_with_zero_displacement_and_velocity",
        }
    ):
        raise ValueError("Keep the fixed neighbor features and original training protocol")
    # The historical configuration supplies the exact old budget/loss/encoder.
    reference = yaml.safe_load((ROOT / config["reference_run"] / "config.yaml").read_text())
    for key in (
        "training",
        "model",
        "splits",
        "rollout",
        "dt",
        "seed",
        "cpu_threads",
        "contact_neighborhood_radius_m",
        "data_directory",
        "legacy_pipeline_config",
    ):
        if config[key] != reference[key]:
            raise ValueError(f"Keep the previous autonomous protocol: {key}")
    return config


def check_historical_a(model, output, reference):
    old = torch.load(
        reference / "autonomous_joint/checkpoint.pt", map_location="cpu", weights_only=False
    )
    new = torch.load(output / "A/checkpoint.pt", map_location="cpu", weights_only=False)
    identical = all(
        torch.equal(value, old["model_state"][key]) for key, value in model.state_dict().items()
    )
    if not identical or new["selected_step"] != old["selected_step"]:
        raise AssertionError("A failed to reproduce the existing autonomous checkpoint")
    return {
        "A_selected_checkpoint_bitwise_equal_historical": identical,
        "A_historical_selected_step": old["selected_step"],
    }


def verify_rollout_reload(legacy, trained, restored, trajectories):
    arrays, start = trajectories["left_deep"], 170
    initial = legacy.gaussian_state(
        arrays["node_position_world"][start], arrays["node_velocity_world"][start]
    )
    external = [IndenterInput.from_arrays(arrays, t) for t in range(start, start + 20)]
    for group in GROUPS:
        original = JointStatePipeline(legacy, trained[group]).rollout(initial, external)
        loaded = restored[group].rollout(initial, external)
        if not all(
            torch.equal(a.state.centers, b.state.centers)
            and torch.equal(a.state.velocities, b.state.velocities)
            for a, b in zip(original, loaded, strict=True)
        ):
            raise AssertionError("Saved/reloaded 20-step predictions differ")
    # Poison every soft/contact label, while preserving known external inputs.
    poisoned = {key: value.copy() for key, value in arrays.items()}
    for key in poisoned:
        if key.startswith(("node_", "contact_")):
            poisoned[key] = np.full(poisoned[key].shape, np.nan)
    external_poisoned = [IndenterInput.from_arrays(poisoned, t) for t in range(start, start + 20)]
    a = restored["B"].rollout(initial, external)
    b = restored["B"].rollout(initial, external_poisoned)
    assert all(
        torch.equal(x.state.centers, y.state.centers)
        and torch.equal(x.state.velocities, y.state.velocities)
        for x, y in zip(a, b, strict=True)
    )
    return {
        "saved_reload_20_steps_bitwise_equal_all_groups": True,
        "B_rollout_all_soft_contact_labels_nan_invariant": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ablate_neighbor_recovery.yaml"
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
        "scripts/ablate_neighbor_recovery.py",
        "scripts/train_joint_state.py",
        "scripts/joint_state_evaluation.py",
        "scripts/analyze_joint_recovery.py",
        "src/contact_gwm/joint_state_dynamics.py",
        "src/contact_gwm/neighbor_dynamics.py",
    )
    for name in sources:
        shutil.copy2(ROOT / name, output / Path(name).name)
    write_json(output / "implementation_sha256.json", {p: file_sha256(ROOT / p) for p in sources})
    mesh, trajectories, splits = load_data(data, config)
    legacy_config = yaml.safe_load((ROOT / config["legacy_pipeline_config"]).read_text())
    legacy = PinnedContactPipeline(
        mesh, data / "scene.xml", ROOT / legacy_config["checkpoint"], legacy_config
    )
    preparation = JointStatePipeline(legacy, make_neighbor_model("A", mesh, config))
    cache, samples = make_cache(preparation, trajectories, splits, mesh, config, output)
    checks = verify_training_inputs(preparation, cache, samples, trajectories)
    # Verify reuse of original samples, labels, masks and current encoded inputs.
    reference = ROOT / config["reference_run"]
    previous = read_npz(reference / "current_inputs_and_labels.npz")
    assert all(np.array_equal(value.numpy(), previous[key]) for key, value in cache.items())
    assert json.loads((reference / "samples.json").read_text()) == samples
    checks["original_cache_and_sample_list_bitwise_equal"] = True
    trained, initial_states, specifications = {}, {}, {}
    for group in GROUPS:
        torch.manual_seed(config["seed"])
        model = make_neighbor_model(group, mesh, config)
        trained[group] = model
        initial_states[group] = copy.deepcopy(model.state_dict())
        specifications[group] = model_specification(model)
    contact = trained["A"].contact_network.state_dict()
    assert all(
        torch.equal(value, trained[g].contact_network.state_dict()[key])
        for g in GROUPS
        for key, value in contact.items()
    )
    checks["identical_initial_contact_parameters_all_groups"] = True
    write_json(output / "model_specifications.json", specifications)
    write_json(
        output / "initialization.json",
        {
            "state_digests": {g: tensor_digest(s) for g, s in initial_states.items()},
            "contact_state_digests": {
                g: tensor_digest(trained[g].contact_network.state_dict()) for g in GROUPS
            },
            "B_shared_initial_columns_equal_A": torch.equal(
                trained["A"].internal_network[0].weight,
                trained["B"].internal_network[0].weight[:, :28],
            ),
            "B_six_added_columns_initially_zero": bool(
                torch.count_nonzero(trained["B"].internal_network[0].weight[:, 28:]) == 0
            ),
            "all_initial_weights_equal": False,
            "explanation": (
                "Share compatible A/B internal tensors and all contact tensors; "
                "C changes internal width"
            ),
        },
    )
    weights = trained["B"].internal_network.neighbor_weights
    train_mask = torch.tensor([s["split"] == "train" for s in samples])
    extra = neighbor_relative_features(cache["features"][train_mask], weights, cache["pinned"])
    write_json(
        output / "neighbor_feature_rule.json",
        {
            "edge_count": len(mesh["edges"]),
            "node_count": len(mesh["node_ids"]),
            "pinned_count": int(cache["pinned"].sum()),
            "degree_min": int((weights > 0).sum(-1).min()),
            "degree_max": int((weights > 0).sum(-1).max()),
            "train_max_abs_normalized_neighbor_channels": extra.abs().amax((0, 1)).tolist(),
            "channels": "mean_j(u_j-u_i)/.01, mean_j(v_j-v_i)/.05",
            "weights": (
                "uniform over unique undirected rest-mesh neighbors; fixed nodes keep u=v=0"
            ),
            "rollout": "recomputed inside model from predicted current features each step",
        },
    )
    summaries, histories = [], {}
    for group in GROUPS:
        architecture = {
            "group": group,
            "autonomous": True,
            "dt": config["dt"],
            **specifications[group],
        }
        result, history = fit_variant(
            trained[group],
            group,
            initial_states[group],
            cache,
            samples,
            config,
            output,
            architecture=architecture,
        )
        summaries.append(result)
        histories[group] = history
    checks.update(check_historical_a(trained["A"], output, reference))
    write_json(output / "training_summaries.json", summaries)
    plot_training(histories, output)
    predictors = {
        "constant_velocity": lambda s, r: legacy.step(s, r, "constant_velocity"),
        "frozen_legacy": lambda s, r: legacy.step(s, r, "mujoco_force"),
    }
    restored = {}
    for group in GROUPS:
        saved = torch.load(output / group / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = make_neighbor_model(saved["architecture"]["group"], mesh, saved["config"])
        model.load_state_dict(saved["model_state"], strict=True)
        model.eval()
        restored[group] = JointStatePipeline(legacy, model)
        predictors[group] = restored[group].step
    evaluation = evaluate_models(
        predictors, legacy.gaussian_state, trajectories, splits, mesh, config, output
    )
    checks.update(evaluation["checks"])
    if checks["rollout_exception_count"] or not checks["common_evaluation_masks"]:
        raise RuntimeError("Incomplete rollout coverage: do not compare unequal windows")
    checks.update(verify_rollout_reload(legacy, trained, restored, trajectories))
    with (
        (output / "signed_node_analysis_log.txt").open("w") as log,
        contextlib.redirect_stdout(log),
    ):
        signed = analyze(output, methods=tuple(predictors))
    from summarize_neighbor_ablation import summarize

    comparison = summarize(output)
    assert before == {path: file_sha256(ROOT / path) for path in before}
    checks["historical_file_count_unchanged_sha256"] = len(before)
    checks["same_minibatch_order_seed_and_training_budget"] = True
    write_json(output / "checks.json", checks)
    write_json(output / "historical_source_hashes.json", before)
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
        "right_test_arrays_loaded": False,
    }
    write_json(output / "environment.json", environment)
    write_json(
        output / "summary.json",
        {
            "environment": environment,
            "checks": checks,
            "training": summaries,
            "model_specifications": specifications,
            "comparison": comparison,
            "signed_probe_nodes": signed["selected_nodes"],
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
