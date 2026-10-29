"""Read fixed search data; keep rollout initialization separate from targets.

No trajectories are generated or relabeled here. The exact central manifests
and teacher-forced caches from contact_response_training/run_001 are verified
before being exposed to the training runner. Right-side arrays are never read.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from audit_contact_mechanism import external_inputs
from joint_state_evaluation import PHASES, _masks

from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.data.mujoco_soft import tetra_volumes
from contact_gwm.pinned_contact_pipeline import IndenterInput
from contact_gwm.search_protocol import MESH_VOLUME_RATIO_THRESHOLD

SOURCE_RUN = Path("outputs/contact_response_training/run_001")
ORDINARY_STARTS = (70, 110, 140, 150, 170, 210)
CENTRAL_NAMES = ("center_shallow", "center_deep")
LEFT_NAMES = ("left_shallow", "left_deep")
BRANCH_STEPS = 20
VALIDATION_STEPS = 5


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _tensor_cache(path, mesh, expected_count):
    arrays = read_npz(path)
    if "node_ids" in arrays and not np.array_equal(arrays.pop("node_ids"), mesh["node_ids"]):
        raise ValueError(f"Changed cache node order: {path}")
    if not np.array_equal(arrays.get("pinned"), mesh["pinned_mask"]):
        raise ValueError(f"Changed fixed-node mask: {path}")
    count, nodes = expected_count, len(mesh["node_ids"])
    shapes = {
        "context": (count, nodes, 36),
        "gates": (count, nodes),
        "features": (count, nodes, 10),
        "velocity": (count, nodes, 3),
        "target_dx": (count, nodes, 3),
        "target_dv": (count, nodes, 3),
        "neighborhood": (count, nodes),
        "pinned": (nodes,),
    }
    if set(arrays) != set(shapes):
        raise ValueError(f"Changed cache fields: {path}")
    for key, value in arrays.items():
        if value.shape != shapes[key] or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} shape/finiteness in {path}")
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def _native_from_record(root, record, mesh):
    if record["trajectory"] not in (*CENTRAL_NAMES, *LEFT_NAMES):
        raise ValueError("Manifest includes a trajectory outside fixed train/development splits")
    path = root / record["file"]
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Changed native trajectory checksum: {path}")
    native = read_npz(path)
    nodes = len(mesh["node_ids"])
    if not np.array_equal(native["node_ids"], mesh["node_ids"]):
        raise ValueError(f"Changed native node order: {path}")
    for key in ("position", "velocity"):
        if native[key].shape != (21, nodes, 3) or not np.isfinite(native[key]).all():
            raise ValueError(f"Invalid native {key}: {path}")
    if native["time"].shape != (21,) or not np.allclose(
        np.diff(native["time"]), 0.01, rtol=0, atol=1e-10
    ):
        raise ValueError(f"Native recording interval changed: {path}")
    pin = mesh["pinned_mask"]
    expected = np.broadcast_to(mesh["rest_positions_world"][pin], native["position"][:, pin].shape)
    if not np.array_equal(native["position"][:, pin], expected) or np.any(
        native["velocity"][:, pin]
    ):
        raise ValueError(f"Native fixed boundary violated: {path}")
    return native


def _ordered_branch_cache(root, mesh, split, expected_cases):
    source = root / SOURCE_RUN
    manifest_name = (
        "central_sample_manifest.csv" if split == "train" else "validation_sample_manifest.csv"
    )
    manifest = _csv(source / manifest_name)
    samples = _csv(source / f"{split}_cache_samples.csv")
    if len(manifest) != expected_cases or len(samples) != expected_cases * BRANCH_STEPS:
        raise ValueError(f"Changed {split} native/cache sample count")
    cache = _tensor_cache(source / f"{split}_branch_cache.npz", mesh, expected_cases * BRANCH_STEPS)
    positions, velocities, robots, rois, minima = [], [], [], [], []
    for case, record in enumerate(manifest):
        if record["split"] != split:
            raise ValueError(f"Wrong split in {manifest_name}")
        allowed = CENTRAL_NAMES if split == "train" else LEFT_NAMES
        if record["trajectory"] not in allowed:
            raise ValueError(f"Wrong trajectory in {manifest_name}")
        native = _native_from_record(root, record, mesh)
        first = case * BRANCH_STEPS
        for step, sample in enumerate(samples[first : first + BRANCH_STEPS]):
            if int(sample["cache_index"]) != first + step or int(sample["step"]) != step:
                raise ValueError(f"Changed {split} teacher-cache ordering")
            if any(
                sample[key] != record[key]
                for key in ("split", "trajectory", "frame", "variant", "file", "sha256")
            ):
                raise ValueError(f"Native/cache case identity mismatch: {split}/{case}")
        ids = slice(first, first + BRANCH_STEPS)
        checks = (
            (cache["velocity"][ids], native["velocity"][:-1]),
            (cache["target_dx"][ids], np.diff(native["position"], axis=0)),
            (cache["target_dv"][ids], np.diff(native["velocity"], axis=0)),
        )
        if not all(
            torch.equal(saved, torch.as_tensor(actual).to(saved.dtype)) for saved, actual in checks
        ):
            raise ValueError(f"Native/cache physical state or supervision mismatch: {split}/{case}")
        roi = cache["neighborhood"][first].clone()
        if not torch.equal(cache["neighborhood"][ids], roi[None].expand(BRANCH_STEPS, -1)):
            raise ValueError(f"Within-branch ROI changed: {split}/{case}")
        positions.append(torch.from_numpy(native["position"].copy()).double())
        velocities.append(torch.from_numpy(native["velocity"].copy()).double())
        robots.append(external_inputs(native))
        rois.append(roi)
        minima.append(
            float(
                (
                    tetra_volumes(native["position"], mesh["tetrahedra"])
                    / mesh["rest_signed_tetra_volumes"]
                ).min()
            )
        )
    return (
        cache,
        manifest,
        torch.stack(positions),
        torch.stack(velocities),
        robots,
        torch.stack(rois),
        min(minima),
    )


def load_search_data(root, trajectories, mesh):
    """Return the fixed 600 ordinary pairs, 108 branch sequences and 90 pairs."""
    root = Path(root)
    source = root / SOURCE_RUN
    ordinary = _tensor_cache(source / "current_inputs_and_labels.npz", mesh, 1500)
    samples = json.loads((source / "samples.json").read_text())
    if len(samples) != 1500:
        raise ValueError("Changed ordinary sample manifest count")
    ids = [index for index, sample in enumerate(samples) if sample["split"] == "train"]
    if len(ids) != 600 or {samples[i]["trajectory"] for i in ids} != set(CENTRAL_NAMES):
        raise ValueError("Only the original 600 central ordinary pairs may train")
    for index in ids:
        sample = samples[index]
        arrays, frame = trajectories[sample["trajectory"]], int(sample["frame"])
        for key, physical in (
            ("velocity", arrays["node_velocity_world"][frame]),
            (
                "target_dx",
                arrays["node_position_world"][frame + 1] - arrays["node_position_world"][frame],
            ),
            (
                "target_dv",
                arrays["node_velocity_world"][frame + 1] - arrays["node_velocity_world"][frame],
            ),
        ):
            if not torch.equal(
                ordinary[key][index], torch.as_tensor(physical).to(ordinary[key].dtype)
            ):
                raise ValueError("Ordinary teacher-cache identity changed")
    ordinary_train = {
        key: value if key == "pinned" else value[ids] for key, value in ordinary.items()
    }
    branches, manifest, x, v, robots, roi, minimum = _ordered_branch_cache(root, mesh, "train", 108)
    pair_arrays = read_npz(source / "train_pair_labels.npz")
    pairs = {key: torch.from_numpy(value) for key, value in pair_arrays.items()}
    if set(pairs) != {"base_index", "variant_index", "target_dx", "target_dv", "neighborhood"}:
        raise ValueError("Changed initial pair fields")
    if pairs["base_index"].shape != (90,) or pairs["variant_index"].shape != (90,):
        raise ValueError("Changed initial response pair count")
    for b, c, target_x, target_v in zip(
        pairs["base_index"],
        pairs["variant_index"],
        pairs["target_dx"],
        pairs["target_dv"],
        strict=True,
    ):
        bi, ci = int(b), int(c)
        if bi % 20 or ci % 20 or not (0 <= bi < 2160 and 0 <= ci < 2160):
            raise ValueError("Response pair is not an initial branch endpoint")
        if not torch.equal(x[bi // 20, 0], x[ci // 20, 0]) or not torch.equal(
            v[bi // 20, 0], v[ci // 20, 0]
        ):
            raise ValueError("Paired branches do not share initial soft x/v")
        if not torch.equal(target_x, (x[ci // 20, 1] - x[bi // 20, 1]).float()) or not torch.equal(
            target_v, (v[ci // 20, 1] - v[bi // 20, 1]).float()
        ):
            raise ValueError("Paired supervision differs from native endpoint difference")
    ordinary_min = min(
        float(
            (
                tetra_volumes(trajectories[name]["node_position_world"], mesh["tetrahedra"])
                / mesh["rest_signed_tetra_volumes"]
            ).min()
        )
        for name in CENTRAL_NAMES
    )
    minimum = min(minimum, ordinary_min)
    if minimum < MESH_VOLUME_RATIO_THRESHOLD:
        raise ValueError(f"Protocol conflict: central native volume ratio {minimum} < 0.1")
    return {
        "ordinary_train": ordinary_train,
        "branch_cache": branches,
        "pairs": pairs,
        "branch_positions": x,
        "branch_velocities": v,
        "branch_robots": robots,
        "branch_roi": roi,
        "branch_manifest": manifest,
        "checks": {
            "ordinary_training_pairs": 600,
            "branch_training_cases": 108,
            "branch_training_pairs": 2160,
            "initial_response_pairs": 90,
            "manifest_cache_native_order_verified": True,
            "teacher_state_and_targets_match_native": True,
            "node_ids_and_fixed_boundary_verified": True,
            "central_native_minimum_volume_ratio": minimum,
            "central_threshold_0_1_passed": True,
            "right_trajectory_arrays_read": False,
        },
    }


def validation_windows(trajectories, mesh, root):
    """Thirty fixed five-step development windows with isolated initial inputs."""
    windows = []
    for name in LEFT_NAMES:
        arrays = trajectories[name]
        masks = _masks(arrays, mesh, 0.02)
        for start in ORDINARY_STARTS:
            windows.append(
                {
                    "initial_x": torch.tensor(arrays["node_position_world"][start]).double(),
                    "initial_v": torch.tensor(arrays["node_velocity_world"][start]).double(),
                    "targets_x": torch.tensor(
                        arrays["node_position_world"][start + 1 : start + 6]
                    ).double(),
                    "targets_v": torch.tensor(
                        arrays["node_velocity_world"][start + 1 : start + 6]
                    ).double(),
                    "external": [
                        IndenterInput.from_arrays(arrays, t) for t in range(start, start + 5)
                    ],
                    "roi": torch.tensor(masks[start]).bool(),
                    "phase": PHASES[int(arrays["phase_id"][start])],
                    "sourcecase": f"ordinary/{name}/{start}",
                }
            )
    _, manifest, x, v, robots, roi, _ = _ordered_branch_cache(Path(root), mesh, "validation", 18)
    for case, record in enumerate(manifest):
        name, start = record["trajectory"], int(record["frame"])
        windows.append(
            {
                "initial_x": x[case, 0].clone(),
                "initial_v": v[case, 0].clone(),
                "targets_x": x[case, 1:6].clone(),
                "targets_v": v[case, 1:6].clone(),
                "external": robots[case][:5],
                "roi": roi[case].clone(),
                "phase": PHASES[int(trajectories[name]["phase_id"][start])],
                "sourcecase": f"branch/{name}/{start}/{record['variant']}",
            }
        )
    if len(windows) != 30 or len({w["sourcecase"] for w in windows}) != 30:
        raise ValueError("Changed five-step validation window coverage")
    return windows
