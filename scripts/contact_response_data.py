"""Central native branching data and current-only teacher-forced input caches."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from audit_contact_mechanism import external_inputs
from validate_contact_pipeline import reference_masks, write_csv, write_json

from contact_gwm.contact_audit import NativeContactIntervention, intervention_robot
from contact_gwm.data.gaussian_pairs import file_sha256, read_npz
from contact_gwm.pinned_contact_pipeline import IndenterInput

ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = ("context", "gates", "features", "velocity")


def collect_central(mesh, trajectories, config, variants, output):
    directory = output / "central_native"
    directory.mkdir()
    solver = NativeContactIntervention(ROOT / config["data_directory"] / "scene.xml", mesh)
    records, checks = [], []
    for name in config["splits"]["train"]:
        arrays = trajectories[name]
        for frame in config["collection"]["start_frames"]:
            for variant in variants:
                robot = intervention_robot(IndenterInput.from_arrays(arrays, frame), variant)
                native, check = solver.rollout(
                    arrays["integration_state"][frame].copy(),
                    robot,
                    config["collection"]["steps"],
                    config["dt"],
                )
                assert np.array_equal(native["position"][0], arrays["node_position_world"][frame])
                assert np.array_equal(native["velocity"][0], arrays["node_velocity_world"][frame])
                path = directory / f"{name}_{frame}_{variant['name']}.npz"
                np.savez_compressed(path, **native)
                loaded = read_npz(path)
                assert all(
                    np.array_equal(v, loaded[k]) and np.isfinite(v).all() for k, v in native.items()
                )
                records.append(
                    {
                        "split": "train",
                        "trajectory": name,
                        "frame": frame,
                        "phase_id": int(arrays["phase_id"][frame]),
                        "variant": variant["name"],
                        "file": str(path.relative_to(ROOT)),
                        "sha256": file_sha256(path),
                    }
                )
                checks.append({**records[-1], **check})
            print(f"Central native data complete: {name}/{frame}, six branches", flush=True)
    write_csv(output / "central_sample_manifest.csv", records)
    write_json(output / "central_collection_checks.json", checks)
    return records


def validation_records(config, variants, output):
    records = []
    directory = ROOT / config["validation_native_directory"]
    for frame in config["collection"]["validation_frames"]:
        for variant in variants:
            path = directory / f"{frame}_{variant['name']}_native.npz"
            records.append(
                {
                    "split": "validation",
                    "trajectory": "left_deep",
                    "frame": frame,
                    "variant": variant["name"],
                    "file": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                }
            )
    write_csv(output / "validation_sample_manifest.csv", records)
    return records


@torch.no_grad()
def branch_cache(records, pipeline, trajectories, mesh, config, output, split):
    values = {key: [] for key in (*INPUT_KEYS, "target_dx", "target_dv", "neighborhood")}
    groups = defaultdict(list)
    for record in records:
        groups[record["trajectory"], record["frame"]].append(record)
    pairs = {
        key: [] for key in ("base_index", "variant_index", "target_dx", "target_dv", "neighborhood")
    }
    samples, pair_samples = [], []
    max_force_error = 0.0
    for (name, frame), group in groups.items():
        natives = {r["variant"]: read_npz(ROOT / r["file"]) for r in group}
        base = natives["base"]
        roi = reference_masks(trajectories[name], mesh, config["contact_neighborhood_radius_m"])[
            frame
        ].copy()
        for native in natives.values():
            assert np.array_equal(native["node_ids"], mesh["node_ids"])
            assert np.array_equal(native["position"][0], base["position"][0])
            assert np.array_equal(native["velocity"][0], base["velocity"][0])
            first, last = native["contact_offsets"][:2]
            points = native["contact_position_world"][first:last]
            if len(points):
                roi |= (
                    np.linalg.norm(base["position"][0, :, None] - points, axis=-1).min(-1)
                    <= config["contact_neighborhood_radius_m"]
                )
        roi &= ~mesh["pinned_mask"]
        start_indices = {}
        for record in group:
            native = natives[record["variant"]]
            start_indices[record["variant"]] = len(values["context"])
            external = external_inputs(native)
            for t, robot in enumerate(external):
                # A teacher-forced CURRENT state; next x/v enter labels below only.
                state = pipeline.legacy.gaussian_state(native["position"][t], native["velocity"][t])
                current = pipeline.current_inputs(state, robot)
                for key in INPUT_KEYS:
                    values[key].append(getattr(current, key))
                values["target_dx"].append(
                    torch.tensor(native["position"][t + 1] - native["position"][t]).float()
                )
                values["target_dv"].append(
                    torch.tensor(native["velocity"][t + 1] - native["velocity"][t]).float()
                )
                values["neighborhood"].append(torch.tensor(roi))
                max_force_error = max(
                    max_force_error,
                    float(
                        np.abs(current.forces.forces.sum(0).numpy() - native["net_force"][t]).max()
                    ),
                )
                samples.append({**record, "step": t, "cache_index": len(samples)})
        b = start_indices["base"]
        for variant, native in natives.items():
            if variant == "base":
                continue
            v = start_indices[variant]
            for key in ("features", "velocity"):
                assert torch.equal(values[key][b], values[key][v])
            pairs["base_index"].append(b)
            pairs["variant_index"].append(v)
            pairs["target_dx"].append(
                torch.tensor(native["position"][1] - base["position"][1]).float()
            )
            pairs["target_dv"].append(
                torch.tensor(native["velocity"][1] - base["velocity"][1]).float()
            )
            pairs["neighborhood"].append(torch.tensor(roi))
            pair_samples.append(
                {
                    "trajectory": name,
                    "frame": frame,
                    "variant": variant,
                    "base_index": b,
                    "variant_index": v,
                }
            )
    cache = {key: torch.stack(value) for key, value in values.items()}
    cache["pinned"] = torch.tensor(mesh["pinned_mask"])
    paired = {
        key: torch.tensor(value, dtype=torch.long) if key.endswith("index") else torch.stack(value)
        for key, value in pairs.items()
    }
    assert all(torch.isfinite(v).all() for v in (*cache.values(), *paired.values()))
    np.savez_compressed(
        output / f"{split}_branch_cache.npz", **{k: v.numpy() for k, v in cache.items()}
    )
    np.savez_compressed(
        output / f"{split}_pair_labels.npz", **{k: v.numpy() for k, v in paired.items()}
    )
    write_csv(output / f"{split}_cache_samples.csv", samples)
    write_csv(output / f"{split}_pair_samples.csv", pair_samples)
    print(
        f"Cached {split}: {len(samples)} current frame pairs, "
        f"{len(pair_samples)} initial response pairs",
        flush=True,
    )
    return (
        cache,
        paired,
        {
            "current_native_force_max_abs_difference_N": max_force_error,
            "same_initial_soft_state_all_six_conditions": True,
        },
    )
