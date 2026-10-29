from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bounded_search_data as data_tools  # noqa: E402

from contact_gwm.data.gaussian_pairs import file_sha256  # noqa: E402


def mesh():
    return {
        "node_ids": np.arange(4, dtype=np.int32),
        "pinned_mask": np.array([True, False, False, False]),
        "rest_positions_world": np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
        ),
        "tetrahedra": np.array([[0, 1, 2, 3]]),
        "rest_signed_tetra_volumes": np.array([1e-6 / 6]),
    }


def native_arrays(m):
    return {
        "node_ids": m["node_ids"].copy(),
        "position": np.repeat(m["rest_positions_world"][None], 21, axis=0),
        "velocity": np.zeros((21, 4, 3)),
        "time": np.arange(21) * 0.01,
        "robot_position": np.repeat(np.array([[0.0, 0.0, 0.022]]), 21, axis=0),
        "robot_velocity": np.zeros((21, 3)),
        "command_position": np.array([0.0, 0.0, 0.022]),
    }


def cache_arrays(m, count=20):
    return {
        "context": np.zeros((count, 4, 36), np.float32),
        "gates": np.zeros((count, 4), np.float32),
        "features": np.zeros((count, 4, 10), np.float32),
        "velocity": np.zeros((count, 4, 3), np.float32),
        "target_dx": np.zeros((count, 4, 3), np.float32),
        "target_dv": np.zeros((count, 4, 3), np.float32),
        "neighborhood": np.repeat((~m["pinned_mask"])[None], count, axis=0),
        "pinned": m["pinned_mask"].copy(),
    }


@pytest.mark.parametrize(
    "bad", ["node_order", "dt", "fixed_position", "fixed_velocity", "checksum"]
)
def test_native_input_rejects_changed_identity_time_or_boundary(tmp_path, bad):
    m = mesh()
    arrays = native_arrays(m)
    if bad == "node_order":
        arrays["node_ids"] = arrays["node_ids"][::-1]
    elif bad == "dt":
        arrays["time"][1] += 0.001
    elif bad == "fixed_position":
        arrays["position"][5, 0, 2] += 0.001
    elif bad == "fixed_velocity":
        arrays["velocity"][5, 0, 2] = 0.1
    path = tmp_path / "native.npz"
    np.savez_compressed(path, **arrays)
    record = {
        "trajectory": "center_shallow",
        "file": path.name,
        "sha256": "bad" if bad == "checksum" else file_sha256(path),
    }
    with pytest.raises(ValueError):
        data_tools._native_from_record(tmp_path, record, m)


def test_tensor_cache_preserves_dtypes_and_rejects_empty_or_changed_node_fields(tmp_path):
    m = mesh()
    arrays = cache_arrays(m, 2)
    arrays["node_ids"] = m["node_ids"].copy()
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, **arrays)
    loaded = data_tools._tensor_cache(path, m, 2)
    assert loaded["context"].dtype.__str__() == "torch.float32"
    assert loaded["neighborhood"].dtype.__str__() == "torch.bool"
    assert "node_ids" not in loaded
    arrays["node_ids"] = m["node_ids"][::-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        data_tools._tensor_cache(path, m, 2)


def test_teacher_cache_native_manifest_order_is_a_contract(tmp_path):
    m = mesh()
    directory = tmp_path / data_tools.SOURCE_RUN
    directory.mkdir(parents=True)
    native = directory / "native.npz"
    np.savez_compressed(native, **native_arrays(m))
    record = {
        "split": "train",
        "trajectory": "center_shallow",
        "frame": "70",
        "variant": "base",
        "file": str(native.relative_to(tmp_path)),
        "sha256": file_sha256(native),
    }
    with (directory / "central_sample_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    samples = [{**record, "step": i, "cache_index": i} for i in range(20)]
    samples[3]["step"] = 4
    with (directory / "train_cache_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    np.savez_compressed(directory / "train_branch_cache.npz", **cache_arrays(m))
    with pytest.raises(ValueError, match="ordering"):
        data_tools._ordered_branch_cache(tmp_path, m, "train", 1)
