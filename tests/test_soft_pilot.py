"""Meaningful optional integration checks for independent soft-body labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("mujoco")
import mujoco

from contact_gwm.data import mujoco_soft


@pytest.fixture
def small_trajectory():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/mujoco_soft_pilot.yaml").read_text()
    )
    config["phases"] = dict.fromkeys(mujoco_soft.PHASES, 0.02)
    config["indenter"]["start_z"] = 0.06
    model = mujoco.MjModel.from_xml_string(mujoco_soft.build_xml(config))
    case = mujoco_soft.cases(config)[-1]
    mesh = mujoco_soft.mesh_from_model(model, mujoco_soft.initialize(model, case, config))
    arrays, _ = mujoco_soft.collect_trajectory(model, case, config, mesh)
    return config, case, mesh, arrays, model


def test_fem_topology_and_fixed_bottom(small_trajectory) -> None:
    _, _, mesh, arrays, _ = small_trajectory
    assert len(mesh["node_ids"]) == 140
    assert mesh["pinned_mask"].sum() == 35
    assert np.all(mesh["rest_positions_world"][mesh["pinned_mask"], 2] == 0.0)
    actual = {tuple(sorted(face)) for face in mesh["surface_triangles"]}
    expected = {tuple(face) for face in mujoco_soft.boundary_faces(mesh["tetrahedra"])}
    assert actual == expected
    np.testing.assert_allclose(abs(mesh["rest_signed_tetra_volumes"]).sum(), 0.06 * 0.04 * 0.03)
    assert arrays["contact_count"].max() == 0


def test_saved_roundtrip_and_independent_replay(tmp_path, small_trajectory) -> None:
    config, case, mesh, arrays, model = small_trajectory
    path = tmp_path / "control.npz"
    np.savez_compressed(path, **arrays)
    saved = mujoco_soft.load_npz(path)
    result = mujoco_soft.validate_trajectory(saved, mesh, config, case)
    assert result["peak_displacement_m"] < 1e-8
    assert mujoco_soft.replay_check(model, saved, result["peak_frame"])["passed"]


def test_reload_rejects_corrupt_time_and_solver_labels(small_trajectory) -> None:
    config, case, mesh, arrays, model = small_trajectory
    bad = {key: value.copy() for key, value in arrays.items()}
    bad["time"][1] += 0.001
    with pytest.raises(ValueError, match="time"):
        mujoco_soft.validate_trajectory(bad, mesh, config, case)
    bad = {key: value.copy() for key, value in arrays.items()}
    bad["node_position_world"][1, 1, 0] += 0.001
    with pytest.raises(ValueError, match="reproduce"):
        mujoco_soft.replay_check(model, bad, 1)
