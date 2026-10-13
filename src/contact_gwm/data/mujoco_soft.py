"""Independent MuJoCo FEM pilot; no Gaussian contact or learned labels are used.

Coordinates are world XYZ (z up), quaternions wxyz, units SI. The block is
anchored to world by its bottom layer. Every saved frame is synchronized by
mj_forward before reading vertices, Jacobians and solver contact forces.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

PHASES = ("approach", "press", "hold", "release", "recover")
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
CONTACT_FIELDS = {
    "contact_position_world": (3, np.float64),
    "contact_frame_world": (9, np.float64),
    "contact_signed_distance": (1, np.float64),
    "contact_wrench_local": (6, np.float64),
    "contact_force_on_block_world": (3, np.float64),
    "contact_geom_ids": (2, np.int32),
    "contact_flex_ids": (2, np.int32),
    "contact_element_ids": (2, np.int32),
    "contact_vertex_ids": (2, np.int32),
    "contact_efc_address": (1, np.int32),
    "contact_dimension": (1, np.int32),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict) -> tuple[int, int]:
    """Require a seven-case, equal-duration pilot with exact sample alignment."""
    if config["solver"]["backend"] != "mujoco_native_cpu":
        raise ValueError("This collector only supports the independent MuJoCo native CPU backend")
    if config["solver"]["version"] != mujoco.__version__:
        raise ValueError("MuJoCo version must match the configured version")
    if config["units"] != {"length": "m", "mass": "kg", "time": "s", "force": "N"}:
        raise ValueError("Pilot units must be SI")
    if set(config["indenter"]["positions"]) != {"left", "center", "right"}:
        raise ValueError("Exactly left/center/right positions are required")
    if set(config["indenter"]["depths"]) != {"shallow", "deep"}:
        raise ValueError("Exactly shallow/deep indentations are required")
    if set(config["phases"]) != set(PHASES):
        raise ValueError("All five protocol phases are required")
    values = [
        config["solver"]["timestep"],
        config["solver"]["sample_interval"],
        *config["phases"].values(),
        *config["block"]["spacing"],
        config["block"]["total_mass"],
        config["block"]["young"],
        *config["indenter"]["depths"].values(),
    ]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Time, spacing, mass, Young modulus and depth must be positive and finite")
    if not 0 <= config["block"]["poisson"] < 0.5:
        raise ValueError("Poisson ratio must be in [0, 0.5)")
    dt = config["solver"]["timestep"]
    stride = config["solver"]["sample_interval"] / dt
    steps = sum(config["phases"].values()) / dt
    if not math.isclose(stride, round(stride), abs_tol=1e-9):
        raise ValueError("Sample interval must be an integer multiple of timestep")
    if not math.isclose(steps, round(steps), abs_tol=1e-8) or round(steps) % round(stride):
        raise ValueError("Duration must align with integration and sample timesteps")
    return round(steps), round(stride)


def build_xml(config: dict) -> str:
    def vector(values: list) -> str:
        return " ".join(str(v) for v in values)

    solver, block, contact, indenter = (
        config[key] for key in ("solver", "block", "contact", "indenter")
    )
    nx, ny, _ = block["count"]
    attributes = (
        f'friction="{vector(contact["friction"])}" '
        f'solref="{vector(contact["solref"])}" '
        f'solimp="{vector(contact["solimp"])}"'
    )
    actuators = "\n".join(
        f'<position name="servo_{axis}" joint="p{axis}" kp="{indenter["kp"]}" '
        f'kv="{indenter["kv"]}"/>'
        for axis in "xyz"
    )
    return f'''<mujoco model="independent_soft_pilot">
  <compiler angle="radian"/>
  <option timestep="{solver["timestep"]}" gravity="{vector(solver["gravity"])}"
    integrator="{solver["integrator"]}" solver="{solver["solver"]}"
    jacobian="{solver["jacobian"]}" iterations="{solver["iterations"]}"
    tolerance="{solver["tolerance"]}"/>
  <worldbody>
    <geom name="base_visual" type="box" pos="0 0 -0.003" size=".04 .03 .003"
      contype="0" conaffinity="0" rgba=".5 .5 .5 1"/>
    <flexcomp name="block" type="grid" count="{vector(block["count"])}"
      spacing="{vector(block["spacing"])}" pos="{vector(block["center"])}"
      dim="3" dof="full" mass="{block["total_mass"]}"
      radius="{block["collision_radius"]}" rgba=".2 .6 .9 1">
      <contact selfcollide="none" internal="false" condim="3" {attributes}/>
      <elasticity young="{block["young"]}" poisson="{block["poisson"]}"
        damping="{block["rayleigh_damping"]}"/>
      <pin gridrange="0 0 0 {nx - 1} {ny - 1} 0"/>
    </flexcomp>
    <body name="indenter">
      <joint name="px" type="slide" axis="1 0 0"/>
      <joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <geom name="sphere" type="sphere" size="{indenter["radius"]}"
        mass="{indenter["mass"]}" rgba=".9 .3 .2 1" {attributes}/>
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>
'''


def cases(config: dict) -> list[dict]:
    result = [
        {
            "name": f"{position}_{depth}",
            "position": position,
            "x": x,
            "depth_name": depth,
            "depth": amount,
            "no_contact": False,
        }
        for position, x in config["indenter"]["positions"].items()
        for depth, amount in config["indenter"]["depths"].items()
    ]
    result.append(
        {
            "name": "no_contact",
            "position": "center",
            "x": 0.0,
            "depth_name": "none",
            "depth": 0.0,
            "no_contact": True,
        }
    )
    return result


def command(t: float, case: dict, config: dict) -> tuple[np.ndarray, np.ndarray, int]:
    block, indenter = config["block"], config["indenter"]
    top = block["center"][2] + (block["count"][2] - 1) * block["spacing"][2] / 2
    touching_z = top + block["collision_radius"] + indenter["radius"]
    high = indenter["start_z"]
    near = touching_z + indenter["approach_clearance"]
    low = touching_z - case["depth"]
    if case["no_contact"]:
        low = indenter["no_contact_low_z"]
        near = (high + low) / 2
    endpoints = ((high, near), (near, low), (low, low), (low, high), (high, high))
    elapsed = 0.0
    for phase, (a, b) in enumerate(endpoints):
        duration = config["phases"][PHASES[phase]]
        if t < elapsed + duration - 1e-12 or phase == len(PHASES) - 1:
            s = np.clip((t - elapsed) / duration, 0.0, 1.0)
            z = a + (b - a) * (1 - np.cos(np.pi * s)) / 2
            vz = (b - a) * np.pi * np.sin(np.pi * s) / (2 * duration)
            return np.array([case["x"], 0.0, z]), np.array([0.0, 0.0, vz]), phase
        elapsed += duration
    raise RuntimeError("Invalid protocol time")


def initialize(model: mujoco.MjModel, case: dict, config: dict) -> mujoco.MjData:
    data = mujoco.MjData(model)
    target, _, _ = command(0.0, case, config)
    addresses = [int(model.joint(f"p{axis}").qposadr[0]) for axis in "xyz"]
    data.qpos[addresses] = target
    data.ctrl[:] = target
    mujoco.mj_forward(model, data)
    return data


def mesh_from_model(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    if model.nflex != 1 or model.flex_dim[0] != 3:
        raise ValueError("Pilot requires exactly one 3D tetrahedral flex")
    mesh = {
        "node_ids": np.arange(model.nflexvert, dtype=np.int32),
        "rest_positions_world": data.flexvert_xpos.copy(),
        "vertex_body_ids": model.flex_vertbodyid.copy(),
        "node_masses": model.body_mass[model.flex_vertbodyid].copy(),
        "pinned_mask": model.flex_vertbodyid == 0,
        "tetrahedra": model.flex_elem.reshape(-1, 4).copy(),
        "surface_triangles": model.flex_shell.reshape(-1, 3).copy(),
        "edges": model.flex_edge.reshape(-1, 2).copy(),
    }
    mesh["rest_signed_tetra_volumes"] = tetra_volumes(
        mesh["rest_positions_world"], mesh["tetrahedra"]
    )
    return mesh


def tetra_volumes(positions: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    vertices = positions[..., tetrahedra, :]
    basis = vertices[..., 1:, :] - vertices[..., :1, :]
    return np.linalg.det(basis) / 6.0


def node_velocities(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    jacobian = np.zeros((3, model.nv))
    velocities = np.zeros((model.nflexvert, 3))
    for i, body_id in enumerate(model.flex_vertbodyid):
        if body_id:
            mujoco.mj_jac(model, data, jacobian, None, data.flexvert_xpos[i], int(body_id))
            velocities[i] = jacobian @ data.qvel
    return velocities


def read_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    rows: dict[str, list] = {key: [] for key in CONTACT_FIELDS}
    sphere = model.geom("sphere").id
    for i in range(data.ncon):
        contact = data.contact[i]
        if sphere not in contact.geom or 0 not in contact.flex:
            raise RuntimeError("Unexpected contact pair in isolated pilot scene")
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        # Local wrench is on contact side 1; frame rows are world basis vectors.
        sign = 1.0 if contact.flex[1] == 0 else -1.0
        force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
        values = {
            "contact_position_world": contact.pos,
            "contact_frame_world": contact.frame,
            "contact_signed_distance": [contact.dist],
            "contact_wrench_local": wrench,
            "contact_force_on_block_world": force,
            "contact_geom_ids": contact.geom,
            "contact_flex_ids": contact.flex,
            "contact_element_ids": contact.elem,
            "contact_vertex_ids": contact.vert,
            "contact_efc_address": [contact.efc_address],
            "contact_dimension": [contact.dim],
        }
        for key, value in values.items():
            rows[key].append(np.array(value, copy=True))
    return {
        key: np.asarray(rows[key], dtype=dtype).reshape(-1, width)
        for key, (width, dtype) in CONTACT_FIELDS.items()
    }


def collect_trajectory(
    model: mujoco.MjModel, case: dict, config: dict, mesh: dict[str, np.ndarray]
) -> tuple[dict, dict]:
    steps, stride = validate_config(config)
    data = initialize(model, case, config)
    dt = model.opt.timestep
    frames: dict[str, list] = {
        key: []
        for key in (
            "time",
            "step_index",
            "phase_id",
            "node_position_world",
            "node_velocity_world",
            "indenter_position_world",
            "indenter_quaternion_wxyz",
            "indenter_velocity_world",
            "action_target_position",
            "action_target_velocity",
            "actuator_force",
            "qpos",
            "qvel",
            "integration_state",
            "net_contact_force_on_block_world",
            "solver_iterations",
            "contact_count",
        )
    }
    contacts: dict[str, list] = {key: [] for key in CONTACT_FIELDS}
    offsets = [0]
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    jacobian = np.zeros((3, model.nv))
    started = time.perf_counter()
    integration_seconds = 0.0
    for step in range(steps + 1):
        expected_time = step * dt
        if not math.isclose(data.time, expected_time, abs_tol=1e-9):
            raise RuntimeError(f"Solver time reset/drift at step {step}: {data.time}")
        target, target_velocity, phase = command(expected_time, case, config)
        data.ctrl[:] = target
        if step % stride == 0:
            mujoco.mj_forward(model, data)
            contact_data = read_contacts(model, data)
            count = len(contact_data["contact_signed_distance"])
            offsets.append(offsets[-1] + count)
            for key, values in contact_data.items():
                contacts[key].append(values)
            mujoco.mj_getState(model, data, state, STATE_SPEC)
            body_id = model.body("indenter").id
            mujoco.mj_jac(model, data, jacobian, None, data.xpos[body_id], body_id)
            values = {
                "time": data.time,
                "step_index": step,
                "phase_id": phase,
                "node_position_world": data.flexvert_xpos.copy(),
                "node_velocity_world": node_velocities(model, data),
                "indenter_position_world": data.xpos[body_id].copy(),
                "indenter_quaternion_wxyz": data.xquat[body_id].copy(),
                "indenter_velocity_world": jacobian @ data.qvel,
                "action_target_position": target,
                "action_target_velocity": target_velocity,
                "actuator_force": data.actuator_force.copy(),
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "integration_state": state.copy(),
                "net_contact_force_on_block_world": contact_data[
                    "contact_force_on_block_world"
                ].sum(axis=0),
                "solver_iterations": data.solver_niter.copy(),
                "contact_count": count,
            }
            for key, value in values.items():
                frames[key].append(value)
            ratios = (
                tetra_volumes(data.flexvert_xpos, mesh["tetrahedra"])
                / mesh["rest_signed_tetra_volumes"]
            )
            if ratios.min() <= config["acceptance"]["min_tetra_volume_ratio"]:
                raise RuntimeError(f"Collapsed/inverted tetrahedron at step {step}")
        warnings = [
            (mujoco.mjtWarning(i).name, warning.number)
            for i, warning in enumerate(data.warning)
            if warning.number
        ]
        if warnings or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"Solver failure at step {step}: {warnings}")
        if step < steps:
            tick = time.perf_counter()
            mujoco.mj_step(model, data)
            integration_seconds += time.perf_counter() - tick
    arrays = {key: np.asarray(values) for key, values in frames.items()}
    arrays["contact_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays.update({key: np.concatenate(values) for key, values in contacts.items()})
    arrays["node_ids"] = mesh["node_ids"].copy()
    stats = {
        "physics_steps": steps,
        "frames": len(arrays["time"]),
        "simulated_seconds": float(data.time),
        "collection_wall_seconds": time.perf_counter() - started,
        "integration_wall_seconds": integration_seconds,
        "solver_warnings": [],
    }
    stats["physics_steps_per_second"] = steps / integration_seconds
    stats["collection_realtime_factor"] = (
        stats["simulated_seconds"] / stats["collection_wall_seconds"]
    )
    return arrays, stats


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_trajectory(arrays: dict, mesh: dict, config: dict, case: dict) -> dict:
    """Validate data reloaded from disk; reject corrupt topology/timing/labels."""
    steps, stride = validate_config(config)
    frame_count = steps // stride + 1
    node_count = len(mesh["node_ids"])
    if not np.array_equal(mesh["node_ids"], np.arange(node_count)):
        raise ValueError("Mesh node IDs must be contiguous and stable")
    if not np.array_equal(arrays["node_ids"], mesh["node_ids"]):
        raise ValueError("Trajectory node IDs differ from the mesh")
    for key in ("tetrahedra", "surface_triangles", "edges"):
        topology = mesh[key]
        if (
            not np.issubdtype(topology.dtype, np.integer)
            or topology.min() < 0
            or topology.max() >= node_count
        ):
            raise ValueError(f"Invalid node indices in {key}")
    for key in ("node_position_world", "node_velocity_world"):
        if arrays[key].shape != (frame_count, node_count, 3):
            raise ValueError(f"Wrong shape for {key}")
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite saved array: {key}")
    expected_steps = np.arange(frame_count) * stride
    if not np.array_equal(arrays["step_index"], expected_steps):
        raise ValueError("Saved integration steps do not match the sample schedule")
    if not np.allclose(
        arrays["time"], expected_steps * config["solver"]["timestep"], atol=1e-9, rtol=0
    ):
        raise ValueError("Saved time is inconsistent with integration steps")
    if set(arrays["phase_id"]) != set(range(len(PHASES))):
        raise ValueError("Incomplete five-phase protocol")
    for i, t in enumerate(arrays["time"]):
        position, velocity, phase = command(float(t), case, config)
        if not (
            np.allclose(arrays["action_target_position"][i], position, atol=1e-12, rtol=0)
            and np.allclose(arrays["action_target_velocity"][i], velocity, atol=1e-12, rtol=0)
            and arrays["phase_id"][i] == phase
        ):
            raise ValueError("Saved actions/phases do not match the complete configuration")
    offsets = arrays["contact_offsets"]
    if offsets.shape != (frame_count + 1,) or offsets[0] != 0 or (np.diff(offsets) < 0).any():
        raise ValueError("Invalid packed contact offsets")
    if not np.array_equal(np.diff(offsets), arrays["contact_count"]):
        raise ValueError("Packed contacts disagree with per-frame counts")
    for key, (width, _) in CONTACT_FIELDS.items():
        if arrays[key].shape != (offsets[-1], width):
            raise ValueError(f"Invalid packed contact shape: {key}")
    for i in range(frame_count):
        forces = arrays["contact_force_on_block_world"][offsets[i] : offsets[i + 1]].sum(0)
        if not np.allclose(forces, arrays["net_contact_force_on_block_world"][i], atol=1e-10):
            raise ValueError("Net force does not match the saved individual contact forces")
    displacement = arrays["node_position_world"] - mesh["rest_positions_world"]
    magnitude = np.linalg.norm(displacement, axis=-1)
    peak = float(magnitude.max())
    fixed_error = float(magnitude[:, mesh["pinned_mask"]].max())
    if fixed_error > config["acceptance"]["max_pinned_displacement"]:
        raise ValueError("Fixed bottom nodes moved")
    if np.abs(arrays["node_velocity_world"][:, mesh["pinned_mask"]]).max() > 1e-12:
        raise ValueError("Fixed bottom nodes have nonzero velocity")
    ratios = (
        tetra_volumes(arrays["node_position_world"], mesh["tetrahedra"])
        / mesh["rest_signed_tetra_volumes"]
    )
    if ratios.min() <= config["acceptance"]["min_tetra_volume_ratio"]:
        raise ValueError("Saved trajectory contains inverted/collapsed tetrahedra")
    if case["no_contact"]:
        if (
            arrays["contact_count"].max() != 0
            or peak > config["acceptance"]["max_no_contact_displacement"]
        ):
            raise ValueError("No-contact control produced contacts or deformation")
    else:
        if (
            arrays["contact_count"].max() == 0
            or peak < config["acceptance"]["min_peak_displacement"]
        ):
            raise ValueError("Pressing did not produce contact and nonzero deformation")
        if magnitude[-1].max() / peak > config["acceptance"]["max_recovery_fraction"]:
            raise ValueError("Recovery residual is too large")
        if arrays["contact_count"][-1] != 0:
            raise ValueError("Indenter did not separate at the end")
        if arrays["net_contact_force_on_block_world"][:, 2].min() >= 0:
            raise ValueError("No downward solver force on the block")
    peak_frame, peak_node = np.unravel_index(magnitude.argmax(), magnitude.shape)
    upper = mesh["rest_positions_world"][:, 2] >= mesh["rest_positions_world"][:, 2].max() - 1e-9
    xy_distance = np.linalg.norm(mesh["rest_positions_world"][:, :2] - [case["x"], 0], axis=1)
    near, far = upper & (xy_distance <= 0.02), upper & (xy_distance >= 0.03)
    return {
        "peak_displacement_m": peak,
        "final_residual_m": float(magnitude[-1].max()),
        "recovery_fraction": float(magnitude[-1].max() / peak) if peak > 1e-12 else 0.0,
        "pinned_max_displacement_m": fixed_error,
        "min_tetra_volume_ratio": float(ratios.min()),
        "peak_frame": int(peak_frame),
        "peak_node_id": int(peak_node),
        "peak_node_rest_position": mesh["rest_positions_world"][peak_node].tolist(),
        "near_top_rms_at_peak_m": float(np.sqrt(np.mean(magnitude[peak_frame, near] ** 2))),
        "far_top_rms_at_peak_m": float(np.sqrt(np.mean(magnitude[peak_frame, far] ** 2))),
        "max_contacts": int(arrays["contact_count"].max()),
        "contact_frames": int(np.count_nonzero(arrays["contact_count"])),
        "max_downward_contact_force_N": float(
            -arrays["net_contact_force_on_block_world"][:, 2].min()
        ),
    }


def replay_check(model: mujoco.MjModel, arrays: dict, peak_frame: int) -> dict:
    """Restore selected engine states and independently recompute node labels."""
    errors = []
    for frame in sorted({0, peak_frame, len(arrays["time"]) - 1}):
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, arrays["integration_state"][frame], STATE_SPEC)
        mujoco.mj_forward(model, data)
        position_error = float(
            np.abs(data.flexvert_xpos - arrays["node_position_world"][frame]).max()
        )
        velocity_error = float(
            np.abs(node_velocities(model, data) - arrays["node_velocity_world"][frame]).max()
        )
        contact_data = read_contacts(model, data)
        force_error = float(
            np.abs(
                contact_data["contact_force_on_block_world"].sum(0)
                - arrays["net_contact_force_on_block_world"][frame]
            ).max()
        )
        if position_error > 1e-12 or velocity_error > 1e-12 or force_error > 1e-5:
            raise ValueError("Saved solver state does not reproduce the node/contact labels")
        errors.append(
            {
                "frame": frame,
                "position_error": position_error,
                "velocity_error": velocity_error,
                "net_force_error": force_error,
            }
        )
    return {"frames": errors, "passed": True}


def validate_batch(directory: Path) -> dict:
    """Standalone disk audit with checksums, topology and MuJoCo state replay."""
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    expected_cases = cases(config)
    if len(manifest["trajectories"]) != 7 or manifest["failures"]:
        raise ValueError("Pilot must contain exactly seven successful trajectories")
    for name, checksum in manifest["shared_sha256"].items():
        if digest(directory / name) != checksum:
            raise ValueError(f"Shared file checksum mismatch: {name}")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    mesh = load_npz(directory / "mesh.npz")
    reference = mesh_from_model(model, initialize(model, expected_cases[0], config))
    for key in reference:
        if not np.array_equal(mesh[key], reference[key]):
            raise ValueError(f"Mesh does not match the independent solver model: {key}")
    reports = []
    for item, case in zip(manifest["trajectories"], expected_cases, strict=True):
        path = directory / item["file"]
        if item["case"] != case or digest(path) != item["sha256"]:
            raise ValueError(f"Case/checksum mismatch: {path.name}")
        arrays = load_npz(path)
        report = validate_trajectory(arrays, mesh, config, case)
        report["replay"] = replay_check(model, arrays, report["peak_frame"])
        reports.append({"name": case["name"], **report})
    result = {"passed": True, "trajectories_checked": 7, "reports": reports}
    (directory / "reload_validation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    """Utility used by tests to audit the stored surface topology."""
    faces = [
        tuple(sorted(face))
        for tet in tetrahedra
        for face in (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        )
    ]
    return np.asarray([face for face, count in Counter(faces).items() if count == 1])
