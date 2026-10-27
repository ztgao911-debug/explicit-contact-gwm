"""Independent physical interventions and signed response scoring for an audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pinned_contact_pipeline import IndenterInput, NativeCurrentContact


def intervention_robot(current: IndenterInput, variant: dict) -> IndenterInput:
    """Different ROBOT initial condition; identical initial soft x/v in all cases.

    Commands are held constant after initialization. This is an initial-condition
    intervention, not a claim that a real robot teleports during its trajectory.
    """
    dp = np.asarray(variant["position_offset_m"], dtype=float)
    dv = np.asarray(variant["velocity_offset_m_s"], dtype=float)
    if dp.shape != (3,) or dv.shape != (3,) or not np.isfinite([dp, dv]).all():
        raise ValueError("Finite three-dimensional robot offsets required")
    return IndenterInput(
        current.position + dp, current.velocity + dv, current.target_position + dp, np.zeros(3)
    )


def signed_response_metrics(predicted, reference, mask, multiplier):
    """Magnitude AND direction; a zero-response baseline is explicit and meaningful."""
    p, r = np.asarray(predicted)[mask], np.asarray(reference)[mask]
    if p.shape != r.shape or not len(p) or not np.isfinite([p, r]).all():
        raise ValueError("Finite responses on the same nonempty node region required")
    pp, rr, pr = float(np.square(p).sum()), float(np.square(r).sum()), float((p * r).sum())
    return {
        "response_rmse": float(np.sqrt(np.square(p - r).sum() / len(p)) * multiplier),
        "reference_response_rms": float(np.sqrt(rr / len(p)) * multiplier),
        "predicted_response_rms": float(np.sqrt(pp / len(p)) * multiplier),
        "projection_gain": pr / rr if rr > 1e-30 else None,
        "amplitude_ratio": np.sqrt(pp / rr).item() if rr > 1e-30 else None,
        "direction_cosine": pr / np.sqrt(pp * rr) if min(pp, rr) > 1e-30 else None,
    }


class NativeContactIntervention:
    """Restore once, change robot initial conditions, integrate independent labels.

    This accepts neither a trajectory nor future soft arrays. Recorded external
    robot motion can subsequently be prescribed to every compared predictor.
    """

    def __init__(self, scene: Path, mesh: dict):
        self.adapter = NativeCurrentContact(scene, mesh)
        self.mj, self.model, self.mesh = self.adapter.mj, self.adapter.model, mesh
        self.ids = np.array(list(self.adapter.node_dofs))
        self.v_addresses = np.asarray([self.adapter.node_dofs[i][1] for i in self.ids])

    def rollout(self, initial_integration_state, robot: IndenterInput, steps, record_dt):
        from .data.mujoco_soft import node_velocities, read_contacts, tetra_volumes

        if steps < 1 or record_dt <= 0:
            raise ValueError("Positive integration horizon required")
        for field in (robot.position, robot.velocity, robot.target_position, robot.target_velocity):
            if np.shape(field) != (3,) or not np.isfinite(field).all():
                raise ValueError("Finite robot state and command required")
        if np.any(robot.target_velocity):
            raise ValueError("This audit holds the initial command constant")
        stride = round(record_dt / self.model.opt.timestep)
        if not np.isclose(stride * self.model.opt.timestep, record_dt, atol=1e-12, rtol=0):
            raise ValueError("Recording interval must align with native timestep")
        data = self.mj.MjData(self.model)
        self.mj.mj_setState(
            self.model, data, initial_integration_state, self.mj.mjtState.mjSTATE_INTEGRATION
        )
        data.qpos[self.adapter.robot_q] = robot.position
        data.qvel[self.adapter.robot_v] = robot.velocity
        # Apply to BASE as well: no stale factual constraint warmstart is privileged.
        data.qacc_warmstart[:] = 0
        initial_time = data.time
        arrays = {
            k: []
            for k in (
                "position",
                "velocity",
                "robot_position",
                "robot_velocity",
                "net_force",
                "net_torque",
                "time",
            )
        }
        contact_arrays, offsets = {}, [0]
        min_volume, pin_error, velocity_error, max_contacts = np.inf, 0.0, 0.0, 0
        pin, rest = self.mesh["pinned_mask"], self.mesh["rest_positions_world"]
        mass = self.mesh["node_masses"]
        for h in range(steps + 1):
            data.ctrl[:] = robot.target_position
            self.mj.mj_forward(self.model, data)
            x = data.flexvert_xpos.copy()
            v = np.zeros_like(x)
            v[self.ids] = data.qvel[self.v_addresses]
            if not np.isfinite([x, v]).all():
                raise FloatingPointError("Non-finite native intervention state")
            if abs(data.time - initial_time - h * record_dt) > 1e-9:
                raise RuntimeError("Native intervention time drift/reset")
            if h in (0, steps):
                velocity_error = max(
                    velocity_error, float(np.abs(v - node_velocities(self.model, data)).max())
                )
            pin_error = max(
                pin_error, float(np.abs(x[pin] - rest[pin]).max()), float(np.abs(v[pin]).max())
            )
            min_volume = min(
                min_volume,
                float(
                    (
                        tetra_volumes(x, self.mesh["tetrahedra"])
                        / self.mesh["rest_signed_tetra_volumes"]
                    ).min()
                ),
            )
            contact = read_contacts(self.model, data)
            for key, value in contact.items():
                contact_arrays.setdefault(key, []).append(value)
            offsets.append(offsets[-1] + len(contact["contact_signed_distance"]))
            com = (x * mass[:, None]).sum(0) / mass.sum()
            values = {
                "position": x,
                "velocity": v,
                "robot_position": data.qpos[self.adapter.robot_q].copy(),
                "robot_velocity": data.qvel[self.adapter.robot_v].copy(),
                "net_force": contact["contact_force_on_block_world"].sum(0),
                "net_torque": np.cross(
                    contact["contact_position_world"] - com, contact["contact_force_on_block_world"]
                ).sum(0),
                "time": data.time,
            }
            for key, value in values.items():
                arrays[key].append(value)
            max_contacts = max(max_contacts, data.ncon)
            if h < steps:
                for _ in range(stride):
                    data.ctrl[:] = robot.target_position
                    self.mj.mj_step(self.model, data)
                    max_contacts = max(max_contacts, data.ncon)
        warnings = [(i, int(w.number)) for i, w in enumerate(data.warning) if w.number]
        if warnings or min_volume <= 0 or pin_error != 0:
            raise RuntimeError(
                f"Native intervention warning/pins/volume: {warnings}, {pin_error}, {min_volume}"
            )
        result = {key: np.asarray(value) for key, value in arrays.items()}
        result.update({key: np.concatenate(value) for key, value in contact_arrays.items()})
        result.update(
            node_ids=self.mesh["node_ids"],
            contact_offsets=np.asarray(offsets),
            command_position=robot.target_position.copy(),
            initial_integration_state=np.array(initial_integration_state, copy=True),
        )
        return result, {
            "native_internal_steps": steps * stride,
            "warnings": warnings,
            "max_fixed_position_or_velocity_error": pin_error,
            "minimum_recorded_tetra_volume_ratio": min_volume,
            "instant_velocity_vs_jacobian_max_error": velocity_error,
            "max_contact_count_including_internal_steps": max_contacts,
            "initial_warmstart_reset_to_zero": True,
        }
