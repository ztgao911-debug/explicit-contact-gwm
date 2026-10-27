"""Matched perturbations and independent native simulation for recovery checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pinned_contact_pipeline import NativeCurrentContact


def normalized_state_norm(delta_x, delta_v, mask, position_unit, velocity_unit):
    """Vector RMS of dimensionless x/v differences, NOT an elastic energy norm."""
    if position_unit <= 0 or velocity_unit <= 0 or not np.any(mask):
        raise ValueError("Positive units and a nonempty region are required")
    if delta_x.shape != delta_v.shape or delta_x.shape[-2:] != (len(mask), 3):
        raise ValueError("State differences must preserve node IDs and XYZ dimensions")
    return np.sqrt(
        (
            np.square(delta_x[..., mask, :] / position_unit).sum(-1)
            + np.square(delta_v[..., mask, :] / velocity_unit).sum(-1)
        ).mean(-1)
    )


def make_perturbations(pinned, settings, seed):
    """All signs/amplitudes reuse the SAME seeded three-dimensional directions."""
    if tuple(settings["signs"]) != (-1, 1) or settings["direction_count"] < 1:
        raise ValueError("Use paired opposite signs and a positive direction count")
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(settings["direction_count"], len(pinned), 3))
    directions[:, pinned] = 0
    scale = np.sqrt(np.square(directions[:, ~pinned]).sum(-1).mean(-1))
    directions /= scale[:, None, None]
    zeros = np.zeros((len(pinned), 3))
    cases = [
        {"case": "base", "kind": "none", "amplitude": 0.0, "level": -1, "direction": -1, "sign": 0}
    ]
    dx, dv = [zeros.copy()], [zeros.copy()]
    for kind, key in (("position", "position_rms_m"), ("velocity", "velocity_rms_m_s")):
        for level, amplitude in enumerate(settings[key]):
            if amplitude <= 0:
                raise ValueError("Perturbation amplitudes must be positive")
            for direction, vector in enumerate(directions):
                for sign in settings["signs"]:
                    cases.append(
                        {
                            "case": f"{kind}_L{level}_D{direction}_S{sign:+d}",
                            "kind": kind,
                            "amplitude": amplitude,
                            "level": level,
                            "direction": direction,
                            "sign": sign,
                        }
                    )
                    dx.append(sign * amplitude * vector if kind == "position" else zeros.copy())
                    dv.append(sign * amplitude * vector if kind == "velocity" else zeros.copy())
    return cases, np.asarray(dx), np.asarray(dv), directions


class NativeRecoveryReplay:
    """Restore ONE initial integration state, perturb slide DOFs, then integrate.

    No source trajectory or future soft state is accepted by rollout(). The
    known constant robot command is the same in all compared simulations.
    """

    def __init__(self, scene: Path, mesh: dict):
        self.adapter = NativeCurrentContact(scene, mesh)
        self.mj, self.model = self.adapter.mj, self.adapter.model
        self.mesh = mesh
        self.ids = np.array(list(self.adapter.node_dofs))
        self.q_addresses = np.asarray([self.adapter.node_dofs[i][0] for i in self.ids])
        self.v_addresses = np.asarray([self.adapter.node_dofs[i][1] for i in self.ids])

    def rollout(self, initial_integration_state, command, delta_x, delta_v, steps, record_dt):
        from .data.mujoco_soft import node_velocities, tetra_volumes

        if steps < 1:
            raise ValueError("Positive rollout length required")
        pinned = self.mesh["pinned_mask"]
        expected_shape = (len(pinned), 3)
        if (
            delta_x.shape != expected_shape
            or delta_v.shape != expected_shape
            or not np.isfinite(delta_x).all()
            or not np.isfinite(delta_v).all()
        ):
            raise ValueError("Finite stable-ID perturbations required")
        if np.any(delta_x[pinned]) or np.any(delta_v[pinned]):
            raise ValueError("Do not perturb fixed bottom nodes")
        stride = round(record_dt / self.model.opt.timestep)
        if not np.isclose(stride * self.model.opt.timestep, record_dt, atol=1e-12, rtol=0):
            raise ValueError("Recording dt must be a multiple of the native solver step")
        data = self.mj.MjData(self.model)
        self.mj.mj_setState(
            self.model, data, initial_integration_state, self.mj.mjtState.mjSTATE_INTEGRATION
        )
        data.qpos[self.q_addresses] += delta_x[self.ids]
        data.qvel[self.v_addresses] += delta_v[self.ids]
        initial_time = data.time
        positions, velocities, robot_positions, robot_velocities = [], [], [], []
        max_contact, max_pin, max_velocity_reader_error, min_volume = 0, 0.0, 0.0, np.inf
        for step in range(steps + 1):
            data.ctrl[:] = command
            self.mj.mj_forward(self.model, data)
            x = data.flexvert_xpos.copy()
            v = np.zeros_like(x)
            # Each movable vertex is carried by three axis-aligned translational
            # slides. This is INSTANTANEOUS velocity, checked against mj_jac.
            v[self.ids] = data.qvel[self.v_addresses]
            if step in (0, steps):
                max_velocity_reader_error = max(
                    max_velocity_reader_error,
                    float(np.abs(v - node_velocities(self.model, data)).max()),
                )
            if not np.isfinite(x).all() or not np.isfinite(v).all():
                raise FloatingPointError("Native replay produced a non-finite soft state")
            if abs(data.time - initial_time - step * record_dt) > 1e-9:
                raise RuntimeError("Native replay time drift/reset")
            max_pin = max(
                max_pin,
                float(np.abs(x[pinned] - self.mesh["rest_positions_world"][pinned]).max()),
                float(np.abs(v[pinned]).max()),
            )
            ratios = (
                tetra_volumes(x, self.mesh["tetrahedra"]) / self.mesh["rest_signed_tetra_volumes"]
            )
            min_volume = min(min_volume, float(ratios.min()))
            max_contact = max(max_contact, data.ncon)
            positions.append(x)
            velocities.append(v)
            robot_positions.append(data.qpos[self.adapter.robot_q].copy())
            robot_velocities.append(data.qvel[self.adapter.robot_v].copy())
            if step < steps:
                for _ in range(stride):
                    data.ctrl[:] = command
                    self.mj.mj_step(self.model, data)
                    max_contact = max(max_contact, data.ncon)
        warnings = [(i, int(w.number)) for i, w in enumerate(data.warning) if w.number]
        if warnings or min_volume <= 0:
            raise RuntimeError(
                f"Native replay warnings/inverted elements: {warnings}, {min_volume}"
            )
        return (
            np.asarray(positions),
            np.asarray(velocities),
            {
                "native_internal_steps": steps * stride,
                "max_contact_count_including_internal_steps": max_contact,
                "max_fixed_position_or_velocity_error": max_pin,
                "nodal_velocity_reader_vs_jacobian_error": max_velocity_reader_error,
                "minimum_tetra_volume_ratio": min_volume,
                "warnings": warnings,
                "robot_positions": np.asarray(robot_positions),
                "robot_velocities": np.asarray(robot_velocities),
            },
        )
