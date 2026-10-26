"""Current Gaussian state -> contacts -> forces -> pinned Gaussian state.

No trajectory, future label, or recorded contact array is accepted by step().
The two analytic branches never run the MuJoCo force solver. MuJoCo is only a
current-state geometry/force oracle here; mj_step is deliberately never called.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contact_force import ContactMaterial, compute_contact_forces
from .contact_geometry import compute_gaussian_contacts
from .spatial_ablation import SpatialEncodedResidual
from .structures import ContactForces, ContactGeometry, GaussianSet

METHODS = ("mujoco_force", "analytic_force", "gaussian_geometry", "constant_velocity")


@dataclass(frozen=True)
class IndenterInput:
    """Known external robot motion/command, all at the CURRENT time, in SI units."""

    position: np.ndarray
    velocity: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray

    @classmethod
    def from_arrays(cls, arrays: dict, frame: int) -> IndenterInput:
        # Only these four external arrays can cross the rollout boundary.
        return cls(
            *(
                np.array(arrays[key][frame], copy=True)
                for key in (
                    "indenter_position_world",
                    "indenter_velocity_world",
                    "action_target_position",
                    "action_target_velocity",
                )
            )
        )


@dataclass(frozen=True)
class StepResult:
    state: GaussianSet
    delta: Tensor
    residual: Tensor
    gates: Tensor
    contacts: ContactGeometry
    forces: ContactForces


def constant_acceleration_update(
    state: GaussianSet, residual: Tensor, rest: Tensor, pinned: Tensor, dt: float
) -> tuple[GaussianSet, Tensor]:
    """Interpret residual as 0.5*a*dt**2, NOT as an endpoint velocity.

    This is a declared approximation to the frozen displacement-only network.
    Endpoint instantaneous v_next = v + 2*residual/dt. Fixed nodes have zero
    TOTAL displacement and zero velocity, irrespective of their input velocity.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    residual = torch.where(pinned[:, None], 0.0, residual)
    delta = torch.where(pinned[:, None], 0.0, state.velocities * dt + residual)
    centers = torch.where(pinned[:, None], rest, state.centers + delta)
    velocity = torch.where(pinned[:, None], 0.0, state.velocities + 2 * residual / dt)
    if not torch.isfinite(centers).all() or not torch.isfinite(velocity).all():
        raise FloatingPointError("Non-finite Gaussian state update")
    return replace(state, centers=centers, velocities=velocity), delta


def nonnegative_two_column_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact 2-variable NNLS by checking the interior and both boundary faces."""
    candidates = [np.zeros(2)]
    fit = np.linalg.lstsq(design, target, rcond=None)[0]
    if np.all(fit >= 0):
        candidates.append(fit)
    for column in range(2):
        x = design[:, column]
        candidate = np.zeros(2)
        candidate[column] = max(0.0, float(x @ target / max(float(x @ x), 1e-30)))
        candidates.append(candidate)
    return min(candidates, key=lambda value: float(np.square(design @ value - target).sum()))


class NativeCurrentContact:
    """Reconstruct full-slide flex DOFs from x/v, without any stored solver state."""

    def __init__(self, scene: Path, mesh: dict) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.rest = mesh["rest_positions_world"]
        self.tets = mesh["tetrahedra"]
        self.pinned = mesh["pinned_mask"]
        self.sphere = self.model.geom("sphere").id
        self.node_dofs = {}
        for i, body in enumerate(mesh["vertex_body_ids"]):
            if self.pinned[i]:
                continue
            first = self.model.body_jntadr[body]
            joints = np.arange(first, first + 3)
            if (
                self.model.body_jntnum[body] != 3
                or not np.all(self.model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_SLIDE)
                or not np.allclose(self.model.jnt_axis[joints], np.eye(3))
            ):
                raise ValueError("This adapter requires three axis-aligned slides per free node")
            self.node_dofs[i] = (self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints])
        self.robot_q = np.array([self.model.joint(f"p{a}").qposadr[0] for a in "xyz"])
        self.robot_v = np.array([self.model.joint(f"p{a}").dofadr[0] for a in "xyz"])

    def query(
        self, state: GaussianSet, robot: IndenterInput, *, with_force: bool
    ) -> tuple[ContactGeometry, np.ndarray | None]:
        mj, model, data = self.mj, self.model, self.data
        x = state.centers.detach().cpu().numpy()
        v = state.velocities.detach().cpu().numpy()
        mj.mj_resetData(model, data)  # No true qacc_warmstart/integration state.
        for node, (q, dof) in self.node_dofs.items():
            data.qpos[q] = x[node] - self.rest[node]
            data.qvel[dof] = v[node]
        data.qpos[self.robot_q] = robot.position
        data.qvel[self.robot_v] = robot.velocity
        data.ctrl[:] = robot.target_position
        if with_force:
            mj.mj_forward(model, data)
        else:
            # Includes collision/constraint geometry, but NO velocity/force solve.
            mj.mj_fwdPosition(model, data)
        points, normals, gaps, velocities, indices, forces = [], [], [], [], [], []
        for k in range(data.ncon):
            c = data.contact[k]
            if self.sphere not in c.geom or 0 not in c.flex or c.dist > 0:
                continue
            side = 1 if c.flex[1] == 0 else 0
            sign = 1.0 if side == 1 else -1.0
            normal = sign * c.frame[:3]
            node = int(np.linalg.norm(x - c.pos, axis=1).argmin())
            # Flex element contact: interpolate current nodal velocities in its
            # tetrahedron. Project midpoint barycentrics to a convex combination.
            # The midpoint is slightly off the material surface (collision skin).
            soft_velocity = v[node]
            if c.vert[side] >= 0:
                soft_velocity = v[c.vert[side]]
            elif c.elem[side] >= 0:
                ids = self.tets[c.elem[side]]
                vertices = x[ids]
                basis = (vertices[1:] - vertices[0]).T
                weights = np.linalg.lstsq(basis, c.pos - vertices[0], rcond=None)[0]
                weights = np.maximum(np.r_[1 - weights.sum(), weights], 0.0)
                weights /= max(weights.sum(), 1e-30)
                soft_velocity = weights @ v[ids]
            points.append(c.pos.copy())
            normals.append(normal.copy())
            gaps.append(float(c.dist))
            velocities.append(robot.velocity - soft_velocity)
            indices.append(node)
            if with_force:
                wrench = np.zeros(6)
                mj.mj_contactForce(model, data, k, wrench)
                forces.append(sign * (c.frame.reshape(3, 3).T @ wrench[:3]))
        if not points:
            return ContactGeometry.empty(state.device, state.dtype), (
                np.zeros((0, 3)) if with_force else None
            )

        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=state.dtype)

        gap = tensor(gaps)
        contacts = ContactGeometry(
            torch.tensor(indices),
            torch.zeros(len(indices), dtype=torch.long),
            tensor(points),
            tensor(normals),
            gap,
            (-gap).clamp_min(0),
            tensor(velocities),
        )
        return contacts, np.asarray(forces).reshape(-1, 3) if with_force else None


class PinnedContactPipeline:
    """Frozen spatial-B local network, current contacts and explicit pinned update."""

    def __init__(self, mesh: dict, scene: Path, checkpoint: Path, config: dict) -> None:
        self.config = config
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.training_config = saved["config"]
        if saved["config"]["seed"] != config["seed"]:
            raise ValueError("Keep the existing fixed seed")
        self.dt = config["dt"]
        if self.dt != 0.01:
            raise ValueError("The frozen network requires 0.01 s")
        self.rest = torch.tensor(mesh["rest_positions_world"], dtype=torch.float64)
        self.pinned = torch.tensor(mesh["pinned_mask"], dtype=torch.bool)
        self.mass = torch.tensor(mesh["node_masses"], dtype=torch.float64)
        self.surface_ids = torch.tensor(np.unique(mesh["surface_triangles"]), dtype=torch.long)
        distance = torch.cdist(self.rest, self.rest)
        distance.fill_diagonal_(torch.inf)
        self.log_scales = (0.5 * distance.min(dim=1).values[:, None].expand(-1, 3)).log()
        self.model = SpatialEncodedResidual(self.rest.float(), **saved["config"]["model"])
        self.model.load_state_dict(saved["model_state"], strict=True)
        self.model.eval()
        self.native = NativeCurrentContact(scene, mesh)
        self.material = ContactMaterial(**config["initial_material"])
        self.material_features = torch.tensor(config["material_features"], dtype=torch.float64)

    def gaussian_state(self, positions, velocities) -> GaussianSet:
        x = torch.as_tensor(positions, dtype=torch.float64).detach().clone()
        v = torch.as_tensor(velocities, dtype=torch.float64).detach().clone()
        x = torch.where(self.pinned[:, None], self.rest, x)
        v = torch.where(self.pinned[:, None], 0.0, v)
        norm = self.training_config["normalization"]
        features = torch.cat(
            (
                self.rest / norm["rest_position"],
                (x - self.rest) / norm["displacement"],
                v / norm["velocity"],
                self.pinned[:, None].double(),
            ),
            dim=1,
        )
        rotations = torch.zeros((len(x), 4), dtype=x.dtype)
        rotations[:, 0] = 1
        return GaussianSet(
            x, self.log_scales.clone(), rotations, x.new_zeros((len(x), 1)), features, v
        )

    def center_of_mass(self, state: GaussianSet) -> Tensor:
        # Actual movable-node mass; pinned nodes have zero stored dynamic mass.
        return (state.centers * self.mass[:, None]).sum(0) / self.mass.sum()

    def gaussian_contacts(self, state: GaussianSet, robot: IndenterInput) -> ContactGeometry:
        ids = self.surface_ids
        cloud = GaussianSet(
            *(
                getattr(state, name)[ids]
                for name in (
                    "centers",
                    "log_scales",
                    "rotations",
                    "opacity_logits",
                    "features",
                    "velocities",
                )
            )
        )
        level = self.config["gaussian_surface_level"]
        sphere = GaussianSet(
            torch.tensor(robot.position)[None],
            torch.full((1, 3), self.config["sphere_radius_m"] / level).double().log(),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=state.dtype),
            state.centers.new_zeros((1, 1)),
            state.centers.new_zeros((1, 10)),
            torch.tensor(robot.velocity)[None],
        )
        contacts = compute_gaussian_contacts(cloud, sphere, surface_level=level)
        return replace(contacts, object_indices=ids[contacts.object_indices])

    def condition(self, state, robot, method) -> tuple[ContactGeometry, ContactForces]:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        com = self.center_of_mass(state)
        if method == "constant_velocity":
            return ContactGeometry.empty(state.device, state.dtype), ContactForces.empty(
                state.device, state.dtype
            )
        if method == "gaussian_geometry":
            contacts = self.gaussian_contacts(state, robot)
            return contacts, compute_contact_forces(contacts, com, self.material)
        contacts, native_force = self.native.query(
            state, robot, with_force=(method == "mujoco_force")
        )
        if method == "analytic_force":
            assert native_force is None
            return contacts, compute_contact_forces(contacts, com, self.material)
        force = torch.as_tensor(native_force, dtype=state.dtype)
        normal = (force * contacts.normals).sum(-1, keepdim=True).clamp_min(0) * contacts.normals
        return contacts, ContactForces(
            force, torch.cross(contacts.points - com, force, dim=-1), normal, force - normal
        )

    @torch.no_grad()
    def step(self, state: GaussianSet, robot: IndenterInput, method: str) -> StepResult:
        # Recompute state-dependent features from the CURRENT predicted x/v.
        state = self.gaussian_state(state.centers.numpy(), state.velocities.numpy())
        contacts, forces = self.condition(state, robot, method)
        if method == "constant_velocity":
            residual, gates = (
                torch.zeros_like(state.centers),
                state.centers.new_zeros(len(self.rest)),
            )
        else:
            norm = self.training_config["normalization"]
            action = torch.tensor(
                np.concatenate(
                    (
                        robot.target_position / norm["action_position"],
                        robot.target_velocity / norm["action_velocity"],
                        robot.position / norm["action_position"],
                        robot.velocity / norm["action_velocity"],
                    )
                ),
                dtype=state.dtype,
            )
            # Parameter-free encoder, using the original feature order/norms.
            from .local_training import ConstantVelocityResidualDeformation

            if not hasattr(self, "encoder"):
                self.encoder = ConstantVelocityResidualDeformation(
                    **self.training_config["model"]
                ).deformation
            context, gates = self.encoder.encode_context(
                state, contacts, forces, action, self.material_features
            )
            raw, _ = self.model.deformation.decode_context(
                self.model.augment_context(context.float())
            )
            residual = torch.where(self.pinned[:, None], 0.0, gates[:, None] * raw.double())
        next_state, delta = constant_acceleration_update(
            state, residual, self.rest, self.pinned, self.dt
        )
        next_state = self.gaussian_state(next_state.centers.numpy(), next_state.velocities.numpy())
        if not all(
            torch.isfinite(value).all()
            for value in (delta, residual, gates, contacts.points, contacts.normals, forces.forces)
        ):
            raise FloatingPointError("Non-finite pipeline prediction")
        return StepResult(next_state, delta, residual, gates, contacts, forces)

    def rollout(self, initial: GaussianSet, robot_inputs: list[IndenterInput], method: str):
        """Only initial soft state plus prescribed external robot inputs are accepted."""
        state = initial
        results = []
        for robot in robot_inputs:
            result = self.step(state, robot, method)
            results.append(result)
            state = result.state
        return results
