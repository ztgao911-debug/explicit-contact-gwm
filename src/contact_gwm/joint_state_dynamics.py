"""Joint discrete position/instantaneous-velocity prediction with internal recovery.

The internal branch sees current soft state even without contact. The contact
branch alone is contact-gated. Both outputs are independently supervised; no
endpoint instantaneous velocity is inferred from a position secant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .local_training import ConstantVelocityResidualDeformation
from .pinned_contact_pipeline import IndenterInput, PinnedContactPipeline, StepResult
from .spatial_ablation import fixed_rest_encoding
from .structures import ContactForces, ContactGeometry, GaussianSet


@dataclass(frozen=True)
class JointIncrement:
    delta_position: Tensor
    delta_velocity: Tensor
    internal_position: Tensor
    internal_velocity: Tensor
    contact_position: Tensor
    contact_velocity: Tensor


class JointStateDynamics(nn.Module):
    """Two equal-width MLPs; autonomous=False is a same-capacity gated control.

    All six output channels now have supervised meaning: position residual (m)
    and instantaneous-velocity increment (m/s). The old unused scale head is
    not reused as a velocity label implicitly; this is a NEW model trained from
    scratch. Internal reference subtraction enforces the rest equilibrium.
    """

    def __init__(
        self,
        rest_positions: Tensor,
        pinned_mask: Tensor,
        hidden_dim: int = 128,
        dt: float = 0.01,
        contact_sigma: float = 0.02,
        max_position_residual: float = 0.001,
        max_velocity_increment: float = 0.1,
        autonomous: bool = True,
    ) -> None:
        super().__init__()
        if abs(dt - 0.01) > 1e-12 or hidden_dim < 1:
            raise ValueError("Use dt=0.01 s and a positive hidden width")
        if contact_sigma <= 0 or min(max_position_residual, max_velocity_increment) <= 0:
            raise ValueError("Contact sigma and output scales must be positive")
        self.dt = dt
        self.contact_sigma = contact_sigma
        self.max_position_residual = max_position_residual
        self.max_velocity_increment = max_velocity_increment
        self.autonomous = autonomous
        self.register_buffer("pinned_mask", pinned_mask.detach().bool().clone())
        self.register_buffer("rest_encoding", fixed_rest_encoding(rest_positions.float()))
        rest_features = torch.cat(
            (
                rest_positions.float() / 0.03,
                torch.zeros((len(rest_positions), 6)),
                pinned_mask[:, None].float(),
            ),
            dim=-1,
        )
        self.register_buffer("rest_features", rest_features)

        def network(input_dim):
            head = nn.Linear(hidden_dim, 6)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                head,
            )

        self.internal_network = network(10 + 18)
        self.contact_network = network(36 + 18)

    def forward(self, context, gates, features, current_velocity) -> JointIncrement:
        count = len(self.pinned_mask)
        if context.shape[-2:] != (count, 36) or features.shape[-2:] != (count, 10):
            raise ValueError("Preserve the stable node order and original current-state features")
        if gates.shape != context.shape[:-1] or current_velocity.shape != features.shape[:-1] + (
            3,
        ):
            raise ValueError("Incompatible current-state or gate dimensions")
        spatial = self.rest_encoding.expand(*features.shape[:-1], 18)
        internal_input = torch.cat((features, spatial), dim=-1)
        equilibrium_input = torch.cat((self.rest_features, self.rest_encoding), dim=-1)
        # Both networks receive only current information. No-contact context may
        # be zero, whereas internal_input ALWAYS retains current displacement/v.
        internal_raw = self.internal_network(internal_input) - self.internal_network(
            equilibrium_input
        )
        contact_raw = self.contact_network(torch.cat((context, spatial), dim=-1))
        internal_gate = 1.0 if self.autonomous else gates[..., None]
        internal = torch.tanh(internal_raw) * internal_gate
        contact = torch.tanh(contact_raw) * gates[..., None]

        def pinned(value):
            return torch.where(self.pinned_mask[..., None], 0.0, value)

        ip = pinned(internal[..., :3] * self.max_position_residual)
        iv = pinned(internal[..., 3:] * self.max_velocity_increment)
        cp = pinned(contact[..., :3] * self.max_position_residual)
        cv = pinned(contact[..., 3:] * self.max_velocity_increment)
        return JointIncrement(pinned(current_velocity * self.dt + ip + cp), iv + cv, ip, iv, cp, cv)


def joint_state_update(
    state: GaussianSet, increment: JointIncrement, rest: Tensor, pinned: Tensor
) -> GaussianSet:
    """A learned finite-time state map; no assumption of constant acceleration.

    This enforces endpoint definitions, NOT a continuous-time force law or
    conservation of elastic energy between samples. Those are unverified here.
    """
    centers = torch.where(pinned[:, None], rest, state.centers + increment.delta_position)
    velocity = torch.where(pinned[:, None], 0.0, state.velocities + increment.delta_velocity)
    if not torch.isfinite(centers).all() or not torch.isfinite(velocity).all():
        raise FloatingPointError("Non-finite joint Gaussian state update")
    return replace(state, centers=centers, velocities=velocity)


@dataclass(frozen=True)
class CurrentJointInputs:
    context: Tensor
    gates: Tensor
    features: Tensor
    velocity: Tensor
    contacts: ContactGeometry
    forces: ContactForces


class JointStatePipeline:
    """The same current-state native contact query for training AND deployment."""

    def __init__(self, legacy: PinnedContactPipeline, model: JointStateDynamics) -> None:
        self.legacy = legacy
        self.model = model
        self.encoder = ConstantVelocityResidualDeformation(
            **legacy.training_config["model"]
        ).deformation

    def current_inputs(self, state: GaussianSet, robot: IndenterInput) -> CurrentJointInputs:
        # No trajectory, current labels or future labels are accepted here.
        state = self.legacy.gaussian_state(state.centers.numpy(), state.velocities.numpy())
        contacts, forces = self.legacy.condition(state, robot, "mujoco_force")
        norm = self.legacy.training_config["normalization"]
        action = torch.cat(
            (
                torch.as_tensor(robot.target_position) / norm["action_position"],
                torch.as_tensor(robot.target_velocity) / norm["action_velocity"],
                torch.as_tensor(robot.position) / norm["action_position"],
                torch.as_tensor(robot.velocity) / norm["action_velocity"],
            )
        )
        context, gate = self.encoder.encode_context(
            state, contacts, forces, action, self.legacy.material_features
        )
        return CurrentJointInputs(
            context.float(),
            gate.float(),
            state.features.float(),
            state.velocities.float(),
            contacts,
            forces,
        )

    @torch.no_grad()
    def step(self, state: GaussianSet, robot: IndenterInput) -> StepResult:
        inputs = self.current_inputs(state, robot)
        increment = self.model(inputs.context, inputs.gates, inputs.features, inputs.velocity)
        # Keep positions in float64; the network and cached training inputs use float32.
        increment = JointIncrement(
            *(getattr(increment, field).double() for field in JointIncrement.__dataclass_fields__)
        )
        next_state = joint_state_update(state, increment, self.legacy.rest, self.legacy.pinned)
        next_state = self.legacy.gaussian_state(
            next_state.centers.numpy(), next_state.velocities.numpy()
        )
        return StepResult(
            next_state,
            increment.delta_position,
            increment.internal_position + increment.contact_position,
            inputs.gates,
            inputs.contacts,
            inputs.forces,
        )

    def rollout(self, initial: GaussianSet, robot_inputs: list[IndenterInput]):
        state, predictions = initial, []
        for robot in robot_inputs:
            result = self.step(state, robot)
            predictions.append(result)
            state = result.state
        return predictions
