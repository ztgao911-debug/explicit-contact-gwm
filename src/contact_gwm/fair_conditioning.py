"""Same-network RAW versus G conditioning with identical current input permissions.

RAW encodes current states and kinematic offsets only; it never queries contact,
force, penetration, a surface normal, or a learned node embedding. Its external
branch gate is one, so learning interaction is possible without a contact gate.
The G arm keeps the existing topology-assisted contact encoder and analytic force.
"""

from __future__ import annotations

import torch

from .differentiable_rollout import DifferentiableContactRollout
from .joint_state_dynamics import CurrentJointInputs, JointStatePipeline
from .structures import ContactForces, ContactGeometry


class RawKinematicEncoder:
    """36 channels: 10 state, 12 tool, 12 relative kinematics, radius and one.

    No padding to create nominal capacity: relative channels are deterministic
    transforms of the same current input. Redundant features and a constant are
    disclosed; equal parameter counts do not imply equal optimization difficulty.
    """

    def __init__(self, normalization, radius_m):
        self.norm = normalization
        self.radius_m = radius_m

    def encode_context(self, state, contacts, forces, action, material_features):
        if action.shape != (12,):
            raise ValueError("Preserve command position/velocity then actual position/velocity")
        # `contacts`, `forces` and material are deliberately never inspected.
        norm, x, v = self.norm, state.centers, state.velocities
        tool_x = action[6:9] * norm["action_position"]
        tool_v = action[9:12] * norm["action_velocity"]
        command_x = action[:3] * norm["action_position"]
        command_v = action[3:6] * norm["action_velocity"]
        n = len(x)
        context = torch.cat(
            (
                state.features,
                action[None].expand(n, -1),
                (tool_x - x) / norm["rest_position"],
                (tool_v - v) / norm["velocity"],
                (command_x - x) / norm["rest_position"],
                (command_v - v) / norm["velocity"],
                x.new_full((n, 1), self.radius_m / norm["rest_position"]),
                x.new_ones(n, 1),
            ),
            dim=-1,
        )
        if context.shape != (n, 36):
            raise ValueError("RAW must preserve the same 36-channel external MLP interface")
        return context, x.new_ones(n)


class EmptyContactConditions:
    """No simulator or geometric contact access in the RAW forward path."""

    def condition(self, state, robot, method):
        if method != "raw":
            raise ValueError("RAW adapter only accepts raw current input")
        return ContactGeometry.empty(state.device, state.dtype), ContactForces.empty(
            state.device, state.dtype
        )


class RawJointPipeline(JointStatePipeline):
    def __init__(self, legacy, model):
        super().__init__(legacy, model, contact_method="none")
        self.encoder = RawKinematicEncoder(
            legacy.training_config["normalization"], legacy.config["sphere_radius_m"]
        )

    def current_inputs(self, state, robot):
        state = self.legacy.gaussian_state(state.centers.numpy(), state.velocities.numpy())
        norm = self.legacy.training_config["normalization"]
        action = torch.cat(
            (
                torch.as_tensor(robot.target_position) / norm["action_position"],
                torch.as_tensor(robot.target_velocity) / norm["action_velocity"],
                torch.as_tensor(robot.position) / norm["action_position"],
                torch.as_tensor(robot.velocity) / norm["action_velocity"],
            )
        )
        contacts, forces = EmptyContactConditions().condition(state, robot, "raw")
        context, gates = self.encoder.encode_context(state, contacts, forces, action, None)
        return CurrentJointInputs(
            context.float(),
            gates.float(),
            state.features.float(),
            state.velocities.float(),
            contacts,
            forces,
        )


def training_path(variant, legacy, model, conditions, contact_encoder):
    if variant == "RAW":
        return DifferentiableContactRollout(
            legacy,
            model,
            EmptyContactConditions(),
            RawKinematicEncoder(
                legacy.training_config["normalization"], legacy.config["sphere_radius_m"]
            ),
            method="raw",
        )
    if variant == "G":
        return DifferentiableContactRollout(
            legacy, model, conditions, contact_encoder, method="gaussian_geometry"
        )
    raise ValueError("Fair comparison only trains RAW and G")
