"""Trainable finite-time rollout, conditional on stop-gradient native contacts.

The Gaussian x/v state, recomputed state features, spatial kernels and neural
updates keep their PyTorch graph. Only the native current-contact solve sees
detached x/v. This is conditional differentiation, NOT a differentiable MuJoCo
solver or a total derivative through contact creation/force changes.
"""

from __future__ import annotations

from dataclasses import replace

import torch


def signed_volume_ratios(positions, tetrahedra, rest_signed_volumes):
    vertices = positions[..., tetrahedra, :]
    a, b, c, d = vertices.unbind(-2)
    volume = ((b - a) * torch.cross(c - a, d - a, dim=-1)).sum(-1) / 6
    return volume / rest_signed_volumes


def mesh_validity_penalty(positions, tetrahedra, rest_signed_volumes, threshold=0.1):
    ratio = signed_volume_ratios(positions, tetrahedra, rest_signed_volumes)
    return torch.relu(threshold - ratio).square().mean()


class DifferentiableContactRollout:
    def __init__(self, legacy, model, conditions, encoder, *, method="mujoco_force"):
        self.legacy, self.model, self.conditions, self.encoder = legacy, model, conditions, encoder
        self.method = method
        self.rest, self.pinned = legacy.rest, legacy.pinned
        self.template = legacy.gaussian_state(
            self.rest.numpy(), torch.zeros_like(self.rest).numpy()
        )
        self.query_count = 0
        self.last_features = None
        self.last_context = None

    def constrain(self, x, v):
        return (
            torch.where(self.pinned[None, :, None], self.rest[None], x),
            torch.where(self.pinned[None, :, None], 0.0, v),
        )

    def state_features(self, x, v):
        norm = self.legacy.training_config["normalization"]
        batch = len(x)
        return torch.cat(
            (
                (self.rest / norm["rest_position"])[None].expand(batch, -1, -1),
                (x - self.rest) / norm["displacement"],
                v / norm["velocity"],
                self.pinned[None, :, None].expand(batch, -1, -1).to(x.dtype),
            ),
            dim=-1,
        )

    def step(self, x, v, robots):
        if len(x) != len(robots):
            raise ValueError("One prescribed current indenter per batch item required")
        x, v = self.constrain(x, v)
        features = self.state_features(x, v)
        contexts, gates = [], []
        norm = self.legacy.training_config["normalization"]
        for i, robot in enumerate(robots):
            attached = replace(self.template, centers=x[i], velocities=v[i], features=features[i])
            # Native adapter is intentionally a stop-gradient condition. Never
            # replace `attached` or the next x/v with this detached solver copy.
            with torch.no_grad():
                detached = replace(
                    attached,
                    centers=x[i].detach(),
                    velocities=v[i].detach(),
                    features=features[i].detach(),
                )
                contacts, forces = self.conditions.condition(detached, robot, self.method)
                self.query_count += 1
            action = torch.cat(
                (
                    x.new_tensor(robot.target_position) / norm["action_position"],
                    x.new_tensor(robot.target_velocity) / norm["action_velocity"],
                    x.new_tensor(robot.position) / norm["action_position"],
                    x.new_tensor(robot.velocity) / norm["action_velocity"],
                )
            )
            context, gate = self.encoder.encode_context(
                attached,
                contacts,
                forces,
                action,
                self.legacy.material_features,
            )
            contexts.append(context.float())
            gates.append(gate.float())
        context, gate = torch.stack(contexts), torch.stack(gates)
        self.last_features, self.last_context = features, context
        increment = self.model(context, gate, features.float(), v.float())
        next_x, next_v = self.constrain(
            x + increment.delta_position.double(),
            v + increment.delta_velocity.double(),
        )
        if not torch.isfinite(next_x).all() or not torch.isfinite(next_v).all():
            raise FloatingPointError("Non-finite attached autoregressive state")
        return next_x, next_v

    def rollout(self, initial_x, initial_v, external_steps, *, teacher_states=None):
        """Only AR initialization is factual; TF A explicitly resets each step.

        `external_steps[t]` is a list of current robot inputs, one per batch.
        Future soft states are not accepted at all in the autoregressive case.
        """
        x, v = self.constrain(initial_x, initial_v)
        states = [(x, v)]
        if teacher_states is not None and len(teacher_states) != len(external_steps):
            raise ValueError("Teacher forcing requires one factual current state per endpoint")
        for t, robots in enumerate(external_steps):
            current_x, current_v = (x, v) if teacher_states is None else teacher_states[t]
            x, v = self.step(current_x, current_v, robots)
            states.append((x, v))
        return states
