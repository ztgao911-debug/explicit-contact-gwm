"""Geometry/gate-preserving force-input removal, for fair fresh G0 training."""

from __future__ import annotations

import torch

from .differentiable_rollout import DifferentiableContactRollout
from .repaired_contact_pipeline import RepairedJointPipeline
from .structures import ContactForces

FORCE_CONTEXT_SLICE = slice(11, 14)  # state10 + gate1, then three log-force channels


class GeometryOnlyConditions:
    """Keep G contact candidates verbatim, remove all force values before encoding.

    The existing G query still computes its analytic force before it is discarded;
    this is an input contribution ablation, not a compute-speed optimization.
    """

    def __init__(self, original):
        self.original = original

    def condition(self, state, robot, method):
        if method != "gaussian_geometry":
            raise ValueError("G0 only uses the frozen G geometry")
        contacts, forces = self.original.condition(state, robot, method)
        zero = torch.zeros_like(forces.forces)
        return contacts, ContactForces(zero, zero.clone(), zero.clone(), zero.clone())


def geometry_only_pipeline(legacy, model, conditions):
    return RepairedJointPipeline(
        legacy, model, GeometryOnlyConditions(conditions), "gaussian_geometry"
    )


def geometry_only_training_path(legacy, model, conditions, encoder):
    return DifferentiableContactRollout(
        legacy, model, GeometryOnlyConditions(conditions), encoder, method="gaussian_geometry"
    )


def without_force_cache(cache):
    """Copy just the context tensor; never mutate the preserved G input cache."""
    if cache["context"].shape[-1] != 36:
        raise ValueError("Preserve the existing 36-channel context")
    context = cache["context"].clone()
    context[..., FORCE_CONTEXT_SLICE] = 0
    return {**cache, "context": context}
