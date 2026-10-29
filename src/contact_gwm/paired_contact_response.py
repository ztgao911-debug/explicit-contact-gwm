"""Current-input paired response supervision and train-only output-bound audit."""

from __future__ import annotations

import math

import torch


def predict_cache(model, cache, indices):
    return model(*(cache[key][indices] for key in ("context", "gates", "features", "velocity")))


def normalized_state_loss(dx, dv, target_dx, target_dv, neighborhood, pinned, settings):
    active = ~pinned
    components = []
    for prediction, target, unit in (
        (dx, target_dx, settings["position_loss_unit_m"]),
        (dv, target_dv, settings["velocity_loss_unit_m_s"]),
    ):
        error = ((prediction - target) / unit).square()
        a = error[:, active].mean()
        b = (error * neighborhood[..., None]).sum() / (3 * neighborhood.sum()).clamp_min(1)
        w = settings["neighborhood_weight"]
        components.append((1 - w) * a + w * b)
    w = settings["velocity_weight"]
    return (1 - w) * components[0] + w * components[1]


def absolute_loss(model, cache, indices, settings):
    prediction = predict_cache(model, cache, indices)
    return normalized_state_loss(
        prediction.delta_position,
        prediction.delta_velocity,
        cache["target_dx"][indices],
        cache["target_dv"][indices],
        cache["neighborhood"][indices],
        cache["pinned"],
        settings,
    )


def paired_loss(model, cache, pairs, indices, settings):
    """Future x/v differences are labels only; neither forward receives them."""
    base = predict_cache(model, cache, pairs["base_index"][indices])
    changed = predict_cache(model, cache, pairs["variant_index"][indices])
    return normalized_state_loss(
        changed.delta_position - base.delta_position,
        changed.delta_velocity - base.delta_velocity,
        pairs["target_dx"][indices],
        pairs["target_dv"][indices],
        pairs["neighborhood"][indices],
        cache["pinned"],
        settings,
    )


def output_bound_audit(cache, pairs, position_cap, velocity_cap, dt=0.01):
    """Necessary component bounds, not a claim that the MLP can fit every label.

    Single-state residual <= cap*(1+gate). In an equal-soft-state initial pair,
    internal and CV terms cancel, so response <= cap*(gate_base+gate_variant).
    No sample is removed if the model cannot represent its target.
    """
    active = ~cache["pinned"]
    absolute_denominator = 1 + cache["gates"][:, active, None]
    response_denominator = (
        cache["gates"][pairs["base_index"]] + cache["gates"][pairs["variant_index"]]
    )[:, active, None]
    result = {}
    for name, cap, absolute, response, tolerance in (
        (
            "position",
            position_cap,
            cache["target_dx"] - cache["velocity"] * dt,
            pairs["target_dx"],
            1e-9,
        ),
        ("velocity", velocity_cap, cache["target_dv"], pairs["target_dv"], 1e-7),
    ):
        a, b = absolute[:, active].abs(), response[:, active].abs()
        nonzero = response_denominator > 1e-8
        ratio = torch.where(nonzero, b / response_denominator.clamp_min(1e-8), 0.0)
        result[name] = {
            "absolute_max_required_cap": float((a / absolute_denominator).max()),
            "pair_max_required_cap_nonzero_gate": float(ratio.max()),
            "absolute_violating_components": int(
                (a > cap * absolute_denominator + tolerance).sum()
            ),
            "pair_violating_components": int((b > cap * response_denominator + tolerance).sum()),
            "zero_gate_pair_nonzero_target_components": int(((~nonzero) & (b > tolerance)).sum()),
            "absolute_components": a.numel(),
            "pair_components": b.numel(),
        }
    return result


def central_derived_caps(audit, original_position=0.001, original_velocity=0.1):
    """Predeclared 25% slack, round UP by powers of two; central labels only."""
    caps = {}
    for name, original, key in (
        ("position", original_position, "max_position_residual"),
        ("velocity", original_velocity, "max_velocity_increment"),
    ):
        required = max(
            audit[name]["absolute_max_required_cap"],
            audit[name]["pair_max_required_cap_nonzero_gate"],
        )
        exponent = max(0, math.ceil(math.log2(max(original, 1.25 * required) / original)))
        caps[key] = original * 2**exponent
    return caps
