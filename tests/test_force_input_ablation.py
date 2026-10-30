from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from test_contact_mechanism_audit import audit_scene as audit_scene
from test_contact_mechanism_audit import touching_robot

from contact_gwm.elastic_area import ElasticAreaMaterial
from contact_gwm.force_input_ablation import (
    FORCE_CONTEXT_SLICE,
    geometry_only_pipeline,
    geometry_only_training_path,
    without_force_cache,
)
from contact_gwm.repaired_contact_pipeline import RepairedContactConditions, RepairedJointPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bounded_search_training import save_milestone  # noqa: E402


def forbidden(*args, **kwargs):
    raise AssertionError("G/G0 may not query native contacts or true forces")


def test_cached_force_removal_changes_only_force_channels_and_not_source():
    context = torch.arange(2 * 5 * 36, dtype=torch.float32).reshape(2, 5, 36)
    cache = {"context": context, "gates": torch.ones(2, 5), "target_dx": torch.ones(2, 5, 3)}
    before = context.clone()
    result = without_force_cache(cache)
    assert torch.equal(context, before)
    assert not result["context"][..., FORCE_CONTEXT_SLICE].any()
    keep = torch.ones(36, dtype=torch.bool)
    keep[FORCE_CONTEXT_SLICE] = False
    assert torch.equal(result["context"][..., keep], context[..., keep])
    assert result["gates"] is cache["gates"]
    assert result["target_dx"] is cache["target_dx"]


def test_live_G0_preserves_G_geometry_gates_and_every_nonforce_input(audit_scene):
    _, _, mesh, _, legacy, model = audit_scene
    conditions = RepairedContactConditions(legacy, mesh, 500.0, ElasticAreaMaterial(6e7), {})
    g = RepairedJointPipeline(legacy, model, conditions, "gaussian_geometry")
    g0 = geometry_only_pipeline(legacy, model, conditions)
    state = legacy.gaussian_state(legacy.rest.numpy(), torch.zeros_like(legacy.rest).numpy())
    with patch.object(legacy.native, "query", forbidden):
        full = g.current_inputs(state, touching_robot())
        removed = g0.current_inputs(state, touching_robot())
    assert full.contacts.num_contacts > 0
    for key in full.contacts.__dataclass_fields__:
        assert torch.equal(getattr(full.contacts, key), getattr(removed.contacts, key))
    assert torch.equal(full.gates, removed.gates)
    assert full.context[..., FORCE_CONTEXT_SLICE].abs().sum() > 0
    assert not removed.context[..., FORCE_CONTEXT_SLICE].any()
    keep = torch.ones(36, dtype=torch.bool)
    keep[FORCE_CONTEXT_SLICE] = False
    assert torch.equal(full.context[..., keep], removed.context[..., keep])
    assert not removed.forces.forces.any()


def test_G0_attached_gradient_boundary_and_inference_consistency(audit_scene):
    _, _, mesh, _, legacy, model = audit_scene
    conditions = RepairedContactConditions(legacy, mesh, 500.0, ElasticAreaMaterial(6e7), {})
    pipeline = geometry_only_pipeline(legacy, model, conditions)
    path = geometry_only_training_path(legacy, model, conditions, pipeline.encoder)
    x = legacy.rest.clone()
    x[~legacy.pinned, 2] -= 0.0001
    v = torch.zeros_like(x)
    v[~legacy.pinned, 2] = 0.0003
    with patch.object(legacy.native, "query", forbidden):
        states = path.rollout(
            x[None].requires_grad_(True), v[None].requires_grad_(True), [[touching_robot()]] * 3
        )
        output = pipeline.step(legacy.gaussian_state(x.numpy(), v.numpy()), touching_robot())
    assert torch.equal(states[1][0][0].detach(), output.state.centers)
    assert torch.equal(states[1][1][0].detach(), output.state.velocities)
    loss = states[-1][0].square().sum() + states[-1][1].square().sum()
    gx, gv = torch.autograd.grad(loss, states[1], retain_graph=True)
    assert torch.isfinite(gx).all() and torch.isfinite(gv).all()
    assert gx.norm() > 0 and gv.norm() > 0
    loss.backward()
    assert not model.contact_network[0].weight.grad[:, FORCE_CONTEXT_SLICE].any()
    for px, pv in states:
        assert torch.equal(px[0, legacy.pinned], legacy.rest[legacy.pinned])
        assert not pv[:, legacy.pinned].any()
    with pytest.raises(TypeError):
        path.rollout(x[None], v[None], [[touching_robot()]], future_forces=torch.ones(1, 3))


def test_milestone_exact_and_prefix_weights_are_distinct_and_prefix_has_no_future(tmp_path):
    model = torch.nn.Linear(2, 1)
    best = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight.add_(1)
    current = copy.deepcopy(model.state_dict())
    save_milestone(tmp_path, 200, model, best, 50, 0.1)
    exact = torch.load(tmp_path / "exact_0200.pt", weights_only=False)
    prefix = torch.load(tmp_path / "prefix_0200.pt", weights_only=False)
    assert torch.equal(exact["model_state"]["weight"], current["weight"])
    assert torch.equal(prefix["model_state"]["weight"], best["weight"])
    assert prefix["selected_step"] == 50 and prefix["prefix_limit"] == 200
    with pytest.raises(ValueError, match="future"):
        save_milestone(tmp_path, 200, model, best, 500, 0.1)
