"""Frozen joint model with repaired geometry and an explicit face-load adapter.

Primary M/F/G compare native force, native geometry + point force, and repaired
topology-assisted geometry + the SAME point force. Area traction is a separate
diagnostic, lumped on original faces before the unchanged historical encoder.
This adapter is invariant to integration sample count, NOT to mesh partition.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .contact_force import ContactMaterial, compute_contact_forces
from .contact_integration import ClippedSurfaceQuadrature, IntegrationSettings
from .continuous_contact_surface import ContinuousContactSurface, triangle_closest_weights
from .elastic_area import ElasticAreaMaterial
from .joint_state_dynamics import CurrentJointInputs, JointStatePipeline
from .structures import ContactForces, ContactGeometry
from .surface_proxies import BoundSurfaceGaussians

CONTACT_METHODS = (
    "mujoco_force",
    "analytic_force",
    "gaussian_geometry",
    "area_geometry",
    "sparse_geometry",
)
METHODS = (*CONTACT_METHODS, "internal_only", "constant_velocity")


def lumped_area_contacts(state, robot, mesh, basis, radius_m, skin_m):
    """Exact per-face resultant line, with material-point instantaneous velocity.

    On a planar face, h=(vertex-c).N is constant and
    W=integral(delta/r dA)=(I.N)/h. q_eff=c+I/W lies on that face and the
    resultant's radial line, preserving both force and torque. This is a
    pressure-weighted representative, not a new dynamic node.
    """
    ids = basis["face_indices"]
    if not len(ids):
        return ContactGeometry.empty(state.device, state.dtype)
    triangles = torch.as_tensor(mesh["surface_triangles"], device=state.device)[ids]
    vertices = state.centers[triangles]
    normals_plane = torch.cross(
        vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1
    )
    normals_plane = normals_plane / normals_plane.norm(dim=-1, keepdim=True)
    center = state.centers.new_tensor(robot.position)
    h = ((vertices[:, 0] - center) * normals_plane).sum(-1)
    if torch.any(h.abs() <= 1e-12):
        raise ValueError("Area resultant requires sphere center off each active face plane")
    vector = basis["face_elastic_vectors_m3"]
    radial_weight = (vector * normals_plane).sum(-1) / h
    if torch.any(radial_weight <= 0) or not torch.isfinite(radial_weight).all():
        raise FloatingPointError("Invalid positive face traction measure")
    material_point = center + vector / radial_weight[:, None]
    bary = triangle_closest_weights(vertices, material_point)
    projected = (vertices * bary[..., None]).sum(1)
    if torch.any((projected - material_point).norm(dim=-1) > 1e-10):
        raise ValueError("Area resultant left its bounded material face")
    velocity = (state.velocities[triangles] * bary[..., None]).sum(1)
    offset = material_point - center
    distance = offset.norm(dim=-1)
    normal = offset / distance[:, None]
    gap = distance - radius_m - skin_m
    if torch.any(gap > 1e-12):
        raise ValueError("Area resultant is outside contact domain")
    points = 0.5 * (material_point - skin_m * normal + center + radius_m * normal)
    anchor = triangles.gather(1, bary.argmax(1)[:, None]).squeeze(1)
    return ContactGeometry(
        anchor,
        torch.zeros_like(anchor),
        points,
        normal,
        gap,
        (-gap).clamp_min(0),
        state.centers.new_tensor(robot.velocity) - velocity,
    )


class RepairedContactConditions:
    def __init__(self, legacy, mesh, point_stiffness_N_m, area_material, integration_settings):
        self.legacy, self.mesh = legacy, mesh
        self.point_material = ContactMaterial(
            normal_stiffness=point_stiffness_N_m,
            normal_damping=0.0,
            tangent_damping=0.0,
            friction_coefficient=0.0,
        )
        self.area_material: ElasticAreaMaterial = area_material
        self.surface = ContinuousContactSurface(
            BoundSurfaceGaussians(
                mesh["rest_positions_world"],
                mesh["surface_triangles"],
                mesh["pinned_mask"],
                max_rest_edge_m=0.002,
                skin_radius_m=0.001,
            )
        )
        self.integration = ClippedSurfaceQuadrature(
            mesh["rest_positions_world"],
            mesh["surface_triangles"],
            mesh["pinned_mask"],
            IntegrationSettings(**integration_settings),
        )
        self.integration_diagnostics = []

    def condition(self, state, robot, method):
        if method not in CONTACT_METHODS:
            raise ValueError(f"Unknown repaired contact condition: {method}")
        # Defend the boundary even when called outside the joint pipeline.
        state = self.legacy.gaussian_state(state.centers.numpy(), state.velocities.numpy())
        origin = self.legacy.center_of_mass(state)
        if method in ("mujoco_force", "analytic_force"):
            contacts, native = self.legacy.native.query(
                state,
                robot,
                with_force=(method == "mujoco_force"),
                velocity_mode="material_surface",
            )
            if method == "mujoco_force":
                force = torch.as_tensor(native, dtype=state.dtype)
                normal = (force * contacts.normals).sum(-1).clamp_min(0)[:, None] * contacts.normals
                return contacts, ContactForces(
                    force,
                    torch.cross(contacts.points - origin, force, dim=-1),
                    normal,
                    force - normal,
                )
            assert native is None
        elif method == "sparse_geometry":
            contacts = self.legacy.gaussian_contacts(state, robot)
        elif method == "gaussian_geometry":
            query = self.surface.query(
                state.centers,
                state.velocities,
                state.centers.new_tensor(robot.position),
                state.centers.new_tensor(robot.velocity),
                self.legacy.config["sphere_radius_m"],
            )
            triangles = torch.as_tensor(self.mesh["surface_triangles"])[query.face_indices]
            anchor = triangles.gather(1, query.face_weights.argmax(1)[:, None]).squeeze(1)
            contacts = replace(query.face_candidates, object_indices=anchor)
        else:
            basis = self.integration.normal_bases(
                state.centers,
                state.velocities,
                state.centers.new_tensor(robot.position),
                state.centers.new_tensor(robot.velocity),
                self.legacy.config["sphere_radius_m"],
                self.surface.binding.skin_radius_m,
                origin,
            )
            self.integration_diagnostics.append(
                {
                    key: basis[key]
                    for key in (
                        "positive_contact_faces",
                        "zero_integral_positive_faces",
                        "unresolved_area_intervals",
                        "unresolved_elastic_intervals",
                        "unresolved_damping_intervals",
                        "evaluated_samples",
                    )
                }
            )
            contacts = lumped_area_contacts(
                state,
                robot,
                self.mesh,
                basis,
                self.legacy.config["sphere_radius_m"],
                self.surface.binding.skin_radius_m,
            )
            force = self.area_material.stiffness_N_m3 * basis["face_elastic_vectors_m3"]
            torque = torch.cross(contacts.points - origin, force, dim=-1)
            expected_force, expected_torque = self.area_material.predict(
                basis["elastic_vector_m3"],
                basis["elastic_torque_m4"],
            )
            torch.testing.assert_close(force.sum(0), expected_force, rtol=1e-12, atol=1e-15)
            torch.testing.assert_close(torque.sum(0), expected_torque, rtol=1e-12, atol=1e-15)
            return contacts, ContactForces(force, torque, force.clone(), torch.zeros_like(force))
        return contacts, compute_contact_forces(contacts, origin, self.point_material)


class RepairedJointPipeline(JointStatePipeline):
    """Only replaces current conditioning; network/update/normalization unchanged."""

    def __init__(self, legacy, model, conditions, method):
        if method not in CONTACT_METHODS:
            raise ValueError("Unknown repaired joint method")
        super().__init__(legacy, model, contact_method="none")
        self.conditions, self.method = conditions, method

    def current_inputs(self, state, robot):
        state = self.legacy.gaussian_state(state.centers.numpy(), state.velocities.numpy())
        contacts, forces = self.conditions.condition(state, robot, self.method)
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
            state,
            contacts,
            forces,
            action,
            self.legacy.material_features,
        )
        return CurrentJointInputs(
            context.float(),
            gate.float(),
            state.features.float(),
            state.velocities.float(),
            contacts,
            forces,
        )


def repaired_pipelines(legacy, model, conditions):
    return {
        **{m: RepairedJointPipeline(legacy, model, conditions, m) for m in CONTACT_METHODS},
        "internal_only": JointStatePipeline(legacy, model, contact_method="none"),
    }
