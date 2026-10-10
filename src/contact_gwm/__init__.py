"""Explicit contact-driven Gaussian world model."""

from .contact_force import ContactMaterial, compute_contact_forces, integrate_rigid_body
from .contact_geometry import compute_gaussian_contacts
from .deformation import ContactGatedDeformation
from .structures import ContactForces, ContactGeometry, GaussianSet, RigidBodyState
from .world_model import ExplicitContactGaussianWorldModel, WorldModelOutput

__all__ = [
    "ContactForces",
    "ContactGatedDeformation",
    "ContactGeometry",
    "ContactMaterial",
    "ExplicitContactGaussianWorldModel",
    "GaussianSet",
    "RigidBodyState",
    "WorldModelOutput",
    "compute_contact_forces",
    "compute_gaussian_contacts",
    "integrate_rigid_body",
]

