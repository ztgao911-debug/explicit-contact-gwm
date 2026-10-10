"""Explicit contact-driven Gaussian world model."""

from .contact_force import ContactMaterial, compute_contact_forces, integrate_rigid_body
from .contact_geometry import compute_gaussian_contacts
from .structures import ContactForces, ContactGeometry, GaussianSet, RigidBodyState

__all__ = [
    "ContactForces",
    "ContactGeometry",
    "ContactMaterial",
    "GaussianSet",
    "RigidBodyState",
    "compute_contact_forces",
    "compute_gaussian_contacts",
    "integrate_rigid_body",
]
