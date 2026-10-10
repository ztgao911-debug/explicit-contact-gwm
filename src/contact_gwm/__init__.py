"""Explicit contact-driven Gaussian world model."""

from .contact_geometry import compute_gaussian_contacts
from .structures import ContactForces, ContactGeometry, GaussianSet, RigidBodyState

__all__ = [
    "ContactForces",
    "ContactGeometry",
    "GaussianSet",
    "RigidBodyState",
    "compute_gaussian_contacts",
]
