"""Single nonnegative area-pressure stiffness, fitted on force labels only.

The model predicts from current geometric integrals. Force labels are accepted
only by the separate calibration function, never by the prediction interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ElasticAreaMaterial:
    stiffness_N_m3: float

    def __post_init__(self):
        if not np.isfinite(self.stiffness_N_m3) or self.stiffness_N_m3 < 0:
            raise ValueError("Area stiffness must be finite and nonnegative, in N/m^3")

    def predict(
        self, elastic_vector_m3: Tensor, elastic_torque_m4: Tensor
    ) -> tuple[Tensor, Tensor]:
        """World force on the object and torque about the supplied basis origin.

        No damping, tangential traction, body update, contact labels or next
        state is used. Force has unit N; torque has unit N*m.
        """
        if elastic_vector_m3.shape != elastic_torque_m4.shape or elastic_vector_m3.shape[-1:] != (
            3,
        ):
            raise ValueError("Force/torque bases must have matching [...,3] shapes")
        if (
            elastic_vector_m3.device != elastic_torque_m4.device
            or elastic_vector_m3.dtype != elastic_torque_m4.dtype
        ):
            raise ValueError("Bases must share dtype/device")
        if (
            not torch.isfinite(elastic_vector_m3).all()
            or not torch.isfinite(elastic_torque_m4).all()
        ):
            raise FloatingPointError("Non-finite current geometric basis")
        return self.stiffness_N_m3 * elastic_vector_m3, self.stiffness_N_m3 * elastic_torque_m4

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stiffness_N_m3": self.stiffness_N_m3,
                    "stiffness_unit": "N/m^3",
                    "damping_N_s_m3": 0.0,
                    "tangential_model": "none",
                    "intercept_N": 0.0,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> ElasticAreaMaterial:
        saved = json.loads(path.read_text())
        if (
            saved.get("schema_version") != 1
            or saved.get("stiffness_unit") != "N/m^3"
            or saved.get("damping_N_s_m3") != 0.0
            or saved.get("tangential_model") != "none"
            or saved.get("intercept_N") != 0.0
        ):
            raise ValueError("Wrong material schema, units, or model scope")
        return cls(float(saved["stiffness_N_m3"]))


def fit_elastic_area_stiffness(
    elastic_vectors_m3: np.ndarray,
    normal_forces_N: np.ndarray,
    *,
    numerical_basis_scale_m3: float = 1e-9,
) -> tuple[ElasticAreaMaterial, dict]:
    """Exact one-column, zero-intercept nonnegative least squares.

    Each row is ONE FRAME's world net normal-force vector, not one candidate.
    Scaling is numerical only and does not reweight frames or change the optimum.
    Caller is responsible for passing central-training rows only.
    """
    basis = np.asarray(elastic_vectors_m3, dtype=np.float64)
    target = np.asarray(normal_forces_N, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != 3 or basis.shape != target.shape or not len(basis):
        raise ValueError("Nonempty, aligned [frames,3] force bases and labels required")
    if not np.isfinite(basis).all() or not np.isfinite(target).all():
        raise FloatingPointError("Non-finite calibration input")
    if not np.isfinite(numerical_basis_scale_m3) or numerical_basis_scale_m3 <= 0:
        raise ValueError("Numerical basis scale must be finite and positive")
    design = basis / numerical_basis_scale_m3
    denominator = float(np.square(design).sum())
    if denominator <= 0:
        raise ValueError("No nonzero geometric basis; stiffness is unidentifiable")
    unconstrained = float((design * target).sum() / denominator)
    coefficient = max(0.0, unconstrained) / numerical_basis_scale_m3
    model = ElasticAreaMaterial(coefficient)
    prediction = coefficient * basis
    info = {
        "fit_frames": len(basis),
        "nonzero_basis_frames": int(np.count_nonzero(np.linalg.norm(basis, axis=1))),
        "numerical_basis_scale_m3": numerical_basis_scale_m3,
        "scaled_design_squared_norm": denominator,
        "design_rank": 1,
        "unconstrained_stiffness_N_m3": unconstrained / numerical_basis_scale_m3,
        "constraint_active": coefficient == 0.0,
        "vector_rmse_N": float(np.sqrt(np.square(prediction - target).sum(-1).mean())),
        "objective_sum_squared_N2": float(np.square(prediction - target).sum()),
    }
    return model, info
