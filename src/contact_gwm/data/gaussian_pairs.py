"""Current-frame MuJoCo to Gaussian adapter, independent of the simulator install.

Future arrays are accessed only when constructing supervision in __getitem__.
frame_inputs() can construct every model input using a single current frame.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from contact_gwm.structures import ContactForces, ContactGeometry, GaussianSet


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@dataclass(frozen=True)
class GaussianFrameInputs:
    node_ids: Tensor
    rest_positions: Tensor
    pinned_mask: Tensor
    gaussians: GaussianSet
    contacts: ContactGeometry
    contact_forces: ContactForces
    action: Tensor
    material_features: Tensor
    neighborhood_mask: Tensor


@dataclass(frozen=True)
class GaussianFramePair:
    inputs: GaussianFrameInputs
    target_delta: Tensor
    target_next_centers: Tensor
    trajectory: str
    frame: int
    time: float
    next_time: float


class GaussianPairDataset(Dataset):
    """Strict stable-ID trajectory loader; no model inputs contain future labels."""

    def __init__(self, directory: Path, config: dict) -> None:
        self.directory = Path(directory)
        self.config = config
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        paths = ["mesh.npz", "manifest.json", *[f"{name}.npz" for name in config["trajectories"]]]
        self.source_hashes = {name: file_sha256(self.directory / name) for name in paths}
        if self.source_hashes["mesh.npz"] != self.manifest["shared_sha256"]["mesh.npz"]:
            raise ValueError("Mesh checksum does not match the independent collection")
        self.mesh = read_npz(self.directory / "mesh.npz")
        self.node_ids = torch.as_tensor(self.mesh["node_ids"].copy(), dtype=torch.long)
        self.rest = torch.as_tensor(self.mesh["rest_positions_world"].copy(), dtype=torch.float32)
        self.pinned = torch.as_tensor(self.mesh["pinned_mask"].copy(), dtype=torch.bool)
        if not np.array_equal(self.mesh["node_ids"], np.arange(len(self.rest))):
            raise ValueError("Mesh node IDs must be ordered and contiguous")
        distances = torch.cdist(self.rest, self.rest)
        distances.fill_diagonal_(torch.inf)
        self.scales = (0.5 * distances.min(dim=1).values)[:, None].expand(-1, 3).clone()
        if not torch.isfinite(self.scales).all() or not torch.all(self.scales > 0):
            raise ValueError("Gaussian initialization requires distinct initial nodes")
        self.log_scales = self.scales.log()
        self.rotations = torch.zeros((len(self.rest), 4))
        self.rotations[:, 0] = 1.0
        self.trajectories: dict[str, dict] = {}
        self.samples: list[tuple[str, int]] = []
        self.selection_counts = {}
        for name in config["trajectories"]:
            if Path(name).name != name:
                raise ValueError("Trajectory names must be plain basenames")
            item = next(
                item for item in self.manifest["trajectories"] if item["file"] == f"{name}.npz"
            )
            if self.source_hashes[f"{name}.npz"] != item["sha256"]:
                raise ValueError(f"Trajectory checksum mismatch: {name}")
            arrays = read_npz(self.directory / f"{name}.npz")
            self._validate_arrays(arrays)
            self.trajectories[name] = arrays
            selected = []
            for t in range(len(arrays["time"]) - 1):
                allowed = arrays["phase_id"][t] in config["selection"]["phase_ids"]
                if config["selection"]["require_both_phases"]:
                    allowed = (
                        allowed and arrays["phase_id"][t + 1] in config["selection"]["phase_ids"]
                    )
                if allowed and self.contact_indices(arrays, t).size:
                    self.samples.append((name, t))
                    selected.append(t)
            self.selection_counts[name] = {
                "pairs": len(selected),
                "press_pairs": sum(arrays["phase_id"][t] == 1 for t in selected),
                "hold_pairs": sum(arrays["phase_id"][t] == 2 for t in selected),
            }
            self.selection_counts[name] = {
                key: int(value) for key, value in self.selection_counts[name].items()
            }
            if not selected:
                raise ValueError(f"No eligible current-contact press/hold pairs: {name}")

    def _validate_arrays(self, arrays: dict) -> None:
        if not np.array_equal(arrays["node_ids"], self.mesh["node_ids"]):
            raise ValueError("Trajectory node IDs do not match the stable mesh order")
        count = len(arrays["time"])
        if arrays["node_position_world"].shape != (count, len(self.rest), 3):
            raise ValueError("Wrong node position shape")
        if arrays["node_velocity_world"].shape != arrays["node_position_world"].shape:
            raise ValueError("Wrong node velocity shape")
        if not np.allclose(np.diff(arrays["time"]), self.config["dt"], atol=1e-9, rtol=0):
            raise ValueError("Adjacent record times must match dt=0.01 s")
        if not np.isclose(self.config["dt"], 0.01, atol=1e-12):
            raise ValueError("This first training run requires dt=0.01 s")
        if not np.allclose(
            arrays["node_position_world"][0], self.mesh["rest_positions_world"], atol=1e-12
        ):
            raise ValueError("Initial node positions differ from the shared rest shape")
        if not np.allclose(
            arrays["node_position_world"][:, self.mesh["pinned_mask"]],
            self.mesh["rest_positions_world"][self.mesh["pinned_mask"]],
            atol=1e-12,
        ):
            raise ValueError("Bottom nodes must remain fixed")
        if arrays["contact_offsets"].shape != (count + 1,):
            raise ValueError("Invalid contact offset shape")
        if arrays["contact_offsets"][0] != 0 or not np.array_equal(
            np.diff(arrays["contact_offsets"]), arrays["contact_count"]
        ):
            raise ValueError("Packed contact counts disagree")
        packed_count = int(arrays["contact_offsets"][-1])
        for key in (
            "contact_position_world",
            "contact_frame_world",
            "contact_signed_distance",
            "contact_force_on_block_world",
            "contact_flex_ids",
        ):
            if len(arrays[key]) != packed_count:
                raise ValueError(f"Wrong contact count in {key}")
        for key, values in arrays.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite source data: {key}")

    def contact_indices(self, arrays: dict, t: int) -> np.ndarray:
        start, end = arrays["contact_offsets"][t : t + 2]
        indices = np.arange(start, end)
        if not len(indices):
            return indices
        forces = arrays["contact_force_on_block_world"][indices]
        active = np.linalg.norm(forces, axis=1) > self.config["selection"]["min_force_N"]
        active &= arrays["contact_signed_distance"][indices, 0] <= 0.0
        return indices[active]

    def frame_inputs(self, trajectory: str, t: int) -> GaussianFrameInputs:
        """Only current frame t, mesh and static material metadata are accessed here."""
        arrays = self.trajectories[trajectory]
        norm = self.config["normalization"]

        def tensor(value: np.ndarray) -> Tensor:
            return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)

        centers = tensor(arrays["node_position_world"][t])
        velocity = tensor(arrays["node_velocity_world"][t])
        features = torch.cat(
            (
                self.rest / norm["rest_position"],
                (centers - self.rest) / norm["displacement"],
                velocity / norm["velocity"],
                self.pinned[:, None].float(),
            ),
            dim=1,
        )
        gaussians = GaussianSet(
            centers=centers,
            log_scales=self.log_scales.clone(),
            rotations=self.rotations.clone(),
            opacity_logits=torch.zeros((len(centers), 1)),
            features=features,
            velocities=velocity,
        )
        indices = self.contact_indices(arrays, t)
        points = tensor(arrays["contact_position_world"][indices])
        if len(indices):
            flex = arrays["contact_flex_ids"][indices]
            if not np.all(np.sum(flex == 0, axis=1) == 1):
                raise ValueError("Expected exactly one block side in each solver contact")
            sign = tensor(np.where(flex[:, 1] == 0, 1.0, -1.0))
            normals = sign[:, None] * tensor(arrays["contact_frame_world"][indices, :3])
            distances = torch.cdist(centers, points)
            object_indices = distances.argmin(dim=0)
            neighbor_mask = (
                distances.min(dim=1).values
                <= self.config["evaluation"]["contact_neighborhood_radius"]
            )
        else:
            normals = torch.zeros((0, 3))
            object_indices = torch.zeros(0, dtype=torch.long)
            neighbor_mask = torch.zeros(len(centers), dtype=torch.bool)
        forces = tensor(arrays["contact_force_on_block_world"][indices])
        normal_forces = (forces * normals).sum(dim=-1, keepdim=True).clamp_min(0) * normals
        gaps = tensor(arrays["contact_signed_distance"][indices, 0])
        contacts = ContactGeometry(
            object_indices=object_indices,
            robot_indices=torch.zeros(len(indices), dtype=torch.long),
            points=points,
            normals=normals,
            signed_gaps=gaps,
            penetrations=(-gaps).clamp_min(0),
            relative_velocities=tensor(arrays["indenter_velocity_world"][t])[None, :]
            - velocity[object_indices],
        )
        response = ContactForces(
            forces=forces,
            torques=torch.cross(points, forces, dim=-1),
            normal_forces=normal_forces,
            friction_forces=forces - normal_forces,
        )
        action = torch.cat(
            (
                tensor(arrays["action_target_position"][t]) / norm["action_position"],
                tensor(arrays["action_target_velocity"][t]) / norm["action_velocity"],
                tensor(arrays["indenter_position_world"][t]) / norm["action_position"],
                tensor(arrays["indenter_velocity_world"][t]) / norm["action_velocity"],
            )
        )
        material = self.manifest["config"]
        material_features = torch.tensor(
            [
                material["block"]["young"] / norm["young"],
                material["block"]["poisson"],
                material["block"]["rayleigh_damping"] / norm["rayleigh_damping"],
                material["contact"]["friction"][0],
            ]
        )
        return GaussianFrameInputs(
            self.node_ids.clone(),
            self.rest.clone(),
            self.pinned.clone(),
            gaussians,
            contacts,
            response,
            action,
            material_features,
            neighbor_mask & ~self.pinned,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GaussianFramePair:
        name, t = self.samples[index]
        arrays = self.trajectories[name]
        # Subtract in float64 before converting: small hold increments must not be rounded away.
        delta = arrays["node_position_world"][t + 1] - arrays["node_position_world"][t]
        return GaussianFramePair(
            inputs=self.frame_inputs(name, t),
            target_delta=torch.tensor(delta, dtype=torch.float32),
            target_next_centers=torch.tensor(
                arrays["node_position_world"][t + 1], dtype=torch.float32
            ),
            trajectory=name,
            frame=t,
            time=float(arrays["time"][t]),
            next_time=float(arrays["time"][t + 1]),
        )
