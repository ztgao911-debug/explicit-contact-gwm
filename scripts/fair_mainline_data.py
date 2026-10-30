"""Re-encode only factual current inputs; preserve labels, sample order and masks."""

from __future__ import annotations

import json

import numpy as np
import torch
from bounded_search_training import check_deadline

from contact_gwm.pinned_contact_pipeline import IndenterInput


@torch.no_grad()
def reencode_training_data(root, source, trajectories, pipeline, deadline, output):
    samples = json.loads(
        (root / "outputs/contact_response_training/run_001/samples.json").read_text()
    )
    ordinary_inputs = []
    for sample in samples:
        if sample["split"] == "train":
            arrays, t = trajectories[sample["trajectory"]], int(sample["frame"])
            ordinary_inputs.append(
                (
                    arrays["node_position_world"][t],
                    arrays["node_velocity_world"][t],
                    IndenterInput.from_arrays(arrays, t),
                )
            )
    branch_inputs = [
        (
            source["branch_positions"][case, t].numpy(),
            source["branch_velocities"][case, t].numpy(),
            source["branch_robots"][case][t],
        )
        for case in range(108)
        for t in range(20)
    ]
    encoded = {}
    for name, inputs, original in (
        ("ordinary_train", ordinary_inputs, source["ordinary_train"]),
        ("branch_cache", branch_inputs, source["branch_cache"]),
    ):
        # Do not copy the historical M context or its contact gates into either arm.
        cache = {k: v for k, v in original.items() if k not in ("context", "gates")}
        values = {k: [] for k in ("context", "gates", "features", "velocity")}
        for index, (x, v, robot) in enumerate(inputs):
            check_deadline(deadline)
            state = pipeline.legacy.gaussian_state(x, v)
            current = pipeline.current_inputs(state, robot)
            for key in values:
                values[key].append(getattr(current, key))
            if index and index % 600 == 0:
                print(f"Re-encoded {name}: {index}/{len(inputs)}", flush=True)
        cache.update({k: torch.stack(v) for k, v in values.items()})
        for key in ("features", "velocity", "target_dx", "target_dv", "neighborhood", "pinned"):
            if not torch.equal(cache[key], original[key]):
                raise ValueError(f"Shared training state/label/mask changed: {name}/{key}")
        if not all(torch.isfinite(v).all() for v in cache.values()):
            raise FloatingPointError("Non-finite factual current conditioning cache")
        np.savez_compressed(output / f"{name}.npz", **{k: v.numpy() for k, v in cache.items()})
        encoded[name] = cache
    return {**source, **encoded}
