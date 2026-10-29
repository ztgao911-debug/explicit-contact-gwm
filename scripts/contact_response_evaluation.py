"""Known-indenter, autonomous-soft-state evaluation of stored native branches."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from audit_contact_mechanism import external_inputs
from validate_contact_pipeline import reference_masks, write_csv, write_json

from contact_gwm.contact_audit import signed_response_metrics
from contact_gwm.data.gaussian_pairs import read_npz
from contact_gwm.data.mujoco_soft import tetra_volumes


@torch.no_grad()
def evaluate_responses(predictors, legacy, trajectories, mesh, config, output, root):
    folder = output / "response_evaluation"
    folder.mkdir()
    rows, absolute, failures, checks = [], [], [], []
    starts = config["collection"]["validation_frames"]
    directory = root / config["validation_native_directory"]
    names = [v["name"] for v in config["_variants"]]
    reference_roi = reference_masks(trajectories["left_deep"], mesh, 0.02)
    horizons = [1, 5, 10, 20]
    for frame in starts:
        native = {v: read_npz(directory / f"{frame}_{v}_native.npz") for v in names}
        roi = reference_roi[frame].copy()
        for arrays in native.values():
            a, b = arrays["contact_offsets"][:2]
            points = arrays["contact_position_world"][a:b]
            if len(points):
                roi |= (
                    np.linalg.norm(native["base"]["position"][0, :, None] - points, axis=-1).min(-1)
                    <= 0.02
                )
        roi &= ~mesh["pinned_mask"]
        masks = {"active": ~mesh["pinned_mask"], "contact_neighborhood": roi}
        np.savez_compressed(folder / f"{frame}_masks.npz", **masks, node_ids=mesh["node_ids"])
        predictions = {}
        for variant, arrays in native.items():
            saved = {"node_ids": mesh["node_ids"]}
            predictions[variant] = {}
            for method, predictor in predictors.items():
                state = legacy.gaussian_state(arrays["position"][0], arrays["velocity"][0])
                x, v = [state.centers.numpy()], [state.velocities.numpy()]
                try:
                    for robot in external_inputs(arrays):
                        result = predictor(state, robot)
                        state = result.state
                        x.append(state.centers.numpy())
                        v.append(state.velocities.numpy())
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    failures.append(
                        {
                            "frame": frame,
                            "variant": variant,
                            "method": method,
                            "completed_steps": len(x) - 1,
                            "error": str(error),
                        }
                    )
                x, v = np.array(x), np.array(v)
                assert np.isfinite(x).all() and np.isfinite(v).all()
                assert np.array_equal(
                    x[:, mesh["pinned_mask"]],
                    np.broadcast_to(
                        mesh["rest_positions_world"][mesh["pinned_mask"]],
                        x[:, mesh["pinned_mask"]].shape,
                    ),
                )
                assert not np.any(v[:, mesh["pinned_mask"]])
                inverted = int(
                    np.any(
                        tetra_volumes(x, mesh["tetrahedra"]) / mesh["rest_signed_tetra_volumes"]
                        <= 0,
                        axis=-1,
                    ).sum()
                )
                checks.append(
                    {
                        "frame": frame,
                        "variant": variant,
                        "method": method,
                        "completed_steps": len(x) - 1,
                        "inverted_states": inverted,
                    }
                )
                predictions[variant][method] = (x, v)
                saved[f"{method}__position"], saved[f"{method}__velocity"] = x, v
                for h in horizons:
                    if h >= len(x):
                        continue
                    for region, mask in masks.items():
                        absolute.append(
                            {
                                "frame": frame,
                                "variant": variant,
                                "method": method,
                                "horizon": h,
                                "region": region,
                                "node_count": int(mask.sum()),
                                "position_rmse_um": float(
                                    np.sqrt(
                                        np.square(x[h, mask] - arrays["position"][h, mask])
                                        .sum(-1)
                                        .mean()
                                    )
                                    * 1e6
                                ),
                                "velocity_rmse_mm_s": float(
                                    np.sqrt(
                                        np.square(v[h, mask] - arrays["velocity"][h, mask])
                                        .sum(-1)
                                        .mean()
                                    )
                                    * 1e3
                                ),
                            }
                        )
            np.savez_compressed(folder / f"{frame}_{variant}_predictions.npz", **saved)
            with np.load(folder / f"{frame}_{variant}_predictions.npz") as f:
                assert all(np.array_equal(f[k], value) for k, value in saved.items())
        for variant in names:
            if variant == "base":
                continue
            for method in predictors:
                x, v = predictions[variant][method]
                bx, bv = predictions["base"][method]
                for h in horizons:
                    if min(len(x), len(bx)) <= h:
                        continue
                    for region, mask in masks.items():
                        row = {
                            "frame": frame,
                            "variant": variant,
                            "method": method,
                            "horizon": h,
                            "region": region,
                            "node_count": int(mask.sum()),
                        }
                        for field, prediction, reference, scale in (
                            (
                                "position",
                                x[h] - bx[h],
                                native[variant]["position"][h] - native["base"]["position"][h],
                                1e6,
                            ),
                            (
                                "velocity",
                                v[h] - bv[h],
                                native[variant]["velocity"][h] - native["base"]["velocity"][h],
                                1e3,
                            ),
                        ):
                            row.update(
                                {
                                    f"{field}_{k}": val
                                    for k, val in signed_response_metrics(
                                        prediction, reference, mask, scale
                                    ).items()
                                }
                            )
                        rows.append(row)
        print(f"Validation response rollout complete: left_deep/{frame}", flush=True)
    write_csv(folder / "per_case_response_metrics.csv", rows)
    write_csv(folder / "absolute_metrics.csv", absolute)
    write_json(folder / "failures.json", failures)
    write_json(folder / "checks.json", checks)
    summary = summarize_response_rows(rows)
    write_csv(folder / "summary.csv", summary)
    return {
        "summary": summary,
        "failures": failures,
        "all_20_step_cases_complete": all(c["completed_steps"] == 20 for c in checks),
        "inverted_states": sum(c["inverted_states"] for c in checks),
    }


def summarize_response_rows(rows):
    """Node-weighted summaries with JSON-safe counts, including zero responses."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["horizon"], row["region"]].append(row)
    summary = []
    for (method, h, region), selected in sorted(groups.items()):
        n = sum(r["node_count"] for r in selected)
        values = {
            "method": method,
            "horizon": h,
            "region": region,
            "comparisons": len(selected),
            "expected_comparisons": 15,
        }
        for field in ("position", "velocity"):
            for metric in ("response_rmse", "reference_response_rms"):
                values[f"{field}_{metric}"] = float(
                    np.sqrt(
                        sum(r["node_count"] * r[f"{field}_{metric}"] ** 2 for r in selected) / n
                    )
                )
            cos = [
                r[f"{field}_direction_cosine"]
                for r in selected
                if r[f"{field}_direction_cosine"] is not None
            ]
            ratios = [
                r[f"{field}_amplitude_ratio"]
                for r in selected
                if r[f"{field}_amplitude_ratio"] is not None
            ]
            values[f"{field}_negative_cosines"] = int(sum(c < 0 for c in cos))
            values[f"{field}_defined_cosines"] = len(cos)
            values[f"{field}_median_amplitude_ratio"] = float(np.median(ratios)) if ratios else None
        summary.append(values)
    return summary
