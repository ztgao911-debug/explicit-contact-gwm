"""Independently audit saved bounded-search evidence without running predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASES = {1: "press", 2: "hold", 3: "release", 4: "recover"}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right):
    return bool(np.isclose(left, right, rtol=1e-9, atol=1e-12))


def independent_physics(x, v, mesh, expected_steps=20):
    """Use determinant volumes, independent of the production cross/dot path."""
    finite = np.isfinite(x).all(axis=(-2, -1)) & np.isfinite(v).all(axis=(-2, -1))
    pins = mesh["pinned_mask"].astype(bool)
    boundaries = np.any(x[:, pins] != mesh["rest_positions_world"][pins], axis=(-2, -1))
    boundaries |= np.any(v[:, pins] != 0, axis=(-2, -1))
    tet = x[:, mesh["tetrahedra"]]
    matrices = np.stack([tet[:, :, j] - tet[:, :, 0] for j in (1, 2, 3)], axis=-1)
    with np.errstate(invalid="ignore", over="ignore"):
        ratios = np.linalg.det(matrices) / (6 * mesh["rest_signed_tetra_volumes"])
    inverted = np.any(ratios <= 0, axis=1)
    minimum = float(np.min(ratios[np.isfinite(ratios)])) if np.isfinite(ratios).any() else None
    eligible = (
        len(x) == expected_steps + 1
        and bool(finite.all())
        and not boundaries.any()
        and np.isfinite(ratios).all()
        and not inverted.any()
    )
    return {
        "eligible": bool(eligible),
        "inverted_states": int(inverted.sum()),
        "nonfinite_states": int((~finite).sum()),
        "fixed_boundary_violating_states": int(boundaries.sum()),
        "minimum_volume_ratio": minimum,
    }


def independent_scores(folder, data, mesh, method="M"):
    """Pool node SSE per frozen phase/region/horizon, then equally average cells."""
    pooled = {}
    with (
        np.load(folder / "ordinary/rollout_predictions.npz", allow_pickle=False) as saved,
        np.load(folder / "ordinary/one_step_predictions.npz", allow_pickle=False) as masks,
    ):
        for key in saved.files:
            parts = key.split("__")
            if len(parts) != 4 or parts[2:] != [method, "position"]:
                continue
            name, start = parts[0], int(parts[1])
            if name not in ("left_shallow", "left_deep"):
                continue
            with np.load(data / f"{name}.npz", allow_pickle=False) as truth:
                phase = PHASES.get(int(truth["phase_id"][start]))
                if phase is None:
                    continue
                for region in ("active", "contact_neighborhood"):
                    mask = (
                        ~mesh["pinned_mask"].astype(bool)
                        if region == "active"
                        else masks[f"{name}__contact_neighborhood_mask"][start].astype(bool)
                    )
                    for horizon in (5, 10, 20):
                        cell = phase, region, horizon
                        psse, vsse, count = pooled.get(cell, (0.0, 0.0, 0))
                        dx = saved[key][horizon] - truth["node_position_world"][start + horizon]
                        dv = (
                            saved[key.removesuffix("position") + "velocity"][horizon]
                            - truth["node_velocity_world"][start + horizon]
                        )
                        pooled[cell] = (
                            psse + float(np.square(dx[mask]).sum()),
                            vsse + float(np.square(dv[mask]).sum()),
                            count + int(mask.sum()),
                        )
    expected = {
        (p, r, h)
        for p in PHASES.values()
        for r in ("active", "contact_neighborhood")
        for h in (5, 10, 20)
    }
    rows = []
    for (phase, region, horizon), (psse, vsse, count) in sorted(pooled.items()):
        rows.append(
            {
                "phase": phase,
                "region": region,
                "horizon": horizon,
                "position_mse_m2": psse / count,
                "velocity_mse_m2_s2": vsse / count,
                "node_count": count,
            }
        )
    complete = set(pooled) == expected and all(row["node_count"] for row in rows)
    pscore = float(np.mean([r["position_mse_m2"] / 0.001**2 for r in rows])) if complete else None
    vscore = float(np.mean([r["velocity_mse_m2_s2"] / 0.1**2 for r in rows])) if complete else None
    return {
        "coverage_complete": complete,
        "cells": rows,
        "position_score": pscore,
        "velocity_score": vscore,
        "joint_score": (pscore + vscore) / 2 if complete else None,
    }


def verify_evaluation(folder, data, mesh, method="M"):
    stored = read_json(folder / "evaluation_summary.json")["per_method"][method]
    scores = independent_scores(folder, data, mesh, method)
    scores_match = scores["coverage_complete"] and all(
        close(scores[name], stored["scores"][name])
        for name in ("joint_score", "position_score", "velocity_score")
    )
    records = []
    ordinary = read_json(folder / "ordinary_physics.json")["windows"]
    with np.load(folder / "ordinary/rollout_predictions.npz", allow_pickle=False) as saved:
        for row in ordinary:
            if row["method"] != method:
                continue
            prefix = f"{row['trajectory']}__{row['start']}__{method}__"
            if prefix + "position" not in saved:
                records.append(
                    {
                        "kind": "ordinary",
                        "saved_failure": row,
                        "matches": not row["eligible"],
                        "missing": True,
                    }
                )
                continue
            result = independent_physics(
                saved[prefix + "position"], saved[prefix + "velocity"], mesh
            )
            records.append(
                {
                    "kind": "ordinary",
                    "trajectory": row["trajectory"],
                    "start": row["start"],
                    "independent": result,
                    "matches": physics_matches(result, row),
                }
            )
    for row in read_json(folder / "response_physics.json")["windows"]:
        if row["method"] != method:
            continue
        path = folder / f"response_evaluation/{row['frame']}_{row['variant']}_predictions.npz"
        if not path.exists():
            records.append(
                {
                    "kind": "intervention",
                    "saved_failure": row,
                    "matches": not row["eligible"],
                    "missing": True,
                }
            )
            continue
        with np.load(path, allow_pickle=False) as saved:
            result = independent_physics(
                saved[f"{method}__position"], saved[f"{method}__velocity"], mesh
            )
        records.append(
            {
                "kind": "intervention",
                "frame": row["frame"],
                "variant": row["variant"],
                "independent": result,
                "matches": physics_matches(result, row),
            }
        )
    return {
        "scores": scores,
        "scores_match": scores_match,
        "physics_windows": records,
        "physics_match": all(r["matches"] for r in records),
        "passed": scores_match and all(r["matches"] for r in records),
    }


def physics_matches(result, stored):
    fields = ("eligible", "inverted_states", "nonfinite_states", "fixed_boundary_violating_states")
    same = all(result[name] == stored[name] for name in fields)
    minimum = result["minimum_volume_ratio"]
    return same and (
        minimum is None
        and stored["minimum_volume_ratio"] is None
        or minimum is not None
        and close(minimum, stored["minimum_volume_ratio"])
    )


def verify(report, output):
    protocol = read_json(report / "protocol.json")
    ledger = read_json(report / "trial_ledger.json", [])
    latest = {}
    for result in ledger:
        latest[result["candidate_id"], result["seed"]] = result
    candidates = {row["id"]: row for row in protocol["candidates"]}
    search = {row["candidate_id"] for row in ledger if row["role"] == "search"}
    replication = {
        (row["candidate_id"], row["seed"]) for row in ledger if row["role"] == "replication"
    }
    checks = {
        "frozen_12_candidates": len(candidates) == 12,
        "candidate_copy_matches": read_json(report / "candidates.json") == protocol["candidates"],
        "search_configuration_limit": len(search) <= 12,
        "replication_training_limit": len(replication) <= 4,
        "all_attempts_match_frozen_configuration": all(
            row["candidate"] == candidates.get(row["candidate_id"]) for row in ledger
        ),
    }
    changed = [
        str(name)
        for name, digest in protocol["data_hashes"].items()
        if not (ROOT / name).exists() or sha256(ROOT / name) != digest
    ]
    checks["frozen_data_unchanged"] = not changed
    digests = {}
    for row in latest.values():
        if row.get("initial_weight_sha256"):
            digests.setdefault(str(row["seed"]), set()).add(row["initial_weight_sha256"])
    checks["same_seed_initialization_consistent"] = all(
        len(values) == 1 for values in digests.values()
    )
    data = ROOT / protocol["configuration"]["data_directory"]
    with np.load(data / "mesh.npz", allow_pickle=False) as saved:
        mesh = {key: saved[key] for key in saved.files}
    trials = []
    for row in latest.values():
        record = {"candidate_id": row["candidate_id"], "seed": row["seed"], "status": row["status"]}
        folder = ROOT / row["artifact_directory"]
        selection = read_json(folder / "checkpoint_selection.json", [])
        if selection:
            finite = [r for r in selection if r["scores"].get("joint_score") is not None]
            chosen = min(finite, key=lambda r: r["scores"]["joint_score"]) if finite else None
            record["step_zero_present"] = selection[0]["update"] == 0
            record["selected_minimum"] = (
                chosen is not None
                and row.get("training", {}).get("selected_step") == chosen["update"]
            )
        else:
            record["checkpoint_selection"] = "not_run_or_interrupted_before_first_validation"
        if row["status"].startswith("complete"):
            record["evaluation"] = verify_evaluation(folder / "evaluation", data, mesh)
        else:
            record["evaluation"] = {
                "status": "not_completed",
                "failure_retained": bool(row.get("error")),
            }
        trials.append(record)
    checks["completed_trial_evidence_matches"] = all(
        (
            record.get("step_zero_present", False)
            and record.get("selected_minimum", False)
            and record["evaluation"].get("passed", False)
        )
        if record["status"].startswith("complete")
        else True
        for record in trials
    )
    artifact = read_json(report / "preflight_artifact_preservation.json")
    reference = None
    if artifact:
        folder = ROOT / artifact["preserved_directory"]
        mismatches = [
            name
            for name, digest in artifact["files_sha256"].items()
            if not (folder / name).exists() or sha256(folder / name) != digest
        ]
        reference = {
            "kind": "preflight_existing_reference_not_search_candidate",
            "hash_mismatches": mismatches,
            **verify_evaluation(folder, data, mesh),
        }
        checks["preserved_reference_hashes_unchanged"] = not mismatches
        checks["preflight_reference_independently_verified"] = reference["passed"]
    gradient = read_json(report / "gradient_preflight.json", {})
    checks["recorded_gradient_preflight_nonzero_finite"] = all(
        np.isfinite(gradient.get(name, np.nan)) and gradient.get(name, 0) > 0
        for name in ("terminal_to_first_x_gradient_norm", "terminal_to_first_v_gradient_norm")
    )
    return {
        "audit_passed": all(checks.values()),
        "checks": checks,
        "formal_search_status": (
            (
                "not_run_budget_exhausted"
                if time.time() >= read_json(report / "budget.json")["deadline_epoch"]
                else "not_run"
            )
            if not ledger
            else "see_trial_records"
        ),
        "search_configurations_started": len(search),
        "replication_trainings_started": len(replication),
        "independent_test": "not_verified",
        "changed_data_files": changed,
        "initialization_digests_by_seed": {k: sorted(v) for k, v in digests.items()},
        "same_seed_initialization_evidence": "not_run" if not digests else "recorded",
        "checkpoint_and_multiseed_evidence": "not_run" if not trials else "see_trial_records",
        "trials": trials,
        "preflight_existing_reference": reference,
        "audit_does_not_imply_dynamics_validation": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/bounded_multistep_search/run_001"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/bounded_multistep_search/run_001"
    )
    args = parser.parse_args()
    result = verify(args.report.resolve(), args.output.resolve())
    destination = args.report / "independent_verification.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": result["audit_passed"],
                "output": str(destination),
                "formal_search_status": result["formal_search_status"],
            }
        )
    )
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
