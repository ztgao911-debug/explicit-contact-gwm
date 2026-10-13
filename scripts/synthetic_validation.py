"""Check controlled contacts and compare CPU rollouts over equal physical time.

Run from the repository root after installing ``pip install -e '.[dev,experiments]'``.
Contact quantities are sampled at the start of each step; rigid states at its end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from contact_gwm import (
    ContactGatedDeformation,
    ContactMaterial,
    ExplicitContactGaussianWorldModel,
    GaussianSet,
    RigidBodyState,
    WorldModelOutput,
)

ROOT = Path(__file__).resolve().parents[1]
OBJECT_RADIUS = 0.11
ROBOT_RADIUS = 0.13
ROBOT_SPEED = -0.20
MASS = 0.25


@dataclass
class Scene:
    local: GaussianSet
    robot: GaussianSet
    rigid: RigidBodyState


class Experiment:
    def __init__(self, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.material = ContactMaterial(**config["material"])
        self.model = (
            ExplicitContactGaussianWorldModel(
                ContactGatedDeformation(**config["deformation"]),
                self.material,
                surface_level=config["surface_level"],
                contact_margin=config["contact_margin"],
                max_contacts=config["max_contacts"],
                gravity=(0.0, 0.0, 0.0),
            )
            .to(device)
            .eval()
        )
        self.action = torch.tensor([0.0, 0.0, ROBOT_SPEED], device=device)
        self.material_features = torch.tensor(
            [
                self.material.normal_stiffness,
                self.material.normal_damping,
                self.material.tangent_damping,
                self.material.friction_coefficient,
            ],
            device=device,
        )
        self.checks: list[dict] = []

    def cloud(self, centers: torch.Tensor, radius: float, velocity: torch.Tensor) -> GaussianSet:
        count = len(centers)
        rotations = centers.new_zeros((count, 4))
        rotations[:, 0] = 1.0
        return GaussianSet(
            centers=centers,
            log_scales=torch.full_like(centers, radius).log(),
            rotations=rotations,
            opacity_logits=centers.new_zeros((count, 1)),
            features=centers.new_zeros((count, self.config["deformation"]["feature_dim"])),
            velocities=velocity.expand_as(centers).clone(),
        )

    def scene(self, robot_x: float = 0.0, robot_z: float = 0.19) -> Scene:
        # Same nine-Gaussian grid, mass and inertia as the original Toy.
        centers = torch.tensor(
            [[x, y, 0.0] for y in (-0.18, 0.0, 0.18) for x in (-0.18, 0.0, 0.18)],
            device=self.device,
        )
        zero = centers.new_zeros(3)
        return Scene(
            local=self.cloud(centers, OBJECT_RADIUS, zero),
            robot=self.cloud(
                centers.new_tensor([[robot_x, 0.0, robot_z]]), ROBOT_RADIUS, self.action
            ),
            rigid=RigidBodyState(
                position=zero.clone(),
                orientation=centers.new_tensor([1.0, 0.0, 0.0, 0.0]),
                linear_velocity=zero.clone(),
                angular_velocity=zero.clone(),
                mass=MASS,
                inertia_diag=centers.new_full((3,), 0.01),
            ),
        )

    def step(self, scene: Scene, dt: float) -> WorldModelOutput:
        return self.model(
            scene.local, scene.robot, scene.rigid, self.action, self.material_features, dt
        )

    def check(self, name: str, expected: str, passed: bool, observed: object) -> None:
        self.checks.append(
            {"name": name, "expected": expected, "passed": bool(passed), "observed": observed}
        )

    @staticmethod
    def metrics(scene: Scene, output: WorldModelOutput) -> dict:
        zero = scene.local.centers.new_zeros(3)
        forces = output.contact_forces_world
        force = forces.forces.sum(0) if forces.forces.numel() else zero
        torque = forces.torques.sum(0) if forces.torques.numel() else zero
        penetrations = output.contacts_world.penetrations
        peak_index = int(output.contact_gates.argmax())
        return {
            "contacts": output.contacts_world.num_contacts,
            "penetration_max": float(penetrations.max()) if penetrations.numel() else 0.0,
            "force_z": float(force[2]),
            "torque_y": float(torque[1]),
            "position_z": float(output.rigid_state.position[2]),
            "velocity_z": float(output.rigid_state.linear_velocity[2]),
            "angular_velocity_y": float(output.rigid_state.angular_velocity[1]),
            "gate_peak_index": peak_index,
            "gate_peak_x": float(scene.local.centers[peak_index, 0]),
            "gate_max": float(output.contact_gates.max()),
            "local_center_change_max": float(
                (output.object_local.centers - scene.local.centers).abs().max()
            ),
            "local_log_scale_change_max": float(
                (output.object_local.log_scales - scene.local.log_scales).abs().max()
            ),
        }

    def single_steps(self) -> tuple[list[dict], list[dict]]:
        dt = self.config["time_step"]
        far = self.scene(robot_z=0.60)
        output = self.step(far, dt)
        row = {"case": "no_contact", **self.metrics(far, output)}
        self.check(
            "no_contact",
            "zero contacts, force and motion; gates zero",
            row["contacts"] == 0
            and row["force_z"] == 0.0
            and torch.equal(output.rigid_state.position, far.rigid.position)
            and torch.equal(output.rigid_state.linear_velocity, far.rigid.linear_velocity)
            and torch.equal(output.rigid_state.orientation, far.rigid.orientation)
            and torch.equal(output.rigid_state.angular_velocity, far.rigid.angular_velocity)
            and row["gate_max"] == 0.0,
            row,
        )
        single = [row]
        level = self.config["surface_level"]
        radius_sum = level * (OBJECT_RADIUS + ROBOT_RADIUS)
        for penetration in (0.005, 0.01, 0.02, 0.03, 0.05):
            scene = self.scene(robot_z=radius_sum - penetration)
            output = self.step(scene, dt)
            expected_force = -(
                self.material.normal_stiffness * penetration
                + self.material.normal_damping * abs(ROBOT_SPEED)
            )
            expected_position = expected_force / MASS * dt**2
            row = {
                "case": f"penetration_{penetration:g}",
                **self.metrics(scene, output),
                "input_penetration": penetration,
                "expected_force_z": expected_force,
                "expected_position_z": expected_position,
            }
            self.check(
                row["case"],
                "one contact; force and displacement match analytic values",
                row["contacts"] == 1
                and math.isclose(row["penetration_max"], penetration, abs_tol=1e-6)
                and math.isclose(row["force_z"], expected_force, rel_tol=1e-5, abs_tol=1e-6)
                and math.isclose(row["position_z"], expected_position, rel_tol=1e-5, abs_tol=1e-7),
                row,
            )
            single.append(row)
        sweep = single[1:]
        self.check(
            "penetration_monotonicity",
            "deeper penetration gives larger downward force and motion",
            all(
                b["force_z"] < a["force_z"] and b["position_z"] < a["position_z"]
                for a, b in zip(sweep, sweep[1:], strict=False)
            ),
            [{"penetration": r["input_penetration"], "force_z": r["force_z"]} for r in sweep],
        )
        gates = []
        for x, expected_index in ((-0.18, 3), (0.0, 4), (0.18, 5)):
            scene = self.scene(robot_x=x)
            output = self.step(scene, dt)
            row = {"case": f"contact_x_{x:g}", **self.metrics(scene, output)}
            passed = row["contacts"] == 1 and row["gate_peak_index"] == expected_index
            if x == 0.0:
                passed = passed and abs(row["torque_y"]) < 1e-6
            else:
                passed = passed and row["torque_y"] * x > 0.0
            self.check(
                row["case"], "gate follows contact; off-center torque has correct sign", passed, row
            )
            gates.append({"robot_x": x, "gates": output.contact_gates.cpu().tolist()})
            single.append(row)
        self.check(
            "single_step_zero_deformation",
            "untrained local center and log-scale changes are zero",
            all(
                r["local_center_change_max"] == r["local_log_scale_change_max"] == 0.0
                for r in single
            ),
            "all single-step cases",
        )
        return single, gates

    def rollout(self, dt: float, duration: float, *, no_contact: bool = False) -> list[dict]:
        scene = self.scene(robot_z=0.60 if no_contact else 0.19)
        rows = []
        impulse_z = 0.0
        for step_index in range(round(duration / dt)):
            output = self.step(scene, dt)
            metrics = self.metrics(scene, output)
            state_tensors = (
                output.rigid_state.position,
                output.rigid_state.linear_velocity,
                output.rigid_state.orientation,
                output.rigid_state.angular_velocity,
                output.object_world.centers,
                output.object_world.log_scales,
            )
            if not all(bool(torch.isfinite(tensor).all()) for tensor in state_tensors):
                raise RuntimeError(f"Non-finite state at dt={dt}, step={step_index}")
            impulse_z += metrics["force_z"] * dt
            rows.append(
                {
                    "run": "no_contact" if no_contact else f"dt_{dt:g}",
                    "dt": dt,
                    "step": step_index,
                    "time_start": step_index * dt,
                    "time_end": (step_index + 1) * dt,
                    "robot_z_start": float(scene.robot.centers[0, 2]),
                    "robot_z_end": float(scene.robot.centers[0, 2] + dt * ROBOT_SPEED),
                    **metrics,
                }
            )
            # Prescribed robot trajectory, not a dynamic body: integrate its commanded velocity.
            scene = Scene(
                output.object_local,
                replace(scene.robot, centers=scene.robot.centers + dt * scene.robot.velocities),
                output.rigid_state,
            )
        label = "no_contact_rollout" if no_contact else f"rollout_dt_{dt:g}"
        self.check(
            label,
            "finite states; zero local deformation; momentum matches contact impulse",
            all(
                r["local_center_change_max"] == r["local_log_scale_change_max"] == 0.0 for r in rows
            )
            and math.isclose(MASS * rows[-1]["velocity_z"], impulse_z, abs_tol=1e-5),
            {
                "steps": len(rows),
                "final_z": rows[-1]["position_z"],
                "final_vz": rows[-1]["velocity_z"],
                "contact_impulse_z": impulse_z,
            },
        )
        if no_contact:
            self.check(
                "no_contact_rollout_rest",
                "no contacts and no motion for the entire rollout",
                all(
                    r["contacts"] == 0
                    and r["gate_max"] == 0.0
                    and r["position_z"] == r["velocity_z"] == 0.0
                    for r in rows
                ),
                len(rows),
            )
        else:
            separated = [r for r in rows if r["contacts"] == 0]
            self.check(
                f"separation_dt_{dt:g}",
                "after contact loss, force is zero and velocity coasts",
                bool(separated)
                and all(r["force_z"] == r["gate_max"] == 0.0 for r in separated)
                and math.isclose(
                    separated[0]["velocity_z"], separated[-1]["velocity_z"], abs_tol=1e-6
                ),
                {"separated_steps": len(separated)},
            )
        return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    directory: Path,
    single: list[dict],
    gates: list[dict],
    runs: list[list[dict]],
    convergence: list[dict],
) -> None:
    # Keep Matplotlib's font cache inside the experiment directory, including in a sandbox.
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = [r for r in single if "input_penetration" in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, measured, expected, label in (
        (axes[0], "force_z", "expected_force_z", "Downward force magnitude"),
        (axes[1], "position_z", "expected_position_z", "Downward displacement magnitude"),
    ):
        ax.plot(
            [r["input_penetration"] for r in sweep],
            [-r[expected] for r in sweep],
            label="Analytic expectation",
        )
        ax.scatter(
            [r["input_penetration"] for r in sweep],
            [-r[measured] for r in sweep],
            label="Measured",
            color="tab:orange",
            zorder=3,
        )
        ax.set(xlabel="Input penetration", ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(directory / "penetration_sweep.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for ax, case in zip(axes, gates, strict=True):
        heat = torch.tensor(case["gates"]).reshape(3, 3).numpy()
        artist = ax.imshow(
            heat,
            origin="lower",
            extent=(-0.27, 0.27, -0.27, 0.27),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        ax.scatter(case["robot_x"], 0.0, marker="x", color="red", label="Robot XY")
        ax.set(
            title=f"Robot x = {case['robot_x']:+.2f}",
            xlabel="Object-local x",
            ylabel="Object-local y",
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(artist, ax=axes, label="Contact gate (not deformation)")
    fig.savefig(directory / "contact_gates.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    for ax, key, label, time_key in zip(
        axes.flat,
        ("position_z", "velocity_z", "force_z", "penetration_max", "contacts", "gate_max"),
        (
            "Object z",
            "Object velocity z",
            "Contact force z",
            "Max penetration",
            "Contact count",
            "Max contact gate",
        ),
        ("time_end", "time_end", "time_start", "time_start", "time_start", "time_start"),
        strict=True,
    ):
        for run in runs:
            times = [r[time_key] for r in run]
            values = [r[key] for r in run]
            if time_key == "time_end":
                times.insert(0, 0.0)
                values.insert(0, 0.0)
            ax.plot(
                times,
                values,
                label=f"dt={run[0]['dt']:g}",
                drawstyle="steps-post" if key == "contacts" else "default",
            )
        ax.set(xlabel="Time (s)", ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Same {runs[0][-1]['time_end']:g} s horizon; robot moves downward; gravity = 0")
    fig.savefig(directory / "rollout_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, label in (
        (axes[0], "position_error", "Final position absolute error"),
        (axes[1], "velocity_error", "Final velocity absolute error"),
    ):
        ax.plot([r["dt"] for r in convergence], [r[key] for r in convergence], "o-")
        ax.set(xlabel="Time step (s)", ylabel=label)
        ax.grid(alpha=0.3)
    fig.suptitle("Error relative to a finer numerical run (not physical ground truth)")
    fig.savefig(directory / "time_step_error.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/toy.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/synthetic_validation")
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--time-steps", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--reference-dt", type=float, default=0.0005)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive and finite")
    for dt in [*args.time_steps, args.reference_dt]:
        if not math.isfinite(dt) or dt <= 0.0 or dt > args.duration:
            parser.error("time steps must be positive, finite and <= duration")
        ratio = args.duration / dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0.0):
            parser.error("duration must be an integer multiple of every time step")
    if args.reference_dt >= min(args.time_steps):
        parser.error("--reference-dt must be smaller than every compared time step")
    args.time_steps = sorted(set(args.time_steps))
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    experiment = Experiment(config, torch.device(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    single, gates = experiment.single_steps()
    no_contact = experiment.rollout(args.time_steps[0], args.duration, no_contact=True)
    runs = [experiment.rollout(dt, args.duration) for dt in args.time_steps]
    reference = experiment.rollout(args.reference_dt, args.duration)
    convergence = [
        {
            "dt": run[0]["dt"],
            "steps": len(run),
            "final_z": run[-1]["position_z"],
            "final_vz": run[-1]["velocity_z"],
            "position_error": abs(run[-1]["position_z"] - reference[-1]["position_z"]),
            "velocity_error": abs(run[-1]["velocity_z"] - reference[-1]["velocity_z"]),
        }
        for run in runs
    ]
    # Record convergence as a measurement, not an assumed physical validation criterion.
    summary = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "matplotlib": version("matplotlib"),
            "architecture": platform.machine(),
            "device": args.device,
        },
        "config": config,
        "duration": args.duration,
        "reference_dt": args.reference_dt,
        "reference_final_z": reference[-1]["position_z"],
        "reference_final_vz": reference[-1]["velocity_z"],
        "checks": experiment.checks,
        "gate_maps": gates,
        "time_step_comparison": convergence,
        "interpretation": "Finite runs do not prove stability. Reference is numerical, not truth. "
        "No gravity, floor or restoring force: a separated object coasts.",
    }
    write_csv(args.output_dir / "single_steps.csv", single)
    write_csv(
        args.output_dir / "rollouts.csv", [r for run in [*runs, reference, no_contact] for r in run]
    )
    write_csv(args.output_dir / "time_step_comparison.csv", convergence)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    make_plots(args.output_dir, single, gates, runs, convergence)
    for check in experiment.checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print("\nTime step comparison (same duration; finer numerical reference):")
    print("dt       steps     final_z     final_vz    position_error   velocity_error")
    for row in convergence:
        print(
            f"{row['dt']:<8g} {row['steps']:<8d} {row['final_z']: .6f}   "
            f"{row['final_vz']: .6f}    {row['position_error']:.6g}      "
            f"{row['velocity_error']:.6g}"
        )
    failures = sum(not check["passed"] for check in experiment.checks)
    print(f"\nChecks: {len(experiment.checks) - failures}/{len(experiment.checks)} passed")
    print(f"Artifacts: {args.output_dir.resolve()}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
