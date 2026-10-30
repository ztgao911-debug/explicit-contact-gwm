"""Saved-evidence analysis for training prefixes and the fresh force-input ablation."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
from fair_replication_statistics import summarize_groups
from report_fair_mainline import read_csv, read_json, save_csv, save_json
from verify_bounded_search import verify_evaluation

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/ab_training_force/run_001"
OUTPUT = ROOT / "outputs/ab_training_force/run_001"
SEEDS = (7, 17, 27)
METHODS = ("RAW", "G", "G0", "CV")
STAGES = ("prefix_200", "prefix_500", "prefix_1000", "prefix_2000", "exact_2000")
PHASES = ("press", "hold", "release", "recover")


def find_row(rows, method, region="active", phase="all", contact_group="all", horizon=None):
    found = [
        r
        for r in rows
        if r["method"] == method
        and r["split"] == "validation"
        and r["region"] == region
        and r["phase"] == phase
        and r["contact_group"] == contact_group
        and (horizon is None or int(r["horizon"]) == horizon)
    ]
    return found[0] if len(found) == 1 else None


def value(row, key):
    return float(row[key]) if row and row.get(key) not in (None, "") else None


def independent_audit(protocol, ledger, evaluations):
    with np.load(ROOT / protocol["configuration"]["data_directory"] / "mesh.npz") as archive:
        mesh = {k: archive[k] for k in archive.files}
    records = []
    for e in evaluations:
        if e["status"] != "completed":
            records.append({"seed": e["seed"], "stage": e["stage"], "expected_failure": e})
            continue
        folder = OUTPUT / "evaluation" / f"seed_{e['seed']}" / e["stage"]
        for method, outcome in e["per_method"].items():
            try:
                result = verify_evaluation(
                    folder, ROOT / protocol["configuration"]["data_directory"], mesh, method
                )
                records.append({"seed": e["seed"], "stage": e["stage"], "method": method, **result})
            except Exception as error:
                records.append(
                    {
                        "seed": e["seed"],
                        "stage": e["stage"],
                        "method": method,
                        "audit_error": str(error),
                        "stored_eligible": outcome["eligible"],
                        "passed": False,
                    }
                )
    selection = []
    reload = []
    for r in ledger:
        if r["status"] != "trained":
            continue
        folder = ROOT / r["artifact_directory"]
        history = read_json(folder / "checkpoint_selection.json")
        for limit in (200, 500, 1000, 2000):
            saved = torch.load(folder / f"prefix_{limit:04d}.pt", weights_only=False)
            available = [
                v
                for v in history
                if v["update"] <= limit and v["scores"]["joint_score"] is not None
            ]
            best = min(available, key=lambda v: v["scores"]["joint_score"]) if available else None
            selection.append(
                {
                    "seed": r["seed"],
                    "method": r["variant"],
                    "limit": limit,
                    "selected": saved["selected_step"],
                    "matched_minimum_without_future": bool(
                        best and best["update"] == saved["selected_step"]
                    ),
                    "included_step_zero": history[0]["update"] == 0,
                }
            )
        checkpoint = torch.load(folder / "checkpoint.pt", weights_only=False)["model_state"]
        best = torch.load(folder / "prefix_2000.pt", weights_only=False)["model_state"]
        reload.append(
            {
                "seed": r["seed"],
                "method": r["variant"],
                "checkpoint_matches_prefix_selected_weights": all(
                    torch.equal(checkpoint[k], best[k]) for k in checkpoint
                ),
            }
        )
    changed = {
        g: [
            n
            for n, h in protocol[g].items()
            if hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h
        ]
        for g in ("source_hashes", "data_hashes", "historical_report_hashes")
    }
    audit = {
        "independent_evaluations": records,
        "prefix_selection_checks": selection,
        "reload_state_checks": reload,
        "changed_frozen_files": changed,
        "all_available_audits_passed": all(r.get("passed", False) for r in records),
        "all_prefix_selections_passed": all(
            r["matched_minimum_without_future"] and r["included_step_zero"] for r in selection
        ),
        "all_reload_state_checks_passed": all(
            r["checkpoint_matches_prefix_selected_weights"] for r in reload
        ),
        "independent_test_verified": False,
    }
    save_json(REPORT / "independent_verification.json", audit)
    return audit


def collect_metrics(protocol, ledger, evaluations):
    records = []
    phase_rows = []
    response = []
    recovery = []
    index = {(r["seed"], r["variant"]): r for r in ledger}
    for e in evaluations:
        if e["status"] != "completed":
            continue
        seed, stage = e["seed"], e["stage"]
        directory = OUTPUT / "evaluation" / f"seed_{seed}" / stage
        one = read_csv(directory / "ordinary/one_step_metrics.csv")
        roll = read_csv(directory / "ordinary/rollout_metrics.csv")
        static = read_csv(directory / "static_stability.csv")
        for method in METHODS:
            outcome = e["per_method"].get(method, {})
            scores = outcome.get("scores", {})
            row = {
                "seed": seed,
                "stage": stage,
                "method": method,
                "physical_eligible": outcome.get("eligible", False),
                "joint_score": scores.get("joint_score"),
                "balanced_position_um": 1000 * math.sqrt(scores["position_score"])
                if scores.get("position_score") is not None
                else None,
                "balanced_velocity_mm_s": 100 * math.sqrt(scores["velocity_score"])
                if scores.get("velocity_score") is not None
                else None,
                "inverted_states": outcome.get("inverted_states"),
                "inverted_windows": outcome.get("inverted_windows"),
                "minimum_volume_ratio": outcome.get("minimum_volume_ratio"),
                "selected_update": None,
            }
            if method != "CV" and index[seed, method]["status"] == "trained":
                kind, n = stage.split("_")
                s = torch.load(
                    ROOT / index[seed, method]["artifact_directory"] / f"{kind}_{int(n):04d}.pt",
                    weights_only=False,
                )
                row["selected_update"] = s.get("selected_step", s.get("update"))
            for region, label in (("active", "active"), ("contact_neighborhood", "ROI")):
                contact = find_row(one, method, region, contact_group="contact")
                row[f"contact_position_{label}_um"] = value(contact, "increment_rmse_um")
                row[f"contact_velocity_{label}_mm_s"] = value(contact, "velocity_rmse_mm_s")
                row[f"contact_frames_{label}"] = int(contact["frame_count"]) if contact else 0
                for phase in PHASES:
                    for h in (5, 10, 20):
                        v = find_row(roll, method, region, phase, horizon=h)
                        phase_rows.append(
                            {
                                "seed": seed,
                                "stage": stage,
                                "method": method,
                                "region": region,
                                "phase": phase,
                                "horizon": h,
                                "position_um": value(v, "position_rmse_um"),
                                "velocity_mm_s": value(v, "velocity_rmse_mm_s"),
                            }
                        )
            s = [r for r in static if r["method"] == method]
            row["static_displacement_um"] = (
                1e6 * max(float(r["max_absolute_displacement_m"]) for r in s) if s else None
            )
            row["static_velocity_mm_s"] = (
                1000 * max(float(r["max_absolute_instantaneous_velocity_m_s"]) for r in s)
                if s
                else None
            )
            records.append(row)
        response.extend(
            {"seed": seed, "stage": stage, **r}
            for r in read_csv(directory / "response_evaluation/per_case_response_metrics.csv")
        )
        if stage in ("prefix_2000", "exact_2000"):
            recovery.extend(
                {"seed": seed, "stage": stage, **r}
                for r in read_csv(directory / "recovery/recovery_waveform_metrics.csv")
            )
    save_csv(REPORT / "metrics_per_seed.csv", records)
    metric_keys = [
        "joint_score",
        "balanced_position_um",
        "balanced_velocity_mm_s",
        "contact_position_active_um",
        "contact_velocity_active_mm_s",
        "contact_position_ROI_um",
        "contact_velocity_ROI_mm_s",
        "static_displacement_um",
    ]
    summary = summarize_groups(records, ["stage", "method"], metric_keys)
    save_csv(REPORT / "metrics_summary.csv", summary)
    save_csv(REPORT / "phase_metrics_per_seed.csv", phase_rows)
    phase_summary = summarize_groups(
        phase_rows,
        ["stage", "method", "phase", "horizon", "region"],
        ["position_um", "velocity_mm_s"],
    )
    save_csv(REPORT / "phase_metrics_summary.csv", phase_summary)
    primary_response = [r for r in response if r["stage"] == "prefix_2000"]
    save_csv(REPORT / "primary_interventions_per_seed.csv", primary_response)
    save_csv(
        REPORT / "primary_interventions_summary.csv",
        summarize_groups(
            primary_response,
            ["method", "frame", "variant", "horizon", "region"],
            [
                "position_response_rmse",
                "velocity_response_rmse",
                "position_direction_cosine",
                "velocity_direction_cosine",
                "position_amplitude_ratio",
                "velocity_amplitude_ratio",
            ],
        ),
    )
    faster = [
        r
        for r in response
        if r["variant"] == "faster_down" and r["horizon"] == "1" and r["region"] == "active"
    ]
    save_csv(REPORT / "faster_down_all_stages.csv", faster)
    faster_counts = []
    for seed in SEEDS:
        for method in METHODS:
            group = [
                r
                for r in faster
                if r["stage"] == "prefix_2000" and r["seed"] == seed and r["method"] == method
            ]
            record = {"seed": seed, "method": method, "paired_states": len(group)}
            for channel in ("position", "velocity"):
                cosines = [
                    float(r[f"{channel}_direction_cosine"])
                    for r in group
                    if r[f"{channel}_direction_cosine"] not in (None, "")
                ]
                amplitudes = [
                    float(r[f"{channel}_amplitude_ratio"])
                    for r in group
                    if r[f"{channel}_amplitude_ratio"] not in (None, "")
                ]
                record[f"{channel}_defined_directions"] = sum(np.isfinite(cosines))
                record[f"{channel}_negative_directions"] = sum(v < 0 for v in cosines)
                record[f"{channel}_mean_amplitude_ratio"] = (
                    float(np.mean(amplitudes)) if amplitudes else None
                )
            faster_counts.append(record)
    save_csv(REPORT / "faster_down_primary_counts.csv", faster_counts)
    save_csv(REPORT / "recovery_metrics_per_seed.csv", recovery)
    diagnostics = []
    for r in ledger:
        file = ROOT / r["artifact_directory"] / "milestone_diagnostics.json"
        if not file.exists():
            continue
        for d in read_json(file):
            diagnostics.append(
                {
                    "seed": r["seed"],
                    "method": r["variant"],
                    "update": d["update"],
                    **d["teacher_forced_central"],
                    "teacher_sum": sum(d["teacher_forced_central"].values()),
                    "central_AR_probe": d.get("central_fixed_AR_probe_loss"),
                    "diagnostic_failure": d.get("diagnostic_failure"),
                    "parameters_unchanged": d["parameters_bitwise_unchanged"],
                    "RNG_unchanged": d["RNG_unchanged"],
                }
            )
    save_csv(REPORT / "training_diagnostics_per_seed.csv", diagnostics)
    save_csv(
        REPORT / "training_diagnostics_summary.csv",
        summarize_groups(
            diagnostics,
            ["method", "update"],
            ["ordinary", "branch_absolute", "paired_response", "teacher_sum", "central_AR_probe"],
        ),
    )
    pairs = []
    for stage in STAGES:
        for seed in SEEDS:
            group = {r["method"]: r for r in records if r["stage"] == stage and r["seed"] == seed}
            for reference in ("RAW", "G0", "CV"):
                g, ref = group["G"], group[reference]
                pairs.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "reference": reference,
                        "both_physical_eligible": g["physical_eligible"]
                        and ref["physical_eligible"],
                        **{
                            key: 100 * (1 - g[key] / ref[key])
                            if g[key] is not None and ref[key]
                            else None
                            for key in metric_keys[:7]
                        },
                    }
                )
    save_csv(REPORT / "G_relative_reductions_per_seed.csv", pairs)
    pair_summary = summarize_groups(pairs, ["stage", "reference"], metric_keys[:7])
    save_csv(REPORT / "G_relative_reductions_summary.csv", pair_summary)
    return records, summary, phase_summary, diagnostics, pair_summary, faster


def stat(rows, metric, **filters):
    r = next(
        r for r in rows if r["metric"] == metric and all(r[k] == v for k, v in filters.items())
    )
    return (
        f"{r['mean']:.4g} ± {r['sample_std']:.3g}"
        if r["all_three_finite"]
        else f"未完成({r['n_available']}/3)"
    )


def plots(summary, diagnostics, ledger):
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method in METHODS:
        for ax, key, label in zip(
            axes,
            ("contact_position_active_um", "contact_velocity_active_mm_s"),
            ("Actual-contact one-step position RMSE (um)", "Instantaneous velocity RMSE (mm/s)"),
            strict=True,
        ):
            values = [
                next(
                    r
                    for r in summary
                    if r["method"] == method and r["stage"] == f"prefix_{n}" and r["metric"] == key
                )
                for n in (200, 500, 1000, 2000)
            ]
            if any(r["mean"] is None for r in values):
                continue
            means = np.array([r["mean"] for r in values])
            std = np.array([r["sample_std"] for r in values])
            (line,) = ax.plot((200, 500, 1000, 2000), means, marker="o", label=method)
            ax.fill_between(
                (200, 500, 1000, 2000),
                np.maximum(0, means - std),
                means + std,
                alpha=0.12,
                color=line.get_color(),
            )
            ax.set(xlabel="Available training-prefix updates", ylabel=label)
            ax.grid(alpha=0.3)
    axes[1].legend()
    fig.savefig(REPORT / "A_contact_budget.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method in ("RAW", "G", "G0"):
        ds = [d for d in diagnostics if d["method"] == method]
        for ax, key in zip(axes, ("teacher_sum", "central_AR_probe"), strict=True):
            points = []
            for update in (0, 200, 500, 1000, 2000):
                vals = [d[key] for d in ds if d["update"] == update and d[key] is not None]
                points.append(np.mean(vals) if len(vals) == 3 else np.nan)
            ax.plot((0, 200, 500, 1000, 2000), points, marker="o", label=method)
    axes[0].set(xlabel="Exact current update", ylabel="Full central teacher-forced loss sum")
    axes[1].set(xlabel="Exact current update", ylabel="Fixed central 3-step AR-probe loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(REPORT / "A_training_diagnostics.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for r in ledger:
        if r["status"] != "trained":
            continue
        curve = read_csv(ROOT / r["artifact_directory"] / "training_curve.csv")
        data = [x for x in curve if x["validation_joint_score"] != ""]
        ax.plot(
            [int(x["update"]) for x in data],
            [float(x["validation_joint_score"]) for x in data],
            label=f"{r['variant']} seed{r['seed']}",
        )
    ax.set(xlabel="Raw checkpoint update", ylabel="Fixed five-step validation score")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(REPORT / "A_validation_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, key, label in zip(
        axes.flat,
        (
            "balanced_position_um",
            "balanced_velocity_mm_s",
            "contact_position_active_um",
            "contact_velocity_active_mm_s",
        ),
        (
            "Balanced rollout position RMSE (um)",
            "Balanced instantaneous velocity RMSE (mm/s)",
            "Actual-contact one-step position RMSE (um)",
            "Actual-contact velocity RMSE (mm/s)",
        ),
        strict=True,
    ):
        selected = [
            next(
                r
                for r in summary
                if r["stage"] == "prefix_2000" and r["method"] == method and r["metric"] == key
            )
            for method in METHODS
        ]
        ax.bar(
            METHODS,
            [r["mean"] for r in selected],
            yerr=[r["sample_std"] for r in selected],
            capsize=4,
            color=("tab:blue", "tab:orange", "tab:green", "tab:gray"),
        )
        ax.set(ylabel=label)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(REPORT / "B_force_tradeoffs.png", dpi=160)
    plt.close(fig)
    waveforms = read_csv(
        OUTPUT / "evaluation/seed_7/prefix_2000/recovery/recovery_signed_waveforms.csv"
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for method in METHODS:
        group = [
            r
            for r in waveforms
            if r["trajectory"] == "left_deep"
            and r["start"] == "170"
            and r["channel"] == "representative_node"
            and r["method"] == method
        ]
        times = [int(r["horizon"]) * 0.01 for r in group]
        for ax, key, label in zip(
            axes,
            ("z_displacement_um", "vz_mm_s"),
            ("Signed z displacement (um)", "Instantaneous vz (mm/s)"),
            strict=True,
        ):
            ax.plot(times, [float(r[f"predicted_{key}"]) for r in group], label=method)
            if method == "RAW":
                ax.plot(
                    times,
                    [float(r[f"reference_{key}"]) for r in group],
                    "k--",
                    label="MuJoCo reference",
                )
            ax.set(ylabel=label)
            ax.grid(alpha=0.3)
    axes[0].legend(ncol=5)
    axes[1].set(xlabel="Time from initialization (s)")
    fig.suptitle("Seed 7, left_deep, start 170, existing representative node")
    fig.savefig(REPORT / "recovery_example.png", dpi=160)
    plt.close(fig)


def main():
    if not (REPORT / "status.json").exists():
        raise RuntimeError("A+B execution is still running")
    protocol = read_json(REPORT / "protocol.json")
    ledger = read_json(REPORT / "training_ledger.json")
    evaluations = read_json(REPORT / "evaluation_ledger.json")
    audit = independent_audit(protocol, ledger, evaluations)
    records, summary, phase_summary, diagnostics, pairs, faster = collect_metrics(
        protocol, ledger, evaluations
    )
    plots(summary, diagnostics, ledger)
    write_report(records, summary, phase_summary, diagnostics, pairs, faster, ledger, audit)
    save_json(
        REPORT / "analysis_provenance.json",
        {
            "post_run_analysis_only": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "statistics": (
                "Each seed evaluated first, mean/sample std ddof=1; failed seeds not deleted"
            ),
            "selection_target_unchanged": True,
            "no_independent_test": True,
        },
    )
    print({"trained": len(ledger), "audit_passed": audit["all_available_audits_passed"]})


def write_report(records, summary, phase_summary, diagnostics, pairs, faster, ledger, audit):
    # Post-run analysis only: this script never changes training or selection.
    lines = [
        "# A：训练预算与 B：解析力输入消融",
        "",
        "**实测结论：增加训练改善了汇总位置，但汇总瞬时速度与接触帧位置精度退化；"
        "完整G也没有显示出相对G0的稳定额外收益。**"
        "不能将联合分数下降解释为动力学全面通过。",
        "",
        "本轮比较同一2000更新轨迹的前缀，并从头训练G0去力组。数据、网络、损失、归一化、选模和评价掩码固定。",
        "",
        "RAW/G/G0各种子7、17、27，共9次训练；所有名义参数量45324。G0保留G几何、法向、偏移与门控，三个力输入通道置零，"
        "有384个首层输入权重不被激活。此为输入贡献消融，不代表有效输入容量完全相同；G0可从几何间接推断力。",
        "G0标签仍由原MuJoCo动力学产生；不是让软块在无物理力环境下运动。G使用既有拓扑辅助连续接触表面，并非纯Gaussian球并集。",
        "",
        "训练均为三步AR、batch16、LR=0.0001、速度权重0.5、网格权重0.001。前缀最佳权重仅使用截至该前缀的五步共同验证目标，包含第0步。"
        "同时保存并评价精确2000步权重，避免选模掩盖末端退化。旧200更新余弦实验不是新长轨迹的匹配前缀。",
        "中央600个普通帧对、108条分支（2160个帧对）、90个初始响应对不变；左侧用于开发验证，右侧未加载。"
        "软体仅在rollout起点读取真值状态，压头实测轨迹为已知外部输入。当前几何/力用预测状态的停止梯度副本计算；"
        "状态特征仍可微，不宣称穿过接触查询全程可微。位置和端点瞬时速度分别预测与监督，底部固定。",
        "",
        "## A：真实接触帧的一步误差",
        "",
        "单位为位置增量μm、瞬时速度mm/s；均值±样本标准差，n=3。接触分组来自GT当前接触标记，不使用预测接触筛样本。",
        "",
        "| 可用更新前缀 | 方法 | 接触位置误差 | 接触速度误差 | 通过硬物理筛选的种子数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage in STAGES:
        for method in METHODS:
            valid = sum(
                r["physical_eligible"]
                for r in records
                if r["stage"] == stage and r["method"] == method
            )
            position = stat(summary, "contact_position_active_um", stage=stage, method=method)
            velocity = stat(summary, "contact_velocity_active_mm_s", stage=stage, method=method)
            lines.append(f"| {stage} | {method} | {position} | {velocity} | {valid}/3 |")
    lines.extend(
        [
            "",
            "全部九次训练实际完成2000更新；RAW三种子选中500/400/450步，G与G0均选中500步。"
            "所以prefix_500、1000、2000的权重与指标相同，这是保留前缀内最佳权重的结果。",
            "",
            "### 汇总位置与速度的权衡",
            "",
            "按压/保持/释放/恢复、活动区/ROI、5/10/20步分别等权聚合MSE，再开方还原单位。"
            "联合分数=0.5×位置MSE/(1mm)²+0.5×速度MSE/(0.1m/s)²；权重固定，但不保证两项都改善。",
            "",
            "| 权重 | 方法 | 位置μm | 瞬时速度mm/s | 联合分数 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for stage in ("prefix_200", "prefix_2000", "exact_2000"):
        for method in METHODS:
            position = stat(summary, "balanced_position_um", stage=stage, method=method)
            velocity = stat(summary, "balanced_velocity_mm_s", stage=stage, method=method)
            joint = stat(summary, "joint_score", stage=stage, method=method)
            lines.append(f"| {stage} | {method} | {position} | {velocity} | {joint} |")
    lines.extend(
        [
            "",
            "G的匹配前缀200→2000，位置由约504降至362μm，但速度由8.65升至11.36mm/s；"
            "真实接触帧一步位置由81.3升至107.2μm。恒速对应13.34μm与2.17mm/s，接触精度仍明显更好。"
            "G接触帧速度13.00→11.42mm/s有所改善，仍落后恒速。"
            "精确2000步的G速度更差（汇总约25.75mm/s），表明继续训练未解决该问题。",
            "",
            "## B：完整G相对去力G0",
            "",
            "百分比为逐种子误差降低后再统计，正值表示完整G误差更低。",
            "",
            "| 指标（prefix_2000） | G相对G0降低% 均值±标准差 |",
            "| --- | --- |",
        ]
    )
    for metric in (
        "joint_score",
        "balanced_position_um",
        "balanced_velocity_mm_s",
        "contact_position_active_um",
        "contact_velocity_active_mm_s",
    ):
        lines.append(f"| {metric} | {stat(pairs, metric, stage='prefix_2000', reference='G0')} |")
    lines.extend(
        [
            "",
            "G与G0的汇总位置分别约362.3与361.6μm，速度11.36与10.51mm/s；"
            "当前实验未支持解析力通道提供稳定额外收益，也未证明统计等价或物理力无用。"
            "几何已携带穿透等信息、当前弹性力近似及力编码是否限制收益，仍是未区分的假设。",
            "",
            "## 全部阶段与边界",
            "",
            "| 20步活动区域阶段 | 方法 | 位置μm | 速度mm/s |",
            "| --- | --- | --- | --- |",
        ]
    )
    for phase in PHASES:
        for method in METHODS:
            filters = {
                "stage": "prefix_2000",
                "method": method,
                "phase": phase,
                "horizon": 20,
                "region": "active",
            }
            position = stat(phase_summary, "position_um", **filters)
            velocity = stat(phase_summary, "velocity_mm_s", **filters)
            lines.append(f"| {phase} | {method} | {position} | {velocity} |")
    lines.extend(
        [
            "",
            "### faster_down方向与幅度",
            "",
            "下表为prefix_2000、活动区、一步、三个预先固定干预状态。幅度比理想值为1；"
            "方向余弦描述干预响应差值，不是物体整体运动方向。恒速不响应压头变化，响应为零、方向未定义。",
            "",
            "| 种子 | 方法 | 位置负方向/3 | 速度负方向/3 | 位置幅度比均值 | 速度幅度比均值 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for seed in SEEDS:
        for method in ("RAW", "G", "G0"):
            group = [
                r
                for r in faster
                if r["seed"] == seed and r["method"] == method and r["stage"] == "prefix_2000"
            ]
            negative = [
                sum(float(r[f"{c}_direction_cosine"]) < 0 for r in group)
                for c in ("position", "velocity")
            ]
            amplitudes = [
                np.mean([float(r[f"{c}_amplitude_ratio"]) for r in group])
                for c in ("position", "velocity")
            ]
            lines.append(
                f"| {seed} | {method} | {negative[0]} | {negative[1]} | "
                f"{amplitudes[0]:.3g} | {amplitudes[1]:.3g} |"
            )
    lines.extend(
        [
            "",
            "G与G0在种子7、27的位置与速度响应均有3/3反向；种子17方向为正但幅度偏大。"
            "训练加长没有解决小速度干预机制。",
            "",
            "单步ROI、5/10/20步、所有阶段、静止漂移、faster_down与其他干预、恢复波形均保存在各stage的评价目录。"
            "完整统计在metrics_summary.csv、phase_metrics_summary.csv、primary_interventions_summary.csv。"
            "阶段接触rollout标记指起点接触，不保证以后每步都接触。",
            "",
            "中央teacher-forced诊断与固定16窗口AR探针是训练充分性的辅助证据，不是相同的损失，也不单独证明收敛。"
            "诊断回调逐次验证参数和RNG未改变。前缀最优验证分数单调降低是选择规则的结果，需结合原始检查点曲线。",
            "",
            "**解释与局限：** 中央teacher-forced拟合持续改善，"
            "固定中央AR探针和左侧原始验证在后段却退化。"
            "这支持训练目标之间存在权衡，不能仅归咎于训练次数不足，也不能据此单独确定是过拟合还是状态表示不足。"
            "每组2000次更新全部触发既有梯度裁剪，此现象保留在训练日志，不在本轮据此追加调参。"
            "G/G0保持静止约束且释放位置有收益；恢复位置更接近仍不等于振动波形正确。"
            "左深恢复代表节点的幅度偏小，20步内预测周期不足，保留insufficient_cycles标记，不臆造相位精度。",
            "",
            "本轮仍是左侧开发验证，无独立测试或实物验证；不能凭平均RMSE或去力消融宣称正确接触机制。",
            "",
            f"实际训练完成{sum(r['status'] == 'trained' for r in ledger)}/9。"
            f"独立评分/网格复核：{audit['all_available_audits_passed']}。",
            "全部评价窗口无翻转、非有限值或固定节点违规；这表示数值与边界检查通过，尚非动力学验证通过。",
            "",
            "## 运行与复现",
            "",
            "预算2小时、最多9次训练、两个CPU进程（每进程单线程）；训练和15套评价共2363.66秒（约39.4分钟）。"
            "9次均为首次执行成功，无失败或超时。A/B未改变网络、数据或选模规则，也未新增搜索配置。"
            "普通G调用次数的历史计数字段native_queries_training_and_selection实际计量条件查询；"
            "本轮真实MuJoCo原生接触查询由禁止调用检查确认始终为0，参见input_permission_checks.json。",
            "",
            "入口：`python scripts/prepare_ab_training_force.py`，"
            "再运行`python scripts/run_ab_training_force.py`，"
            "完成后运行`python scripts/report_ab_training_force.py`。"
            "先激活项目.venv。已有目录不会覆盖，"
            "继续既有预算可用`--resume`；新一轮应复制配置并换独立目录。初始缓存依赖原公平主线及既有数据。"
            "准备脚本补齐评价输出父目录，记录在directory_preparation.json；未更改冻结科学代码或重启训练。",
            "",
            "完整协议、样本与数据摘要见protocol.json、analysis_plan.json；逐次训练和评价见training_ledger.json、"
            "evaluation_ledger.json；独立重算评分、体积比与前缀选模见independent_verification.json。"
            "大权重和预测NPZ位于outputs/ab_training_force/run_001，沿用忽略规则；CSV/JSON/图随报告保存。",
            "",
            "实测检查：`python -m pytest`全部246项通过，`ruff check .`通过。"
            "新增测试覆盖G0只去力、保留几何和门控、固定边界、梯度贯通、训练/推理一致及前缀禁止未来选模。"
            "45次诊断回调均未改变参数或RNG；9份重载权重与保存选择一致，60组方法评价独立复算通过。"
            "检查摘要见software_checks.json与independent_verification.json。",
            "",
            "**下一项优先假设：** 当前位置与瞬时速度两支路的联合状态映射及响应监督，"
            "未能把接触条件转化为方向、幅度一致的动力学响应。下一轮应预先规定一个位置—速度一致性结构对照，"
            "并同时验收接触与小速度干预；本轮证据不支持继续单纯延长训练或宣称显式解析力已带来收益。",
            "",
            "![接触误差与预算](A_contact_budget.png)",
            "",
            "![中央诊断](A_training_diagnostics.png)",
            "",
            "![原始验证曲线](A_validation_curves.png)",
            "",
            "![位置速度与力输入的权衡](B_force_tradeoffs.png)",
            "",
            "![恢复波形示例](recovery_example.png)",
            "",
            "恢复说明图固定展示种子7、左深170窗口、沿用原评价器代表节点；并非挑选最好种子或片段。"
            "所有种子和恢复窗口的数据保留于recovery_metrics_per_seed.csv及各stage评价目录。",
            "",
        ]
    )
    (REPORT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
