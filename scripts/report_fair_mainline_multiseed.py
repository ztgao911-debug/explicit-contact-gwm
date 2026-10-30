"""Independent audits and complete seed-7/17/27 statistics; no training or tuning."""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from fair_replication_statistics import summarize_groups
from report_fair_mainline import read_csv, read_json, save_csv, save_json
from verify_bounded_search import verify_evaluation

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/fair_mainline_multiseed/run_001"
OUTPUT = ROOT / "outputs/fair_mainline_multiseed/run_001"
SEEDS = (7, 17, 27)
METHODS = ("RAW", "G", "CV")
PHASES = ("press", "hold", "release", "recover")


def balanced_rows(folder, seed):
    balanced, phases = [], []
    for filename, horizons, position_key in (
        ("one_step_metrics.csv", (1,), "increment_rmse_um"),
        ("rollout_metrics.csv", (5, 10, 20), "position_rmse_um"),
    ):
        raw = read_csv(folder / "evaluation/ordinary" / filename)
        for method in METHODS:
            for horizon in horizons:
                for region in ("active", "contact_neighborhood"):
                    subset = [
                        r
                        for r in raw
                        if r["method"] == method
                        and r["split"] == "validation"
                        and r["phase"] in PHASES
                        and r["region"] == region
                        and r["contact_group"] == "all"
                        and (horizon == 1 or int(r["horizon"]) == horizon)
                    ]
                    if len(subset) != 4:
                        continue
                    balanced.append(
                        {
                            "seed": seed,
                            "method": method,
                            "horizon": horizon,
                            "region": region,
                            "position_rmse_um": math.sqrt(
                                np.mean([float(r[position_key]) ** 2 for r in subset])
                            ),
                            "velocity_rmse_mm_s": math.sqrt(
                                np.mean([float(r["velocity_rmse_mm_s"]) ** 2 for r in subset])
                            ),
                        }
                    )
                    phases.extend(
                        {
                            "seed": seed,
                            "method": method,
                            "horizon": horizon,
                            "region": region,
                            "phase": r["phase"],
                            "position_rmse_um": float(r[position_key]),
                            "velocity_rmse_mm_s": float(r["velocity_rmse_mm_s"]),
                        }
                        for r in subset
                    )
    return balanced, phases


def format_stat(rows, method, metric):
    row = next(r for r in rows if r["method"] == method and r["metric"] == metric)
    if not row["all_three_finite"]:
        return f"未完成（{row['n_available']}/3）"
    return f"{row['mean']:.4g} ± {row['sample_std']:.3g}"


def main():
    protocol = read_json(REPORT / "protocol.json")
    parent_report = ROOT / protocol["configuration"]["replication"]["parent_report"]
    parent_output = ROOT / protocol["configuration"]["replication"]["parent_output"]
    report_paths = {7: parent_report, **{s: REPORT / f"seed_{s}" for s in (17, 27)}}
    output_paths = {7: parent_output, **{s: OUTPUT / f"seed_{s}" for s in (17, 27)}}
    summaries = {s: read_json(report_paths[s] / "evaluation_summary.json") for s in SEEDS}
    parent_ledger = read_json(parent_report / "training_ledger.json")
    new_ledger = read_json(REPORT / "training_ledger.json")
    training = {(7, r["variant"]): r for r in parent_ledger}
    training.update({(int(r["seed"]), r["variant"]): r for r in new_ledger})
    data = ROOT / protocol["configuration"]["data_directory"]
    with np.load(data / "mesh.npz") as saved:
        mesh = {k: saved[k] for k in saved.files}
    independent = {
        str(s): {
            m: verify_evaluation(output_paths[s] / "evaluation", data, mesh, m)
            for m in METHODS
            if m in summaries[s]
        }
        for s in SEEDS
    }
    selection = {}
    for seed in SEEDS:
        for method in ("RAW", "G"):
            result = training[seed, method]
            if result["status"] != "trained":
                selection[f"{seed}_{method}"] = {"status": result["status"], "selected": None}
                continue
            history = read_json(output_paths[seed] / method / "checkpoint_selection.json")
            available = [r for r in history if r["scores"]["joint_score"] is not None]
            best = min(available, key=lambda r: r["scores"]["joint_score"])
            selection[f"{seed}_{method}"] = {
                "update_zero_included": history[0]["update"] == 0,
                "minimum_common_score_selected": best["update"]
                == result["training"]["selected_step"],
                "selected": best["update"],
                "status": result["status"],
            }
    cv_checks = {}
    for seed in (17, 27):
        matched, arrays = True, 0
        for filename in ("rollout_predictions.npz", "one_step_predictions.npz"):
            with (
                np.load(parent_output / "evaluation/ordinary" / filename) as a,
                np.load(output_paths[seed] / "evaluation/ordinary" / filename) as b,
            ):
                keys = [k for k in a.files if "__CV__" in k]
                assert keys
                matched = matched and all(k in b and np.array_equal(a[k], b[k]) for k in keys)
                arrays += len(keys)
        cv_checks[str(seed)] = {"bitwise_equal_to_seed7": matched, "arrays_checked": arrays}
    changed = {
        group: [
            n
            for n, h in protocol[group].items()
            if hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h
        ]
        for group in ("source_hashes", "data_hashes")
    }
    seed7_changed = [
        n
        for n, h in protocol["seed7_report_hashes_before"].items()
        if hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h
    ]
    old_artifacts_changed = [
        n
        for n, record in protocol["seed7_large_artifact_hashes"].items()
        if hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != record["sha256"]
    ]
    audit = {
        "independent_per_seed_method": independent,
        "checkpoint_selection": selection,
        "CV_deterministic_reference_checks": cv_checks,
        "changed_frozen_files": changed,
        "original_seed7_report_changed": seed7_changed,
        "original_seed7_large_artifacts_changed": old_artifacts_changed,
        "audit_passed": all(v["passed"] for d in independent.values() for v in d.values())
        and not any(changed.values())
        and not seed7_changed
        and not old_artifacts_changed
        and all(c["bitwise_equal_to_seed7"] for c in cv_checks.values())
        and all(
            c.get("update_zero_included") and c.get("minimum_common_score_selected")
            for c in selection.values()
            if c["status"] == "trained"
        ),
        "independent_test_verified": False,
    }
    save_json(REPORT / "independent_verification.json", audit)
    assert audit["audit_passed"]
    metrics, balanced, phase_records, response_records, recovery_records, phase_statuses = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seed in SEEDS:
        static = read_csv(report_paths[seed] / "evaluation/static_stability.csv")
        for method in METHODS:
            outcome = summaries[seed].get(method, {})
            scores = outcome.get("scores", {})
            train = training.get((seed, method), {})
            row = {
                "seed": seed,
                "method": method,
                "training_status": train.get("status", "untrained_deterministic_reference"),
                "selected_update": train.get("training", {}).get("selected_step"),
                "physically_valid_finite_pins_mesh": outcome.get("eligible", False),
                "joint_score": scores.get("joint_score"),
                "position_rmse_um": 1000 * math.sqrt(scores["position_score"])
                if scores.get("position_score") is not None
                else None,
                "velocity_rmse_mm_s": 100 * math.sqrt(scores["velocity_score"])
                if scores.get("velocity_score") is not None
                else None,
                "inverted_states": outcome.get("inverted_states"),
                "inverted_windows": outcome.get("inverted_windows"),
                "minimum_volume_ratio": outcome.get("minimum_volume_ratio"),
                "static_displacement_um": 1e6
                * max(
                    float(r["max_absolute_displacement_m"]) for r in static if r["method"] == method
                ),
                "static_velocity_mm_s": 1000
                * max(
                    float(r["max_absolute_instantaneous_velocity_m_s"])
                    for r in static
                    if r["method"] == method
                ),
            }
            metrics.append(row)
        b, p = balanced_rows(report_paths[seed], seed)
        balanced.extend(b)
        phase_records.extend(p)
        response_records.extend(
            {"seed": seed, **r}
            for r in read_csv(
                report_paths[seed] / "evaluation/response_evaluation/per_case_response_metrics.csv"
            )
        )
        recovery = read_csv(
            report_paths[seed] / "evaluation/recovery/recovery_waveform_metrics.csv"
        )
        recovery_records.extend({"seed": seed, **r} for r in recovery)
        for method in METHODS:
            for signal in ("position", "velocity"):
                counts = Counter(
                    r[f"{signal}_phase_status"] for r in recovery if r["method"] == method
                )
                phase_statuses.extend(
                    {
                        "seed": seed,
                        "method": method,
                        "signal": signal,
                        "phase_status": k,
                        "channels": n,
                    }
                    for k, n in counts.items()
                )
    summary = summarize_groups(
        metrics,
        ["method"],
        [
            "joint_score",
            "position_rmse_um",
            "velocity_rmse_mm_s",
            "static_displacement_um",
            "static_velocity_mm_s",
        ],
    )
    save_csv(REPORT / "per_seed_metrics.csv", metrics)
    save_csv(REPORT / "multiseed_summary.csv", summary)
    save_csv(REPORT / "balanced_metrics_per_seed.csv", balanced)
    balanced_summary = summarize_groups(
        balanced, ["method", "horizon", "region"], ["position_rmse_um", "velocity_rmse_mm_s"]
    )
    save_csv(REPORT / "balanced_metrics_summary.csv", balanced_summary)
    save_csv(REPORT / "phase_metrics_per_seed.csv", phase_records)
    phase_summary = summarize_groups(
        phase_records,
        ["method", "horizon", "region", "phase"],
        ["position_rmse_um", "velocity_rmse_mm_s"],
    )
    save_csv(REPORT / "phase_metrics_summary.csv", phase_summary)
    phase_wins = []
    for phase in PHASES:
        for horizon in (1, 5, 10, 20):
            for region in ("active", "contact_neighborhood"):
                group = {
                    (r["seed"], r["method"]): r
                    for r in phase_records
                    if r["phase"] == phase and r["horizon"] == horizon and r["region"] == region
                }
                for reference in ("RAW", "CV"):
                    for metric in ("position_rmse_um", "velocity_rmse_mm_s"):
                        wins = sum(
                            group[s, "G"][metric] < group[s, reference][metric] for s in SEEDS
                        )
                        phase_wins.append(
                            {
                                "phase": phase,
                                "horizon": horizon,
                                "region": region,
                                "reference": reference,
                                "metric": metric,
                                "G_better_seed_count": wins,
                                "seed_count": 3,
                            }
                        )
    save_csv(REPORT / "phase_win_counts.csv", phase_wins)
    save_csv(REPORT / "intervention_metrics_per_seed.csv", response_records)
    save_csv(
        REPORT / "intervention_metrics_summary.csv",
        summarize_groups(
            response_records,
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
    save_csv(REPORT / "recovery_metrics_per_seed.csv", recovery_records)
    save_csv(
        REPORT / "recovery_metrics_summary.csv",
        summarize_groups(
            recovery_records,
            ["method", "trajectory", "start", "channel"],
            [
                "position_rmse",
                "velocity_rmse",
                "position_bias",
                "velocity_bias",
                "position_amplitude_error",
                "velocity_amplitude_error",
                "position_phase_error_degrees",
                "velocity_phase_error_degrees",
            ],
        ),
    )
    save_csv(REPORT / "recovery_phase_status_counts.csv", phase_statuses)
    paired = []
    for seed in SEEDS:
        g = next(r for r in metrics if r["seed"] == seed and r["method"] == "G")
        for reference in ("RAW", "CV"):
            ref = next(r for r in metrics if r["seed"] == seed and r["method"] == reference)
            paired.append(
                {
                    "seed": seed,
                    "reference": reference,
                    **{
                        f"{k}_reduction_percent": 100 * (1 - g[k] / ref[k])
                        if g[k] is not None and ref[k]
                        else None
                        for k in ("joint_score", "position_rmse_um", "velocity_rmse_mm_s")
                    },
                }
            )
    paired_summary = summarize_groups(
        paired,
        ["reference"],
        [
            f"{k}_reduction_percent"
            for k in ("joint_score", "position_rmse_um", "velocity_rmse_mm_s")
        ],
    )
    save_csv(REPORT / "paired_reductions_per_seed.csv", paired)
    save_csv(REPORT / "paired_reductions_summary.csv", paired_summary)
    faster = [
        r
        for r in response_records
        if r["variant"] == "faster_down" and r["horizon"] == "1" and r["region"] == "active"
    ]
    save_csv(REPORT / "faster_down_one_step.csv", faster)
    directions = []
    for seed in SEEDS:
        for method in METHODS:
            selected = [r for r in faster if r["seed"] == seed and r["method"] == method]
            row = {"seed": seed, "method": method, "cases": len(selected)}
            for signal in ("position", "velocity"):
                cosines = [
                    float(r[f"{signal}_direction_cosine"])
                    for r in selected
                    if r[f"{signal}_direction_cosine"] != ""
                ]
                row[f"{signal}_negative"] = sum(v < 0 for v in cosines)
                row[f"{signal}_positive"] = sum(v > 0 for v in cosines)
                row[f"{signal}_undefined"] = len(selected) - len(cosines)
                row[f"{signal}_amplitude_ratio_mean"] = float(
                    np.mean([float(r[f"{signal}_amplitude_ratio"]) for r in selected])
                )
            directions.append(row)
    save_csv(REPORT / "faster_down_direction_counts.csv", directions)
    plot_results(metrics, balanced, faster)
    write_report(metrics, summary, paired, paired_summary, directions, phase_summary, training)
    save_json(
        REPORT / "analysis_provenance.json",
        {
            "analysis_only_not_new_training": True,
            "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "statistics_script_sha256": hashlib.sha256(
                (ROOT / "scripts/fair_replication_statistics.py").read_bytes()
            ).hexdigest(),
            "official_summary_requires_all_three_finite_seeds": True,
            "invalid_finite_metrics_not_removed_from_descriptive_statistics": True,
            "no_statistical_significance_claim": True,
        },
    )
    print({"audit_passed": audit["audit_passed"], "summary": summary, "paired": paired_summary})


def plot_results(metrics, balanced, faster):
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, ylabel in zip(
        axes,
        ("position_rmse_um", "velocity_rmse_mm_s"),
        ("Position RMSE (um)", "Instantaneous velocity RMSE (mm/s)"),
        strict=True,
    ):
        for i, method in enumerate(METHODS):
            values = [r[key] for r in metrics if r["method"] == method]
            ax.errorbar(i, np.mean(values), yerr=np.std(values, ddof=1), color="black", capsize=4)
            for j, seed in enumerate(SEEDS):
                row = next(r for r in metrics if r["seed"] == seed and r["method"] == method)
                ax.scatter(
                    i + (j - 1) * 0.08,
                    row[key],
                    color=f"C{j}",
                    label=f"seed {seed}" if i == 0 else None,
                )
        ax.set(xticks=range(3), xticklabels=METHODS, ylabel=ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(REPORT / "seed_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for i, region in enumerate(("active", "contact_neighborhood")):
        for j, key in enumerate(("position_rmse_um", "velocity_rmse_mm_s")):
            ax = axes[i, j]
            for method in METHODS:
                values = np.array(
                    [
                        [
                            next(
                                r[key]
                                for r in balanced
                                if r["seed"] == seed
                                and r["method"] == method
                                and r["horizon"] == h
                                and r["region"] == region
                            )
                            for h in (1, 5, 10, 20)
                        ]
                        for seed in SEEDS
                    ]
                )
                mean, std = values.mean(0), values.std(0, ddof=1)
                (line,) = ax.plot((1, 5, 10, 20), mean, marker="o", label=method)
                ax.fill_between(
                    (1, 5, 10, 20),
                    np.maximum(0, mean - std),
                    mean + std,
                    alpha=0.12,
                    color=line.get_color(),
                )
                for row in values:
                    ax.plot((1, 5, 10, 20), row, alpha=0.25, linewidth=0.7, color=line.get_color())
            ax.set(xlabel="Prediction steps", ylabel=key, title=region)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.savefig(REPORT / "horizons_mean_std.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for seed in SEEDS:
        for method in ("RAW", "G"):
            rows = [r for r in faster if r["seed"] == seed and r["method"] == method]
            for ax, signal in zip(axes, ("position", "velocity"), strict=True):
                ax.plot(
                    [int(r["frame"]) for r in rows],
                    [float(r[f"{signal}_direction_cosine"]) for r in rows],
                    linestyle="-" if method == "G" else "--",
                    color=f"C{SEEDS.index(seed)}",
                    marker="o",
                    label=f"{method} seed{seed}",
                )
    for ax, title in zip(
        axes, ("Position response cosine", "Velocity response cosine"), strict=True
    ):
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(xlabel="Intervention start frame", ylabel=title, ylim=(-1, 1))
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(REPORT / "faster_down_directions.png", dpi=160)
    plt.close(fig)


def write_report(metrics, summary, paired, paired_summary, directions, phase_summary, training):
    lines = [
        "# 公平主线对照：三种子复核",
        "",
        "已完成RAW/G在种子17、27的四次新增训练；种子7沿用原记录，未重训或覆盖。"
        "本轮没有搜索参数，仍是小预算开发验证，独立测试未验证。",
        "",
        "## 协议与统计",
        "",
        "网络45324参数、三步展开、200更新、batch16、LR=0.0001、速度权重0.5、网格权重0.001、"
        "损失/选模/掩码/数据划分全部固定。同种子RAW/G共享初始权重和批次；跨种子改变初始化和批次随机性。"
        "G自由递推重算当前接触与解析力，前向不读原生接触/力。当前输入缓存只读复用原先验证的确定性编码。",
        "",
        "每个种子先独立评价，再计算均值和样本标准差（ddof=1）。误差汇总依旧按四阶段、两区域、"
        "5/10/20步等权；不先平均预测再评分，不丢弃失败或退化种子。标准差不是置信区间，n=3不作显著性声明。"
        "CV为同一个确定性参照，重复评价只用于一致性核查，±0不是泛化不确定性估计。",
        "",
        "## 全部种子的实际结果",
        "",
        "| 种子 | 方法 | 联合分数 | 位置RMSE μm | 速度RMSE mm/s | "
        "选中更新 | 翻转状态/窗口 | 最低体积比 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in metrics:
        lines.append(
            f"| {r['seed']} | {r['method']} | {r['joint_score']:.6f} | "
            f"{r['position_rmse_um']:.3f} | {r['velocity_rmse_mm_s']:.3f} | "
            f"{r['selected_update']} | {r['inverted_states']}/{r['inverted_windows']} | "
            f"{r['minimum_volume_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| 方法 | 联合分数 均值±样本标准差 | 位置μm | 速度mm/s | 静止对照最大位移μm |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for method in METHODS:
        lines.append(
            f"| {method} | "
            + " | ".join(
                format_stat(summary, method, k)
                for k in (
                    "joint_score",
                    "position_rmse_um",
                    "velocity_rmse_mm_s",
                    "static_displacement_um",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 成对收益是否复现",
            "",
            "以下是每个种子的误差降低百分比，再跨种子统计；正值改善、负值退化。"
            "它不同于直接拿两组均值作比值。",
            "",
            "| 对照 | 位置降低% 均值±标准差 | 速度降低% 均值±标准差 | "
            "G位置较优种子数 | G速度较优种子数 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for reference in ("RAW", "CV"):
        vals = []
        for metric in (
            "position_rmse_um_reduction_percent",
            "velocity_rmse_mm_s_reduction_percent",
        ):
            s = next(
                r for r in paired_summary if r["reference"] == reference and r["metric"] == metric
            )
            vals.append(f"{s['mean']:.2f} ± {s['sample_std']:.2f}")
        group = [r for r in paired if r["reference"] == reference]
        wins = [
            sum(r[f"{key}_reduction_percent"] > 0 for r in group)
            for key in ("position_rmse_um", "velocity_rmse_mm_s")
        ]
        lines.append(f"| {reference} | {vals[0]} | {vals[1]} | {wins[0]}/3 | {wins[1]}/3 |")
    lines.extend(
        [
            "",
            "种子7的G位置优势没有在两个新增种子相对CV复现；G速度三种子均未超过CV。"
            "这里指四阶段/区域/时域等权汇总。相对RAW，G在种子7和17较优，在种子27略差，"
            "不能声称汇总指标稳定领先。",
            "",
            "## 阶段、时域与机制局限",
            "",
            "单步及5/10/20步、两区域的完整均值/标准差见balanced_metrics_summary.csv，"
            "四个主阶段见phase_metrics_summary.csv；以下列出20步活动区域，避免只展示汇总分数。"
            "接近与静止不参与主分数，完整评价分别保存在原种子7目录及seed_17/seed_27/evaluation/。",
            "",
            "| 阶段 | 方法 | 位置μm 均值±标准差 | 速度mm/s 均值±标准差 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for phase in PHASES:
        for method in METHODS:
            rows = [
                r
                for r in phase_summary
                if r["method"] == method
                and r["phase"] == phase
                and r["horizon"] == 20
                and r["region"] == "active"
            ]
            lines.append(
                f"| {phase} | {method} | "
                + " | ".join(
                    format_stat(rows, method, k) for k in ("position_rmse_um", "velocity_rmse_mm_s")
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "局部收益也有重复出现：20步活动区域中，G在释放阶段的位置和速度、恢复阶段的位置，"
            "均有3/3种子优于CV；恢复速度仍3/3更差，按压/保持的位置和速度均3/3更差。"
            "这说明收益有明确阶段边界，而非所有收益都未复现。完整逐阶段胜出计数在phase_win_counts.csv。",
            "",
            "| 种子 | 方法 | faster_down位置负方向/例数 | 速度负方向/例数 | "
            "位置幅度比均值 | 速度幅度比均值 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in directions:
        lines.append(
            f"| {r['seed']} | {r['method']} | {r['position_negative']}/{r['cases']} | "
            f"{r['velocity_negative']}/{r['cases']} | "
            f"{r['position_amplitude_ratio_mean']:.3f} | "
            f"{r['velocity_amplitude_ratio_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "G的位置方向在种子17三例均为正，种子7三例均为负，种子27两例为负；"
            "这说明机制响应明显依赖种子。方向为正也不等于幅度正确或完整机制通过。"
            "CV零响应的方向未定义，表中的0个负方向不代表通过，详见faster_down_direction_counts.csv的undefined列。",
            "",
            "全部五类干预、响应幅度及恢复偏差/幅度/相位可识别状态均跨种子保存。"
            "不足周期或零响应的方向/相位不填0。网格有限性、固定边界和无翻转与机制正确性分开判断。"
            "G静止对照不漂移来自门控/静止参考先验；RAW漂移和两模型恢复波形的偏差继续保留。",
            "",
            "## 运行与检查",
            "",
            "四次新增训练均完成200更新，无算法失败或超时；所有种子/方法评价窗口无翻转。"
            "每个方法每个种子35个普通及18个干预窗口，失败窗口没有删除。"
            "包括种子7的全部评分与网格均用独立NumPy复核，CV跨种子预测逐位一致。"
            "原报告、权重、数据和缓存摘要不变；右侧数组未加载。242项测试和Ruff通过。",
            "",
            "| 新种子 | 方法 | 训练秒数 | 含重载秒数 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for seed in (17, 27):
        for method in ("RAW", "G"):
            r = training[seed, method]
            lines.append(
                f"| {seed} | {method} | {r['training']['training_elapsed_seconds']:.3f} | "
                f"{r['elapsed_seconds']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 本轮结论",
            "",
            "1. 数值有效性在本轮六个模型中得到重复观察，静止先验的约束作用也保持。",
            "2. G相对RAW的平均收益存在，但不是每个种子都领先；相对CV的汇总稳定优势未得到支持。"
            "释放20步位置/速度及恢复位置的局部收益在3/3种子重复出现。",
            "3. faster_down的位置方向随种子变化，恢复波形仍未验证正确；不能宣布可靠接触动力学。",
            "4. 左侧仍是开发验证，没有独立测试结论；不根据本轮复核结果追加参数搜索或挑选最好种子。",
            "",
            "这轮重复的是同一短预算训练流程，不能证明模型充分收敛，也不能把平均收益全部归于解析力。"
            "后续应围绕主线预先规定训练预算和公平基线的充分性检查，再冻结方法作独立评价。",
            "",
            "入口：`python scripts/replicate_fair_mainline.py`（已有运行拒绝覆盖）。"
            "只重建审计、统计与图：`python scripts/report_fair_mainline_multiseed.py`。"
            "新权重、预测和源码快照位于outputs/fair_mainline_multiseed/run_001/。",
            "",
            "![每个种子与均值标准差](seed_comparison.png)",
            "",
            "![时域误差，细线为单种子](horizons_mean_std.png)",
            "",
            "![faster_down逐种子响应方向](faster_down_directions.png)",
            "",
        ]
    )
    (REPORT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
