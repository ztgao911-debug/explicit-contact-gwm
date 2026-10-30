"""Audit the frozen fair pilot and publish all methods, phases and failures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from verify_bounded_search import verify_evaluation

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/fair_mainline/run_001"
OUTPUT = ROOT / "outputs/fair_mainline/run_001"
METHODS = ("RAW", "G", "CV")
PHASES = ("press", "hold", "release", "recover")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = read_json(REPORT / "protocol.json")
    status = read_json(REPORT / "status.json")
    ledger = read_json(REPORT / "training_ledger.json")
    summary = read_json(REPORT / "evaluation_summary.json")
    permissions = read_json(REPORT / "input_permission_checks.json")
    with np.load(ROOT / protocol["configuration"]["data_directory"] / "mesh.npz") as saved:
        mesh = {key: saved[key] for key in saved.files}
    independent = {
        method: verify_evaluation(
            OUTPUT / "evaluation", ROOT / protocol["configuration"]["data_directory"], mesh, method
        )
        for method in METHODS
    }
    selection_checks = {}
    for method in ("RAW", "G"):
        history = read_json(OUTPUT / method / "checkpoint_selection.json")
        selected = next(r for r in ledger if r["variant"] == method)["training"]["selected_step"]
        best = min(history, key=lambda r: r["scores"]["joint_score"])
        selection_checks[method] = history[0]["update"] == 0 and best["update"] == selected
    changed = {
        group: [
            name
            for name, digest in protocol[group].items()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        ]
        for group in ("source_hashes", "data_hashes")
    }
    audit = {
        "per_method": independent,
        "checkpoint_zero_and_common_minimum": selection_checks,
        "changed_frozen_files": changed,
        "input_permissions": permissions,
        "both_twenty_step_reload_checks": [r.get("reload") for r in ledger],
        "audit_passed": all(r["passed"] for r in independent.values())
        and all(selection_checks.values())
        and not any(changed.values())
        and permissions["native_contact_or_force_query_calls"] == 0,
        "independent_test_verified": False,
    }
    save_json(REPORT / "independent_verification.json", audit)
    assert audit["audit_passed"]
    # Preserve the original frozen file. The executable, per-arm protocol and
    # native-query guard disambiguate inherited generic M-only descriptions.
    save_json(
        REPORT / "metadata_clarifications.json",
        {
            "original_protocol_unchanged": True,
            "inherited_fixed_training_condition": protocol["fixed"]["training_condition"],
            "actual_training_conditions": {
                "RAW": protocol["RAW_condition"],
                "G": protocol["G_condition"],
            },
            "superseded_template_description": (
                "The fixed.training_condition M description is inherited and inapplicable here; "
                "frozen RAW_condition/G_condition and executable paths define this experiment."
            ),
            "legacy_training_summary_fields": (
                "The legacy native_queries counter counts conditioner calls, not native queries;"
                " RAW gradients follow current kinematics, G geometry/force are stop-gradient."
            ),
            "actual_native_contact_or_force_queries": permissions[
                "native_contact_or_force_query_calls"
            ],
            "no_changes_to_training_selection_or_scores": True,
        },
    )
    balanced = []
    phase_rows = []
    for filename, horizons, position_key in (
        ("one_step_metrics.csv", (1,), "increment_rmse_um"),
        ("rollout_metrics.csv", (5, 10, 20), "position_rmse_um"),
    ):
        raw = read_csv(OUTPUT / "evaluation/ordinary" / filename)
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
                    assert len(subset) == 4
                    balanced.append(
                        {
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
                    phase_rows.extend(
                        {
                            "method": method,
                            "horizon": horizon,
                            "region": region,
                            "phase": r["phase"],
                            "position_rmse_um": float(r[position_key]),
                            "velocity_rmse_mm_s": float(r["velocity_rmse_mm_s"]),
                        }
                        for r in subset
                    )
    save_csv(REPORT / "balanced_metrics.csv", balanced)
    save_csv(REPORT / "phase_metrics.csv", phase_rows)
    comparison = []
    for method in METHODS:
        scores, outcome = summary[method]["scores"], summary[method]
        row = {
            "method": method,
            "eligible_finite_pins_mesh": outcome["eligible"],
            "joint_score": scores["joint_score"],
            "balanced_position_rmse_um": 1000 * math.sqrt(scores["position_score"]),
            "balanced_velocity_rmse_mm_s": 100 * math.sqrt(scores["velocity_score"]),
            "inverted_states": outcome["inverted_states"],
            "inverted_windows": outcome["inverted_windows"],
            "minimum_volume_ratio": outcome["minimum_volume_ratio"],
        }
        for reference in ("RAW", "CV"):
            for channel in ("position", "velocity"):
                row[f"{channel}_rmse_reduction_vs_{reference}_percent"] = 100 * (
                    1
                    - math.sqrt(
                        scores[f"{channel}_score"]
                        / summary[reference]["scores"][f"{channel}_score"]
                    )
                )
        comparison.append(row)
    save_csv(REPORT / "method_comparison.csv", comparison)
    response = read_csv(OUTPUT / "evaluation/response_evaluation/per_case_response_metrics.csv")
    faster = [r for r in response if r["variant"] == "faster_down" and r["horizon"] == "1"]
    save_csv(REPORT / "faster_down_one_step.csv", faster)
    static = read_csv(OUTPUT / "evaluation/static_stability.csv")
    static_summary = [
        {
            "method": method,
            "maximum_no_contact_displacement_um": 1e6
            * max(float(r["max_absolute_displacement_m"]) for r in static if r["method"] == method),
            "maximum_no_contact_instantaneous_velocity_mm_s": 1000
            * max(
                float(r["max_absolute_instantaneous_velocity_m_s"])
                for r in static
                if r["method"] == method
            ),
        }
        for method in METHODS
    ]
    save_csv(REPORT / "static_summary.csv", static_summary)
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for method in ("RAW", "G"):
        curve = read_csv(OUTPUT / method / "training_curve.csv")
        axes[0].plot(
            [int(r["update"]) for r in curve],
            [float(r["validation_joint_score"]) for r in curve],
            marker="o",
            label=method,
        )
    for method in METHODS:
        a = [r for r in balanced if r["method"] == method and r["region"] == "active"]
        axes[1].plot(
            [r["horizon"] for r in a], [r["position_rmse_um"] for r in a], marker="o", label=method
        )
        axes[2].plot(
            [r["horizon"] for r in a],
            [r["velocity_rmse_mm_s"] for r in a],
            marker="o",
            label=method,
        )
    for ax, ylabel in zip(
        axes,
        (
            "Fixed 5-step joint selection score",
            "Position RMSE (um)",
            "Instantaneous velocity RMSE (mm/s)",
        ),
        strict=True,
    ):
        ax.set(xlabel="Updates" if ax is axes[0] else "Prediction steps", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.savefig(REPORT / "comparison.png", dpi=160)
    plt.close(fig)
    waves = read_csv(OUTPUT / "evaluation/recovery/recovery_signed_waveforms.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method in METHODS:
        a = [
            r
            for r in waves
            if r["method"] == method
            and r["trajectory"] == "left_deep"
            and r["start"] == "170"
            and r["channel"] == "representative_node"
        ]
        for ax, channel, unit in zip(axes, ("z_displacement", "vz"), ("um", "mm_s"), strict=True):
            ax.plot(
                [int(r["horizon"]) for r in a],
                [float(r[f"predicted_{channel}_{unit}"]) for r in a],
                label=method,
            )
            if method == "G":
                ax.plot(
                    [int(r["horizon"]) for r in a],
                    [float(r[f"reference_{channel}_{unit}"]) for r in a],
                    color="black",
                    linestyle="--",
                    label="MuJoCo label",
                )
    for ax, ylabel in zip(
        axes, ("Signed displacement z (um)", "Instantaneous velocity vz (mm/s)"), strict=True
    ):
        ax.set(xlabel="Prediction steps", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.savefig(REPORT / "recovery.png", dpi=160)
    plt.close(fig)
    g = next(r for r in comparison if r["method"] == "G")
    velocity_cv = g["velocity_rmse_reduction_vs_CV_percent"]
    velocity_comparison = (
        f"速度降低{velocity_cv:.2f}%" if velocity_cv >= 0 else f"速度反而增加{-velocity_cv:.2f}%"
    )
    lines = [
        "# 公平主线对照：RAW 与显式接触 G（短版）",
        "",
        "本轮已实际完成两次从头训练和恒速对照。属于单种子、小预算开发验证，不是独立测试或全局最优搜索。",
        "",
        "## 冻结设置与公平性",
        "",
        "两组相同网络45324个参数、相同初始权重、固定空间编码、输出上限16mm/1.6m/s、"
        "种子7、三步自回归、200更新、batch16、Adam LR=0.0001、速度权重0.5、网格权重0.001。"
        "相同批次顺序、普通绝对监督+三步端点平均+初始成对响应监督，损失乘100和余弦调度一致。",
        "",
        "RAW拥有当前软体x/v/rest、压头实际x/v和指令x/v，36维状态/相对运动编码，外支路门控恒为1。"
        "G使用已有拓扑辅助接触表面、固定中央标定点式解析力及接触门控。"
        "两组都从头训练，RAW不是冻结模型的输入置零消融；G训练缓存与自回归均重新计算G条件。",
        "",
        "这比较的是整体输入表示与门控归纳偏置：G有已知拓扑、标定力律及接触外零响应先验。"
        "不能把结果只归因于力特征，相同参数量也不保证相同优化难度。G是拓扑辅助表面，尚非纯Gaussian球并集。",
        "",
        "固定中央600普通帧对、108分支及90成对响应标签；左侧普通轨迹和18分支作开发验证，"
        "右侧数组未读取。真实下一帧只作监督/评分；压头轨迹已知，软体状态自由递推。"
        "历史M输入上下文和门控在两组中均被替换，前向原生接触/力查询次数为0。",
        "",
        "五步共同目标包含第0步，50更新检查一次，两组均选中第50更新；最终统一评价1/5/10/20步。"
        "所有方法共享GT定义的评价掩码和四阶段/两区域/预测步长等权规则，不删除失败窗口。",
        "",
        "## 实测收益与代价",
        "",
        f"G相对RAW的汇总位置RMSE降低{g['position_rmse_reduction_vs_RAW_percent']:.2f}%，"
        f"瞬时速度RMSE降低{g['velocity_rmse_reduction_vs_RAW_percent']:.2f}%。"
        f"相对CV位置降低{g['position_rmse_reduction_vs_CV_percent']:.2f}%，"
        f"{velocity_comparison}。",
        "",
        "| 方法 | 固定联合分数 | 等权位置RMSE μm | 速度RMSE mm/s | 翻转状态/窗口 | 最低体积比 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in comparison:
        lines.append(
            f"| {r['method']} | {r['joint_score']:.6f} | {r['balanced_position_rmse_um']:.3f} | "
            f"{r['balanced_velocity_rmse_mm_s']:.3f} | "
            f"{r['inverted_states']}/{r['inverted_windows']} | "
            f"{r['minimum_volume_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "均通过有限性、固定边界和网格翻转硬筛选；这不包含静止漂移或干预方向正确性。",
            "",
            "| 静止对照 | 最大位移μm | 最大瞬时速度mm/s |",
            "| --- | --- | --- |",
        ]
    )
    for r in static_summary:
        lines.append(
            f"| {r['method']} | {r['maximum_no_contact_displacement_um']:.6g} | "
            f"{r['maximum_no_contact_instantaneous_velocity_mm_s']:.6g} |"
        )
    lines.extend(
        [
            "",
            "RAW在无接触对照出现明显漂移，G/CV接近数值零。"
            "这与显式门控/静止平衡先验的作用一致，尚不能据此证明学到了正确接触响应。",
            "",
            "| 步数 | 区域 | RAW位置μm | G位置μm | CV位置μm | "
            "RAW速度mm/s | G速度mm/s | CV速度mm/s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for h in (1, 5, 10, 20):
        for region in ("active", "contact_neighborhood"):
            group = {
                r["method"]: r for r in balanced if r["horizon"] == h and r["region"] == region
            }
            lines.append(
                f"| {h} | {region} | "
                + " | ".join(
                    f"{group[m][key]:.3f}"
                    for key in ("position_rmse_um", "velocity_rmse_mm_s")
                    for m in METHODS
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "上述RMSE先对四阶段MSE等权再开方，一步位置为增量RMSE。"
            "全部按压、保持、释放、恢复及两区域在phase_metrics.csv；接近/静止和完整轨迹指标在evaluation/。",
            "",
            "G没有在全部单项上胜过RAW：一步ROI速度为7.529对5.405mm/s，"
            "五步ROI位置为303.853对302.566μm，均略差或更差。"
            "相对CV，G按压/保持仍明显退化，汇总改善不能代替这些阶段的正确性。",
            "",
            "| 活动区域阶段 | 步数 | RAW位置μm | G位置μm | CV位置μm | "
            "RAW速度mm/s | G速度mm/s | CV速度mm/s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for phase in PHASES:
        for h in (5, 10, 20):
            group = {
                r["method"]: r
                for r in phase_rows
                if r["phase"] == phase and r["horizon"] == h and r["region"] == "active"
            }
            lines.append(
                f"| {phase} | {h} | "
                + " | ".join(
                    f"{group[m][key]:.3f}"
                    for key in ("position_rmse_um", "velocity_rmse_mm_s")
                    for m in METHODS
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 干预与恢复：仍失败的部分",
            "",
            "| faster_down起点 | 方法 | 位置方向余弦 | 速度方向余弦 | 位置幅度比 | 速度幅度比 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in faster:
        if r["region"] != "active":
            continue
        values = [
            r[k]
            for k in (
                "position_direction_cosine",
                "velocity_direction_cosine",
                "position_amplitude_ratio",
                "velocity_amplitude_ratio",
            )
        ]
        lines.append(
            f"| {r['frame']} | {r['method']} | "
            + " | ".join(f"{float(v):.4f}" if v else "未定义（零响应）" for v in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "RAW和G的三例faster_down位置方向均为负；G速度方向为正但幅度偏低。"
            "CV的方向在零响应时未定义，不能计为机制通过。所有五类干预及多步结果完整保留在evaluation/response_evaluation/。",
            "",
            "恢复有符号位移/瞬时速度、幅度和相位可识别状态保存在evaluation/recovery/。"
            "代表点仍不能正确重现回弹振动和静止形状；低于CV的漂移不等于恢复动力学正确。"
            "不足两个可识别周期时，相位保持未验证。",
            "",
            "## 运行与检查",
            "",
            f"从本轮开始到训练/评价结束墙钟{status['wall_elapsed_seconds']:.2f}s，含实现和预检；"
            "两组均200更新，但G需要重算接触，不能将相同更新数称作相同计算量。",
            "",
            "| 模型 | 训练秒数 | 含缓存/重载秒数 | 梯度裁剪更新数 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for r in ledger:
        lines.append(
            f"| {r['variant']} | {r['training']['training_elapsed_seconds']:.3f} | "
            f"{r['elapsed_seconds']:.3f} | {r['training']['clipped_updates']}/200 |"
        )
    lines.extend(
        [
            "",
            "两组三步末端到第一步梯度有限且非零，训练与推理前向逐位一致，实际权重20步保存重载一致。"
            "独立NumPy重算每方法24评分单元及35普通+18干预网格窗口通过，145个冻结输入摘要无变化。"
            "233项测试及Ruff通过。原生查询被运行时禁止，实际调用0次。",
            "",
            "原始冻结协议保留。继承模板的fixed.training_condition=M描述不适用本轮，"
            "以冻结的RAW_condition/G_condition及执行路径为准；旧训练日志的native_queries字段计数条件构建调用，"
            "不是原生查询。两处描述在metadata_clarifications.json明确澄清，不改训练、选模和评分。",
            "",
            "## 结论与边界",
            "",
            "预算内观察：G比这次RAW学习基线误差更低，并避免了静止漂移；相对CV收益有限且分阶段不一致。"
            "可重复性：当前仅种子7，没有多种子支持。机制：faster_down位置方向和恢复波形仍失败。"
            "独立测试：未验证，左侧仅开发验证。",
            "",
            "这是一轮相同短预算下的整体方法对照，不能说明RAW已充分训练，"
            "不能将全部收益归于接触力，也不能声称显式模型理解物理或已解决泛化。"
            "下一轮优先固定本协议做多种子复核，再决定哪些收益值得独立测试；不根据本轮结果追加参数搜索。",
            "",
            "训练入口：`python scripts/run_fair_mainline.py`（已运行目录拒绝覆盖）。"
            "重建审计和图表：`python scripts/report_fair_mainline.py`。"
            "完整输入缓存、检查点、预测NPZ和源快照位于忽略的outputs/fair_mainline/run_001/。",
            "",
            "![公平对照曲线](comparison.png)",
            "",
            "![恢复波形](recovery.png)",
            "",
        ]
    )
    (REPORT / "REPORT.md").write_text("\n".join(lines))
    save_json(
        REPORT / "analysis_provenance.json",
        {
            "analysis_only_after_training": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_and_scores_unchanged": True,
        },
    )
    print(json.dumps({"audit_passed": audit["audit_passed"], "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
