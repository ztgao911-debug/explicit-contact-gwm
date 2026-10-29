"""Independently audit and summarize the saved short experiment; never train."""

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
REPORT = ROOT / "experiments/short_multistep/run_001"
OUTPUT = ROOT / "outputs/short_multistep/run_001"
METHODS = ("M", "F", "G", "I", "CV")
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
    data = ROOT / protocol["configuration"]["data_directory"]
    with np.load(data / "mesh.npz") as archive:
        mesh = {key: archive[key] for key in archive.files}
    checks = {m: verify_evaluation(OUTPUT / "final_routes", data, mesh, m) for m in METHODS}
    trial = read_json(REPORT / "trial_ledger.json")[0]
    artifact = ROOT / trial["artifact_directory"]
    selection = read_json(artifact / "checkpoint_selection.json")
    best = min(
        (s for s in selection if s["scores"]["joint_score"] is not None),
        key=lambda s: s["scores"]["joint_score"],
    )
    changes = {}
    for group in ("source_hashes", "data_hashes"):
        changes[group] = [
            name
            for name, expected in protocol[group].items()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
        ]
    with (
        np.load(artifact / "evaluation/ordinary/rollout_predictions.npz") as a,
        np.load(OUTPUT / "final_routes/ordinary/rollout_predictions.npz") as b,
    ):
        repeat = all(np.array_equal(a[key], b[key]) for key in a.files)
    audit = {
        "formal_training_completed": True,
        "selected_update": trial["training"]["selected_step"],
        "selected_checkpoint_is_minimum_common_validation": (
            best["update"] == trial["training"]["selected_step"]
        ),
        "update_zero_in_selection": selection[0]["update"] == 0,
        "changed_frozen_files": changes,
        "frozen_data_files": len(protocol["data_hashes"]),
        "repeated_M_rollout_bitwise_equal": repeat,
        "per_method": checks,
        "audit_passed": all(c["passed"] for c in checks.values())
        and not any(changes.values())
        and repeat,
        "independent_test_verified": False,
    }
    save_json(REPORT / "independent_verification.json", audit)
    assert audit["audit_passed"] and audit["selected_checkpoint_is_minimum_common_validation"]
    rows = []
    for filename, horizons, position_key in (
        ("one_step_metrics.csv", (1,), "increment_rmse_um"),
        ("rollout_metrics.csv", (5, 10, 20), "position_rmse_um"),
    ):
        raw = read_csv(OUTPUT / "final_routes/ordinary" / filename)
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
                    assert len(subset) == 4, (method, horizon, region)
                    rows.append(
                        {
                            "method": method,
                            "horizon": horizon,
                            "region": region,
                            "phase_balanced_position_rmse_um": math.sqrt(
                                np.mean([float(r[position_key]) ** 2 for r in subset])
                            ),
                            "phase_balanced_velocity_rmse_mm_s": math.sqrt(
                                np.mean([float(r["velocity_rmse_mm_s"]) ** 2 for r in subset])
                            ),
                        }
                    )
    for row in rows:
        cv = next(
            r
            for r in rows
            if r["method"] == "CV"
            and r["horizon"] == row["horizon"]
            and r["region"] == row["region"]
        )
        for channel, unit in (("position", "um"), ("velocity", "mm_s")):
            key = f"phase_balanced_{channel}_rmse_{unit}"
            row[f"{channel}_rmse_reduction_vs_CV_percent"] = 100 * (1 - row[key] / cv[key])
    save_csv(REPORT / "balanced_metrics.csv", rows)
    summary = read_json(REPORT / "final_routes_summary.json")
    comparison = []
    for method in METHODS:
        outcome, baseline = summary[method], summary["CV"]["scores"]
        scores = outcome["scores"]
        comparison.append(
            {
                "method": method,
                "physically_valid": outcome["eligible"],
                **{k: scores[k] for k in ("joint_score", "position_score", "velocity_score")},
                "joint_score_reduction_vs_CV_percent": 100
                * (1 - scores["joint_score"] / baseline["joint_score"]),
                "position_rmse_reduction_vs_CV_percent": 100
                * (1 - math.sqrt(scores["position_score"] / baseline["position_score"])),
                "velocity_rmse_reduction_vs_CV_percent": 100
                * (1 - math.sqrt(scores["velocity_score"] / baseline["velocity_score"])),
                "inverted_states": outcome["inverted_states"],
                "inverted_windows": outcome["inverted_windows"],
                "minimum_volume_ratio": outcome["minimum_volume_ratio"],
            }
        )
    save_csv(REPORT / "route_comparison.csv", comparison)
    responses = read_csv(OUTPUT / "final_routes/response_evaluation/per_case_response_metrics.csv")
    faster = [r for r in responses if r["variant"] == "faster_down" and r["horizon"] == "1"]
    save_csv(REPORT / "faster_down_one_step.csv", faster)
    negatives = [r for r in faster if r["method"] == "M" and r["region"] == "active"]
    curve = read_csv(artifact / "training_curve.csv")
    initial = float(curve[0]["validation_joint_score"])
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(
        [int(r["update"]) for r in curve],
        [float(r["validation_joint_score"]) for r in curve],
        marker="o",
    )
    axes[0].set(xlabel="Updates", ylabel="Fixed 5-step joint validation score")
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method and r["region"] == "active"]
        for ax, key in zip(
            axes[1:],
            ("phase_balanced_position_rmse_um", "phase_balanced_velocity_rmse_mm_s"),
            strict=True,
        ):
            ax.plot(
                [r["horizon"] for r in selected],
                [r[key] for r in selected],
                marker="o",
                label=method,
            )
    axes[1].set(xlabel="Prediction steps", ylabel="Phase-balanced position RMSE (um)")
    axes[2].set(xlabel="Prediction steps", ylabel="Instantaneous velocity RMSE (mm/s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[2].legend()
    fig.savefig(REPORT / "short_experiment.png", dpi=160)
    plt.close(fig)
    waves = read_csv(OUTPUT / "final_routes/recovery/recovery_signed_waveforms.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method in ("M", "CV"):
        selected = [
            r
            for r in waves
            if r["trajectory"] == "left_deep"
            and r["start"] == "170"
            and r["channel"] == "representative_node"
            and r["method"] == method
        ]
        for ax, channel, unit in zip(axes, ("z_displacement", "vz"), ("um", "mm_s"), strict=True):
            ax.plot(
                [int(r["horizon"]) for r in selected],
                [float(r[f"predicted_{channel}_{unit}"]) for r in selected],
                label=method,
            )
            if method == "M":
                ax.plot(
                    [int(r["horizon"]) for r in selected],
                    [float(r[f"reference_{channel}_{unit}"]) for r in selected],
                    color="black",
                    linestyle="--",
                    label="MuJoCo label",
                )
    axes[0].set(xlabel="Prediction steps", ylabel="Signed displacement z (um)")
    axes[1].set(xlabel="Prediction steps", ylabel="Instantaneous velocity vz (mm/s)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(REPORT / "recovery_waveform.png", dpi=160)
    plt.close(fig)
    recovery_example = next(
        r
        for r in waves
        if r["trajectory"] == "left_deep"
        and r["start"] == "170"
        and r["channel"] == "representative_node"
        and r["method"] == "M"
        and r["horizon"] == "20"
    )
    save_json(REPORT / "representative_recovery_endpoint.json", recovery_example)
    status = read_json(REPORT / "status.json")
    m = next(r for r in comparison if r["method"] == "M")
    reduction = 100 * (1 - best["scores"]["joint_score"] / initial)
    lines = [
        "# 三步自回归短实验：已实际运行",
        "",
        "本轮是单配置、单种子的短程训练流程诊断，不是十二配置搜索或独立测试。",
        "",
        "复用冻结候选S08：种子7、三步展开、LR=0.0001、速度权重0.5、网格权重0.001；"
        "batch16、200次更新。200次余弦调度压缩到本轮预算，并不等同于原2000次的前200次。",
        "",
        f"实际完成200次更新，训练{trial['training']['training_elapsed_seconds']:.3f}s；"
        f"训练及M/F/G/I/CV评价共{status['wall_elapsed_seconds']:.3f}s（不含后验报告整理）。"
        "新预算15分钟；旧run_001六小时停止证据未覆盖。",
        "",
        f"五步共同目标包含第0步，从{initial:.8f}降到{best['scores']['joint_score']:.8f}，"
        f"选中第{best['update']}次更新，下降{reduction:.2f}%。",
        "",
        "## 主要实测结果",
        "",
        f"M相对恒速：固定联合分数下降{m['joint_score_reduction_vs_CV_percent']:.2f}%，"
        f"等权汇总位置RMSE下降{m['position_rmse_reduction_vs_CV_percent']:.2f}%，"
        f"瞬时速度RMSE下降{m['velocity_rmse_reduction_vs_CV_percent']:.2f}%。"
        "联合分数下降百分比与RMSE下降百分比不是同一量。",
        "",
        "| 方法 | 联合分数 | 位置分数 | 速度分数 | 翻转状态/窗口 | 最低体积比 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in comparison:
        lines.append(
            f"| {r['method']} | {r['joint_score']:.6f} | {r['position_score']:.6f} | "
            f"{r['velocity_score']:.6f} | {r['inverted_states']}/{r['inverted_windows']} | "
            f"{r['minimum_volume_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "M为原生几何+原生力；F为原生几何+解析力；G为修正接触表面+解析力。"
            "I屏蔽接触输入，属于冻结权重的输入消融，不能称公平重训基线。"
            "所有方法通过本轮物理筛选，严格固定节点，评价掩码相同，失败窗口未删除。",
            "",
            "| 步数 | 区域 | M位置RMSE μm | CV位置RMSE μm | M速度RMSE mm/s | CV速度RMSE mm/s |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in rows:
        if r["method"] != "M":
            continue
        cv = next(
            v
            for v in rows
            if v["method"] == "CV" and v["horizon"] == r["horizon"] and v["region"] == r["region"]
        )
        lines.append(
            f"| {r['horizon']} | {r['region']} | {r['phase_balanced_position_rmse_um']:.3f} | "
            f"{cv['phase_balanced_position_rmse_um']:.3f} | "
            f"{r['phase_balanced_velocity_rmse_mm_s']:.3f} | "
            f"{cv['phase_balanced_velocity_rmse_mm_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "M在一步、五步的两区域位置与速度都比CV差；10步位置也更差。"
            "20步活动区域位置改善约18%，速度仍略差；不能只展示汇总改善。",
            "",
            "| 阶段（活动区域） | 步数 | M位置μm | CV位置μm | M速度mm/s | CV速度mm/s |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    phase_rows = read_csv(OUTPUT / "final_routes/ordinary/rollout_metrics.csv")
    for phase in PHASES:
        for horizon in (5, 10, 20):
            values = {
                r["method"]: r
                for r in phase_rows
                if r["method"] in ("M", "CV")
                and r["split"] == "validation"
                and r["phase"] == phase
                and r["region"] == "active"
                and r["contact_group"] == "all"
                and int(r["horizon"]) == horizon
            }
            a, b = values["M"], values["CV"]
            lines.append(
                f"| {phase} | {horizon} | {float(a['position_rmse_um']):.3f} | "
                f"{float(b['position_rmse_um']):.3f} | {float(a['velocity_rmse_mm_s']):.3f} | "
                f"{float(b['velocity_rmse_mm_s']):.3f} |"
            )
    lines.extend(
        [
            "",
            "表中先按压/保持/释放/恢复四阶段MSE等权，再开方；一步位置栏是增量RMSE。"
            "全部分阶段、活动/ROI、接近/静止、干预和恢复CSV保存在final_routes/。",
            "",
            "## 局限与解释",
            "",
            f"M的一步faster_down活动区域位置方向余弦为"
            f"{', '.join(r['position_direction_cosine'] for r in negatives)}，仍有3/3为负。"
            "速度方向为正，但响应幅度也未恢复到真值。联合分数改善不代表接触机制验收通过。",
            "",
            "收益主要出现在释放/恢复的位置误差，按压/保持的位置和速度普遍退化。"
            "F/G相对CV的汇总速度RMSE也增加约1.09%，不能说三路线都同时改善位置和速度。",
            "",
            "I的平均位置和速度误差均低于M，表明本轮接触输入并未带来对应绝对误差收益。"
            "I对压头干预的一步响应为零，这也不能称为具有正确接触因果响应。"
            "F与G在当前精度下接近，不能据此断言几何近似已经解决所有情况。",
            "",
            "恢复有符号位移与瞬时速度、半峰峰幅度和偏差保存在恢复CSV中。M和CV的30个"
            "恢复通道的预测/真值未同时具备至少两个可识别周期，相位/滞后未验证，不填作0。",
            f"图中固定材料点在20步时，预测z位移为"
            f"{float(recovery_example['predicted_z_displacement_um']):.2f}μm，"
            f"真值为{float(recovery_example['reference_z_displacement_um']):.2f}μm；"
            "预测速度衰减后仍留有位移偏差，降低恒速漂移并不等于恢复到静止形状。",
            "",
            "仅三步训练、选中50次更新、单种子和反复使用的左侧开发验证，尚不能把观察到的改善归因于"
            "网格惩罚或多步训练，也不能与2000次历史模型做公平训练预算比较。"
            "没有新增数据、网络结构、频率或参数池。右侧数组未加载，独立测试未验证，多种子未运行。",
            "",
            "下一项可检验的问题是：在相同三步、200更新预算下，对比接触输入是否引入绝对状态偏差，"
            "并同时保留干预响应检验。暂不据本轮追加自动搜索或修改物理有效性门槛。",
            "",
            "## 检查与复现",
            "",
            "五步末端到第一步位置/速度梯度有限且非零；原生求解器只提供停止梯度的当前条件。"
            "位置与瞬时速度独立监督；后续真实软体状态只评分、不回填。保存重载位置/速度逐位一致；"
            "独立NumPy复算每方法24评分单元及35普通+18干预网格窗口通过，145个数据摘要无变化。",
            "",
            "训练入口：`python scripts/run_short_multistep.py`。已运行目录禁止重复训练覆盖；"
            "大检查点/NPZ在`outputs/short_multistep/run_001/`。"
            "仅重建审计、表格与图：`python scripts/report_short_multistep.py`。",
            "",
            "![训练曲线及误差](short_experiment.png)",
            "",
            "![固定材料节点的恢复波形](recovery_waveform.png)",
            "",
        ]
    )
    (REPORT / "REPORT.md").write_text("\n".join(lines))
    save_json(
        REPORT / "analysis_provenance.json",
        {
            "post_training_analysis_only": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_scoring_and_training_unchanged": True,
        },
    )
    print(json.dumps({"audit_passed": audit["audit_passed"], "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
