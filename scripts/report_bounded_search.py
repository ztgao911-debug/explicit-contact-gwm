"""Publish the fixed-budget search evidence without changing any scores."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def json_file(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def number(value):
    return f"{float(value):.6g}" if value not in (None, "") else "未完成"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/bounded_multistep_search/run_001"
    )
    args = parser.parse_args()
    report = args.report.resolve()
    protocol = json_file(report / "protocol.json")
    if protocol is None:
        raise FileNotFoundError("Frozen protocol is required before reporting")
    ledger = json_file(report / "trial_ledger.json", [])
    current = {}
    for row in ledger:
        current[row["candidate_id"], row["seed"]] = row
    trials = list(current.values())
    ranking = read_csv(report / "candidate_rankings.csv")
    status = json_file(report / "status.json", {"status": "running"})
    baseline = json_file(report / "baseline_summary.json", {})
    selected = json_file(report / "selected_for_replication.json", {"candidate_ids": []})[
        "candidate_ids"
    ]
    lines = [
        "# 固定预算多步自动实验搜索",
        "",
        "本轮属于开发验证，不是独立测试或全局最优参数证明。",
        "",
        f"参考提交：`{protocol['reference_head']}`；分支：`{protocol['branch']}`。",
        f"状态：**{status['status']}**。搜索上限12配置、种子7；最多2候选×17/27复核；总墙钟上限6小时。",
        "",
    ]

    def paragraph(text):
        lines.extend([text, ""])

    def table(headers, rows):
        lines.extend(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        )
        lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        lines.append("")

    valid = sorted(
        [r for r in ranking if r.get("eligible") == "True"], key=lambda r: float(r["joint_score"])
    )
    if not trials and status["status"] == "budget_stop":
        paragraph(
            "**本轮未完成正式自动搜索：0个候选开始训练，0次优化器更新，0次多种子复核。** "
            "原六小时墙钟截止前没有启动正式候选；停止记录不重置预算。"
            "因此没有预算内最优配置，也不能将未搜索解释为模型或候选算法失败。"
        )
    elif valid:
        best = valid[0]
        paragraph(
            f"预算内种子7有效候选中，联合排名第一为 **{best['candidate_id']}**，"
            f"联合分数 **{number(best['joint_score'])}**。"
            "这仅表示通过本轮物理门槛后的开发验证排名。"
        )
    else:
        paragraph(
            "**本轮预算内未找到通过全部物理有效性检查的配置。** "
            "没有放宽门槛、删除失败窗口或追加第13个配置。"
            if status["status"] != "running"
            else "搜索仍在进行，当前尚无完成的有效候选。"
        )
    search_started = status.get(
        "search_configurations_started", sum(r["role"] == "search" for r in trials)
    )
    replications_started = status.get(
        "replication_trainings_started", sum(r["role"] == "replication" for r in trials)
    )
    paragraph(
        f"已开始的不同搜索配置：{search_started}；复核训练：{replications_started}；"
        f"墙钟时间：{number(status.get('wall_elapsed_seconds'))} s。"
        "算法失败计入配置预算，技术尝试单独保存。"
    )
    stop = json_file(report / "budget_stop_details.json", {})
    if stop:
        paragraph(
            f"预算开始／截止（北京时间）：{stop.get('start_beijing')} ／ "
            f"{stop.get('deadline_beijing')}。发现已超时：{stop.get('observed_beijing')}。"
            "墙钟包括实现、检查与期间未推进实验的时间，不等于CPU计算时间。"
            "不能用0.18秒的微基准反推出已执行2000次训练；这里没有这样的训练。"
        )
    lines.extend(["## 冻结协议与训练路径", ""])
    paragraph(
        "沿用同一45324参数网络、空间编码、归一化、接触几何、解析力参数、数据划分与ROI。"
        "每支路上限固定为16 mm和1.6 m/s。训练池固定为中央普通600帧对、"
        "已有108中央分支及90初始响应对；"
        "左侧普通轨迹和18分支仅做开发验证，右侧不参与搜索。"
    )
    paragraph(
        "A在五个端点前逐帧读取真实当前状态，B只在起点初始化并五步自回归，C在B上加网格惩罚。"
        "三组共享相同五端点、普通绝对监督和初始响应监督。其余9组在固定池无放回生成，候选清单先于正式训练保存。"
    )
    paragraph(
        "专用训练路径保留x/v、重算状态特征、距离核与网络更新的梯度；MuJoCo查询读取预测状态的detached副本，"
        "返回接触量停止梯度。因此是条件梯度近似，不是穿过求解器全程可微。位置和端点瞬时速度独立监督。"
    )
    check = json_file(report / "gradient_preflight.json", {})
    paragraph(
        f"实际五步梯度预检：末端损失对第一步位置／速度梯度范数分别为"
        f" **{number(check.get('terminal_to_first_x_gradient_norm'))} / "
        f"{number(check.get('terminal_to_first_v_gradient_norm'))}**；"
        "固定节点严格约束，未执行优化器更新。无接触时沿用原零接触上下文，内部状态特征仍保留梯度。"
    )
    paragraph(
        "展开项按所有监督端点求平均。网格项为mean(relu(0.1−V/V_rest)²)，0.1是训练惩罚阈值，"
        "物理筛选中的翻转阈值则是V/V_rest≤0。中央原生数据已先检查通过，未根据左侧结果修改阈值。"
    )
    paragraph(
        "每组最多2000更新、batch16；相同更新数不等于相同计算量。所有候选用同一30个五步AR窗口、"
        "固定位置／速度0.5权重、固定单位0.001 m／0.1 m/s及四阶段／两区域等权选模，包含第0步。"
        "选模不加候选网格项，不随训练速度权重改变。"
    )
    table(
        ["配置", "训练模式", "LR", "展开步数", "速度权重", "网格权重"],
        [
            [
                c["id"],
                c["mode"],
                c["learning_rate"],
                c["horizon"],
                c["velocity_weight"],
                c["mesh_weight"],
            ]
            for c in protocol["candidates"]
        ],
    )
    lines.extend(["## 候选排名与物理有效性", ""])
    paragraph(
        "排名使用位置和瞬时速度等权的固定归一化平方误差，按压／保持／释放／恢复、活动／ROI、5／10／20步均等权。"
        "接近、静止、干预和恢复波形另报。任何评价窗口出现非有限值、固定边界破坏、翻转或缺失，均不进入有效排名。"
    )
    table(
        [
            "配置",
            "状态",
            "有效",
            "联合分数",
            "位置分数",
            "速度分数",
            "翻转状态/窗口",
            "最低体积比",
            "选中步",
            "秒",
        ],
        [
            [
                r["candidate_id"],
                r["status"],
                r["eligible"],
                number(r["joint_score"]),
                number(r["position_score"]),
                number(r["velocity_score"]),
                f"{r['inverted_states']}/{r['inverted_windows']}",
                number(r["minimum_volume_ratio"]),
                r["selected_step"],
                number(r["elapsed_seconds"]),
            ]
            for r in ranking
        ],
    )
    if baseline:
        table(
            ["固定参照（未训练）", "有效", "联合", "位置", "速度"],
            [
                [
                    name,
                    row["eligible"],
                    *(
                        number(row["scores"].get(k))
                        for k in ("joint_score", "position_score", "velocity_score")
                    ),
                ]
                for name, row in baseline.items()
            ],
        )
    paragraph(
        "E0为已有扩上限绝对监督检查点，CV为固定边界恒速基线，均未新训练、不计作搜索配置。"
        "主CSV同时保存联合、位置、速度各自的有效排名；单项最佳不自动代表其他指标改善。"
    )
    smoke = json_file(report / "evaluator_smoke_check.json", {})
    if smoke:
        m = smoke["per_method"]["M"]
        table(
            ["实际运行的预检（已有E0，未训练）", "实测值"],
            [
                [
                    "固定联合／位置／速度分数",
                    "/".join(
                        number(m["scores"][k])
                        for k in ("joint_score", "position_score", "velocity_score")
                    ),
                ],
                ["普通／干预20步窗口数", "35 / 18"],
                ["翻转状态／窗口", f"{m['inverted_states']} / {m['inverted_windows']}"],
                ["最低体积比", number(m["minimum_volume_ratio"])],
                ["评价秒数", number(smoke["elapsed_seconds"])],
            ],
        )
        paragraph(
            "这次已有检查点评价仅验证固定评价器和证据保存，不是本轮训练收益。"
            "详细位置／瞬时速度、阶段／区域、干预和恢复CSV及曲线位于"
            "`preflight_existing_E0/`，大预测NPZ保存在同名outputs目录。"
            "本轮CV及最终M/F/G/I尚未运行；不得拿历史结果冒充这些未执行比较。"
        )
    lines.extend(["## 多种子复核", ""])
    paragraph(
        f"选定候选：`{', '.join(selected) or '无'}`。"
        + (
            "复核规则固定为原配置、完整报告17／27，不能只挑最好种子。"
            if selected
            else "未启动复核，种子17／27没有训练结果或统计量。"
        )
    )
    table(
        ["候选", "指标", "n", "均值", "样本标准差", "三种子完成", "三种子均有效"],
        [
            [
                r["candidate_id"],
                r["metric"],
                r["n"],
                number(r["mean"]),
                number(r["sample_std"]),
                r["all_three_completed"],
                r["all_three_physically_valid"],
            ]
            for r in read_csv(report / "multiseed_summary.csv")
        ],
    )
    paragraph(
        "标准差使用ddof=1；不足三种子明确标记未完成。历史E0只有既有种子7，不能把复核均值与它的比较称为匹配多种子的因果改善或统计显著性。"
    )
    lines.extend(["## 干预、恢复与三路线", ""])
    faster, responses = [], []
    for result in trials:
        published = ROOT / result.get(
            "published_directory",
            str(report / "trials" / f"{result['candidate_id']}_seed_{result['seed']}"),
        )
        path = published / "evaluation/response_evaluation/per_case_response_metrics.csv"
        for r in read_csv(path):
            if r["variant"] == "faster_down" and r["horizon"] == "1" and r["region"] == "active":
                faster.append({"candidate_id": result["candidate_id"], "seed": result["seed"], **r})
        for r in read_csv(published / "evaluation/response_evaluation/summary.csv"):
            responses.append({"candidate_id": result["candidate_id"], "seed": result["seed"], **r})
    if faster:
        with (report / "faster_down_acceptance.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(faster[0]))
            writer.writeheader()
            writer.writerows(faster)
        table(
            ["候选/种子", "起点", "一步位置响应余弦", "预测幅度μm", "真实幅度μm"],
            [
                [
                    r["candidate_id"] + "/" + str(r["seed"]),
                    r["frame"],
                    number(r["position_direction_cosine"]),
                    number(r["position_predicted_response_rms"]),
                    number(r["position_reference_response_rms"]),
                ]
                for r in faster
            ],
        )
    final = json_file(report / "final_routes_summary.json", [])
    table(
        ["候选", "种子", "路线", "物理有效", "联合分数", "翻转状态/窗口"],
        [
            [
                r["candidate_id"],
                r["seed"],
                method,
                m["eligible"],
                number(m["scores"]["joint_score"]),
                f"{m['inverted_states']}/{m['inverted_windows']}",
            ]
            for r in final
            if "per_method" in r
            for method, m in r["per_method"].items()
        ],
    )
    paragraph(
        "M/F/G分别是原生几何原生力、原生几何解析力、修正表面解析力；F/G公式与参数相同。"
        "I是冻结输入消融，不能冒充公平重训的无接触基线。每条路线的物理状态单独报告。"
    )
    paragraph(
        "恢复文件保存有符号z位移和瞬时vz，包括活动/ROI均值及固定材料点，报告幅度误差和相位/滞后代理。"
        "若窗口不足两个周期，相位明确为未验证，不填0。联合分数降低不等于faster_down、响应幅度或恢复波形全部通过。"
    )
    lines.extend(["## 停止、限制与复现", ""])
    paragraph(
        "本轮没有独立测试结果，右侧未参与搜索，左侧已反复选模，因此结论限于开发验证。"
        "若预算内没有物理有效且有收益的配置，结论为本轮预算内未找到；不追加候选、放宽筛选或删除失败窗口。"
    )
    if not trials:
        paragraph(
            "本轮预算内未找到配置的直接原因是正式搜索未启动，而不是已证明12个配置均无效。"
            "位置与速度能否共同改善、网格惩罚能否降低翻转、faster_down方向能否修复，均仍未验证。"
            "不能据此选择下一项模型结构修改；下面的结构解释只保留为已有问题的待验假设。"
        )
    paragraph(
        "若出现持续的速度/网格问题，下一结构假设是当前独立位置/速度离散映射及局部接触聚合仍不足以约束恢复动力学；"
        "这属于待验证假设，不因完成运行而视为已证实。"
    )
    paragraph(
        "复现：先执行`python scripts/run_bounded_search.py --prepare-only`完成预检与冻结，"
        "再执行`python scripts/run_bounded_search.py`；"
        "最终用`python scripts/report_bounded_search.py`重建此报告。"
        "源代码/数据摘要、协议、候选、停止及检查JSON随实验目录保存；"
        "训练执行时还会保存逐更新曲线和选模记录。大权重和NPZ位于忽略的outputs目录。"
    )
    if not trials:
        paragraph(
            "当前run_001截止时间已经过期，上述搜索入口会继续拒绝候选训练并整理停止状态，"
            "不会自动续期。若另行启动新一轮，需要新的明确预算与独立运行目录；"
            "不能修改本轮budget.json伪装成仍在原预算内。当前没有候选权重、训练曲线或选模记录可供展示。"
        )
    (report / "REPORT.md").write_text("\n".join(lines))

    if ranking:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(ROOT / "outputs/bounded_multistep_search/run_001/matplotlib")
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for r in ranking:
            if r["position_score"] in ("", None) or r["velocity_score"] in ("", None):
                continue
            valid_row = r["eligible"] == "True"
            x, y = float(r["position_score"]), float(r["velocity_score"])
            ax.scatter(
                x, y, marker="o" if valid_row else "x", color="tab:blue" if valid_row else "tab:red"
            )
            ax.annotate(r["candidate_id"], (x, y), fontsize=8)
        ax.set(
            xlabel="Fixed normalized position score",
            ylabel="Fixed normalized instantaneous-velocity score",
            title="Development validation; X = physically invalid",
        )
        ax.grid(alpha=0.25)
        fig.savefig(report / "position_velocity_tradeoff.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for result in trials:
            if result["seed"] != 7:
                continue
            published = ROOT / result.get("published_directory", "")
            curve = read_csv(published / "training_curve.csv")
            usable = [r for r in curve if r.get("validation_joint_score") not in ("", None)]
            if usable:
                ax.semilogy(
                    [int(r["update"]) for r in usable],
                    [float(r["validation_joint_score"]) for r in usable],
                    label=result["candidate_id"],
                )
        ax.set(xlabel="Optimizer updates", ylabel="Fixed five-step joint selection score")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.25)
        fig.savefig(report / "fixed_validation_curves.png", dpi=160)
        plt.close(fig)
    print(
        json.dumps(
            {
                "report": str(report / "REPORT.md"),
                "status": status,
                "independent_test_verified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
