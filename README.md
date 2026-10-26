# Explicit Contact-Driven Gaussian World Model

> 面向局部可变形物体的显式接触驱动 Gaussian 世界模型

本项目研究机器人与局部可变形物体交互过程中的三维未来状态预测。模型以
3D Gaussian 作为场景状态表示，从机器人与物体的 Gaussian 几何关系中解析
接触位置、接触法向、距离或穿透深度以及相对速度，再利用显式接触响应分别
驱动物体的整体近刚体运动和接触区域局部形变。

项目目前处于早期开发阶段，代码与实验结果将持续更新。

## 研究动机

现有视觉世界模型通常直接学习从当前状态和机器人动作到未来状态的映射：

$$
(\mathcal G_t,a_t)\longrightarrow \mathcal G_{t+1}.
$$

这种端到端预测方式容易将整体运动与局部形变混合在一起，也难以解释机器人
在什么位置接触物体、接触产生了多大的力，以及局部形变为什么发生。

本项目在状态转移过程中显式引入接触变量：

$$
(\mathcal G_t^{\mathrm{obj}},\mathcal G_t^{\mathrm{robot}})
\longrightarrow C_t^g
\longrightarrow (\mathbf f_t,\boldsymbol\tau_t)
\longrightarrow \mathcal G_{t+1},
$$

其中

$$
C_t^g=
\left\{
\mathbf p_{t,k},
\mathbf n_{t,k},
d_{t,k},
\mathbf v_{t,k}^{\mathrm{rel}}
\right\}_{k=1}^{K}
$$

表示接触点、接触法向、距离或穿透深度以及相对速度。

## 方法概览

模型的单步状态转移包含以下过程：

1. 将目标物体和机器人表示为两组 3D Gaussian；
2. 根据 Gaussian 几何关系解析显式接触变量；
3. 根据接触状态和材料参数计算接触力与力矩；
4. 由合力和合力矩预测物体整体平移与旋转；
5. 通过接触门控网络预测接触邻域的局部非刚性形变；
6. 在物体中心坐标系中组合局部形变与整体位姿；
7. 通过可微 Gaussian 渲染生成未来 RGB-D 观测；
8. 递归执行状态转移，实现未来多步预测。

整体建模思想为：

$$
\text{物体未来运动}
=
\text{整体近刚体运动}
+
\text{接触区域局部形变}.
$$

## 主要研究内容

### 1. Gaussian 几何接触解析

推理阶段不依赖仿真器提供的接触标签，而是从机器人 Gaussian 与物体
Gaussian 的相对位置和几何尺度中计算接触点、法向、距离与相对速度。

### 2. 显式接触响应

将接触力分解为法向弹性力、阻尼力和切向摩擦力，并根据接触位置计算作用于
物体质心的力矩。接触力用于连接机器人动作与物体状态变化。

### 3. 整体运动与局部形变解耦

接触产生的合力和合力矩驱动物体的整体近刚体运动；接触点、接触力、机器人
动作和材料属性共同驱动接触区域的局部形变。

### 4. 接触门控局部形变

第 $i$ 个 Gaussian 的局部门控权重由其中心与接触点之间的距离确定：

$$
w_i^t=
\max_k\exp\left(
-\frac{\|\mathbf c_i^t-\mathbf p_{t,k}\|^2}{2\sigma^2}
\right).
$$

靠近接触点的 Gaussian 获得较大的形变权重，远离接触区域的 Gaussian 主要
跟随物体整体运动。

### 5. 接触因果验证

除常规状态误差和渲染指标外，本项目计划通过以下反事实干预检验模型是否学习
到接触与形变之间的关系：

- 移除接触，局部形变应明显减弱；
- 移动接触点，主要形变区域应随之迁移；
- 缩放接触力，形变幅值应保持合理的单调变化；
- 释放物体，弹性材料应产生恢复过程。

## 仓库结构

```text
explicit-contact-gwm/
├── configs/                 # 训练与实验配置
├── scripts/                 # 数据处理、训练和评估入口
├── src/contact_gwm/
│   ├── structures.py        # Gaussian、刚体与接触状态
│   ├── math3d.py            # 四元数与坐标变换
│   ├── contact_geometry.py  # 显式接触几何解析
│   ├── contact_force.py     # 接触响应与整体运动
│   ├── deformation.py       # 接触门控局部形变
│   └── world_model.py       # 组合式 Gaussian 状态更新
└── tests/                   # 几何与物理模块测试
```

## 开发计划

- [x] 明确整体方法框架与状态变量
- [x] 实现 Gaussian 接触几何解析原型
- [x] 实现显式接触力与整体近刚体更新原型
- [x] 实现接触门控局部形变网络原型
- [x] 完成核心模块的单步组合与合成示例
- [ ] 构建 ManiSkill 仿真数据采集流程
- [ ] 使用仿真真值验证接触解析与接触力
- [ ] 建立物体中心 3D Gaussian 表示
- [ ] 接入可微 Gaussian 渲染器
- [ ] 完成单步预测与多步未来展开
- [ ] 加入非穿透、平滑与释放恢复约束
- [ ] 完成反事实实验和消融实验

## 数据与实验环境

计划首先在 ManiSkill 仿真环境中构建机器人操作局部可变形物体的数据集。训练
阶段可以使用仿真器输出的物体状态、接触点、法向和接触力作为监督；推理阶段
仅使用 RGB-D、机器人状态、动作、相机参数以及当前 Gaussian 状态。

## 安装

项目要求 Python 3.10 以上，建议使用 Python 3.10 或 3.11。
macOS / Apple Silicon 的 CPU 验证直接安装 PyTorch 即可，无需 CUDA。
本项目也已在 Apple M1、macOS 15.6.1、Python 3.13.5 上完成 CPU 验证，
环境版本、实测结果和复现步骤见 [CPU 验证记录](docs/cpu-validation.md)。
若后续在 NVIDIA GPU 上运行，再根据 CUDA 环境选择对应的 PyTorch。

```bash
git clone https://github.com/ztgao911-debug/explicit-contact-gwm.git
cd explicit-contact-gwm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

运行合成接触示例：

```bash
python scripts/toy_rollout.py --device cpu
```

运行测试：

```bash
python -m pytest
```

ManiSkill 和可微 3D Gaussian Splatting 渲染器将在后续实验阶段接入。

## 可控合成实验

该脚本验证无接触静止、穿透深度与接触力的关系、接触门控与力矩的位置变化，
并在相同物理时长内比较不同时间步的多步运行结果。
绘图依赖为可选安装项，不影响原有 Toy 和测试。

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,experiments]"
python scripts/synthetic_validation.py --device cpu
```

默认结果保存在 `outputs/synthetic_validation/`，包含 CSV 数据、JSON 检查结果和
四张 PNG 图；任意行为检查失败时脚本返回非零退出码。
默认进行 23 项合成检查，它们与原有 7 个 pytest 测试分别统计。
实测结果、每步含义及限制见 [合成验证记录](docs/synthetic-validation.md)。

## 独立 MuJoCo 软体试采

使用 MuJoCo 原生 CPU 的三维四面体弹性软块，底部固定，球形压头由位置伺服器驱动。
采集左 / 中 / 右 × 浅 / 深共六条按压轨迹及一条无接触对照，
每条包含接近、按压、保持、释放和恢复观察。
节点与接触标签直接来自独立求解器，配置见 `configs/mujoco_soft_pilot.yaml`。

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,simulation]"
python scripts/collect_soft_pilot.py
python scripts/collect_soft_pilot.py --validate-only data/mujoco_soft_pilot/pilot_001
python -m pytest
```

默认输出 `data/mujoco_soft_pilot/pilot_001/`；已有目录会被拒绝覆盖，
再次试采可用 `--output-dir` 指定新目录。
数据、场景 XML、配置、求解器状态、检查报告和调试图保存在一起。
七条试采的实测结果与字段说明见 [MuJoCo 软体试采记录](docs/mujoco-soft-pilot.md)。

## Gaussian 局部形变首次训练

将 `pilot_001` 的中央浅 / 深按压轨迹连接到接触门控形变网络：每个节点对应一个
Gaussian，固定尺度与旋转，硬约束底部节点，只预测相邻 0.01 秒记录帧的中心增量。
本轮使用当前帧 MuJoCo 接触与力真值作为调试条件，拟合按压 / 保持帧对。

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,experiments]"
python scripts/train_soft_local.py
python -m pytest
```

默认结果在 `outputs/soft_local_training/run_001/`，包含模型、损失、预测、误差及曲线。
已有目录不会被覆盖；重跑可指定 `--output-dir outputs/soft_local_training/run_002`。
数据目录保持只读。输入、初始化规则、基线比较和实际运行结果见
[首次 CPU 训练记录](docs/soft-local-training.md)。本轮成绩仅代表训练帧对上的一步拟合。

## 恒速残差与完整轨迹划分比较

在相同当前状态 / 接触真值条件下，比较静止、恒速、直接位移网络和恒速残差网络。
中央两条训练，左侧两条验证，右侧两条测试；网络按验证损失选定后再评估测试集。
原直接位移训练入口和 `outputs/soft_local_training/run_001/` 保留。

```bash
source .venv/bin/activate
python scripts/compare_soft_models.py
python scripts/compare_soft_models.py --evaluate-only outputs/soft_residual_comparison/run_001
python -m pytest
```

新增配置 `configs/compare_soft_residual.yaml` 继承原归一化、网络规模及帧筛选规则。
默认输出 `outputs/soft_residual_comparison/run_001/`；重跑时需指定其他 `--output-dir`。
本轮测试集上恒速基线优于残差网络，完整实测结果见
[恒速残差比较记录](docs/soft-residual-comparison.md)。本轮始终是提供真实当前状态的一步预测。

## 恒速残差诊断与小样本拟合

只使用中央训练轨迹、左侧验证轨迹，统计恒速残差的大小和空间分布，
再用固定 8 帧对检查原网络的拟合能力，并仅改变输出上限做一次对照。

```bash
source .venv/bin/activate
python scripts/diagnose_soft_residual.py
python scripts/diagnose_soft_residual.py \
  --config configs/diagnose_soft_residual_scaled.yaml \
  --output-dir outputs/soft_residual_diagnostics/run_002_output_scale \
  --compare-to outputs/soft_residual_diagnostics/run_001
python -m pytest
```

结果保存在独立的 `outputs/soft_residual_diagnostics/`，已有目录会拒绝覆盖。
本次两种设置均未达到小样本误差降低 50% 的诊断标准，仅缩小输出上限没有改善。
逐项实测结果、图表说明和新目录复现命令见
[恒速残差诊断记录](docs/soft-residual-diagnostics.md)。本轮成绩不能作为泛化结果。

## 单节点、单帧拟合与梯度检查

使用一帧中央深按压数据，检查单节点与整帧拟合、逐层梯度、数值差分，
并运行独立节点参数和 L-BFGS 两个诊断对照。原模型及历史输出均保留。

```bash
source .venv/bin/activate
python scripts/diagnose_soft_optimization.py
python scripts/diagnose_soft_optimization.py \
  --verify-only outputs/soft_optimization_diagnostics/run_001
python -m pytest
```

默认结果在 `outputs/soft_optimization_diagnostics/run_001/`，已有目录拒绝覆盖。
本轮单节点和独立节点对照可拟合，原共享 MLP 的单帧拟合仍未达到 90% 误差降幅标准。
梯度实测、各对照的用途及复现步骤见[优化诊断记录](docs/soft-optimization-diagnostics.md)。

## 单帧损失尺度对照

保持第 71 帧、原网络、Adam 和训练预算不变，只将原损失乘以 1、100、10000，
记录梯度裁剪和实际参数更新，并用公共物理误差比较结果。

```bash
source .venv/bin/activate
python scripts/diagnose_soft_loss_scale.py
python scripts/diagnose_soft_loss_scale.py \
  --verify-only outputs/soft_loss_scale_diagnostics/run_001
python -m pytest
```

默认输出 `outputs/soft_loss_scale_diagnostics/run_001/`，已有目录拒绝覆盖。
×100 在本帧有有限改善，×10000 更差；三组均未达到强拟合标准。
实测指标、逐层更新分析及复现步骤见[损失尺度对照记录](docs/soft-loss-scale-diagnostics.md)。

## 固定 8 帧对损失尺度复核

复用原固定中央样本，在相同 2000 步预算下比较损失 ×1、×100，
分别统计按压 / 保持和活动节点 / 接触邻域误差。

```bash
source .venv/bin/activate
python scripts/diagnose_soft_loss_scale.py \
  --config configs/diagnose_soft_eight_pair_loss_scale.yaml \
  --output-dir outputs/soft_eight_pair_loss_scale/run_001
python scripts/diagnose_soft_loss_scale.py \
  --verify-only outputs/soft_eight_pair_loss_scale/run_001
```

×100 的有限改善在这批拟合样本上保持，但仍未达到原小样本拟合标准。
完整实测与复现说明见[8 帧对复核记录](docs/soft-eight-pair-loss-scale.md)。

## 固定 8 帧对空间编码消融

在相同损失 ×100 和 2000 步预算下，比较原 MLP、追加固定静止坐标正弦 / 余弦编码的 MLP，
以及参数量接近的加宽原 MLP。频率固定为 0.5、1、2，不搜索频率或学习节点编号嵌入。

```bash
source .venv/bin/activate
python scripts/ablate_soft_spatial_encoding.py
python scripts/ablate_soft_spatial_encoding.py \
  --verify-only outputs/soft_spatial_encoding_ablation/run_001
```

默认输出 `outputs/soft_spatial_encoding_ablation/run_001/`，已有目录拒绝覆盖。
空间编码组本次总体拟合优于原模型和近参数量对照，但保持阶段活动节点误差退化，
三组均未达到小样本诊断标准；完整数据和限制见[空间编码消融记录](docs/soft-spatial-encoding-ablation.md)。

## 固定边界接触管线与短时预测

使用相同的既有局部网络，比较 MuJoCo 几何 + 原生力、MuJoCo 几何 + 解析力、
高斯几何 + 解析力，并与恒速基线比较。软块底部固定，不执行自由刚体积分。
5 / 10 / 20 步实验给定压头实际轨迹，软体位置、速度和接触只由预测状态递推。

```bash
source .venv/bin/activate
python scripts/validate_contact_pipeline.py
python -m pytest
ruff check .
```

依赖已有试采数据及空间编码 B 检查点，默认写入新的
`outputs/contact_pipeline_validation/run_NNN/`，不覆盖历史输出。
本轮管线已实际运行，但整体预测未超过恒速；高斯几何有漏检，现有接触门控也无法表示无接触恢复。
配置、完整指标、速度更新近似与唯一后续优先项见
[固定边界接触管线实验记录](docs/contact-pipeline-validation.md)。

## 引用

本项目仍在开发中。论文与引用信息将在正式发布后补充。

## License

This project is released under the MIT License.

