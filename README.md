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

## 引用

本项目仍在开发中。论文与引用信息将在正式发布后补充。

## License

This project is released under the MIT License.

