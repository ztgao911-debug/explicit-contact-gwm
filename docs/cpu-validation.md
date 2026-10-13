# 首次 CPU 运行验证

验证日期：2026-10-12（Asia/Shanghai）。起始提交：`271baf0`。
使用本地仓库，origin 与用户提供的 GitHub 地址一致；开始时工作区干净。

## 环境与操作目的

| 项目 | 实测 |
| --- | --- |
| 机器 | MacBook Air，Apple M1，8 核，16 GB 内存 |
| 系统 / 架构 | macOS 15.6.1 / arm64 |
| Python | 3.13.5 |
| Git | 2.50.1（Apple Git-155） |
| 环境 | 仓库根目录 `.venv`，由现有 Python 创建 |
| pip | 26.2.1 |
| PyTorch | 2.14.1 |
| NumPy / PyYAML | 2.5.4 / 6.0.3 |
| pytest | 9.1.1 |
| 运行设备 | CPU，显式指定 `--device cpu` |

1. 阅读 README、pyproject.toml、配置、核心模块和四个测试文件，确认入口和依赖。
2. 检查系统、架构、Python 和 Git，确保安装适配 Apple arm64 的二进制包。
3. 建立 `.venv`，隔离项目依赖；以 editable 模式安装项目，代码修改能直接生效。
4. 运行 `python -m pytest`，验证现有几何、接触力、形变门控与单步组合测试。
5. 运行 CPU Toy，检查从接触解析到力、整体运动及局部形变的完整单步链路。
6. 运行 `python -m pip check`，确认安装后的依赖版本没有冲突。

README 推荐 Python 3.10/3.11，项目配置允许 `>=3.10`。
本次使用机器已有的 Python 3.13.5，并通过实际安装和测试确认此组合可运行。
[PyTorch 官方 macOS 安装文档](https://pytorch.org/get-started/locally/)
列出 Python 3.10–3.14 的支持范围。

## 安装问题与处理

- 沙箱内首次联网失败，需要在获得联网执行权限后下载依赖。
- Python.org 安装的 Python 缺少默认 CA 文件，导致 pip HTTPS 证书校验失败。
  安装命令通过 `--cert /opt/anaconda3/ssl/cert.pem` 使用本机已有的 Conda CA 证书。
  证书校验保持开启；没有修改系统 Python 或 Conda 环境。
- 最终 `pip install -e '.[dev]'` 成功；`pip check` 输出
  `No broken requirements found.`，退出码为 0。
- 没有发现需要修改核心模型的安装或运行错误。Toy 仅增加两行诊断输出，
  显示物体局部中心及 log-scale 的最大绝对变化。README 更新 CPU 命令与记录链接。

## 预期与实测

| 验收项 | 预期 | 实测 |
| --- | --- | --- |
| 现有测试 | 7 个通过 | `7 passed in 5.83s`，退出码 0 |
| 接触识别 | 有接触 | `contacts: 1` |
| 物体接触合力 | z 分量为负 | `[0.0, 0.0, -42.399986267089844]` |
| 物体整体运动 | z 位置减小 | 从 `[0, 0, 0]` 到 `[0.0, 0.0, -0.06783997267484665]` |
| 局部中心变化 | 零初始化网络输出 0 | 最大绝对变化 `0.0` |
| 局部尺度变化 | 零初始化网络输出 0 | log-scale 最大绝对变化 `0.0` |

测试原始输出：

```text
.......                                                                  [100%]
7 passed in 5.83s
```

增加诊断输出后，CPU Toy 实测输出（退出码 0）：

```text
contacts: 1
net force on object: [0.0, 0.0, -42.399986267089844]
next rigid position: [0.0, 0.0, -0.06783997267484665]
contact gates: [0.562030017375946, 0.7283307313919067, 0.562030017375946, 0.7283307313919067, 0.9438387155532837, 0.7283307313919067, 0.562030017375946, 0.7283307313919067, 0.562030017375946]
max local center change: 0.0
max local log-scale change: 0.0
```

## 如何理解这些数值

本例以 z 正方向为向上，机器人位于物体上方并向下运动。
中央物体 Gaussian 半径为 0.11，机器人 Gaussian 半径为 0.13，
中心距离为 0.19，因此按本例几何定义，初始穿透深度约为
`0.11 + 0.13 - 0.19 = 0.05`。

按配置预期，弹性力大小约为 `800 × 0.05 = 40`，阻尼贡献为
`12 × 0.20 = 2.4`，故合力约为向下 42.4。
时间步为 0.02，质量为 0.25，初速度为 0，半隐式 Euler 更新预期位移为
`(−42.4 / 0.25) × 0.02² = −0.06784`。
这与实测吻合，微小差异来自浮点计算。Toy 的重力设为 0，
因此这里的整体位移由接触力驱动。

中央 Gaussian 的门控约为 0.94384，角落约为 0.56203，体现接触邻域权重更高。
门控不是形变量：形变网络最后一层的权重和偏置都初始化为零，
因此即使有接触、门控非零，当前局部中心及尺度变化仍严格为零。
这是未训练模型的预期初始行为。

本次结果只验证现有合成单步基线能够运行，尚未验证真实材料参数、
仿真真值精度、多步稳定性或学习后的形变效果。

## 在当前机器复现

在仓库根目录执行：

```bash
source .venv/bin/activate
python -m pytest
python scripts/toy_rollout.py --device cpu
python -m pip check
```

已有 `.venv` 无需重复安装。IDE 的 Python 解释器也应选择 `.venv/bin/python`。

若在同类 macOS arm64 环境重新建立环境，使用此次依赖快照：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu-macos-arm64.txt
python -m pip install -e '.[dev]'
python -m pytest
python scripts/toy_rollout.py --device cpu
```

快照记录已安装的第三方依赖版本，不含项目自身，亦非跨平台通用锁文件。
如果本机仍有相同 CA 问题，且 `/opt/anaconda3/ssl/cert.pem` 存在，
在上述每条 pip install 命令中加入 `--cert /opt/anaconda3/ssl/cert.pem`。
其他机器应使用自身有效的 CA 证书路径。
