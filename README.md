# 多智能体强化学习算法对比

在统一环境中比较算法，并考察同一算法跨场景的表现。合作场景包括 Spread、LBF 和 RWARE，Tag 用固定对手研究追捕方的协作。

## 快速开始

按 [安装与验收](docs/SETUP.md) 激活项目的 Python 3.12 环境，再从仓库根目录运行。输出目录使用一个新名称。

```bash
python -m teamwork.cli train --config configs/algorithms/ippo.yaml --env spread --steps 1024 --seed 0 --out runs/ippo_spread_check
python -m teamwork.cli evaluate --checkpoint runs/ippo_spread_check/model.pt --episodes 3 --seed 10000 --out runs/ippo_spread_check/eval.json
```

以上命令用于接入检查。正式实验使用 [实验协议](docs/EXPERIMENTS.md) 约定的训练预算、种子和指标。

## 连续动作实验

MuJoCo 双 Panda 场景包括协作抬升和接力交接。双臂交接可在 WSL/Ubuntu 使用 Conda，通过一条命令复现示范采集、行为克隆和留出评估。[复现步骤](docs/ENVIRONMENT_DEVELOPMENT.md#mujoco-双臂交接)

## 算法与场景配置

| 想做什么 | 从哪里开始 |
|---|---|
| 找算法名、场景配置或 SDK 支持范围 | [算法目录与分工](docs/ALGORITHMS.md) |
| 运行对照、核对指标或复现实验 | [实验协议](docs/EXPERIMENTS.md) |
| 修改算法、环境或评估代码 | [开发指南](docs/DEVELOPMENT.md) |
| 用 Python SDK 提交任务或加载策略 | [SDK 接入说明](docs/API_INTEGRATION.md) |
| 查看 IPPO/IQL 接入记录 | [SDK 验收记录](docs/API_ACCEPTANCE.md) |
| 了解环境来源及备选场景 | [环境参考](docs/references/ENVIRONMENTS.md) |

## 仓库导航

- [`src/teamwork/algorithms/`](src/teamwork/algorithms/) 提供算法注册、训练和策略加载。
- [`src/teamwork/environments/`](src/teamwork/environments/) 提供仿真环境适配。
- [`src/teamwork/evaluation/`](src/teamwork/evaluation/) 定义公共指标、对比和图表。
- [`configs/algorithms/`](configs/algorithms/) 保存算法配置，运行产物写入本地 `runs/`。
- [`tests/`](tests/) 与 [`scripts/smoke.py`](scripts/smoke.py) 检查接口和短程训练评估链路。

![研究流程](docs/assets/architecture.svg)
