# 开发指南

先跑一遍 [快速开始](../../README.md#快速开始)，确认能训练、评估并生成回放。然后按你要改的地方找文件。

## 代码放在哪里

| 要改什么 | 打开哪个文件 |
|---|---|
| 训练参数 | [`configs/algorithms/`](../../configs/algorithms/) 下的 YAML。 |
| 算法 | [`algorithms/<算法>/algorithm.py`](../../src/teamwork/algorithms/)。训练循环和网络可单独写在 `trainer.py`、`networks.py` 或 `learner.py`。 |
| 仿真场景 | [每个场景的 `environment.py`](../../src/teamwork/environments/)。修改观测、动作空间、奖励或回合结束条件。 |
| 评分规则 | [`evaluation/metrics/<场景>/rules.py`](../../src/teamwork/evaluation/metrics/)。例如 Spread 的覆盖率。 |
| 公共训练入口 | [`experiments/runner.py`](../../src/teamwork/experiments/runner.py)。它选算法、设置随机种子，再记录训练耗时。 |
| Python SDK | [`api/`](../../src/teamwork/api/)。这里列出可调用的算法，校验请求，管理任务和文件。 |

具体算法和场景各放一个目录。公共接口、注册表和多个实现共用的代码放在上层。比如双臂交接的 [环境](../../src/teamwork/environments/robotics/two_arm_handover/)、[评分](../../src/teamwork/evaluation/metrics/two_arm_handover/rules.py) 和 [手写示范策略](../../src/teamwork/algorithms/baselines/two_arm_handover/policy.py) 分别在三个位置。双臂轨迹绘图在 [机器人评估目录](../../src/teamwork/evaluation/robotics/)。

## 新增算法

1. 先看 [PPO 的入口](../../src/teamwork/algorithms/ppo/algorithm.py) 和 [算法接口](../../src/teamwork/algorithms/base.py)。实现 `train(config, out)` 来训练并保存模型，再实现 `load_policy(...)` 来加载模型。评估器只需要调用策略的 `reset()` 和 `act(observations)`。
2. 在 [算法注册表](../../src/teamwork/algorithms/registry.py) 加上算法名和创建函数。检查 `action_kind` 是否与场景匹配。离散动作是从几个选项里选一个，连续动作是输出有范围的数值，例如控制机械臂的位置变化。
3. 在 [`configs/algorithms/`](../../configs/algorithms/) 加一份配置。先在一个场景跑短训练，确认模型能保存、能加载并得到评分。再用统一预算比较效果。

策略选动作时只接收自己能看到的信息，也叫局部观测。训练时，用于估计价值的 critic 可以读取 `state()` 提供的全局状态。评分器用 `task_snapshot()` 取任务数据，例如目标的位置。带记忆的策略在每回合调用 `reset()` 清掉上回合的内部状态。

要新增场景，请从 [环境指南](../environments/README.md#接入或编写环境) 开始。

## 训练会写出什么

每次运行使用一个新目录。 `config.json` 保存配置，`episodes.csv` 逐回合记录累计步数和每个智能体的原始回报，`training.json` 记录耗时和预算。模型文件也叫 checkpoint，包含模型参数、配置和实际步数，由算法的 `load_policy()` 读取。环境执行一次 `step()` 就计一步。

如果要从 Python 启动训练，可以调用同一个入口：

```python
from pathlib import Path

import yaml

from teamwork.experiments.runner import run_training

config = yaml.safe_load(Path("configs/algorithms/ippo.yaml").read_text(encoding="utf-8"))
checkpoint = run_training({**config, "total_steps": 1024}, Path("runs/ippo-python-check"))
```

## 检查改动

在仓库根目录运行。只改了一个模块时，可以先用 `pytest -q <测试文件>` 跑对应测试。

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python scripts/smoke.py --skip-tests
```

最后一条会跑短训练、模型评估和绘图，方便检查流程。正式实验的步数、种子和指标见 [实验规则](../experiments/README.md)。本地 `runs/`、缓存和 Python 环境由 Git 忽略。
