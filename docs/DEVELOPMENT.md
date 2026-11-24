# 开发指南

命令均在仓库根目录运行。环境和依赖见 [安装与验收](SETUP.md)。算法和场景的选择见 [算法目录](ALGORITHMS.md)。

## 代码职责

| 改动 | 入口 | 职责 |
|---|---|---|
| 算法接入 | [`algorithms/base.py`](../src/teamwork/algorithms/base.py)、[`algorithms/registry.py`](../src/teamwork/algorithms/registry.py) | 实现 `Algorithm` 和 `Policy`，注册 CLI 标识 |
| 算法实现 | [`algorithms/`](../src/teamwork/algorithms/) | `algorithm.py` 适配训练、策略加载和动作接口；训练循环与网络按需要放入 `trainer.py`、`networks.py` 或 `learner.py` |
| 场景或动作空间 | [`environments/`](../src/teamwork/environments/) | 输出按智能体命名的观测、奖励、`terminated` 和 `truncated` |
| 实验元数据 | [`experiments/runner.py`](../src/teamwork/experiments/runner.py) | 调度训练并记录种子、预算和耗时 |
| 评分与绘图 | [`evaluation/`](../src/teamwork/evaluation/) | 维护公共指标、跨种子比较和绘图 |
| 后端任务与会话 | [`api/`](../src/teamwork/api/) | 维护 SDK 目录、请求合同、任务、模型和产物 |

策略使用自己的观测，集中 critic 可以在训练时使用环境的 `state()`。`task_snapshot()` 供评估读取，循环策略在每回合调用 `reset()`。

## 配置和产物

配置放在 [`configs/algorithms/`](../configs/algorithms/)，选择场景时同时核对环境参数和奖励约定。训练预算的单位是整个环境的一次联合交互步。

`Algorithm.train(config, out)` 返回 checkpoint 路径。每次运行使用独立目录，记录 `config.json`、`episodes.csv` 和模型。公共 runner 额外记录 `training.json`。回合 CSV 包含 `episode`、`steps` 和各智能体原始回报。Checkpoint 包含配置、实际步数和模型状态，由算法的 `load_policy` 加载。

通过 Python 公共入口训练的示例：

```python
from pathlib import Path

import yaml

from teamwork.experiments.runner import run_training

config = yaml.safe_load(Path("configs/algorithms/mappo/spread.yaml").read_text(encoding="utf-8"))
checkpoint = run_training(config, Path("runs/mappo_spread_s0"))
```

各算法的短程验收和适用边界记录在方法页。方法页依次说明 **方法、训练与评估、验证与边界**。公共参数和指标的定义集中在 [实验协议](EXPERIMENTS.md)。

## 格式与验收

使用锁定版本的 Ruff。格式整理可指定改动的文件，提交前可检查整个仓库。

```bash
python -m ruff format .
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python scripts/smoke.py --skip-tests
```

`scripts/smoke.py` 检查短程训练、评估、比较和绘图。正式性能对照见 [实验协议](EXPERIMENTS.md)。运行产物、缓存和本机解释器目录位于 Git 忽略项中。
