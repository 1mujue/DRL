# HAPPO / HASAC（离散）方法与实验说明

HAPPO 和离散 HASAC 都能通过公共命令训练，也能从 Python 用 `AlgorithmAPI` 调用。每个智能体根据自己的观测从几个动作里选一个。和 MAPPO 或 QMIX 对比时，用同样的场景设置和训练步数，训练奖励取所有智能体的平均值（`reward_aggregation: mean`）。

## 方法

### HAPPO：异构 actor + 顺序更新

- 结构：每个 agent 一套独立 actor（不共享参数）+ 集中 critic；actor 只读自己的局部观测，
  critic 输入为 `[global_state, agent_one_hot]`。
- 与 MAPPO 共用同一套网络、价值归一化和 GAE，差异只在更新顺序：按 `actor_order` 逐个 agent
  更新，每个 agent 的代理目标乘上「已更新 agent 在**当前参数**下的重要性比率」，再对累计乘积裁剪。
- 前序比率在每次参数更新后重算（`happo.trainer.prefix_ratio`）；`updates.csv` 的
  `prefix_ratio_deviation` 若恒为 0，说明顺序更新退化成了同时更新。
- `share_policy` 必须为 `false`；Tag 需要分阵营封装，当前不支持。
- 已知近似：裁剪对象是累计比率乘积而非逐 agent 代理目标；不含论文的 M^{i} 优势分解与信赖域线搜索。
  这些偏差记录在每次运行的 `training_contract.json`。

### HASAC（离散）

- 结构：每个 agent 一个 categorical actor（异构）+ 集中双 soft-Q，critic 输入
  `[central_state, 联合动作 one-hot]`；每个 agent 一个可学温度（目标熵 `target_entropy_scale·log|A|`）。
- 更新：逐 agent 顺序软优势。前序动作取样自**刚更新**的当前策略，自身动作对离散动作空间做精确
  枚举期望，后续 agent 沿用回放中的旧动作；目标网络按 `tau` 软同步，回合回放区分终止与截断。
- 与论文的偏差：论文主线为连续动作（squashed Gaussian + 重参数化采样）。本实现是离散适配，
  不声称复现论文数值结果，偏差逐项写入 `training_contract.json`。
- Tag 需要分阵营训练，当前不支持。

## 训练与评估

### 配置与预算

| 算法 | 入口配置 | 场景配置 |
|---|---|---|
| HAPPO | `configs/algorithms/happo.yaml` | `configs/algorithms/happo/{spread,lbf,lbf_partial,rware}.yaml` |
| HASAC | `configs/algorithms/hasac.yaml` | `configs/algorithms/hasac/{spread,lbf,lbf_partial,rware}.yaml` |

预算单位为联合环境步；`episodes.csv` 与公共评估始终记录环境原始个体回报，
训练奖励变换只在算法内部使用并记录在 `training_contract.json`。

### 调用方式

按照 [安装指南](../guides/SETUP.md) 激活项目解释器，从仓库根目录运行。

直接使用公共 Python API：

```python
from pathlib import Path

import yaml

from teamwork.algorithms import make_algorithm

config = yaml.safe_load(Path("configs/algorithms/happo/spread.yaml").read_text(encoding="utf-8"))
checkpoint = make_algorithm("happo").train(config, Path("runs/happo_spread_s0"))
policy = make_algorithm("happo").load_policy(checkpoint, device="cpu")
```

或通过后端 SDK：

```python
from teamwork.api import AlgorithmAPI

with AlgorithmAPI("runs/backend_algorithms") as api:
    run = api.train({"algorithm": "happo", "env": "spread", "steps": 100000, "seed": 0})
    result = api.evaluate(run["run_id"], {"episodes": 10, "seed": 10000})
```

两者写入同样的产物：`config.json`、`episodes.csv`、`updates.csv`、
`training_contract.json`、`model.pt`（按 `checkpoint_interval` 另存 `checkpoints/`）。

## 验证与边界

```bash
python -m pytest tests/algorithms/test_happo.py tests/algorithms/test_hasac.py -q
python scripts/verify_algorithm_api.py --out runs/api_happo_hasac \
    --algorithms happo hasac_discrete --steps 10000 --episodes 10
```

短程（数千步）只证明流程与接口可用，不构成收敛证据；正式实验的种子、预算与指标版本由全组统一，
参见 [实验说明](../experiments/README.md)。
