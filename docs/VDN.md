# VDN 接入与使用

本模块根据 [VDN 论文](https://arxiv.org/abs/1706.05296) 的价值相加分解原理在项目内编写，使用 PyTorch 实现训练；没有下载或复制其他 MARL 项目的训练框架。它是经典算法的项目实现，不是提出了一个新算法，也不宣称复现论文全部细节。

## 方法

每个 agent 有独立的 `观测维度 → hidden_size → hidden_size → 动作数` utility 网络，隐藏层使用 ReLU。默认 `hidden_size=64`，不共享参数，允许不同 agent 的观测维度和动作数量不同。

联合 Q 值是各 agent 所选动作 Q 值的和：

```text
Q_tot = sum_i Q_i(o_i, a_i)
y = r_team + gamma * (1 - terminated) * sum_i max_a Q_i_target(o_i_next, a)
loss = SmoothL1(Q_tot, y)
```

整个联合 transition 存入同一回放池，batch 中各 agent 的数据始终来自同一时间步。联合 loss 一次反向传播到所有 utility 网络，使用 Adam 与范数 10 的梯度裁剪；目标网络按联合环境步数周期同步。纯超时截断保留最后观测并估计后续价值，真正任务终止则不再 bootstrap。

训练使用线性退火的 epsilon-greedy 探索；默认从 1.0 在 5 万步降到 0.05。评估默认各 agent 独立取局部 Q 的最大值；`Policy.act(..., deterministic=False)` 使用固定 0.05 的 epsilon-greedy。策略不读取其他 agent 的观测、`env.state()` 或 `task_snapshot()`。这是一版前馈 VDN，没有循环网络、Double DQN 或完整续训支持。

## 训练与评估

注册名为 `vdn`，配置为 `configs/algorithms/vdn.yaml`。沿用已有 `Algorithm.train`、`Algorithm.load_policy` 和 `Policy.act` 接口，CLI、公共评估器和绘图器可以直接调用。

按照 [安装指南](SETUP.md) 激活项目解释器，从仓库根目录运行。

```bash
# 短程检查：512 步足够越过默认的 256 步预热，并产生参数更新。
python -m teamwork.cli train --config configs/algorithms/vdn.yaml --steps 512 --out runs/vdn_spread_check
python -m teamwork.cli evaluate --checkpoint runs/vdn_spread_check/model.pt --episodes 5 --out runs/vdn_spread_check/eval.json --gif runs/vdn_spread_check/demo.gif
python -m teamwork.cli plot --csv runs/vdn_spread_check/episodes.csv --out runs/vdn_spread_check/training_curve.png

# 正式预算起点：10 万联合环境步；是否足够收敛需实际实验判断。
python -m teamwork.cli train --config configs/algorithms/vdn.yaml --seed 0 --out runs/vdn_spread_s0

# 更换合作场景：在新环境中重新训练。
python -m teamwork.cli train --config configs/algorithms/vdn.yaml --env lbf --steps 512 --out runs/vdn_lbf_check
python -m teamwork.cli train --config configs/algorithms/vdn.yaml --env rware --steps 512 --out runs/vdn_rware_check
```

输出目录必须不存在，以免覆盖实验；重复运行时换一个目录名。少于预热步数或不足一个 batch 的运行可以生成未更新的模型，检查 `updates` 是否大于 0。

后端也可以直接调用 Python API：

```python
from pathlib import Path
import yaml

from teamwork.experiments.runner import run_training
from teamwork.evaluation.evaluator import evaluate, save_result

config = yaml.safe_load(Path("configs/algorithms/vdn.yaml").read_text(encoding="utf-8"))
config["total_steps"] = 512
checkpoint = run_training(config, "runs/vdn_api_check")
result = evaluate(config["env"], checkpoint=checkpoint, episodes=5)
save_result(result, "runs/vdn_api_check/eval.json")
```

`run_training` 会通过注册表调用 `VDNAlgorithm.train`，并额外生成公共 `training.json`。直接调用 `make_algorithm("vdn").train(config, out)` 也能获得模型路径。

新版仓库的统一 SDK 也可管理训练任务、模型和评估结果：

```python
from teamwork.api import AlgorithmAPI, TrainingRequest

with AlgorithmAPI("runs/vdn_sdk") as api:
    record = api.train(TrainingRequest("vdn", steps=512), timeout=120)
    evaluation = api.evaluate(record["run_id"], {"episodes": 5})
```

`api.submit_training(...)` 提交后台任务后，可用 `api.status(run_id)` 查询进度；VDN 按 `log_frequency` 和最终步数输出进度。SDK 的 `parameters` 接受目录中列出的数值超参数；`team_reward` 等字符串选项使用算法默认值，需要调整时使用上面的完整配置调用。

完整配置的种子范围为 `0` 到 `2**32 - 1`，设备支持 `cpu` 或本机可用的 `cuda`／`cuda:N`，数值超参数不接受布尔值或数字字符串。`Policy.act` 会拒绝非有限值和维度不匹配的局部观测。

## 验证与边界

### 团队奖励与比较边界

`team_reward: mean` 表示训练用 `r_team = mean_i(r_i)`；也可显式设为 `sum`。这项转换只在 VDN 模块中进行，保存到配置、模型和来源记录里，不修改环境自身奖励，也不修改公共评估指标。

Spread 默认奖励包含个体碰撞项；LBF/RWARE 也可能使用个体奖励，不能直接假定它们天然共享奖励。VDN 在这里通过明确的聚合构造合作目标。正式和其他算法比较前，必须先由全组统一训练奖励约定；公共 `compare` 会检查当前评估结果中的奖励聚合与缩放。

支持现有 Spread、LBF、RWARE 的固定 agent、向量观测、离散动作模式。Tag 含有敌我双方，当前 VDN 会拒绝，不能把双方合成一个共享奖励团队。自定义合作环境需提供每个 agent 的最终观测，并让所有 agent 同时结束回合。当前 LBF/RWARE 公共适配器尚不支持生成 GIF，行为动画请使用 Spread。

### 输出与检查

| 文件 | 内容 |
|---|---|
| `model.pt` | 每个 agent 的网络权重、完整配置、实际步数、更新次数、平均 loss |
| `config.json` | 合并默认值后的实际配置及环境参数 |
| `episodes.csv` | 已完成回合的原始个体回报，与公共绘图兼容 |
| `updates.csv` | 每次参数更新的步数、更新编号、loss、epsilon |
| `vdn_training.json` | 实际步数、回合数、更新次数与 loss 汇总 |
| `provenance.json` | 算法来源、奖励公式、设备、源码 SHA-256、相关包版本 |
| `training.json` | 公共入口记录的预算、种子与训练耗时 |
| `eval.json` | 调用公共评估器后生成的逐回合指标与统计 |

检查点用于加载执行策略；不含优化器、回放池和随机状态，不能当作完整续训快照。短程检查只证明调用、更新、保存、加载、评估和可视化流程可用，不代表任务已收敛。

```bash
python -m pytest -q tests/algorithms/test_vdn.py
```

测试覆盖可手算的联合 TD 目标、终止与 bootstrap、简单可加合作收益的学习、参数实际更新、固定种子的复现、局部执行、配置拒绝，以及三个合作环境的训练和评估。未安装额外环境依赖时，LBF/RWARE 测试会跳过。
