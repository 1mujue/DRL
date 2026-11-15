# IPPO 与 DQN-based IQL

两种方法均使用每个 agent 的局部向量观测、离散动作和原始个体奖励，各 agent 参数独立。
项目统一场景为 Spread、Tag、LBF、RWARE；指标沿用 [评估协议 v1](EXPERIMENTS.md)。
方法实现与短程运行检查只能证明程序可训练、可评估；算法表现需要正式多种子实验。

## 方法

| 方法 | 学习对象 | 更新方式 | 配置 |
|---|---|---|---|
| IPPO | 独立 actor 与本地 value critic，64×64 Tanh MLP | 同策略 rollout、GAE、PPO clipped objective、value loss 与 entropy bonus | `configs/algorithms/ippo.yaml` |
| IQL | 独立动作价值 Q，64×64 ReLU MLP | 独立经验回放、目标网络、Huber TD loss、ε-greedy | `configs/algorithms/iql.yaml` |

IPPO 在现有实现上接入独立配置与固定对手训练。IPPO 的 critic 输入仅为本地观测。
IPPO 默认 GAE λ=0.95、γ=0.99、clip=0.2、学习率 3e-4，value 系数 0.5、entropy 系数 0.01、梯度裁剪 0.5。

IQL 指 Independent Q-Learning。每个 agent 的回放只保存自身 `(obs, action, reward, next_obs, terminated)`，
用本地奖励分别优化 `Huber(Q(obs, action), reward + γ(1−terminated) max Q_target(next_obs))`。
这采用标准 DQN 目标。默认回放容量 50000、batch=64、学习率 5e-4、γ=0.99、梯度裁剪 10；
从第 1000 个联合环境 step 开始，每 4 step 更新一次，每 200 step 硬同步目标网络。
ε 从 1.0 线性衰减到 0.05，衰减长度 50000 step；评估默认 argmax，非确定性评估使用 ε=0.05。
实现采用前馈网络；部分可观测场景中的记忆能力与参数共享可以在后续研究中单独考察。

两种方法都在真实终止时停止 bootstrap，在时间上限截断时使用 reset 前的最终观测 bootstrap。
预算按联合环境 step 计算；换场景重新训练。环境训练奖励保持原始个体奖励，与指标计算独立。

参考：[IPPO 论文](https://arxiv.org/abs/2011.09533)、[EPyMARL IQL 配置](https://github.com/uoe-agents/epymarl/blob/main/src/config/algs/iql.yaml)。
当前实现按本项目公共接口编写，具体网络、优化器和预算以保存的配置和上述说明为准。

业务后端的统一调用方式与交付验收见 [算法 API 接入说明](API_INTEGRATION.md)。

## 安装与单次运行

使用仓库安装入口 `python3 scripts/setup.py` 创建 Python 3.12 与 `.venv`。下列命令在仓库根目录运行；
Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。

```bash
.venv/bin/python -m teamwork.cli train --config configs/algorithms/ippo.yaml --env spread --seed 0 --out runs/ippo_spread_s0
.venv/bin/python -m teamwork.cli train --config configs/algorithms/iql.yaml --env spread --seed 0 --out runs/iql_spread_s0
.venv/bin/python -m teamwork.cli evaluate --checkpoint runs/iql_spread_s0/model.pt --episodes 100 --seed 10000 --out runs/iql_spread_s0/eval.json
```

`--env` 支持四种场景，并使用所选场景的协议默认参数。`--steps` 可覆盖训练预算。
IQL 的短程检查需至少达到 warmup 且足够组成一个 batch；默认预算小于 1000 时不会更新参数。

Tag 训练选定一个阵营，另一阵营始终使用固定的规则或随机策略：

```bash
.venv/bin/python -m teamwork.cli train --config configs/algorithms/iql.yaml --env tag --team predators --opponent rule --out runs/iql_tag_s0
.venv/bin/python -m teamwork.cli evaluate --checkpoint runs/iql_tag_s0/model.pt --team predators --opponent rule --episodes 100 --out runs/iql_tag_s0/eval.json
```

仅创建、更新和保存所选阵营的网络。固定对手的动作在每步统一合并，双方原始回报都写入 CSV。
IPPO 使用相同参数。配置字段为 `train_team/train_opponent`；省略时沿用全体 agent 独立学习的行为。
固定对手模型的评估必须指定训练时相同的阵营与对手。
公共 `Algorithm/Policy` 方法签名与指标公式保持兼容；`Policy.act` 允许仅返回受控阵营动作，
评估器补齐固定对手动作。新增训练参数与评估字段是与组员对接时需要同步的内容。

## 批量实验

先做单种子短程流程检查：

```bash
.venv/bin/python scripts/run_ippo_iql.py --steps 2048 --seeds 0 --episodes 3 --out runs/ippo_iql_check
```

正式实验默认四场景、两个算法、训练种子 0–4，每个模型 100000 联合 step、100 个测试回合，测试种子从 10000 起：

```bash
.venv/bin/python scripts/run_ippo_iql.py --out runs/ippo_iql_experiment
```

运行前由全组确认预算。默认实验共 40 个模型，合计 4000000 联合 step；耗时取决于硬件和环境。
`--envs spread lbf` 可缩小场景范围。Tag 批量实验固定 `predators + rule`。
输出目录必须不存在，避免覆盖旧实验。实验根目录的 `experiment.json` 保存矩阵；每个模型保存：

- `config.json`、`model.pt`、`training.json`、`provenance.json`：参数、模型、实际步数、耗时与依赖/源码信息。
- `episodes.csv`：完整训练回合的原始个体回报；预算结束时未完成的回合不计作完整回合。
- IQL 的 `updates.csv`：每次更新的 agent、步数、TD loss 和探索率；检查点的 `updates` 记录各 agent 更新次数。
- `eval.json`：固定协议的逐回合评估与汇总；有完整训练回合时生成 `training_curve.png`。
- 场景目录的 `comparison.json/png`：同场景的多训练种子汇总，使用模型均值与训练种子间样本标准差。

检查点支持执行策略加载；重新运行会开始一次新训练。正式报告应包含四场景分别的结果、失败案例、
训练耗时与机器条件；单种子短程检查不提供收敛或优劣结论。

## 验证

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke.py --skip-tests
```

测试覆盖 IQL 的 TD 目标、回放拷贝与淘汰、探索率、参数更新、模型加载、同种子复现、
单步奖励任务中的动作学习与目标网络冻结、
四环境接入，以及 IPPO/IQL 的 Tag 固定阵营训练与评估条件检查。
`scripts/smoke.py` 对 IPPO、集中 critic PPO 和 IQL 各训练 1024 step，验证 CLI 评估、比较和绘图。
