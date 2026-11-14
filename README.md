# 多智能体强化学习算法对比

研究：**同场景比较算法，同算法跨场景验证。** 每人认领至少2个DRL算法，共用环境和固定指标。
目前已注册 IPPO、DQN-based IQL、集中 critic PPO 基线、前馈 MAPPO、QMIX、COMA 和 MAAC；调用名与场景配置见下表。候选方法的研究方向见 [算法清单与认领](docs/ALGORITHMS.md)，实际接入情况以本表和注册表为准。

IPPO/IQL 的方法说明、四场景批量实验和 Tag 固定对手训练见 [IPPO 与 IQL](docs/IPPO_IQL.md)。
供业务后端调用的训练任务、模型推理和结果查询 API 见 [算法接入交付](docs/API_INTEGRATION.md)。

## 算法与场景配置

场景栏填写已有的配置入口；**空白表示尚未提供场景设计或配置，留给其他成员补充，不代表算法在理论上不适用。** 未接入的候选方法仅预留算法名，注册名、场景和调用栏保持空白，接入后再填写。

| 算法 | 注册名 | Spread：目标覆盖 | Tag：追捕对抗 | LBF：合作采集 | RWARE：机器人仓储 | Python 调用入口 |
|---|---|---|---|---|---|---|
| MAPPO（前馈） | `mappo_ff` | [3 agent](configs/algorithms/mappo/spread.yaml) | | [全场视野](configs/algorithms/mappo/lbf.yaml)、[局部视野](configs/algorithms/mappo/lbf_partial.yaml) | [2 agent](configs/algorithms/mappo/rware.yaml) | `make_algorithm("mappo_ff")` |
| QMIX（局部 GRU） | `qmix` | [3 agent](configs/algorithms/qmix/spread.yaml) | | [全场视野](configs/algorithms/qmix/lbf.yaml)、[局部视野](configs/algorithms/qmix/lbf_partial.yaml) | [2 agent](configs/algorithms/qmix/rware.yaml) | `make_algorithm("qmix")` |
| IPPO 基线 | `ippo` | [独立配置](configs/algorithms/ippo.yaml) | [固定队伍](docs/IPPO_IQL.md) | [场景覆盖](docs/IPPO_IQL.md) | [场景覆盖](docs/IPPO_IQL.md) | `make_algorithm("ippo")` |
| 集中 critic PPO 基线 | `mappo` | [共用配置](configs/algorithms/ppo.yaml) | | | | `make_algorithm("mappo")` |
| DQN-based IQL | `iql` | [独立配置](configs/algorithms/iql.yaml) | [固定队伍](docs/IPPO_IQL.md) | [场景覆盖](docs/IPPO_IQL.md) | [场景覆盖](docs/IPPO_IQL.md) | `make_algorithm("iql")` |
| VDN | | | | | | |
| COMA | `coma` | [默认配置](configs/algorithms/coma.yaml) | | | | `make_algorithm("coma")` |
| MAAC | `maac` | [默认配置](configs/algorithms/maac.yaml) | | | | `make_algorithm("maac")` |
| QTRAN | | | | | | |
| QPLEX | | | | | | |
| HAPPO | | | | | | |
| HASAC | | | | | | |
| MAT / MAT-Dec | | | | | | |
| SMPE² | | | | | | |

`mappo` 是旧集中 critic PPO 基线，独立 MAPPO 使用 `mappo_ff`。运行旧基线 `mappo` 时，需把共用 `ppo.yaml` 读入配置的 `algorithm` 改为 `mappo`，或通过 CLI 指定 `--algorithm mappo`。

MAPPO/QMIX 的成对场景配置固定相同环境条件、训练预算和 `reward_aggregation: mean`；训练日志和公共评估仍记录原始个体回报。局部视野 LBF 的 `sight=2`，全场视野为 `sight=8`。当前 MAPPO 没有循环记忆；上述配置已做短程流程验证，尚不表示已收敛。Tag 暂无这两个方法的固定队伍训练配置，场景栏留空。

### Python API 调用

以 MAPPO 的 Spread 配置为例；调用 QMIX 时，将配置路径换成 `configs/algorithms/qmix/spread.yaml`，其余代码相同。其他场景使用表中对应配置；以下代码从项目根目录运行。

```python
from pathlib import Path

import yaml

from teamwork.algorithms import make_algorithm
from teamwork.environments.registry import make_env

config_path = Path("configs/algorithms/mappo/spread.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
algorithm = make_algorithm(config["algorithm"])
out = Path("runs") / config["algorithm"] / "spread_s0"
checkpoint = algorithm.train(config, out)

env = make_env(config["env"], **config["env_kwargs"])
try:
    observations, _ = env.reset(seed=10000)
    policy = algorithm.load_policy(checkpoint, env, observations)
    policy.reset()  # 每回合清空循环策略的隐藏状态。
    while env.agents:
        actions = policy.act(observations, deterministic=True)
        observations, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

`out` 必须是尚不存在的实验目录；改场景时同步更换输出目录名。排错可先减小 `config["total_steps"]`，预算单位为联合环境步。需要统一耗时报告 `training.json` 时，用 `teamwork.experiments.runner.run_training(config, out)` 替换上述 `algorithm.train(config, out)`。

所有已注册方法均通过 `Algorithm.train`、`Algorithm.load_policy` 和 `Policy.act` 调用。MAPPO/QMIX 还支持 `algorithm.load_policy(checkpoint, device="cpu")`，无需创建仿真环境即可加载执行策略；原 PPO 基线加载仍需传入 `env` 和 `observations`。场景指标统计请使用下方公共 `evaluate` 入口。

## 目录结构

```text
src/teamwork/
├── algorithms/                 ← 主要工作区：各自的算法
│   ├── base.py                 # Algorithm / Policy 公共合同
│   ├── registry.py             # 新算法在这里注册
│   ├── baselines.py            # 随机与规则对照
│   └── ppo/                    # 现有 IPPO / 集中 critic PPO
│       ├── algorithm.py        # 接口、策略动作与模型加载
│       └── trainer.py          # 网络、采样、损失与参数更新
├── environments/               ← 接入或修改仿真环境
│   ├── base.py                 # Environment 合同与空间描述
│   ├── registry.py             # 环境工厂注册
│   ├── mpe.py                  # MPE2 Spread / Tag
│   └── gym_tuple.py            # LBF / RWARE 共同适配
├── evaluation/                 ← 全组共同维护，算法不自行计分
│   ├── metrics.py              # 每个场景的固定指标和默认条件
│   ├── evaluator.py            # 运行模型，输出逐回合指标
│   ├── compare.py              # 可比性检查与跨训练种子汇总
│   └── plots.py                # 训练曲线与算法对比图
├── experiments/runner.py        ← 实验调度、种子、预算、耗时
└── cli.py                      ← 命令行参数，不放算法实现

configs/algorithms/ppo.yaml      # 当前 PPO 家族的共用参数起点
configs/algorithms/ippo.yaml     # IPPO 的独立配置
configs/algorithms/iql.yaml      # IQL 的独立配置
tests/                          # 按 algorithms / environments / evaluation 对应放测试
scripts/setup.py                # Windows / Ubuntu 共用安装入口
scripts/smoke.py                # 测试、训练、评估、图表的一键验收
docs/                           # 算法清单、固定指标、安装说明
runs/                           # 自动生成的模型和结果，不提交 Git
```

![研究流程](docs/assets/architecture.svg)

## 开发新算法配置

1. `src/teamwork/algorithms/<方法名>/`：实现 `Algorithm` 和 `Policy`，训练循环由方法自己决定。
2. `algorithms/registry.py`：调用 `register_algorithm("方法名", YourAlgorithm)`。
3. `configs/algorithms/<方法名>.yaml`：写该方法的配置，不沿用其他算法的超参数。
4. `tests/algorithms/test_<方法名>.py`：验证更新、保存加载和适用场景。

改环境写 `environments/`；定义指标写 `evaluation/metrics.py`；改绘图写 `evaluation/plots.py`。
当前只支持固定 agent、向量观测和离散动作；不要为尚未实现的方法建立空目录。

## 运行示例（以Windows为例）

> 在Linux,macOS上的环境配置问题,可以参考 [SETUP](docs/SETUP.md)

```powershell
.\.venv\Scripts\python.exe -m teamwork.cli train --config configs/algorithms/ppo.yaml --algorithm ippo --env spread --steps 2048 --out runs/ippo_spread_s0
.\.venv\Scripts\python.exe -m teamwork.cli evaluate --checkpoint runs/ippo_spread_s0/model.pt --episodes 10 --out runs/ippo_spread_s0/eval.json --gif runs/ippo_spread_s0/demo.gif
```

切换场景用 `--env`，切换已注册方法用 `--algorithm`。Tag 比较需固定另一方；详见 [固定指标与比较规则](docs/EXPERIMENTS.md)。
备用环境资料在 [references](docs/references/ENVIRONMENTS.md)。

`.github/` 保留 CI 与协作模板，`.vscode/` 只是编辑辅助；环境、缓存、`*.egg-info` 等自动目录无需阅读或提交。

<details>
<summary>接入时需要遵守的接口与输出格式</summary>

- 环境返回按 agent 命名的字典，保留 `terminated/truncated`；训练 critic 可用 `state()`，actor 只用自己的观测。
- `task_snapshot()` 仅供评估，不作为执行策略输入；循环策略在每回合 `Policy.reset()` 清空隐藏状态。
- `Algorithm.train(config, out)` 返回模型路径，预算按联合环境 step，不能覆盖已有实验。
- 保存 `config.json`、`episodes.csv`、模型；公共入口另写 `training.json`。CSV 含 episode、steps 与各 agent 回报。
- 检查点用 `torch.save({"config": config, "steps": 实际步数, ...自有状态...}, path)`，内容为 tensors/基础类型。
- 算法自行实现 `load_policy`；现有 PPO 检查点支持评估，不含完整续训状态。跨场景通常需要重新训练。

</details>
