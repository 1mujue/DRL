# MAT-Dec 接入与使用

注册名为 `mat_dec`，配置文件为 `configs/algorithms/mat_dec.yaml`。已接入项目的 `Algorithm.train`、`Algorithm.load_policy` 和 `Policy.act`，可以使用公共 CLI、评估器和绘图器。

这是依据 [MAT 论文](https://arxiv.org/abs/2205.14953) 及作者代码结构编写的项目实现，不是提出新算法，也没有将整个 GitHub 项目复制进来。结构核对参考 [官方仓库](https://github.com/PKU-MARL/Multi-Agent-Transformer/tree/be3ff49c8264d454c1fe2c41582aa2bfc98498c8)，提交号 `be3ff49c8264d454c1fe2c41582aa2bfc98498c8`；具体差异在下文列明。

## MAT-Dec 中 Transformer 在哪里

官方 [`ma_transformer.py`](https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/be3ff49c8264d454c1fe2c41582aa2bfc98498c8/mat/algorithms/mat/algorithm/ma_transformer.py) 的 `Decoder(dec_actor=True)` 分支使用局部 MLP actor；Transformer encoder 用联合观测产生训练所需的价值估计。MAT-Dec 不等同于 MAT 的自回归动作 decoder。

```text
执行：agent_i 自己的局部观测 → MLP actor_i → 自己的动作
训练：各 agent 局部观测 → 观测嵌入 → Transformer encoder → 每个 agent 的价值估计
                    回报、价值、GAE → PPO 更新 actor 与 critic
```

默认每个 agent 独立维护 actor；`share_actor: true` 可以共享 actor 参数，但要求观测和动作空间完全一致。参数共享也不会让 actor 看到其他 agent 的观测。

`MATDecPolicy.act` 只调用对应 agent 的 actor，完全不调用 critic，不读取全局状态、评估快照或其他 agent 已选择的动作。critic 使用联合局部观测，只存在于训练路径，不调用隐藏状态 `env.state()`。

## 运行命令

在项目根目录使用安装好的解释器；安装说明见 [SETUP.md](SETUP.md)。Windows：

```powershell
# 接入验收：512 个联合环境步，默认会产生 32 次 optimizer 更新。
.\.venv\Scripts\python.exe -m teamwork.cli train --config configs/algorithms/mat_dec.yaml --steps 512 --out runs/mat_dec_spread_check
.\.venv\Scripts\python.exe -m teamwork.cli evaluate --checkpoint runs/mat_dec_spread_check/model.pt --episodes 5 --out runs/mat_dec_spread_check/eval.json --gif runs/mat_dec_spread_check/demo.gif
.\.venv\Scripts\python.exe -m teamwork.cli plot --csv runs/mat_dec_spread_check/episodes.csv --out runs/mat_dec_spread_check/training_curve.png

# 正式预算的初始配置：10 万步，效果需要后续实验验证。
.\.venv\Scripts\python.exe -m teamwork.cli train --config configs/algorithms/mat_dec.yaml --seed 0 --out runs/mat_dec_spread_s0

# 换环境重新训练；共享默认设置来自公共指标协议。
.\.venv\Scripts\python.exe -m teamwork.cli train --config configs/algorithms/mat_dec.yaml --env lbf --steps 512 --out runs/mat_dec_lbf_check
.\.venv\Scripts\python.exe -m teamwork.cli train --config configs/algorithms/mat_dec.yaml --env rware --steps 512 --out runs/mat_dec_rware_check
```

Linux 使用 `.venv/bin/python`。输出目录必须尚不存在，以免覆盖实验。LBF/RWARE 需要额外环境依赖；公共适配器尚不支持它们的 GIF 输出。

后端直接调用 Python API：

```python
from pathlib import Path
import yaml

from teamwork.experiments.runner import run_training
from teamwork.evaluation.evaluator import evaluate, save_result

config = yaml.safe_load(Path("configs/algorithms/mat_dec.yaml").read_text(encoding="utf-8"))
config["total_steps"] = 512
checkpoint = run_training(config, "runs/mat_dec_api_check")
result = evaluate(config["env"], checkpoint=checkpoint, episodes=5)
save_result(result, "runs/mat_dec_api_check/eval.json")
```

`make_algorithm("mat_dec").train(config, out)` 可直接返回模型路径；`run_training` 则会额外生成公共 `training.json`。网页后端可以把同步训练调用放进后台任务，供后续 HTTP API 使用。

## 参数与训练流程

默认：嵌入维度 64、1 个注意力头、1 个 encoder block、独立 actor；actor 使用 LayerNorm、两层 GELU 隐藏层和分类动作输出。critic 将 agent 作为 token，使用无因果掩码的多头 self-attention、残差连接、LayerNorm 和逐 token 价值头。dropout 固定为 0，保证采样与 PPO 更新计算的动作概率一致。

相同观测维度的 agent 共享 critic 的观测嵌入；不同维度时使用独立投影到同一嵌入维度，这是项目适配的扩展。actor 也支持不同 agent 的观测维度与动作数量。

训练为 on-policy PPO：

1. 采集最多 `rollout_steps=256` 个联合环境步，并记录动作、旧动作概率和价值。
2. 用 `gamma=0.99`、`gae_lambda=0.95` 计算每个 agent 的 GAE 和价值目标；超时保留最终观测并 bootstrap，每次 reset 都截断 GAE 轨迹。
3. 打乱联合时间步，保持同一时间步的所有 agent 一起进入 critic。默认每 batch 64 个时间步、4 个 epoch。
4. 对每个 agent 的动作概率比使用 `clip=0.2` 的 PPO 裁剪，组合价值损失和熵项；Adam 学习率为 `0.0003`，梯度范数上限为 `0.5`。

同官方结构一致的部分：`dec_actor=True` 的局部 MLP、集中 Transformer encoder、逐 agent PPO 概率比。项目适配的差异：单环境采样而非向量并行；PyTorch 标准多头注意力；可支持异构空间；明确的团队奖励聚合；固定 agent，未接入动作掩码、循环网络、连续动作、ValueNorm 和完整续训。默认超参数是项目起点，没有照搬官方 MPE 的 2000 万步预算或宣称完整论文复现。

## 奖励、环境与公平比较

默认训练奖励为 `team_reward: mean`，即 `r_team = mean_i(r_i)`，给各 agent 的价值学习使用同一团队回报。也可以明确配置 `sum`。配置、模型和 `provenance.json` 都记录奖励口径；环境原始奖励与公共评估指标保持原定义。

目前支持 Spread、LBF、RWARE 合作任务。Tag 包含对抗双方，不能直接取敌我平均奖励；在没有阵营训练封装前，本实现会明确拒绝 Tag。

MAT-Dec 和 VDN 的默认训练奖励都为 `mean`，两者可以作为同一合作研究轨的起点。正式比较仍需保持场景参数、交互预算、测试种子和训练奖励一致；当前公共 `compare` 尚不校验训练奖励转换，不能将不同奖励配置直接混排。与 IPPO 等方法比较也应先统一训练奖励。

## 文件与验证

| 文件 | 内容 |
|---|---|
| `model.pt` | actor 和 critic 参数、完整配置、空间描述、步数、rollout 数和更新次数 |
| `config.json` | 合并默认值后的实际训练配置 |
| `episodes.csv` | 已完成回合的原始个体奖励，供公共训练曲线使用 |
| `updates.csv` | 每个 optimizer 更新的 loss、策略损失、价值损失、熵、近似 KL、裁剪比例和梯度范数 |
| `mat_dec_training.json` | 训练步数、回合数、rollout 数、更新次数和最后一次更新统计 |
| `provenance.json` | 来源、参考提交号、执行/训练信息、奖励定义、源码 SHA-256 和依赖版本 |
| `training.json` | 公共入口生成的种子、预算、设备与耗时 |
| `eval.json` | 公共评估器输出的逐回合指标及汇总 |

检查点可加载执行策略，不含优化器、随机数状态和未完成的 rollout，不能当作完整续训快照。512 步验证用于确认真实参数更新和完整流程，不能证明任务收敛或算法优于其他方法。

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/algorithms/test_mat_dec.py
```

测试检查局部执行时禁止调用 critic、critic 对队友信息的使用、共享与独立 actor、异构空间、采样和更新概率一致性、可手算的 GAE/PPO 目标、actor/critic 实际更新、精确步数预算、固定种子复现、保存加载及合作环境的训练评估。
