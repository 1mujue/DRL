# QTRAN-base 与 QPLEX

两种方法共用 QMIX 的局部 Q 网络、回合回放和目标网络更新机制。各自的 `learner.py` 实现联合价值函数与损失，执行时只加载局部 Q 网络。默认使用带上一动作、agent ID 的共享 GRU 和 Double Q。

- **QTRAN-base** 训练一个集中式联合 Q 和状态偏置。在局部贪心动作上约束等式，在采样动作上惩罚非最优约束的负残差。`updates.csv` 记录 `td_loss`、`opt_loss`、`nopt_loss`。参考 [QTRAN 论文](https://proceedings.mlr.press/v97/son19a.html)。
- **QPLEX** 使用状态条件的正仿射变换和多头正权重，分别混合局部价值与动作优势。局部贪心时优势为零，其他动作的优势非正。参考 [论文第 4 节](https://arxiv.org/html/2008.01062)与[作者实现](https://github.com/wjh720/QPLEX)。

默认合作场景是 Spread。用 `--env lbf` 或 `--env rware` 选择其他合作场景。训练按 `reward_aggregation: mean` 使用个体奖励均值，原始个体回报仍用于日志及公共评估。正式横向比较时核对各方法的奖励聚合方式。Tag 需另行实现阵营隔离与固定对手训练。当前检查点支持推理和公共评估。

## QPLEX 梯度消融

`configs/algorithms/qplex.yaml` 的 `gradient_mode` 默认为 `dueling`。将它设为 `dmaq` 可复现作者 DMAQ 中的梯度路径：价值支路使用已执行动作的局部 Q，缩放后的优势差停止梯度，混合权重仍从优势项接收梯度。两种模式采用代数等价的联合价值公式。项目内的网络、优化器和采样均使用原配置。 [作者的 DMAQ mixer](https://github.com/wjh720/QPLEX/blob/master/pymarl-master/src/modules/mixers/dmaq_general.py)

梯度模式记录在 `config.json`、`training_contract.json` 和 checkpoint 中。做消融时准备两份配置，只修改 `gradient_mode`，固定场景、动作屏蔽、训练种子、评估种子和步数预算。按梯度模式汇总模型，再按训练种子配对比较。

专项用例覆盖 QPLEX 的前向值、合法动作掩码和反向梯度，以及 QTRAN-base 的最优约束、非最优约束、截断 bootstrap 和损失加权。在项目根目录执行 `python -m pytest -q tests/algorithms/test_qtran_qplex.py`。

## LBF 动作屏蔽对照

可在算法 YAML 中加入 `mask_unavailable_actions: true`，然后使用 `--env lbf`。这个实验选项适用于 QTRAN、QPLEX 和 QMIX，默认关闭。启用后，探索、贪心动作、回放中的目标动作及 QPLEX 的局部最大 Q 都依据当前可执行动作。掩码取自 LBF 2.0 提供给每个玩家的可执行动作，仍由各自的局部 Q 选择动作。

公共 `evaluate --checkpoint` 按保存的配置自动使用屏蔽。独立调用 `load_policy` 时需要提供该策略实际交互的 LBF 环境。报告实验时标明是否启用该选项，按同一环境配置、预算和种子做消融对照。

## LBF 改进配置

- **QTRAN 成功回放**使用 [qtran/lbf_positive_replay.yaml](../configs/algorithms/qtran/lbf_positive_replay.yaml)。额外保留最近 64 个有正奖励的任务终止回合，每批最多先抽 2 个，其余从普通回放均匀抽取。一个批次内的同一回合只使用一次。日志记录保留与实际抽中的成功回合数。
- **带示范的 QPLEX**使用 [qplex/lbf_guided.yaml](../configs/algorithms/qplex/lbf_guided.yaml)。在默认两人一食物 LBF 中，每 10 回合使用一次联合规则策略采集，示范步计入训练预算。局部 Q 网络同时接受 TD 更新和成功回合的动作监督。评估使用模型的局部贪心动作和 LBF 动作屏蔽。

在项目根目录运行，对种子 `0、1、2` 使用各自的新目录：

```bash
python -m teamwork.cli train \
  --config configs/algorithms/qtran/lbf_positive_replay.yaml --seed 0 --out runs/qtran_lbf_retained_s0
python -m teamwork.cli train \
  --config configs/algorithms/qplex/lbf_guided.yaml --seed 0 --out runs/qplex_lbf_guided_s0
```

带示范的 QPLEX 每 1 万步保存 checkpoint。用验证种子 `30000–30199` 在预先指定的步数节点选出跨训练种子平均采集成功率最高的一个共同步数，平局取较早节点。再用独立种子 `60000–60199` 评估该节点与同预算的基线。报告注明示范频率和动作监督权重。评估 JSON 会记录训练变体，`compare` 会拒绝混合汇总同一算法的不同变体。

### 已完成的 LBF 对照

三次独立训练使用种子 `0、1、2`。表中的每个数值来自相同的 200 个独立评估种子 `60000–60199`。基线和改进都使用动作屏蔽。

| 算法和评估步数 | 基线成功次数 | 改进后成功次数 |
|---|---:|---:|
| QTRAN，100,000 | 34、5、13 | 144、39、125 |
| QPLEX，30,000 | 0、0、0 | 129、120、94 |

QTRAN 只更改成功回合的保留和抽样。QPLEX 使用带示范和动作监督的变体，按验证种子从 10,000、20,000、30,000、40,000、50,000 步中选择。三种子的验证成功次数之和在 30,000 步为 332/600，50,000 步降到 157/600。单种子诊断中，DMAQ 梯度、扩大均匀回放或单独示范均未使 QPLEX 完成采集。成功回放单项在三个 QPLEX 种子上也未获得成功。

## 一起测试

在 WSL 打开项目。先确认 Python 路径和 CPU 版 PyTorch，然后查看训练帮助里是否列出 `qtran`、`qplex`。

```bash
cd /home/mujue/proj/deepRL/DeepRL/teamwork
conda activate deeprl-teamwork
python -m teamwork.cli doctor
python -m teamwork.cli train --help
```

确认后，在相同的 Spread 场景各训练 900 个联合步。每回合最多 50 步，默认需积累 16 个回合才开始更新，因此 900 步足以检查实际更新。使用新目录，重复运行时更换 `trial`。

```bash
mkdir -p runs
trial=$(mktemp -d runs/q-decomp-smoke-XXXXXX)
for algo in qtran qplex; do
  python -m teamwork.cli train \
    --config "configs/algorithms/$algo.yaml" --steps 900 --out "$trial/$algo" || break
  head -n 3 "$trial/$algo/updates.csv"
done
```

检查两份 `updates.csv` 都有数据行，损失有限，终端显示 `updates` 大于零。然后用新模型走同一个评估入口。

```bash
for algo in qtran qplex; do
  python -m teamwork.cli evaluate \
    --checkpoint "$trial/$algo/model.pt" --episodes 5 --seed 10000 \
    --out "$trial/$algo/eval.json" || break
done
python -m teamwork.cli compare "$trial/qtran/eval.json" "$trial/qplex/eval.json" \
  --metric final_coverage --out "$trial/comparison.json" --plot "$trial/comparison.png"
```

比较成功后可用同样流程验证 LBF，只需替换为新目录、训练时加 `--env lbf`，并将比较指标改为 `food_collected_fraction`。短程测试验收训练更新、模型加载和结果格式。正式结果需要更长预算与多个训练种子。
