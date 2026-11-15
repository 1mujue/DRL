# QTRAN-base 与 QPLEX

两种方法共用 QMIX 的局部 Q 网络、回合回放和目标网络更新机制。各自的 `learner.py` 实现联合价值函数与损失，执行时只加载局部 Q 网络。默认使用带上一动作、agent ID 的共享 GRU 和 Double Q。

- **QTRAN-base** 训练一个集中式联合 Q 和状态偏置。在局部贪心动作上约束等式，在采样动作上惩罚非最优约束的负残差。`updates.csv` 记录 `td_loss`、`opt_loss`、`nopt_loss`。参考 [QTRAN 论文](https://proceedings.mlr.press/v97/son19a.html)。
- **QPLEX** 使用状态条件的正仿射变换和多头正权重，分别混合局部价值与动作优势。局部贪心时优势为零，其他动作的优势非正。参考 [论文第 4 节](https://arxiv.org/html/2008.01062)与[作者实现](https://github.com/wjh720/QPLEX)。

默认合作场景是 Spread。用 `--env lbf` 或 `--env rware` 选择其他合作场景。训练按 `reward_aggregation: mean` 使用个体奖励均值，原始个体回报仍用于日志及公共评估。正式横向比较时核对各方法的奖励聚合方式。Tag 需另行实现阵营隔离与固定对手训练。当前检查点支持推理和公共评估。

## LBF 动作屏蔽对照

可在算法 YAML 中加入 `mask_unavailable_actions: true`，然后使用 `--env lbf`。这个实验选项适用于 QTRAN、QPLEX 和 QMIX，默认关闭。启用后，探索、贪心动作、回放中的目标动作及 QPLEX 的局部最大 Q 都依据当前可执行动作。掩码取自 LBF 2.0 提供给每个玩家的可执行动作，仍由各自的局部 Q 选择动作。

公共 `evaluate --checkpoint` 按保存的配置自动使用屏蔽。独立调用 `load_policy` 时需要提供该策略实际交互的 LBF 环境。报告实验时标明是否启用该选项，按同一环境配置、预算和种子做消融对照。

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
