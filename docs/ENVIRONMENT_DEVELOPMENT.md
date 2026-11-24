# 环境开发指南

每个场景提供并行的 `reset`、`step`、观测和动作空间。算法通过 PettingZoo 风格的字典为每个智能体提供动作，一次联合 `step` 计作一个训练交互步。

## 接入或编写环境

1. 在 [环境目录](../src/teamwork/environments/) 实现 `Environment`。适配第三方环境时在构造阶段按需导入依赖。编写新环境时让 `reset(seed=...)` 可以复现初始状态，并明确任务终止和时间截断。参考 [MPE 适配](../src/teamwork/environments/mpe.py) 与 [双臂适配](../src/teamwork/environments/two_arm_lift.py)。
2. 在 [环境注册表](../src/teamwork/environments/registry.py) 登记工厂、动作类型、依赖与评估所需源码。动作可以是 `Discrete` 或有界的向量 `Box`。CLI 自动列出注册场景，训练入口提前检查算法和场景的动作类型。
3. 在 [评估协议实现](../src/teamwork/evaluation/metrics.py) 定义默认参数、指标和每回合计数规则。`task_snapshot()` 提供计分事实，`state()` 供集中 critic 使用，执行策略只接收自己的观测。测试同种子复现、边界动作、终止/截断和短程训练评估。
4. 给可训练算法添加 [场景配置](../configs/algorithms/)，用同一预算与评估种子横向比较。改动正式指标定义时更新协议版本，旧检查点可以用新评估器重新打分。

## MuJoCo 双臂抬升

两个智能体分别控制一台 Panda，默认使用官方 Panda 的末端位姿控制器和夹爪动作。执行观测为自身本体状态、共享锅位姿、分配给自己的把手位置。回合最长 200 个控制步，控制频率 20 Hz。默认将 robosuite 的稠密团队奖励发给两只机械臂，独立的抬升成功判定见 [实验协议](EXPERIMENTS.md)。

连续 IPPO 与连续 MAPPO 共用 PPO 循环和有界高斯 actor，critic 分别读取自身观测与联合状态。机器人场景目前通过 CLI 与公共 Python runner 使用。

### WSL 与 Conda

在 WSL 或 Ubuntu 的项目根目录运行。已有机器人环境直接激活。

```bash
conda create -n deeprl-mujoco --override-channels -c conda-forge python=3.12 pip
conda activate deeprl-mujoco
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[robotics]' -c requirements-robotics.txt
```

默认运行不需要渲染。有 EGL 的机器可用 `MUJOCO_GL=egl` 录制回放。

### 短程验收

```bash
set -e
python -m teamwork.cli metrics --env two_arm_lift
mkdir -p runs
trial=$(mktemp -d runs/two-arm-XXXXXX)
for algo in ippo mappo; do
  python -m teamwork.cli train --config "configs/algorithms/robotics/$algo.yaml" \
    --steps 512 --seed 0 --out "$trial/$algo"
  python -m teamwork.cli evaluate --checkpoint "$trial/$algo/model.pt" \
    --episodes 3 --seed 10000 --out "$trial/$algo/eval.json"
done
python -m teamwork.cli compare "$trial/ippo/eval.json" "$trial/mappo/eval.json" \
  --metric stable_dual_lift --out "$trial/comparison.json"
```

短程预算用于检查更新、评估和比较。正式实验需使用独立训练种子，并保留双臂同时抓握占比、最大抬升高度和回放以解释失败。

```bash
for policy in scripted_lift scripted_single_arm random; do
  python -m teamwork.cli evaluate --env two_arm_lift --policy "$policy" \
    --episodes 3 --seed 10000 --out "$trial/$policy.json"
done
MUJOCO_GL=egl python -m teamwork.cli evaluate --env two_arm_lift --policy scripted_lift \
  --episodes 1 --seed 10000 --out "$trial/demo.json" --gif "$trial/demo.gif"
```

脚本正对照按接近、闭合、同步抬升的顺序执行。单臂对照给另一台机械臂发送零位姿增量及打开夹爪指令。两臂与单臂脚本共享相同初始摆位、控制周期和评分协议。录像为脚本策略回放。

### 训练稳定抬升策略

`mappo_stable.yaml` 为双臂场景增加本臂末端相对初始把手的位姿误差和回合时钟。示范拟合从训练回合的控制命令学习位姿反馈和逐步偏置。MAPPO 随后在相同场景微调有界高斯策略，集中 critic 读取联合状态。测试只加载保存的策略参数，每只机械臂使用自己的观测。

```bash
trial=$(mktemp -d runs/stable-lift-XXXXXX)
python scripts/train_stable_lift.py --out "$trial/training" \
  --demo-seed 31000 --demo-episodes 16 --ppo-steps 2048
python -m teamwork.cli evaluate --checkpoint "$trial/training/ppo/model.pt" \
  --episodes 30 --seed 50000 --out "$trial/ppo_heldout.json"
MUJOCO_GL=egl python -m teamwork.cli evaluate --checkpoint "$trial/training/ppo/model.pt" \
  --episodes 1 --seed 50000 --out "$trial/demo.json" --gif "$trial/demo.gif"
```

运行前先创建 `runs/`。示范数据和两个训练阶段的检查点分别保存在 `training/demonstrations.npz`、`training/behavior_clone/model.pt` 和 `training/ppo/model.pt`。如果已有同配置的示范文件，可用 `--demonstrations` 复用。示范交互步数和在线 PPO 步数分别记录。复评沿用 `stable_dual_lift` 的固定判定。这组结果用于验证学习策略的抬升能力，算法对比时要保持观测、示范预算和在线预算一致。

### 多算法对照

在 `train_stable_lift.py` 中用 `--algorithm` 切换算法。每个算法都从同一份示范拟合局部 actor，再使用自己的在线更新规则。

| 算法组 | CLI 注册名 | 训练差异 |
|---|---|---|
| PPO | `ippo_continuous`、`mappo_continuous` | 局部 value 与集中 value |
| 顺序 PPO | `happo_continuous` | 共享集中 value，按随机顺序更新 actor，已更新策略的概率比重加权后续优势 |
| Transformer critic | `mat_dec_continuous` | 使用仓库的 MAT-Dec 观测 token 编码器，按 agent 输出 value，执行使用局部 actor |
| DDPG | `iddpg_continuous`、`maddpg_continuous` | 确定性 actor、局部或集中单 Q |
| TD3 | `itd3_continuous`、`matd3_continuous` | 局部或集中双 Q、目标动作平滑和延迟 actor 更新 |
| SAC | `isac_continuous`、`masac_continuous`、`hasac_continuous` | 分别使用局部双 Q、集中双 Q、共享集中双 Q 加顺序策略更新 |

SAC 系列使用固定温度和每个 actor 独立的可学习标准差。集中 Q 接收联合状态和动作。MADDPG/MATD3 的 actor 更新使用其他 actor 当前动作的冻结样本。MAT-Dec 连续版继承仓库的去中心化 actor 结构和注意力 critic。实现参考 [HAPPO](https://arxiv.org/abs/2109.11251)、[MAT](https://arxiv.org/abs/2205.14953)、[MADDPG](https://arxiv.org/abs/1706.02275)、[MATD3](https://arxiv.org/abs/1910.01465)、[SAC](https://arxiv.org/abs/1812.05905) 和 [HARL / HASAC](https://arxiv.org/abs/2304.09870)。

例如，复用已有的示范数据来训练 MATD3。

```bash
python scripts/train_stable_lift.py --algorithm matd3_continuous \
  --demonstrations "$trial/training/demonstrations.npz" \
  --online-steps 2048 --seed 0 --out "$trial/matd3"
python -m teamwork.cli evaluate --checkpoint "$trial/matd3/online/model.pt" \
  --episodes 30 --seed 60000 --out "$trial/matd3-eval.json"
```

PPO 方案的在线检查点位于 `ppo/model.pt`，DDPG、TD3 和 SAC 方案位于 `online/model.pt`。比较时固定示范文件、在线环境步数与评估种子，并在报告中记录离线示范步数。示范初始化后的成功率适合验收稳定执行。区分方法效果时需要预先规定多训练种子、从零训练或小样本条件。

### 质量鲁棒性试验

`evaluate --pot-mass-scale` 从已有检查点评估。倍率同时作用于锅的质量和惯量，保持几何、观察和成功判定。标定完成后固定倍率与新的测试种子。用 `--record-trace` 保存每步高度、倾角与两只夹爪的抓握状态。

```bash
python -m teamwork.cli evaluate --checkpoint "$checkpoint" --pot-mass-scale 8 \
  --episodes 8 --seed 62000 --record-trace --out runs/heavy_lift.json
python -m teamwork.cli plot-lift-trace --evaluation runs/heavy_lift.json \
  --seed 62000 --out runs/heavy_lift.png
```

多种子矩阵可通过下列命令复现。先用同一个示范文件运行 `scripts/train_stable_lift.py`，为三种算法分别保存训练种子 `0`、`1000` 和 `2000` 的模型。将训练根目录赋给 `study`。

```bash
python scripts/evaluate_lift_mass.py \
  --checkpoints "$study"/training/{ippo_continuous,mappo_continuous}/seed_*/ppo/model.pt \
    "$study"/training/hasac_continuous/seed_*/online/model.pt \
  --behavior-clone "$study"/training/mappo_continuous/seed_0/behavior_clone/model.pt \
  --mass-scales 1 6 7 8 --episodes 8 --seed 62000 --out "$study/evaluation"
```

输出包含逐模型的轨迹、每个倍率的成功率与峰值高度，以及行为克隆对照。在线模型的误差统计基于独立训练种子。机器人任务源码更新后应从检查点重新评估，旧版 JSON 保留作归档。


### 固定质量与随机质量训练

`hasac_mass_study.yaml` 定义一项配对试验。两组都从同一个名义质量下的 HASAC 行为克隆检查点热启动，共用训练种子、锅初始摆位和在线交互预算。固定组每回合使用 4.75 倍锅质量，随机组每回合在 3.0 至 6.5 倍内均匀抽取。质量抽样使用独立的随机数流，具体取值随检查点存入 `episode_resets.json`。

```bash
clone=runs/example/behavior_clone/model.pt
python scripts/study_lift_mass_training.py --behavior-clone "$clone"   --config configs/algorithms/robotics/hasac_mass_study.yaml   --out runs/hasac_mass_training --workers 6
```

`clone` 可通过 `scripts/train_stable_lift.py --algorithm hasac_continuous --online-steps 0` 生成。固定评估倍率会覆盖检查点里的训练质量分布。比较仍按实际训练设置分组，研究脚本单独核验两组只在质量分布上不同，再计算配对种子差值。网格、示范与在线预算见 `study_plan.json` 和 `summary.json`。


### 抓握反馈与起抬时序

时序试验对已训练的 `lift_pose` 策略改起抬规则，各组复用完全相同的 actor 权重和预抓握动作。控制器在原起抬阶段的入口检查条件，后续抬升动作沿用原策略。

| 起抬规则 | 相位从抓握转入抬升的条件 |
|---|---|
| `clock` | 继续按原时钟推进 |
| `local_grasp` | 本臂的夹爪连续 4 个控制步抓住其指定把手 |
| `mutual_grasp` | 本臂连续抓握达标，并收到对侧同样达标的一个就绪位 |

接触来自仿真的本地夹爪触觉信号。就绪位在两臂间即时传递，策略只用本臂信号和接收到的对侧就绪位。延迟从指定臂第一次请求闭合时算起，只暂缓夹爪闭合，六维位姿动作照常执行。起抬相位和是否提前起抬写进轨迹，和高度、倾角、两臂实际接触一起显示。任务成功仍由统一指标判定。

```bash
python scripts/study_lift_timing.py   --checkpoints "$study"/training/hasac_continuous/seed_*/online/model.pt   --delays 0 15 30 45 --delay-arm arm_1 --seed 66000 --episodes 10   --out runs/lift_timing --workers 6

python -m teamwork.cli evaluate --checkpoint "$checkpoint" --lift-timing mutual_grasp   --gripper-delay-steps 45 --delayed-arm arm_1 --seed 66000 --episodes 1   --record-trace --out runs/lift_timing_trace.json
python -m teamwork.cli plot-lift-trace --evaluation runs/lift_timing_trace.json   --out runs/lift_timing_trace.png
```

`study` 是前面鲁棒性实验的根目录，`checkpoint` 是单个支持 `lift_pose` 的模型。比较图区分稳定抬升成功率和至少一臂提前进入起抬阶段的回合占比。每个延迟档按相同测试摆位比较。

## MuJoCo 双臂交接

激活 [机器人 Conda 环境](#wsl-与-conda)，在仓库根目录运行。

```bash
python scripts/study_handover.py --out runs/handover
```

默认采集 12 条成功示范，做行为克隆，再让脚本和克隆策略分别在相同的 20 个新摆位上评估。每回合 400 步，在线训练 0 步。成功要求先递出臂持物，再共同抓握，最后接收臂独持，锤子连续 10 步高出桌面超过 0.10 米。种子和预算见 [实验配置](../configs/experiments/handover.yaml)，指标见 [实验协议](EXPERIMENTS.md)。

- `summary.json` 记录两种策略的成功数、测试种子和训练预算。
- `handover.png` 展示一个克隆回合的高度与抓握顺序。
- `training/behavior_clone/model.pt` 是可用公共 `evaluate` 命令加载的检查点。

重跑时换一个输出目录。默认实验固定锤子几何与引导阶段，本地参考结果为脚本与克隆各 18/20。原始回合见 `scripted.json` 和 `clone.json`。
