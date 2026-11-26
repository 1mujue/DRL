# 固定评估协议 v1.0.0

这个页面回答两个问题：每个场景怎么算分，以及哪些结果可以放在一起比较。

同一场景的所有算法都用同一套 [计分代码](../../src/teamwork/evaluation/metrics/)。想快速查一个指标，可以运行 `python -m teamwork.cli metrics --env spread`。评分从环境执行第一步后开始。双臂场景输出有范围的连续动作，其余场景使用离散动作。

## 场景与指标

| 场景 / 默认条件 | 固定指标及含义 |
|---|---|
| Spread：3 agent、3 目标、50 步，`local_ratio=0.5` | `final_coverage`：最后一步目标覆盖占比，最近 agent 距离严格 <0.1；`success`：最后一步全覆盖；`ever_success`：途中曾全覆盖；`final_mean_landmark_distance`：最终最近距离均值；`collision_pairs_per_step`：无序接触对×步总数/步数；`mean_agent_return`：平均个体未折扣回报 |
| Tag：3 追捕、1 prey、2 障碍、50 步 | `capture_success`：每只 prey 在回合内均曾被接触；`captured_prey_fraction`：曾接触的不同 prey 占比；`first_contact_step`：首次接触步数，未接触为 null；`contact_pairs_per_step`：跨阵营接触对×步/步数；`prey_outside_rate`：任一坐标绝对值 >1 的 prey×步占比；`predator_return/prey_return`：分别的阵营平均回报 |
| LBF：`Foraging-8x8-2p-1f-coop-v3` | `food_collected_fraction`：采集食物格数/reset 后实际食物格数，不按等级加权；`success`：剩余食物为 0；`completion_step`：成功完成步数，失败为 null；`mean_agent_return` |
| RWARE：`rware-tiny-2ag-v2` | `deliveries`：请求货架交付次数，重复请求后再次交付也计数；`deliveries_per_100_steps`：100×交付次数/实际步数；`mean_agent_return`。不定义虚构的成功率 |
| 双臂抬升：两台 Panda、20 Hz、200 步 | `stable_dual_lift`：两个夹爪分别抓住指定把手、锅底高出桌面严格超过 0.10 m、倾角不超过 15°，连续保持至少 10 步；`first_stable_lift_step`：首次完成步数，失败为 null；`dual_grasp_fraction`：双臂同时抓握步数占比；`peak_lift_height_m`：锅底最大相对高度；`mean_agent_return` |
| 双臂交接：两台 Panda、20 Hz、200 步 | `handover_success`：递出臂先独立抓握，随后两臂共同抓握，再由接收臂独立握住手柄并保持任务高度 >0.10 m 连续 10 步；交接中双方同时失去接触后须重新达到递出和共同抓握阶段；`first_handover_step`：首次完成步数，失败为 null；`mutual_grasp_fraction`、`receiver_grasp_fraction`：相应抓握步数占比；`peak_object_height_m`；`mean_agent_return` |

接触均按距离严格小于两半径之和计数；不计自身，持续接触逐步重复计数，并非独立碰撞事件。
Tag 越界阈值是项目几何指标，不等同于奖励边界；多 prey 的捕获成功不要求同时发生。双方回报不能相加排名。
RWARE 2.0 支持 INDIVIDUAL/GLOBAL 交付计数，拒绝不能从奖励区分交付和归还的 TWO_STAGE。

## 比较的必要条件

- 要比较两个学习模型，先核对场景设置、观测范围、动作空间、训练步数、评估种子和评分版本。每个场景按自己的指标汇总。
- 横向比较时，训练奖励的取法、缩放、动作屏蔽和训练阵营也要相同。例如 Spread 的 `local_ratio=0` 时，每个智能体收到同样的奖励，取一个或取平均等价，奖励缩放仍需一致。规则和随机基线只需与学习模型采用相同的评估设置。
- Tag 例子里，`--team predators --opponent rule` 表示让待测策略负责追捕方，让规则策略扮演逃跑方。固定对手训练和评估要使用同一阵营及对手。CLI 支持 `ippo`、`iql`、`centralized_ppo`，SDK 支持前两种。
- 测试时默认关闭训练用的随机探索。随机基线的动作采样也会设置种子。要测动作屏蔽等单项设置的影响，可以分别训练，再按相同训练和评估种子成对比较。
- 评估结果中的 `benchmark_id` 由场景设置、动作和观测空间、指标、评估种子、对手、依赖及相关源码生成。`compare` 用它检查评估设置，同时核对训练预算、训练条件、训练种子和 Tag 对手。查奖励和动作屏蔽用 `training_conditions`，查训练阵营和对手用 `training_setup`。LBF 的全视野或局部视野应在 `sight` 中写明。

## 统计与复现

- 先用一个训练种子检查流程。正式实验建议用训练种子 0–4、评估种子 10000–10099。已有场景基准时，按基准中列出的种子运行。
- 先对每个模型的测试回合取平均，再对各训练种子等权取平均，并报告种子间的样本标准差。用同一个模型测 100 回合时，训练种子仍只有一个。
- null 不当作 0，报告有效样本数；仅一个训练种子时种子间标准差为 null。完成/接触时间须与成功率同时解释。
- 想看哪个算法学得快，可在相同训练步数保存模型，再用相同测试回合比较。训练回报在 `episodes.csv`，实际耗时在 `training.json`。跨机器比较耗时时写明硬件。
- LBF 超时和 RWARE max_steps 按截断处理，RWARE inactivity 按终止处理。
- 改指标含义需递增版本并重新评估；旧模型可加载，旧评估 JSON 不直接混用。
- 源码归档结构调整会更新评估指纹。汇总新旧运行时，可用同一版本的评估器从原检查点重新生成评估 JSON。
- 学习模型评估记录 `training_env_kwargs` 和 `training_initialization`。比较器检查训练场景、初始化方法、示范文件 SHA-256 和示范交互步数，通用模型热启动也核对源模型 SHA-256。共享示范初始化的算法可以有各自的热启动权重，示范文件和预算须一致。
- 抬升可用 `--pot-mass-scale` 执行评估专用的质量扰动。不同倍率有各自的评估指纹，保留原成功判定。仅做行为克隆的对照与在线算法分别汇总，以免把 0 在线步和非零预算混为一组。
- 夹爪延迟试验保持抬升指标不变。延迟臂和延迟步数进入评估指纹，起抬策略单独记录。普通比较入口只汇总同一时序规则的结果，配对试验用同一模型与相同测试摆位对照三种规则。
- 旧版 schema 2 评估可在同一旧 benchmark 内汇总，比较结果显示 `training_conditions: null`，表示当时的训练条件无法从结果核验。正式合并旧 checkpoint 与新实验结果时，应使用当前评估器重新生成 JSON。

## Spread 统一基准

[共享配置](../../configs/experiments/spread.yaml) 固定训练预算、种子、评估回合和场景参数。从仓库根目录运行：

```bash
python scripts/run_spread_benchmark.py --out runs/spread-study-1 --workers 4
```

默认训练并评估 QTRAN、QPLEX、QMIX 和前馈 MAPPO，使用相同的团队平均奖励。可用 `--algorithms qtran qplex` 只运行其中两种。每个算法的各训练种子各生成一份检查点和评估 JSON；输出目录还包含随机与规则基线及覆盖率、平均目标距离、全覆盖成功率的对比图。

每个模型固定训练 **500,000 个联合交互步**，相当于 10,000 个 50 步回合。各算法使用训练种子 0、1、2，最终评估统一用 50000–50099，避开长程训练中可能用到的回合种子。训练预算是实验控制条件，评分仍用本页的固定指标。

### 预算标定

在同一训练轨迹的 10、20、30、40、50 万步读取检查点，分别用评估种子 41000–41049 测量。每格先按回合取均值，再按三个训练种子等权汇总。下表记录选定预算时的一组校准结果。

| 联合交互步 | QPLEX 平均最终距离 ↓ | QPLEX 最终覆盖率 ↑ | 前馈 MAPPO 平均最终距离 ↓ | 前馈 MAPPO 最终覆盖率 ↑ |
|---:|---:|---:|---:|---:|
| 100,000 | 0.568 | 3.1% | 0.443 | 4.2% |
| 200,000 | 0.481 | 3.6% | 0.413 | 5.8% |
| 300,000 | 0.622 | 1.8% | 0.336 | 6.7% |
| 400,000 | 0.455 | 3.8% | 0.271 | 14.0% |
| 500,000 | 0.447 | 4.2% | 0.215 | 21.3% |

QPLEX 在 40–50 万步间的平均改善较小，前馈 MAPPO 同期仍有明显提升，因此选 50 万步作为当前共同预算。

QPLEX 训练种子 0 延长到 100 万步。校准回合中，距离从 0.452 降到 0.408，覆盖率从 4.67% 升到 8.67%。同一对检查点在留出回合中的距离从 0.492 升到 0.500，覆盖率从 4.33% 降到 2.00%。这次延长训练的收益未在留出回合复现，判断范围为一个训练种子。前馈 MAPPO 在 50 万步时仍在改善。

### 50 万步留出评估

每个学习算法有 3 个独立训练种子，每个模型评估 100 回合。规则和随机基线各跑相同的 100 回合。

| 策略 | 最终覆盖率 ↑ | 平均最终距离 ↓ | 最终全覆盖回合 |
|---|---:|---:|---:|
| QTRAN | 2.67% | 0.520 | 0/300 |
| QPLEX | 3.67% | 0.486 | 0/300 |
| QMIX | 1.00% | 0.753 | 0/300 |
| 前馈 MAPPO | 18.67% | 0.234 | 1/300 |
| 规则基线 | 7.33% | 0.326 | 0/100 |
| 随机基线 | 2.00% | 0.911 | 0/100 |

## 命令示例

```powershell
python -m teamwork.cli compare runs/ippo_spread_s0/eval.json runs/centralized_ppo_spread_s0/eval.json --metric final_coverage --out runs/comparison.json --plot runs/comparison.png
```

前提是两份评估已在相同条件下生成；本文成功/捕获定义是项目研究协议。
任务依据：[Spread](https://mpe2.farama.org/environments/simple_spread/)、[Tag](https://mpe2.farama.org/environments/simple_tag/)、
[LBF](https://github.com/uoe-agents/lb-foraging)、[RWARE](https://github.com/uoe-agents/robotic-warehouse)。
