# 固定评估协议 v1.0.0

**指标随场景固定，算法不得自行改分数。** 唯一公式在 `src/teamwork/evaluation/metrics.py`，可用 `cli metrics --env <场景>` 查询。
本页为精简说明，指标代码和版本未变。场景默认使用固定 agent 和向量观测，`two_arm_lift` 和 `two_arm_handover` 使用有界连续动作，其余使用离散动作，reset 状态不计分，每次联合 step 后采样。

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

- 同一场景设置、动作/信息条件、指标版本、测试种子和训练预算。换场景需分表，不把不同单位拼成总分。
- Tag 固定另一方：`--team predators --opponent rule`，或对应 prey 设置；双方同时更换只作观察。
  CLI 的 `ippo`、`iql`、`mappo` 支持固定对手训练。SDK 提供其中的 IPPO 和 IQL。固定对手训练的模型要求用同一阵营与对手评估。
  检查点配置记录 `train_team/train_opponent`，评估结果的 `training_setup` 记录训练条件。
- 评估默认 greedy，训练按各算法的采样规则执行。随机基线固定动作空间种子。
- 学习模型的评估 JSON 在 `training_conditions` 中记录生效的奖励聚合、原配置的聚合方式、奖励除数及动作屏蔽，在 `training_setup` 中记录阵营及对手。环境参数和观测空间见 `benchmark.env_kwargs` 与 `benchmark.spaces`。全视野和局部视野请显式写入 LBF 的 `sight`。
- JSON 记录协议、环境参数/空间、相关依赖、测试种子、对手和源码校验值，生成 `benchmark_id`。
  `compare` 检查 benchmark、预算、训练条件、独立训练种子以及 Tag 评估对手。
- 横向比较学习模型时统一奖励聚合、奖励缩放、动作屏蔽及训练阵营。Spread 使用 `local_ratio=0` 时，取单个智能体的奖励、取团队平均或原样使用共享奖励等价。奖励缩放仍参与校验。随机和规则基线只需匹配评估协议。比较屏蔽等条件的效果时，对每个设置分别汇总，再按相同训练种子和评估种子配对。

## 统计与复现

- 正式建议训练种子 0–4、测试种子 10000–10099，先用单种子排错。
- 先求每个模型的测试回合均值，再等权汇总独立训练种子，报告样本标准差；一个模型的 100 回合不是 100 次训练。
- null 不当作 0，报告有效样本数；仅一个训练种子时种子间标准差为 null。完成/接触时间须与成功率同时解释。
- 学习效率需在相同步数节点保存模型、统一评估；`episodes.csv` 只是训练回报诊断曲线。
  墙钟时间记录在 `training.json`，跨机器解释时附硬件条件。
- LBF 超时和 RWARE max_steps 按截断处理，RWARE inactivity 按终止处理。
- 改指标含义需递增版本并重新评估；旧模型可加载，旧评估 JSON 不直接混用。
- 新增环境时公共评估器的源码校验值已更新。需要与新版结果合并时，可直接从旧检查点重新生成评估 JSON。
- 学习模型评估记录 `training_env_kwargs` 和 `training_initialization`。比较器检查训练场景、初始化方法、示范文件 SHA-256 和示范交互步数，通用模型热启动也核对源模型 SHA-256。共享示范初始化的算法可以有各自的热启动权重，示范文件和预算须一致。
- 抬升可用 `--pot-mass-scale` 执行评估专用的质量扰动。不同倍率有各自的评估指纹，保留原成功判定。仅做行为克隆的对照与在线算法分别汇总，以免把 0 在线步和非零预算混为一组。
- 夹爪延迟试验保持抬升指标不变。延迟臂和延迟步数进入评估指纹，起抬策略单独记录。普通比较入口只汇总同一时序规则的结果，配对试验用同一模型与相同测试摆位对照三种规则。
- 旧版 schema 2 评估可在同一旧 benchmark 内汇总，比较结果显示 `training_conditions: null`，表示当时的训练条件无法从结果核验。正式合并旧 checkpoint 与新实验结果时，应使用当前评估器重新生成 JSON。

## 命令示例

```powershell
python -m teamwork.cli compare runs/ippo_spread_s0/eval.json runs/mappo_spread_s0/eval.json --metric final_coverage --out runs/comparison.json --plot runs/comparison.png
```

前提是两份评估已在相同条件下生成；本文成功/捕获定义是项目研究协议。
任务依据：[Spread](https://mpe2.farama.org/environments/simple_spread/)、[Tag](https://mpe2.farama.org/environments/simple_tag/)、
[LBF](https://github.com/uoe-agents/lb-foraging)、[RWARE](https://github.com/uoe-agents/robotic-warehouse)。
