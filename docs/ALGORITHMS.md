# 算法目录与分工

## 已接入的算法

以下算法通过 `python -m teamwork.cli train` 和 `make_algorithm(注册名)` 接入。SDK 指 `AlgorithmAPI`，支持的训练场景由 `catalog()` 返回。场景空白表示这张表暂无对应实验配置。

| 算法 | 注册名 | Spread（目标覆盖） | Tag（追捕对抗） | LBF（合作采集） | RWARE（机器人仓储） | SDK |
|---|---|---|---|---|---|---|
| MAPPO（前馈） | `mappo_ff` | [3 agent](../configs/algorithms/mappo/spread.yaml) |  | [全场视野](../configs/algorithms/mappo/lbf.yaml)、[局部视野](../configs/algorithms/mappo/lbf_partial.yaml) | [2 agent](../configs/algorithms/mappo/rware.yaml) | — |
| QMIX（局部 GRU） | `qmix` | [3 agent](../configs/algorithms/qmix/spread.yaml) |  | [全场视野](../configs/algorithms/qmix/lbf.yaml)、[局部视野](../configs/algorithms/qmix/lbf_partial.yaml) | [2 agent](../configs/algorithms/qmix/rware.yaml) | — |
| IPPO 基线 | `ippo` | [独立配置](../configs/algorithms/ippo.yaml) | [固定队伍](IPPO_IQL.md) | [场景覆盖](IPPO_IQL.md) | [场景覆盖](IPPO_IQL.md) | 支持 |
| 集中 critic PPO 基线 | `mappo` | [共用配置](../configs/algorithms/ppo.yaml) | 固定对手（CLI） |  |  | — |
| DQN-based IQL | `iql` | [独立配置](../configs/algorithms/iql.yaml) | [固定队伍](IPPO_IQL.md) | [场景覆盖](IPPO_IQL.md) | [场景覆盖](IPPO_IQL.md) | 支持 |
| COMA | `coma` | [默认配置](../configs/algorithms/coma.yaml) |  |  |  | — |
| VDN | `vdn` | [默认配置](../configs/algorithms/vdn.yaml) |  | [切换场景](VDN.md) | [切换场景](VDN.md) | 支持 |
| MAT-Dec | `mat_dec` | [默认配置](../configs/algorithms/mat_dec.yaml) |  | [切换场景](MAT_DEC.md) | [切换场景](MAT_DEC.md) | 支持 |
| MAAC | `maac` | [默认配置](../configs/algorithms/maac.yaml) |  |  |  | — |
| QTRAN | `qtran` | [默认配置](../configs/algorithms/qtran.yaml) |  | `--env lbf` | `--env rware` | — |
| QPLEX | `qplex` | [默认配置](../configs/algorithms/qplex.yaml) |  | `--env lbf` | `--env rware` | — |
| HAPPO（异构 actor，顺序更新） | `happo` | [3 agent](../configs/algorithms/happo/spread.yaml) |  | [全场视野](../configs/algorithms/happo/lbf.yaml)、[局部视野](../configs/algorithms/happo/lbf_partial.yaml) | [2 agent](../configs/algorithms/happo/rware.yaml) | 支持 |
| HASAC（离散，软优势分解） | `hasac_discrete` | [3 agent](../configs/algorithms/hasac/spread.yaml) |  | [全场视野](../configs/algorithms/hasac/lbf.yaml)、[局部视野](../configs/algorithms/hasac/lbf_partial.yaml) | [2 agent](../configs/algorithms/hasac/rware.yaml) | 支持 |

`mappo` 是使用 `ppo.yaml` 的集中 critic PPO 基线，通过 `--algorithm mappo` 选择。前馈 MAPPO 的注册名为 `mappo_ff`。Tag 固定对手训练由 `ippo`、`iql`、`mappo` 提供，SDK 开放其中的 IPPO 和 IQL。

### 方法说明

- [IPPO 与 IQL](IPPO_IQL.md)
- [HAPPO 与离散 HASAC](HAPPO_HASAC.md)
- [QTRAN-base 与 QPLEX](QTRAN_QPLEX.md)
- [VDN](VDN.md)与 [MAT-Dec](MAT_DEC.md)
- 其他方法的实现位于 [算法源码](../src/teamwork/algorithms/)，各自的基础配置见上表。

## 候选方法

默认选择离散动作版本。合作类方法优先在 Spread、LBF 验证，RWARE 为扩展场景。

| 算法（论文/年份） | 值得研究的问题 | 预计难度 / 接入重点 |
|---|---|---|
| [DQN-based IQL](https://github.com/uoe-agents/epymarl/blob/main/src/config/algs/iql.yaml)，经典深度独立学习基线 | 各自学习 Q 值，能否自然形成协作？ | 低–中；经验回放、目标网络；勿与离线 RL 的同名 IQL 混淆 |
| [VDN](https://arxiv.org/abs/1706.05296)，2017/2018 | 团队价值相加分解，比独立学习好在哪里？ | 中；共享奖励、联合回放 |
| [QMIX](https://proceedings.mlr.press/v80/rashid18a.html)，2018 | 单调混合网络能否改善信用分配？ | 中；共享奖励、mixer、全局训练状态 |
| [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794)，2018 | 用反事实基线估计个体贡献是否有效？ | 中–高；共享奖励、集中 Q critic |
| [MAAC](https://proceedings.mlr.press/v97/iqbal19a.html)，2019 | 注意力 critic 如何选择有用的队友信息？ | 中–高；注意力、离散策略与回放；可考察个体奖励/对抗 |
| [QTRAN](https://proceedings.mlr.press/v97/son19a.html)，2019 | 放松价值分解限制，是否提高表达能力？ | 高；联合 Q 与约束损失 |
| [QPLEX](https://arxiv.org/abs/2008.01062)，2021 | 更强的 IGM 价值分解能否带来实际收益？ | 高；dueling/mixer、共享奖励 |
| [IPPO](https://arxiv.org/abs/2011.09533)，2020 | 本地 critic 在协作/非平稳环境中有多强？ | 低–中；已有起点，需正式实验 |
| [MAPPO](https://arxiv.org/abs/2103.01955)，2022 | 集中 critic 相比本地 critic 改善多少？ | 中；已有简化起点，完整复现仍需实现细节核对 |
| [HAPPO](https://arxiv.org/abs/2109.11251)，2022 | 顺序更新与同时更新有什么差异？ | 中–高；异构策略、顺序更新与重要性权重 |
| [MAT / MAT-Dec](https://arxiv.org/abs/2205.14953)，2022 | Transformer/序列建模是否有助于协作？ | 高；MAT-Dec 是同一方法族的变体；必须审计执行时信息访问 |
| [HASAC](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html)，2024 | 最大熵与顺序更新能否改善探索、样本效率？ | 高；离散 SAC、回放、温度和顺序更新 |
| [SMPE²](https://proceedings.mlr.press/v267/kontogiannis25a.html)，2025 | 局部状态建模与探索如何改善部分可观测协作？ | 高；belief/探索模块；作者已在 MPE、LBF、RWARE 评估 |

**可选连续动作研究线**：[MADDPG](https://arxiv.org/abs/1706.02275) 与 [MATD3 参考实现](https://github.com/PKU-MARL/HARL)。
需要连续动作适配和独立实验条件，不纳入当前离散动作横向表；不能直接将输出取整后称为原算法。

## 前期分工示例

| 研究线 | 可认领 1–3 个 | 推荐主方法 | 对照问题 |
|---|---|---|---|
| A：策略优化基线 | IPPO、MAPPO | MAPPO（复用 IPPO 对照） | 全局 critic 的价值 |
| B：基础价值分解 | IQL、VDN、QMIX | QMIX，按 IQL→VDN→QMIX 递进 | 独立学习、相加分解、单调分解 |
| C：贡献与注意力 | COMA、MAAC | 二选一，另一个扩展 | 反事实贡献与相关信息选择 |
| D：价值分解表达力 | QTRAN、QPLEX | 二选一，另一个扩展 | 分解约束与表达能力 |
| E：顺序更新与熵 | HAPPO、HASAC | HAPPO，HASAC 为进阶 | 更新顺序、探索与样本效率 |
| F：较新表示方法 | MAT/MAT-Dec、SMPE² | 选一个主方法 | 序列建模或 belief/探索的作用 |

研究线和认领数量以小组分工为准。成员在公共环境上训练和验证所选方法，并提供行为可视化。

每人交付方法说明、实现与配置、原始结果、多种子汇总、行为解释及失败分析；交叉复核另一人的结果。

**组长额外统一实验矩阵、预算和指标版本** 。第 1 周认领并冻结条件，第 2 周跨场景跑通，第 3 周多种子比较，第 4 周完成图表、报告与复现。公共接口和指标如果需要更改，请直接在 **群中说明**， 需要及时告知大家。

训练奖励、执行信息、固定对手与可比性要求见 [实验协议](EXPERIMENTS.md)。

## 参考实现入口

- [EPyMARL](https://github.com/uoe-agents/epymarl)：IQL、VDN、QMIX、QTRAN、COMA、IPPO、MAPPO 的研究参考。
- [MAAC 作者代码](https://github.com/shariqiqbal2810/MAAC)、[QPLEX 作者代码](https://github.com/wjh720/QPLEX)。
- [MAPPO 作者代码](https://github.com/marlbenchmark/on-policy)、[HARL 作者代码](https://github.com/PKU-MARL/HARL)：HAPPO、HASAC 等。
- [MAT 作者代码](https://github.com/PKU-MARL/Multi-Agent-Transformer)、[SMPE² 作者代码](https://github.com/andreaskontogiannis/smpe)。
