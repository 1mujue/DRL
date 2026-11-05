# 算法认领清单

每人认领 1–3 个，先完成 1 个主方法，其余作为扩展。难度是针对本项目的工程估计。
目前代码只有 IPPO 与集中 critic PPO 基线；下面其余方法均待接入，不能直接传给 `--algorithm`。

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

## 六人的认领示例

| 研究线 | 可认领 1–3 个 | 推荐主方法 | 对照问题 |
|---|---|---|---|
| A：策略优化基线 | IPPO、MAPPO | MAPPO（复用 IPPO 对照） | 全局 critic 的价值 |
| B：基础价值分解 | IQL、VDN、QMIX | QMIX，按 IQL→VDN→QMIX 递进 | 独立学习、相加分解、单调分解 |
| C：贡献与注意力 | COMA、MAAC | 二选一，另一个扩展 | 反事实贡献与相关信息选择 |
| D：价值分解表达力 | QTRAN、QPLEX | 二选一，另一个扩展 | 分解约束与表达能力 |
| E：顺序更新与熵 | HAPPO、HASAC | HAPPO，HASAC 为进阶 | 更新顺序、探索与样本效率 |
| F：较新表示方法 | MAT/MAT-Dec、SMPE² | 选一个主方法 | 序列建模或 belief/探索的作用 |

**注意:** 每人至少选择两个算法开展研究，并且具体的算法不需要和上面的表格列举的算法一致，也不需要按照上述研究线的设定开展。

在选择研究的DRL算法后，需要在已有的环境中完成训练、验证工作，并提供可视化的结果说明（组长觉得可视化的内容和展示方式也需要统一，但是现在还没有完成实验，所以暂不考虑），

每人交付方法说明、实现与配置、原始结果、多种子汇总、行为解释及失败分析；交叉复核另一人的结果。

**组长额外统一实验矩阵、预算和指标版本** 。第 1 周认领并冻结条件，第 2 周跨场景跑通，第 3 周多种子比较，第 4 周完成图表、报告与复现。公共接口和指标如果需要更改，请直接在 **群中说明**， 需要及时告知大家。

## 认领前必须确认的三件事

1. **奖励合同。** VDN/QMIX/QTRAN/QPLEX/COMA 等合作方法不能将敌我双方混成一个共享奖励团队。
   当前 Spread 默认含个体碰撞项，LBF/RWARE 也可能返回个体奖励；不能直接视为严格共享奖励。
   全组应先确定合作比较轨：例如统一 `r_team = mean_i(r_i)`，并让该轨所有方法采用同一训练奖励约定。
   这项转换尚未实现，当前比较工具也尚未校验训练奖励变换，实施时需记录进训练元数据并增加检查；本次不改变现有指标 v1。
   [EPyMARL 的奖励支持说明](https://github.com/uoe-agents/epymarl)明确区分共享奖励和一般个体奖励方法。
2. **执行信息。** 当前约定 actor 只用自己的局部观测。MAT 的联合/自回归执行不能默认算作同等信息条件；
   优先审计 MAT-Dec，若仍访问联合观测，应单列执行信息条件，禁止直接和严格分散策略混排。
   Tag 的合作方法需按阵营训练并固定另一方；现有代码仅支持评估时替换对手，尚无通用固定对手训练封装。
3. **参考代码不是现成插件。** 官方实现可能依赖旧 Gym、不同 MPE 或 Python 版本。先移植算法到公共接口，
   再在本项目统一场景中实验。 注意： 我们的项目保证了对不同算法的比较，是在同一个环境完成的，所以 **不需要参考该算法在论文中的表现** 。

## 参考实现入口

- [EPyMARL](https://github.com/uoe-agents/epymarl)：IQL、VDN、QMIX、QTRAN、COMA、IPPO、MAPPO 的研究参考。
- [MAAC 作者代码](https://github.com/shariqiqbal2810/MAAC)、[QPLEX 作者代码](https://github.com/wjh720/QPLEX)。
- [MAPPO 作者代码](https://github.com/marlbenchmark/on-policy)、[HARL 作者代码](https://github.com/PKU-MARL/HARL)：HAPPO、HASAC 等。
- [MAT 作者代码](https://github.com/PKU-MARL/Multi-Agent-Transformer)、[SMPE² 作者代码](https://github.com/andreaskontogiannis/smpe)。
