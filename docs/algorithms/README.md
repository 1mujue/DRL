# 算法目录与分工

## 分工内容

| 人员姓名 | 领取的算法 | CLI 入口 |
|---|---|---|
| 赵龙淳 | QTRAN、QPLEX | `qtran`、`qplex` |
| 杜敏 | COMA、MAAC | `coma`、`maac` |
| 印煜天 | HAPPO、HASAC | `happo`、`hasac_discrete` |
| 夏宝啟 | MAPPO、QMIX | `mappo_ff`、`qmix` |
| 睢德昊 | IPPO、DQN-based IQL | `ippo`、`iql` |
| 罗雪洋 | VDN、MAT-Dec | `vdn`、`mat_dec` |

正式分工使用离散动作入口。MAPPO 的入口为 `mappo_ff`，HASAC 的入口为 `hasac_discrete`。集中 critic PPO 基线使用独立入口 `centralized_ppo`。

每人负责自己领取的算法，在公共场景中训练和评估。交付代码、配置、原始数据和多种子汇总，配上回放或图表说明策略表现与失败原因。每人再复核一位组员的结果。

**组长统一实验场景、训练步数和评分版本。** 要改公共接口或指标时，请先在群里说明。

训练奖励、执行信息、固定对手与可比性要求见 [实验协议](../experiments/README.md)。

## 训练入口与场景配置

要跑一个算法，先选场景，再点表里的链接拿到 YAML，交给 `python -m teamwork.cli train --config`。格子里的「3 个智能体」表示配置中的人数。「独立配置」表示该算法有单独的 YAML。空格表示本表暂时没有该组合的配置。

SDK 是供其他 Python 程序调用的接口。支持时，可通过 `AlgorithmAPI` 提交训练，用 `catalog()` 查看可选场景。需要在终端运行时，直接用 CLI。

| 算法 | 注册名 | Spread（目标覆盖） | Tag（追捕对抗） | LBF（合作采集） | RWARE（机器人仓储） | SDK |
|---|---|---|---|---|---|---|
| MAPPO（前馈） | `mappo_ff` | [3 个智能体](../../configs/algorithms/mappo/spread.yaml) |  | [全场视野](../../configs/algorithms/mappo/lbf.yaml)、[局部视野](../../configs/algorithms/mappo/lbf_partial.yaml) | [2 个智能体](../../configs/algorithms/mappo/rware.yaml) | — |
| QMIX（局部 GRU） | `qmix` | [3 个智能体](../../configs/algorithms/qmix/spread.yaml) |  | [全场视野](../../configs/algorithms/qmix/lbf.yaml)、[局部视野](../../configs/algorithms/qmix/lbf_partial.yaml) | [2 个智能体](../../configs/algorithms/qmix/rware.yaml) | — |
| IPPO 基线 | `ippo` | [独立配置](../../configs/algorithms/ippo.yaml) | [固定对手](IPPO_IQL.md) | [运行示例](IPPO_IQL.md) | [运行示例](IPPO_IQL.md) | 支持 |
| 集中 critic PPO 基线 | `centralized_ppo` | [基线配置](../../configs/algorithms/centralized_ppo.yaml) | 固定对手（CLI） |  |  | — |
| DQN-based IQL | `iql` | [独立配置](../../configs/algorithms/iql.yaml) | [固定对手](IPPO_IQL.md) | [运行示例](IPPO_IQL.md) | [运行示例](IPPO_IQL.md) | 支持 |
| COMA | `coma` | [默认配置](../../configs/algorithms/coma.yaml) |  |  |  | — |
| VDN | `vdn` | [默认配置](../../configs/algorithms/vdn.yaml) |  | [切换场景](VDN.md) | [切换场景](VDN.md) | 支持 |
| MAT-Dec | `mat_dec` | [默认配置](../../configs/algorithms/mat_dec.yaml) |  | [切换场景](MAT_DEC.md) | [切换场景](MAT_DEC.md) | 支持 |
| MAAC | `maac` | [默认配置](../../configs/algorithms/maac.yaml) |  |  |  | — |
| QTRAN | `qtran` | [默认配置](../../configs/algorithms/qtran.yaml) |  | `--env lbf` | `--env rware` | — |
| QPLEX | `qplex` | [默认配置](../../configs/algorithms/qplex.yaml) |  | `--env lbf` | `--env rware` | — |
| HAPPO（异构 actor，顺序更新） | `happo` | [3 个智能体](../../configs/algorithms/happo/spread.yaml) |  | [全场视野](../../configs/algorithms/happo/lbf.yaml)、[局部视野](../../configs/algorithms/happo/lbf_partial.yaml) | [2 个智能体](../../configs/algorithms/happo/rware.yaml) | 支持 |
| HASAC（离散，软优势分解） | `hasac_discrete` | [3 个智能体](../../configs/algorithms/hasac/spread.yaml) |  | [全场视野](../../configs/algorithms/hasac/lbf.yaml)、[局部视野](../../configs/algorithms/hasac/lbf_partial.yaml) | [2 个智能体](../../configs/algorithms/hasac/rware.yaml) | 支持 |

Tag 固定对手训练由 `ippo`、`iql`、`centralized_ppo` 提供，SDK 开放其中的 IPPO 和 IQL。

### 集中 critic PPO 基线

每个智能体都有自己的 actor 和 critic。Actor 根据自己看到的信息选动作，critic 在训练时估计接下来能拿到多少奖励。集中 critic PPO 的 critic 能看到全局状态，IPPO 的 critic 只看该智能体自己的观测。用相同场景、奖励和训练参数对比，可以检验全局信息是否有帮助。

前馈 MAPPO（`mappo_ff`）也使用集中 critic。仓库为它维护独立实现及配置，支持参数共享、特征和价值归一化。和集中 critic PPO 基线比较时，仍需检查奖励聚合等实验条件。

兼容旧实验时，`mappo` 指向 `centralized_ppo`，`mappo_continuous` 指向 `centralized_ppo_continuous`。评估旧检查点时使用对应的新名称输出结果。

### 方法说明

- [IPPO 与 IQL](IPPO_IQL.md)
- [HAPPO 与离散 HASAC](HAPPO_HASAC.md)
- [QTRAN-base 与 QPLEX](QTRAN_QPLEX.md)
- [VDN](VDN.md)与 [MAT-Dec](MAT_DEC.md)
- 其他方法的实现位于 [算法源码](../../src/teamwork/algorithms/)，各自的基础配置见上表。

## 方法背景与扩展候选

正式分工见页首。下表补充论文背景和扩展候选。合作类方法优先在 Spread、LBF 验证，RWARE 为扩展场景。

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
| [MAPPO](https://arxiv.org/abs/2103.01955)，2022 | 集中 critic 相比本地 critic 改善多少？ | 中；局部 actor、集中 critic，支持参数共享和归一化 |
| [HAPPO](https://arxiv.org/abs/2109.11251)，2022 | 顺序更新与同时更新有什么差异？ | 中–高；异构策略、顺序更新与重要性权重 |
| [MAT / MAT-Dec](https://arxiv.org/abs/2205.14953)，2022 | Transformer/序列建模是否有助于协作？ | 高；MAT-Dec 是同一方法族的变体；必须审计执行时信息访问 |
| [HASAC](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html)，2024 | 最大熵与顺序更新能否改善探索、样本效率？ | 高；离散 SAC、回放、温度和顺序更新 |
| [SMPE²](https://proceedings.mlr.press/v267/kontogiannis25a.html)，2025 | 局部状态建模与探索如何改善部分可观测协作？ | 高；belief/探索模块；作者已在 MPE、LBF、RWARE 评估 |

连续动作变体使用双臂环境和独立配置，详见 [机器人环境与算法](../environments/README.md#多算法对照)。

## 参考实现入口

- [EPyMARL](https://github.com/uoe-agents/epymarl)：IQL、VDN、QMIX、QTRAN、COMA、IPPO、MAPPO 的研究参考。
- [MAAC 作者代码](https://github.com/shariqiqbal2810/MAAC)、[QPLEX 作者代码](https://github.com/wjh720/QPLEX)。
- [MAPPO 作者代码](https://github.com/marlbenchmark/on-policy)、[HARL 作者代码](https://github.com/PKU-MARL/HARL)：HAPPO、HASAC 等。
- [MAT 作者代码](https://github.com/PKU-MARL/Multi-Agent-Transformer)、[SMPE² 作者代码](https://github.com/andreaskontogiannis/smpe)。
