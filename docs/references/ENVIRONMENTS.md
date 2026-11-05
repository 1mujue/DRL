# 备用多智能体环境调研

检索日期：2026-11-04。以下“适合程度”是依据四周、六人、各自电脑开发的条件作出的工程判断。
“已集成”只表示适配和短程检查，当前运行方法见 [安装说明](../SETUP.md)；不表示算法已经收敛。

## 推荐顺序

| 环境 | 协作/对抗及输入 | 适合研究什么 | 集成与算力判断 | 本项目建议 |
|---|---|---|---|---|
| MPE2 Spread / Tag | 覆盖协作 / 追逃对抗，向量观测，离散或连续动作 | 非平稳性、集中 critic、对手变化 | PettingZoo Parallel API；轻量 | 已选主线，保留 |
| Level-Based Foraging (LBF) | 采集协作，也可分配个体回报；格点向量、6 个离散动作 | 稀疏奖励、信用分配、协作必要性 | Gymnasium tuple 需适配；CPU 可开发 | 首选外部验证，已提供适配与配置 |
| Robot Warehouse (RWARE) | 仓储协作，局部向量观测、离散动作 | 局部可观测、拥堵、探索 | tuple 需适配；仿真轻，成功奖励稀疏 | 已提供适配；训练难度通常高于入门 Spread，作为备选 |
| PettingZoo Pistonball | 多活塞推球协作，RGB 观测 | 视觉协作、参数共享 | Parallel API；需要图像预处理和 CNN | 展示效果好，但偏离结构化输入主线，暂不安装 |
| PettingZoo Multiwalker | 多机器人共同抬包裹，向量观测、连续动作 | 连续控制、团队失败传播 | Parallel API；Box2D 等额外依赖 | 第二阶段；当前离散 actor 不能直接训练 |
| MAgent2 Battle / Adversarial Pursuit | 大规模格点群体协作对抗 | agent 数扩展、局部观测 | PettingZoo API；官方支持 Linux/macOS | 需独立 Linux/WSL 环境，四周内非首选 |
| SMACv2 | 队伍配合对抗脚本敌方；局部观测、离散动作掩码 | 单位微操、分布泛化 | 需要 StarCraft II、地图与额外接口 | 依赖较重；不是双方都学习的自博弈平台，作后续基准 |
| JaxMARL / SMAX | JAX 原生 SMAC 类微操，无需 SC2 引擎 | 批量环境与训练加速 | JAX PRNG/函数式接口，需重做适配；原生 Windows CUDA 不支持 | 有 Linux/WSL 与 JAX 经验再选 |
| MaMuJoCo | 将机器人动作划分给多个 agent，连续控制 | 连续动作协作、因子化控制 | Parallel API；MuJoCo，官方不支持 Windows | 研究扩展，当前不安装 |

## 外部测试怎样形成有效结论

首选方案：Spread/Tag 完成研究主线，再在 **LBF 小地图、2 agent、1 food、强制协作** 上重新训练 IPPO 和 MAPPO-style。
用相同交互预算与种子集合比较两方法；再逐步增加食物数量或缩小视野。不要把不同环境的原始奖励相加。
RWARE 先跑 tiny、2 agent；若随机探索长时间无奖励，缩小任务或调整请求密度，同时完整记录环境变化。

如果老师要求额外的“双方可学习对抗环境”，优先评估 MAgent2 的小规模 Battle；
LBF/RWARE 主要增加协作任务多样性，并不能替代 Tag 的对抗研究。
外部环境的验证一般是重新训练方法，不能据此声称同一模型零样本迁移。

## 安装和接口注意

- MPE2 已是独立包，使用 `from mpe2 import simple_spread_v3`，不照搬旧 `pettingzoo.mpe` 教程。
- LBF/RWARE 官方文档中混有旧 Gym 示例。以本次安装版本的 Gymnasium 五返回值接口和测试结果为准。
- Pistonball 可选依赖为 `pettingzoo[butterfly]`；Multiwalker 位于 SISL，连续 actor 和依赖应另开分支验证。
- JAX GPU 路径宜 Linux/WSL；别因电脑有 NVIDIA GPU 就假定 Windows 上所有 MARL 框架可直接加速。
- 环境与算法库是两回事。SB3、RLlib、TorchRL/BenchMARL 等是算法/训练工具，不能作为“另一个任务环境”计数。

## 官方来源

1. [MPE2](https://mpe2.farama.org/)、[Spread](https://mpe2.farama.org/environments/simple_spread/)、[Tag](https://mpe2.farama.org/environments/simple_tag/)
2. [LBF 作者仓库](https://github.com/uoe-agents/lb-foraging)、[PyPI 发布](https://pypi.org/project/lbforaging/)
3. [RWARE 作者仓库](https://github.com/uoe-agents/robotic-warehouse)、[PyPI 接口变更](https://pypi.org/project/rware/)
4. [Pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/)
5. [Multiwalker](https://pettingzoo.farama.org/environments/sisl/multiwalker/)
6. [MAgent2 支持平台](https://magent2.farama.org/)
7. [SMACv2 作者仓库](https://github.com/oxwhirl/smacv2)
8. [JaxMARL](https://github.com/bold-lab-ai/JaxMARL)、[SMAX](https://jaxmarl.foersterlab.com/environments/smax/)、[JAX 支持平台](https://docs.jax.dev/en/latest/installation.html)
9. [Gymnasium-Robotics / MaMuJoCo](https://github.com/Farama-Foundation/Gymnasium-Robotics)
