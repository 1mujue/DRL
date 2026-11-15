# 算法 API 接入验收记录

日期：2026-11-12。验收对象为本分支的 IPPO/IQL Python SDK。

运行环境：macOS-26.5.1-arm64-arm-64bit，Python 3.12.11，PyTorch 2.7.1，CPU。
本次所有模型使用训练种子 0、100000 联合环境步；每个模型评估 10 回合，测试种子 10000–10009。
Tag 固定追捕阵营，对手为规则策略。

## 实际验证

- `ruff check .` 通过。
- 完整测试：108 passed；第三方 PettingZoo 环境入口产生 2 条弃用警告。
- 最小调用示例完成两个算法的训练、模型调用、评估和比较。
- 四环境共 8 个模型训练成功，实际训练步数均为 100000。
- IQL 所有受控 agent 均产生梯度更新；检查点记录各自的更新次数。
- 所有模型均完成保存加载、确定性批量推理和至少一个完整环境回合的有效动作检查。
- 每个环境均通过同预算、同评估种子的算法结果比较检查。
- wheel 安装到独立目录后，在 `/tmp` 工作目录通过已安装包完成训练和评估，确认调用不依赖源码工作目录。

## 本次单种子结果

训练耗时为本次运行记录；其他校验进程可能并行运行，耗时不作为独立性能基准。
各场景使用不同指标，按场景分别解释。

| 环境 | 算法 | 实际步数 | 训练耗时（秒） | 任务指标 | 10 回合均值 |
|---|---|---:|---:|---|---:|
| spread | ippo | 100000 | 48.34 | final_coverage | 0.0667 |
| spread | iql | 100000 | 52.90 | final_coverage | 0.1000 |
| tag | ippo | 100000 | 56.38 | capture_success | 0.3000 |
| tag | iql | 100000 | 61.19 | capture_success | 0.1000 |
| lbf | ippo | 100000 | 33.27 | success | 0.0000 |
| lbf | iql | 100000 | 36.50 | success | 0.0000 |
| rware | ippo | 100000 | 40.32 | deliveries_per_100_steps | 0.0000 |
| rware | iql | 100000 | 42.09 | deliveries_per_100_steps | 0.0000 |

这份记录证明接入链路可运行。LBF 与 RWARE 的任务指标仍为 0，当前模型的任务效果需要继续训练、调参和分析。
正式算法表现比较还需按全组协议完成多训练种子实验，并与规则/随机基线一起解释。

## 产物与复现

- 原始总报告：`runs/api_acceptance_20261112/acceptance.json`。
- 每个模型的配置、训练日志、回报 CSV、权重、来源信息和评估 JSON 保存在该目录的 `workspace/` 下。
- SDK wheel 与交付 ZIP 保存在 `runs/api_delivery_20261112/` 及同级交付文件中，均作为生成产物保存。

```bash
.venv/bin/python scripts/verify_algorithm_api.py --out runs/api_acceptance_new --steps 100000 --episodes 10
```

接口合同与组长调用示例见 [API_INTEGRATION](API_INTEGRATION.md)。
公共后端需持有 API 实例并包装 Web 路由，跨进程崩溃恢复由服务层另行实现。

## 共享仓库接入检查（2026-11-14）

交付分支为 `feat/ippo-iql`。合入公共主分支 `657eae4` 后，保留 MAPPO、QMIX、COMA、MAAC
及原有 PPO 基线的注册入口，并补齐 README 中 IPPO/IQL 的场景说明。

- `ruff check .` 通过。
- `python -m pytest -q`：197 passed，2 条第三方弃用警告。
- `python scripts/smoke.py --skip-tests` 通过：IPPO、集中 critic PPO 和 IQL 各训练 1024 step，完成 CLI 评估、比较和绘图。
- `python examples/algorithm_api.py --workspace <新目录>` 通过：IPPO/IQL 各训练 4096 step，完成 SDK 推理、每模型 3 回合评估及结果比较。

本节记录合入公共代码后的本机回归验证。四场景 100000 step 结果仍以上文单种子验收为准。
公共 Web 后端尚待调用方联调验收。
