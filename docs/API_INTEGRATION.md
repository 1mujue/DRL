# IPPO / IQL 算法接入交付

本模块供小组业务后端通过 Python 调用，API 版本为 `1.0`。
负责算法目录、请求校验、训练任务、模型加载、观测到动作的推理、统一评估及产物查询。
Web 路由、用户业务与服务器部署由项目的公共后端承接。

## 安装与验收

在 Python 3.12 环境中安装仓库：

```bash
python -m pip install -e '.[dev,extra-envs]' -c requirements.lock.txt
```

本地也可使用项目安装脚本 `python3 scripts/setup.py`。以下命令从仓库根目录运行；
Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。

```bash
.venv/bin/python examples/algorithm_api.py --workspace runs/api_example
.venv/bin/python scripts/verify_algorithm_api.py --out runs/api_acceptance --steps 100000 --episodes 10
```

第一个示例对两个算法完成训练、模型调用、评估与比较。
第二个命令在四环境各训练两个模型，验收报告位于输出目录的 `acceptance.json`。
报告包含实际训练步数、IQL 更新次数、模型 SHA256、训练耗时、动作有效性检查、逐项评估汇总和产物路径。
已完成的本机实测见 [接入验收记录](API_ACCEPTANCE.md)。
验收使用单个训练种子；正式统计实验的种子与预算由全组统一，参见 [实验说明](IPPO_IQL.md)。

## 后端最小调用

```python
from teamwork.api import AlgorithmAPI, TrainingRequest

# 在后端应用启动时创建一个实例，在应用关闭时调用 close()。
with AlgorithmAPI("runs/backend_algorithms", max_concurrent=1) as api:
    request = TrainingRequest(
        algorithm="iql",
        env="spread",
        steps=100000,
        seed=0,
        parameters={"lr": 0.0005, "batch_size": 64},
    )
    run = api.train(request, timeout=600)
    run_id = run["run_id"]
    evaluation = api.evaluate(run_id, {"episodes": 10, "seed": 10000}, timeout=120)
    artifacts = api.artifacts(run_id)
```

请求也可传入普通字典，适合由业务后端从 JSON 构建。
返回值由字典、列表、数值、字符串和空值组成，可直接 JSON 序列化。
推理输入额外支持 NumPy 数组；模型文件由 SDK 按运行 ID 定位。

## 调用入口

| 方法 | 输入 | 输出 / 行为 |
|---|---|---|
| `AlgorithmAPI.catalog()` | 无 | API 版本、算法描述、参数默认值、环境默认设置、指标与依赖安装状态 |
| `validate_training(request)` | `TrainingRequest` 或字典 | 完整训练配置与环境空间；执行环境构造检查 |
| `submit_training(request)` | 同上 | 启动训练进程，立即返回运行 ID 和状态 |
| `train(request, timeout=None)` | 同上 | 提交并等待训练成功，失败时抛结构化异常 |
| `status(run_id)` | 运行 ID | 状态、时间、配置、最近日志步数、错误信息 |
| `wait(run_id, timeout=None)` | 运行 ID | 等待终态；超时时任务继续运行 |
| `cancel(run_id)` | 运行 ID | 结束本实例管理的训练进程，保留已有产物 |
| `logs(run_id, limit=8192)` | 运行 ID、末尾字节数 | 训练标准输出与标准错误日志 |
| `list_runs()` | 无 | 工作目录中的运行记录，按创建时间倒序 |
| `artifacts(run_id)` | 运行 ID | 存在的产物名称、绝对路径和字节数 |
| `open_policy(run_id, seed=0)` | 成功运行的 ID | 可复用 `PolicySession`，模型加载一次 |
| `predict(run_id, observations, deterministic=True)` | ID、局部观测 | 加载模型并执行一次推理 |
| `evaluate(run_id, request=None, timeout=None)` | ID、评估参数 | 评估引用与完整结果；保存 JSON 和日志 |
| `evaluation(run_id, evaluation_id)` | 评估引用 | 读取已有评估 JSON |
| `compare(references, metric)` | 评估引用列表、指标名 | 通过公共可比性检查的算法结果汇总 |
| `export_manifest(run_id, path)` | ID、输出文件 | 导出运行信息和产物索引，便于交付 |
| `close()` | 无 | 取消本实例尚未结束的训练进程并关闭 API |

## 训练请求

| 字段 | 默认值 | 规则 |
|---|---|---|
| `algorithm` | 必填 | `ippo` 或 `iql` |
| `env` | `spread` | `spread`、`tag`、`lbf`、`rware` |
| `steps` | `100000` | 正整数；联合环境交互步数 |
| `seed` | `0` | 0 至 2³²−1 的整数 |
| `device` | `cpu` | `cpu` 或本机可用的 `cuda` |
| `parameters` | `{}` | 所选算法的参数覆盖，允许项与默认值由 catalog 返回 |
| `env_kwargs` | `{}` | 场景参数覆盖，和公共协议默认设置合并 |
| `team` | `all` | Tag 必须明确传入 `predators` 或 `prey` |
| `opponent` | `None` | Tag 必须明确传入 `rule` 或 `random` |

未知字段、未知算法参数、非有限数值、错误的数据类型、不可用设备和无效阵营组合会在创建任务前被拒绝。
环境参数会通过真实环境构造和 reset 校验。IQL 检查回放容量与 batch、探索率上下界；
算法默认参数在规范化配置中展开并随模型保存。

Tag 示例：

```python
run = api.train({
    "algorithm": "ippo", "env": "tag", "steps": 100000,
    "team": "predators", "opponent": "rule",
})
result = api.evaluate(run["run_id"], {"episodes": 10})
```

SDK 自动按训练时的阵营与对手评估。固定阵营策略只返回受控阵营动作；
项目后端驱动完整环境时，应按相同配置补齐对手动作。

## 非阻塞训练与生命周期

```python
run = api.submit_training({"algorithm": "iql", "steps": 100000})
run_id = run["run_id"]
status = api.status(run_id)
log_tail = api.logs(run_id)
# 后续请求按 run_id 查询；业务需要停止时调用 api.cancel(run_id)。
```

训练在独立 Python 进程中执行，避免全局随机种子和 Torch 线程设置影响业务进程。
`max_concurrent` 限制单个 API 实例同时运行的训练数；达到上限返回 `BUSY`，调用方可稍后重试。
API 实例应在后端应用生命周期内复用。上下文退出或显式关闭时会回收仍在运行的训练进程。

| 状态 | 含义 |
|---|---|
| `created` | 已写入规范化请求与环境空间 |
| `running` | 训练子进程已启动 |
| `succeeded` | 子进程成功退出，模型产物存在 |
| `failed` | 进程启动或训练失败，错误和日志保留 |
| `cancelled` | 调用方或 API 关闭流程结束了训练 |

`progress.logged_steps` 来自算法日志，是最近已报告的步数；运行中的百分比可能落后于实际进度。
成功后百分比为 1。超时只结束 `wait/train` 的等待，调用方仍可查询或取消任务。
评估的超时会停止该评估子进程。

运行清单保存在工作目录，可由新实例读取已结束任务。正在运行的进程由创建它的 API 实例管理；
其他实例读取时 `managed=False`，等待或取消会返回 `UNMANAGED_TASK`。
后端应为一个工作目录指定一个长期运行的实例；进程崩溃恢复与分布式任务队列属于公共服务的后续职责。

## 加载与复用推理模型

```python
with api.open_policy(run_id, seed=42) as policy:
    print(policy.agents)  # 受控 agent 的 ID
    actions = policy.predict(observations)
    batch_actions = policy.predict_batch([observations, observations])
    policy.reset()       # 每个新回合调用
```

`observations` 格式为 `{agent_id: [数值, ...]}`，允许提供全部 agent 的局部观测，
必须包含所有受控 agent，且各向量长度与模型所用场景一致。无效长度、非有限值或未知 agent 会返回校验错误。
`predict_batch` 按顺序处理多个观测字典，在执行前校验整批输入。

执行策略只访问局部观测。确定性推理返回 argmax；随机推理采用算法既定采样方式。
随机推理会话保存自己的 CPU RNG 状态，支持通过 `reset(seed=42)` 重现采样序列。
频繁推理应复用 `open_policy` 返回的会话；便捷方法 `api.predict` 每次加载模型。
调用方负责关闭自己创建的会话。

## 评估与结果比较

```python
first = api.evaluate(first_run_id, {"episodes": 100, "seed": 10000})
second = api.evaluate(second_run_id, {"episodes": 100, "seed": 10000})
comparison = api.compare([first, second], "final_coverage")
```

`EvaluationRequest` 只允许 `episodes` 与 `seed`，使用模型中保存的环境设置。
评估过程在独立进程中执行，调用方随机数状态保持独立。
引用包括 `run_id/evaluation_id`；不同环境、不同评估种子、不同预算或重复训练种子等条件沿用公共比较器的检查。
跨场景结果分别展示；指标定义见 [EXPERIMENTS](EXPERIMENTS.md)。

## 结构化异常

```python
from teamwork.api import AlgorithmAPIError

try:
    run = api.train({"algorithm": "iql", "steps": 100000})
except AlgorithmAPIError as exc:
    payload = exc.to_dict()
    # {"code": "...", "message": "...", "details": {...}}
```

主要错误码：`INVALID_REQUEST`、`BUSY`、`TIMEOUT`、`RUN_NOT_FOUND`、`RUN_NOT_READY`、
`ARTIFACT_NOT_FOUND`、`START_FAILED`、`TRAINING_FAILED`、`EVALUATION_FAILED`、
`EVALUATION_NOT_FOUND`、`INVALID_CHECKPOINT`、`INVALID_ACTION`、`INCOMPARABLE_RESULTS`、
`UNMANAGED_TASK`、`INVALID_MANIFEST`、`API_CLOSED`、`SESSION_CLOSED`。
训练失败时可通过运行 ID 查询完整日志；重试会创建新运行记录，已有产物保留。

## 产物布局与集成边界

```text
workspace/
  run_<id>/
    manifest.json            # 任务状态、请求配置、环境空间与评估索引
    request.json             # 原始 SDK 请求
    resolved.json            # 展开的算法与环境参数
    train.log                # 训练进程日志
    training/                # 公共训练器的原有产物
      model.pt
      config.json
      episodes.csv
      updates.csv            # IQL
      training.json
      provenance.json
    evaluations/
      eval_<id>.json
      eval_<id>.log
```

运行 ID 和评估 ID 由 SDK 生成，读取接口校验 ID 格式与工作目录边界。
训练器、环境接口和固定指标沿用项目公共实现；SDK 不为两种算法另建计分规则。
现有 CLI 和研究用脚本继续可用。模型支持加载执行策略；再次训练会创建一次新运行。

组长集成时需要确认：后端持有 API 实例的方式、工作目录位置、训练并发上限，以及部署平台的 Python/依赖版本。
接口采用 JSON 可序列化数据，便于公共后端封装为后续 Web API。

## 生成交付包

```bash
.venv/bin/python -m pip wheel . --no-deps --no-build-isolation --wheel-dir runs/api_delivery
.venv/bin/python scripts/package_algorithm_delivery.py --wheel runs/api_delivery/drl_teamwork-0.1.0-py3-none-any.whl --acceptance runs/api_acceptance --out runs/algorithm_handoff
```

打包脚本要求四环境验收通过，并核对模型 SHA256。交付 ZIP 包含 wheel、接口说明、调用示例、
已训练模型与原始验收产物，附带源码提交号、可迁移路径索引与文件校验清单。
