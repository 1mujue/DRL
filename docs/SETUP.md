# 安装与运行

## WSL 使用 Conda

已安装 Miniconda 的成员可以在项目根目录创建独立环境。已有 `deeprl-teamwork` 环境时，直接激活即可。

```bash
conda create -n deeprl-teamwork --override-channels -c conda-forge python=3.12 pip
conda activate deeprl-teamwork
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,extra-envs]' -c requirements.lock.txt
```

如果 VS Code 还指向旧解释器，在 WSL 窗口用 `Python: Select Interpreter` 选择 `deeprl-teamwork`。
本项目 QTRAN 和 QPLEX 的测试步骤见 [方法说明](QTRAN_QPLEX.md)。

## 使用项目虚拟环境的安装方式

先安装 Git、可用的 Python（3.10+）和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
如果已有 Python，可用 `python -m pip install uv` 安装 uv。

在项目根目录，两种系统都使用同一个安装脚本：

```bash
python scripts/setup.py
```

它创建独立 Python 3.12、`.venv`，安装版本约束下的依赖；默认 CPU。
安装后用项目解释器验收：

```powershell
# Windows
.\.venv\Scripts\python.exe scripts/smoke.py
```

```bash
# Ubuntu / WSL
.venv/bin/python scripts/smoke.py
```

完整验收会跑测试、短程训练、模型评估和图表，结果写入新建的 `runs/smoke-*/`，不是收敛实验。
`python -m teamwork.cli --help` 查看入口；有 NVIDIA GPU 时可选 `python scripts/setup.py --backend cu128`，需另外验证 CUDA。
Windows 和 Ubuntu 必须各自创建 `.venv`，不共享环境目录；当前完整实测平台是 Windows，Ubuntu 仍待实际训练与 CI 验证。
`.github/workflows/ci.yml` 已为两种系统配置同一验收脚本；GitHub 远程仓库尚未创建。

依赖由 `pyproject.toml` 声明、`requirements.lock.txt` 约束。本机虚拟环境和下载缓存不提交 Git。
