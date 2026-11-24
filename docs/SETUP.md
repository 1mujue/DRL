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

## 使用项目虚拟环境的安装方式

先安装 Git、可用的 Python（3.10+）和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
如果已有 Python，可用 `python -m pip install uv` 安装 uv。

在项目根目录，两种系统都使用同一个安装脚本：

```bash
python scripts/setup.py
```

它创建独立 Python 3.12、`.venv`，安装版本约束下的依赖；默认 CPU。
Windows 在 PowerShell 激活项目环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux、WSL 和 macOS 在相应的 shell 中激活：

```bash
source .venv/bin/activate
```

## 验收

激活环境后，在仓库根目录运行：

```bash
python -m teamwork.cli doctor
python scripts/smoke.py
```

Smoke 流程包括测试、短程训练、模型评估和图表，结果写入 `runs/smoke-*/`。GitHub Actions 在 Windows 和 Ubuntu 使用相同脚本。Windows 与 WSL 分别安装各自的环境并选择相应解释器。

`python -m teamwork.cli --help` 查看入口。有 NVIDIA GPU 时可通过 `python scripts/setup.py --backend cu128` 安装并检查 CUDA 设备。

依赖由 `pyproject.toml` 声明、`requirements.lock.txt` 约束。算法执行示例见 [算法目录](ALGORITHMS.md)。
