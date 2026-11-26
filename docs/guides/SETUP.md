# 安装与运行

安装完成后，用 [README 的快速开始](../../README.md#快速开始) 跑第一个例子。下列命令从仓库根目录运行。

## WSL 使用 Conda

在 WSL 或 Ubuntu 上可以用 Miniconda。MPE2、LBF、RWARE 和 MuJoCo 都装到同一个 `deeprl-teamwork` 环境。

```bash
conda create -n deeprl-teamwork --override-channels -c conda-forge python=3.12 pip
conda activate deeprl-teamwork
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
env -u PYTHONPATH python -m pip install -r requirements.txt
```

环境已经建好时，激活后执行最后一行即可。项目把 NumPy 固定为 1.26.4，因为 robosuite 1.5.2 的依赖需要这个版本范围。上面的 `env -u PYTHONPATH` 只对这条命令生效，可避免从 ROS Humble 的路径加载 Python 模块。项目的 pytest 配置也会关闭第三方插件的自动加载。

VS Code 选错 Python 时，在 WSL 窗口用 `Python: Select Interpreter` 选择 `deeprl-teamwork`。

## 使用项目虚拟环境的安装方式

需要 Git、Python 3.10 或更新版本和 uv。已有 Python 时，用 `python -m pip install uv` 安装 uv，再运行：

```bash
python scripts/setup.py
```

脚本会在 `.venv` 安装 Python 3.12 和项目依赖，默认使用 CPU。Windows 在 PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux、WSL 和 macOS 在终端运行：

```bash
source .venv/bin/activate
```

## 检查安装

```bash
python -m teamwork.cli doctor
python scripts/smoke.py
```

第一条列出 Python、依赖版本和 GPU 状态。第二条会跑测试、短训练、模型评估和绘图，输出到 `runs/smoke-*/`。GitHub Actions 在 Windows 和 Ubuntu 上运行同一个脚本。Windows 和 WSL 各用自己的 Python 环境。

WSL 或 Linux 上的 GIF 渲染报 SDL 显示错误时，可先运行 `export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`，再重试评估。命令详情用 `python -m teamwork.cli --help` 查看。需要在 NVIDIA GPU 上运行可使用 `python scripts/setup.py --backend cu128`。

完整依赖见 [requirements.txt](../../requirements.txt)，版本约束见 [requirements.lock.txt](../../requirements.lock.txt)。
