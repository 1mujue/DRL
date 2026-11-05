"""Windows / Ubuntu 共用安装入口：python scripts/setup.py [--backend cpu|cu128]。"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Create the project's Python 3.12 environment")
    parser.add_argument("--backend", choices=["cpu", "cu128"], default="cpu")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        parser.exit(1, "Install uv first: python -m pip install uv\n")
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (ROOT / ".venv").exists() and not python.is_file():
        parser.exit(1, "This .venv belongs to another OS. Use a separate checkout.\n")
    env = dict(
        os.environ,
        UV_PYTHON_INSTALL_DIR=str(ROOT / ".python"),
        UV_CACHE_DIR=str(ROOT / ".uv-cache"),
    )
    if not python.is_file():
        subprocess.run(
            [uv, "venv", "--python", "3.12", "--managed-python", "--seed", ".venv"],
            cwd=ROOT,
            env=env,
            check=True,
        )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--torch-backend",
            args.backend,
            "-e",
            ".[dev,extra-envs]",
            "-c",
            "requirements.lock.txt",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run([str(python), "-m", "teamwork.cli", "doctor"], cwd=ROOT, check=True)
    print(f"Ready. Verify with: {python} scripts/smoke.py")


if __name__ == "__main__":
    main()
