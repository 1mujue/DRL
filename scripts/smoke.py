"""跨平台验收：测试、短程训练、模型加载、固定指标与图表，结果保存在 runs/。"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Test the research workflow on this OS")
    parser.add_argument("--skip-tests", action="store_true", help="Only exercise CLI workflows")
    args = parser.parse_args()
    env = dict(
        os.environ,
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        PYGAME_HIDE_SUPPORT_PROMPT="1",
        MPLBACKEND="Agg",
        PYTHONIOENCODING="utf-8",
    )

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)

    run("-m", "teamwork.cli", "doctor")
    if not args.skip_tests:
        run("-m", "pytest", "-q")
    (ROOT / "runs").mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="smoke-", dir=ROOT / "runs"))
    for algorithm in ("ippo", "mappo"):
        run(
            "-m",
            "teamwork.cli",
            "train",
            "--config",
            "configs/algorithms/ppo.yaml",
            "--algorithm",
            algorithm,
            "--env",
            "spread",
            "--steps",
            "64",
            "--out",
            out / algorithm,
        )
        run(
            "-m",
            "teamwork.cli",
            "evaluate",
            "--checkpoint",
            out / algorithm / "model.pt",
            "--episodes",
            "2",
            "--out",
            out / f"{algorithm}.json",
            "--gif",
            out / f"{algorithm}.gif",
        )
    run(
        "-m",
        "teamwork.cli",
        "compare",
        out / "ippo.json",
        out / "mappo.json",
        "--metric",
        "final_coverage",
        "--out",
        out / "comparison.json",
        "--plot",
        out / "comparison.png",
    )
    run(
        "-m",
        "teamwork.cli",
        "plot",
        "--csv",
        out / "ippo" / "episodes.csv",
        "--out",
        out / "training_curve.png",
    )
    print(f"Workflow passed. Short-run checks are not convergence evidence. Results: {out}")


if __name__ == "__main__":
    main()
