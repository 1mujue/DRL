"""算法研究共用的实验入口：统一种子、环境默认值与耗时记录。"""

import importlib.metadata
import json
import platform
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..evaluation.metrics import resolved_env_kwargs
from .runs import atomic_write_json, reserve_run, utc_now


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _provenance(device):
    root = Path(__file__).resolve().parents[3]

    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None

    revision = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return {
        "git_commit": revision,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": device,
        "steps_unit": "joint environment transitions",
    }


def _save_common_artifacts(out, config):
    # 保留算法自行补充后的配置与来源文件，不覆盖算法专属元数据。
    if not (out / "config.json").exists():
        atomic_write_json(out / "config.json", config)
    if not (out / "provenance.json").exists():
        atomic_write_json(out / "provenance.json", _provenance(config.get("device", "cpu")))


def run_training(config, out=None, runs_root="runs"):
    # 延迟导入，避免公共实验入口依赖某种算法的内部模块。
    from ..algorithms.registry import make_algorithm

    config = deepcopy(config)
    config["env_kwargs"] = resolved_env_kwargs(config["env"], config.get("env_kwargs"))
    if config["total_steps"] < 1:
        raise ValueError("total_steps must be positive")
    # 提前验证可保存性，避免训练结束时才发现配置无法写入 JSON。
    json.dumps(config, allow_nan=False)
    algorithm = make_algorithm(config["algorithm"])
    record = reserve_run(config, out, runs_root)
    started = time.perf_counter()
    report = {
        "algorithm": config["algorithm"],
        "training_seed": config["seed"],
        "requested_joint_steps": config["total_steps"],
        "timing_scope": "algorithm setup, collection, updates and checkpoint writing",
        "device": config.get("device", "cpu"),
        "run_id": record.data["run_id"],
    }
    try:
        seed_all(config["seed"])
        started = time.perf_counter()
        checkpoint = Path(algorithm.train(config, record.path)).resolve()
        elapsed = time.perf_counter() - started
        if not checkpoint.is_file():
            raise ValueError("Algorithm.train must return an existing checkpoint")
        if not checkpoint.is_relative_to(record.path):
            raise ValueError("Algorithm checkpoint must be inside its run directory")
        _save_common_artifacts(record.path, config)
        report.update(status="completed", wall_seconds=elapsed, checkpoint=str(checkpoint))
        atomic_write_json(record.path / "training.json", report)
        record.update(
            status="completed", finished_at=utc_now(), wall_seconds=elapsed,
            checkpoint=str(checkpoint),
        )
    except BaseException as error:
        elapsed = time.perf_counter() - started
        failure = {"type": type(error).__name__, "message": str(error)}
        try:
            record.update(
                status="failed", finished_at=utc_now(), wall_seconds=elapsed, error=failure
            )
            if record.path.is_dir() and record.path.resolve() == record.path:
                _save_common_artifacts(record.path, config)
                report.update(status="failed", wall_seconds=elapsed, error=failure)
                atomic_write_json(record.path / "training.json", report)
        except (OSError, ValueError, TypeError) as metadata_error:
            error.add_note(f"Could not finish run metadata: {metadata_error}")
        raise
    return checkpoint
