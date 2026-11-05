"""算法研究共用的实验入口：统一种子、环境默认值与耗时记录。"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..evaluation.metrics import resolved_env_kwargs


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def run_training(config, out):
    # 延迟导入，避免公共实验入口依赖某种算法的内部模块。
    from ..algorithms.registry import make_algorithm

    config = {**config, "env_kwargs": resolved_env_kwargs(config["env"], config.get("env_kwargs"))}
    if config["total_steps"] < 1:
        raise ValueError("total_steps must be positive")
    seed_all(config["seed"])
    started = time.perf_counter()
    checkpoint = make_algorithm(config["algorithm"]).train(config, out)
    elapsed = time.perf_counter() - started
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("Algorithm.train must return an existing checkpoint")
    report = {
        "algorithm": config["algorithm"],
        "training_seed": config["seed"],
        "requested_joint_steps": config["total_steps"],
        "wall_seconds": elapsed,
        "timing_scope": "algorithm setup, collection, updates and checkpoint writing",
        "device": config.get("device", "cpu"),
    }
    (Path(out) / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return checkpoint
