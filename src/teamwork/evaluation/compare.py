"""同一场景下跨算法、跨训练种子的统一比较；跨场景请分别生成表格。"""

import json
from pathlib import Path

import numpy as np


def compare_results(paths, metric):
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not results:
        raise ValueError("At least one evaluation result is required")
    if any(r.get("schema_version") != 2 for r in results):
        raise ValueError(
            "Old evaluation format: re-evaluate models with the current metric protocol"
        )
    if len({r["benchmark_id"] for r in results}) != 1:
        raise ValueError(
            "Incomparable settings: environment, metrics, seeds, opponent or code differ"
        )
    if not all(r["comparison_allowed"] for r in results):
        raise ValueError("Tag comparison requires the same fixed opponent and evaluated team")
    lift_timing = results[0].get("lift_timing")
    if any(result.get("lift_timing") != lift_timing for result in results):
        raise ValueError("Lift timing modes differ; use the paired lift timing study")
    budgets = {r["training_steps"] for r in results if r["training_steps"] is not None}
    if len(budgets) > 1:
        raise ValueError("Training interaction budgets differ")
    trained = [result for result in results if result["training_steps"] is not None]
    training_env = trained[0].get("training_env_kwargs") if trained else None
    if any(result.get("training_env_kwargs") != training_env for result in trained):
        raise ValueError("Training environment differs or is missing")
    initialization = trained[0].get("training_initialization") if trained else None
    if any(result.get("training_initialization") != initialization for result in trained):
        raise ValueError(
            "Training initialization differs or is missing: match demonstration source, "
            "demonstration budget, and initialization method"
        )
    if initialization and (
        initialization["method"] == "checkpoint_warm_start"
        and not initialization.get("checkpoint_sha256")
        and len(trained) > 1
    ):
        raise ValueError("Warm-start checkpoint identity is missing")
    definitions = {d["name"]: d for d in results[0]["benchmark"]["protocol"]["metrics"]}
    if metric not in definitions:
        raise ValueError(f"Unknown metric {metric!r}; choose from {tuple(definitions)}")
    grouped = {}
    for result in results:
        grouped.setdefault(result["algorithm"], []).append(result)
    algorithms = {}
    for algorithm, group in grouped.items():
        seeds = [r["training_seed"] for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seed for {algorithm}; do not count it twice")
        if any(r["summary"][metric]["mean"] is None for r in group):
            raise ValueError("Metric is undefined for a run; choose an unconditional metric first")
        means = [r["summary"][metric]["mean"] for r in group]
        algorithms[algorithm] = {
            "mean": float(np.mean(means)),
            "std_across_training_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "training_seeds": seeds,
            "n_models": len(group),
            "per_model_means": means,
        }
    return {
        "benchmark_id": results[0]["benchmark_id"],
        "env": results[0]["env"],
        **({"lift_timing": lift_timing} if lift_timing is not None else {}),
        "metric": definitions[metric],
        "training_steps": next(iter(budgets), None),
        "training_initialization": initialization,
        "training_env_kwargs": training_env,
        "aggregation": "equal-weight model means; sample std over independent training seeds",
        "algorithms": algorithms,
    }
