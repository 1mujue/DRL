"""Train and evaluate the shared Spread benchmark, or evaluate its existing checkpoints."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import yaml

from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import evaluate, save_result
from teamwork.evaluation.metrics import resolved_env_kwargs
from teamwork.evaluation.plots import plot_comparison
from teamwork.experiments.runner import run_training

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiments/spread.yaml"


def evaluate_model(config, *, evaluation, training, reuse, out):
    algorithm, seed, steps = config["algorithm"], config["seed"], config["total_steps"]
    run = training / algorithm / f"seed_{seed}"
    print(f"{algorithm} seed={seed} steps={steps}", flush=True)
    if reuse:
        snapshot = run / "checkpoints" / f"step_{steps:09d}.pt"
        checkpoint = snapshot if snapshot.is_file() else run / "model.pt"
    else:
        checkpoint = run_training(config, run)
    result = evaluate(
        "spread", checkpoint=checkpoint, env_kwargs=config["env_kwargs"], **evaluation
    )
    if (result["algorithm"], result["training_seed"], result["training_steps"]) != (
        algorithm,
        seed,
        steps,
    ):
        raise ValueError(
            f"Checkpoint {checkpoint} has unexpected algorithm, seed or training steps"
        )
    evaluation_path = out / "evals" / f"{algorithm}-seed{seed}.json"
    save_result(result, evaluation_path)
    return evaluation_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--algorithms", nargs="+", help="Subset of the algorithm presets")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-training", type=Path, help="Existing algorithm/seed_N directories")
    args = parser.parse_args()

    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    algorithms = args.algorithms or list(plan["algorithms"])
    seeds = plan["training_seeds"]
    if len(set(algorithms)) != len(algorithms) or set(algorithms) - plan["algorithms"].keys():
        parser.error("choose distinct algorithms from " + ", ".join(plan["algorithms"]))
    if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1:
        parser.error("use at least one training seed, distinct seeds, and a positive worker count")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args.out.mkdir(parents=True, exist_ok=False)
    env_kwargs = resolved_env_kwargs("spread", plan["env_kwargs"])
    evaluation = plan["evaluation"]
    training = args.reuse_training or args.out / "training"
    configs = []
    for algorithm in algorithms:
        preset = args.config.parent / plan["algorithms"][algorithm]
        config = yaml.safe_load(preset.read_text(encoding="utf-8"))
        if config["algorithm"] != algorithm:
            raise ValueError(f"Preset {preset} does not configure {algorithm}")
        for seed in seeds:
            configs.append(
                {
                    **config,
                    "env": "spread",
                    "env_kwargs": env_kwargs,
                    "total_steps": plan["total_steps"],
                    "seed": seed,
                }
            )
    save_result(
        {**plan, "selected_algorithms": algorithms, "env_kwargs": env_kwargs},
        args.out / "experiment.json",
    )
    evaluate_model_at_budget = partial(
        evaluate_model,
        evaluation=evaluation,
        training=training,
        reuse=args.reuse_training is not None,
        out=args.out,
    )
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        paths = list(pool.map(evaluate_model_at_budget, configs))
    for policy in ("random", "rule"):
        result = evaluate("spread", policy=policy, env_kwargs=env_kwargs, **evaluation)
        out = args.out / "evals" / f"{policy}.json"
        save_result(result, out)
        paths.append(out)
    for metric in plan["metrics"]:
        comparison = compare_results(paths, metric)
        save_result(comparison, args.out / f"{metric}.json")
        plot_comparison(comparison, args.out / f"{metric}.png")
    print(f"Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
