"""IPPO/IQL 四场景实验：统一预算、训练种子、评估种子及 Tag 固定对手。"""

import argparse
import csv
import json
import os
from pathlib import Path

import yaml

from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import evaluate, save_result
from teamwork.evaluation.plots import plot_comparison, plot_training_curve
from teamwork.experiments.runner import run_training

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "spread": "final_coverage",
    "tag": "capture_success",
    "lbf": "success",
    "rware": "deliveries_per_100_steps",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--envs", nargs="+", choices=tuple(METRICS), default=list(METRICS))
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1:
        parser.error("steps and episodes must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.envs)) != len(args.envs):
        parser.error("seeds and environments must be unique")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        **vars(args),
        "out": str(args.out),
        "algorithms": ["ippo", "iql"],
        "tag_team": "predators",
        "tag_opponent": "rule",
    }
    save_result(manifest, args.out / "experiment.json")
    for environment in args.envs:
        evaluations = []
        for algorithm in ("ippo", "iql"):
            base = yaml.safe_load((ROOT / f"configs/algorithms/{algorithm}.yaml").read_text())
            for seed in args.seeds:
                config = {**base, "env": environment, "seed": seed, "total_steps": args.steps}
                if environment == "tag":
                    config.update(train_team="predators", train_opponent="rule")
                out = args.out / environment / f"{algorithm}_s{seed}"
                print(f"Training {algorithm} | {environment} | seed={seed}", flush=True)
                checkpoint = run_training(config, out)
                if algorithm == "iql":
                    import torch

                    updates = torch.load(checkpoint, weights_only=True)["updates"]
                    if not all(updates.values()):
                        print(
                            "IQL has no gradient updates: increase --steps beyond warmup.",
                            flush=True,
                        )
                result = evaluate(
                    environment,
                    checkpoint=checkpoint,
                    episodes=args.episodes,
                    seed=args.eval_seed,
                    team="predators" if environment == "tag" else "all",
                    opponent="rule" if environment == "tag" else None,
                )
                path = out / "eval.json"
                save_result(result, path)
                evaluations.append(path)
                with (out / "episodes.csv").open() as stream:
                    has_episodes = next(csv.DictReader(stream), None) is not None
                if has_episodes:
                    plot_training_curve(out / "episodes.csv", out / "training_curve.png")
        comparison = compare_results(evaluations, METRICS[environment])
        save_result(comparison, args.out / environment / "comparison.json")
        plot_comparison(comparison, args.out / environment / "comparison.png")
    print(
        json.dumps(
            {"results": str(args.out), "steps_per_model": args.steps, "training_seeds": args.seeds},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
