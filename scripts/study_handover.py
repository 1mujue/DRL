"""Reproduce the scripted and behavior-cloned two-arm handover comparison."""

import argparse
from pathlib import Path

import torch
import yaml
from train_handover_clone import train_handover_clone

from teamwork.algorithms.baselines import SCRIPTED_HANDOVER_POLICY
from teamwork.algorithms.names import canonical_algorithm_name
from teamwork.evaluation.evaluator import evaluate, save_result
from teamwork.evaluation.plots import plot_handover_trace


def run_study(plan_path, out):
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    config_path = plan_path.parent / plan["training_config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["algorithm"] = canonical_algorithm_name(config["algorithm"])
    demos, evaluation = plan["demonstrations"], plan["evaluation"]
    demo_seeds = range(demos["start_seed"], demos["start_seed"] + demos["max_attempts"])
    eval_seeds = range(evaluation["seed"], evaluation["seed"] + evaluation["episodes"])
    if not eval_seeds or eval_seeds.start < 0:
        raise ValueError("Evaluation requires a nonnegative seed and at least one episode")
    if max(demo_seeds.start, eval_seeds.start) < min(demo_seeds.stop, eval_seeds.stop):
        raise ValueError("Demonstration and evaluation seed ranges must be disjoint")

    # CPU control needs only one Torch thread for these small networks.
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    save_result({**plan, "training_config": config}, out / "config.json")
    checkpoint, training = train_handover_clone(config, out / "training", **demos)
    settings = {"env_kwargs": config["env_kwargs"], **evaluation, "record_trace": True}
    results = {}
    for label, options in (
        ("scripted", {"policy": SCRIPTED_HANDOVER_POLICY}),
        ("clone", {"checkpoint": checkpoint}),
    ):
        print(f"Evaluating {label}", flush=True)
        report = evaluate(config["env"], **settings, **options)
        save_result(report, out / f"{label}.json")
        results[label] = report
    scripted, clone = results["scripted"], results["clone"]
    if scripted["benchmark_id"] != clone["benchmark_id"]:
        raise ValueError("Scripted and cloned policies were evaluated with different protocols")

    metrics = {}
    for label, report in results.items():
        episodes = report["episodes"]
        count = sum(int(row["metrics"]["handover_success"]) for row in episodes)
        metrics[label] = {
            "successes": count,
            "episodes": len(episodes),
            "success_rate": count / len(episodes),
        }
        print(f"{label}: {count}/{len(episodes)} successes", flush=True)
    sample = next(
        (row for row in clone["episodes"] if row["metrics"]["handover_success"]),
        clone["episodes"][0],
    )
    plot_handover_trace(out / "clone.json", out / "handover.png", seed=sample["seed"])
    report_path = out / "summary.json"
    save_result(
        {
            "benchmark_id": clone["benchmark_id"],
            "metric": "handover_success",
            "demonstration_seeds": training["demonstration_seeds"],
            "demonstration_joint_steps": training["demonstration_joint_steps"],
            "online_joint_steps": training["online_joint_steps"],
            "evaluation_seeds": list(eval_seeds),
            "plotted_seed": sample["seed"],
            "results": metrics,
        },
        report_path,
    )
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/handover.yaml"))
    parser.add_argument(
        "--out", type=Path, required=True, help="New directory for experiment output"
    )
    args = parser.parse_args()
    try:
        print(run_study(args.config, args.out))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
