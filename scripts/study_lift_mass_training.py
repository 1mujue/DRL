"""Compare fixed-mean and randomized mass training from one HASAC demonstration checkpoint."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, stdev

import torch
import yaml
from evaluate_lift_mass import evaluate_checkpoint

from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import save_result
from teamwork.experiments.runner import run_training

GROUPS = ("fixed", "randomized")
ALGORITHM = "hasac_continuous"
ENVIRONMENT = "two_arm_lift"


def train_model(job):
    group, config, destination = job
    checkpoint = run_training(config, destination)
    resets = json.loads((destination / "episode_resets.json").read_text(encoding="utf-8"))
    samples = [entry["infos"]["arm_0"]["pot_mass_scale"] for entry in resets]
    expected_count = math.ceil(config["total_steps"] / config["env_kwargs"]["horizon"])
    seed = config["seed"]
    if [entry["seed"] for entry in resets] != list(range(seed, seed + expected_count)):
        raise ValueError(f"Unexpected online reset seeds in {destination}")
    if group == "fixed":
        if any(value != config["env_kwargs"]["pot_mass_scale"] for value in samples):
            raise ValueError(f"Unexpected fixed mass in {destination}")
    else:
        low, high = config["env_kwargs"]["pot_mass_range"]
        if any(not low <= value <= high for value in samples):
            raise ValueError(f"Randomized mass outside the planned range in {destination}")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data["steps"] != config["total_steps"] or data["updates"] < 1:
        raise ValueError(f"Incomplete online training in {destination}")
    print(f"{group} seed={seed} trained, mean mass scale={mean(samples):.3f}", flush=True)
    return {
        "group": group,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "episodes_started": len(samples),
        "sampled_mass_mean": mean(samples),
        "sampled_mass_min": min(samples),
        "sampled_mass_max": max(samples),
        "actor_updates": data["updates"],
    }


def summarize_paired_level(scale, models, evaluations, control, plan):
    metrics = ("stable_dual_lift", "peak_lift_height_m")
    grouped = {group: [] for group in GROUPS}
    for model, results in zip(models, evaluations, strict=True):
        grouped[model["group"]].append(results[scale][0])
    summaries = {
        group: {metric: compare_results(paths, metric) for metric in metrics}
        for group, paths in grouped.items()
    }
    fixed, randomized = (summaries[group]["stable_dual_lift"] for group in GROUPS)
    conditions = {
        group: summaries[group]["stable_dual_lift"]["training_env_kwargs"] for group in GROUPS
    }
    expected_mass = {
        "fixed": {"pot_mass_scale": plan["fixed_mass_scale"]},
        "randomized": {"pot_mass_range": plan["random_mass_range"]},
    }
    for group, settings in conditions.items():
        actual = {key: value for key, value in settings.items() if key.startswith("pot_mass_")}
        if actual != expected_mass[group]:
            raise ValueError("Training mass settings differ from the study design")
        shared = {key: value for key, value in settings.items() if key not in actual}
        if shared != control["training_env_kwargs"]:
            raise ValueError("Experiment groups differ in non-mass environment settings")

    initial = control["training_initialization"]
    online = fixed["training_initialization"]
    if (
        fixed["benchmark_id"] != randomized["benchmark_id"]
        or fixed["benchmark_id"] != control["benchmark_id"]
        or fixed["training_steps"] != randomized["training_steps"]
        or fixed["training_steps"] != plan["training"]["total_steps"]
        or online != randomized["training_initialization"]
        or initial["method"] != "behavior_cloning"
        or online["method"] != "behavior_cloning_then_online"
        or any(
            initial[key] != online[key]
            for key in ("demonstration_sha256", "demonstration_joint_steps")
        )
    ):
        raise ValueError("Unexpected budget, initialization or evaluation settings across groups")

    paired = {}
    for metric in metrics:
        first = summaries["fixed"][metric]["algorithms"][ALGORITHM]
        second = summaries["randomized"][metric]["algorithms"][ALGORITHM]
        if first["training_seeds"] != second["training_seeds"]:
            raise ValueError("Training seed pairs differ")
        deltas = [
            random_score - fixed_score
            for fixed_score, random_score in zip(
                first["per_model_means"], second["per_model_means"], strict=True
            )
        ]
        paired[metric] = {
            "seeds": first["training_seeds"],
            "randomized_minus_fixed": deltas,
            "mean": mean(deltas),
            "std_across_pairs": stdev(deltas) if len(deltas) > 1 else None,
        }
    return {"groups": summaries, "behavior_clone": control["summary"], "paired": paired}


def plot_study(report, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("stable_dual_lift", "Stable lift fraction"),
        ("peak_lift_height_m", "Mean peak height (m)"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    scales = report["plan"]["evaluation"]["mass_scales"]
    cases = [report["results"][f"{scale:g}"] for scale in scales]
    positions = list(range(len(scales)))
    for axis, (metric, label) in zip(axes, panels, strict=True):
        for group, color, marker in (("fixed", "#315b95", "o"), ("randomized", "#9b481f", "s")):
            values = [case["groups"][group][metric]["algorithms"][ALGORITHM] for case in cases]
            axis.errorbar(
                positions,
                [entry["mean"] for entry in values],
                [entry["std_across_training_seeds"] or 0 for entry in values],
                marker=marker,
                markerfacecolor="none",
                color=color,
                capsize=3,
                label=f"{group} online training",
            )
        axis.plot(
            positions,
            [case["behavior_clone"][metric]["mean"] for case in cases],
            ":D",
            color="#525252",
            markerfacecolor="none",
            label="behavior clone",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    axes[-1].set_xticks(positions, [f"{scale:g}×" for scale in scales])
    axes[-1].set_xlabel("Evaluation pot mass multiplier")
    figure.suptitle("HASAC | shared demonstration checkpoint and online budget")
    axes[0].set_title("Error bars show standard deviation across training seeds", fontsize=9)
    figure.tight_layout()
    figure.savefig(out, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/algorithms/robotics/hasac_mass_study.yaml")
    )
    parser.add_argument("--behavior-clone", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    low, high = plan["random_mass_range"]
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        parser.error("random_mass_range must have finite increasing positive bounds")
    seeds = plan["training_seeds"]
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        parser.error("provide at least two distinct nonnegative training seeds")
    plan["fixed_mass_scale"] = (low + high) / 2
    origin = torch.load(args.behavior_clone, map_location="cpu", weights_only=True)
    nominal = origin["config"]
    if (
        nominal["algorithm"] != ALGORITHM
        or nominal["env"] != ENVIRONMENT
        or origin["steps"] != 0
        or nominal.get("behavior_cloning", {}).get("method") != "behavior_cloning"
        or any(key.startswith("pot_mass_") for key in nominal["env_kwargs"])
    ):
        parser.error("use a nominal TwoArmLift HASAC behavior-clone checkpoint")
    test = plan["evaluation"]
    steps = plan["training"]["total_steps"]
    if type(steps) is not int or steps < 1:
        parser.error("training.total_steps must be a positive integer")
    if (
        type(test["start_seed"]) is not int
        or test["start_seed"] < 0
        or type(test["episodes"]) is not int
        or test["episodes"] < 1
    ):
        parser.error("evaluation requires a nonnegative start seed and positive episode count")
    scales = test["mass_scales"]
    if (
        not scales
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in scales
        )
        or len(scales) != len(set(scales))
    ):
        parser.error("evaluation.mass_scales must contain distinct positive finite values")
    reset_count = math.ceil(steps / nominal["env_kwargs"]["horizon"])
    training_resets = {seed + index for seed in seeds for index in range(reset_count)}
    evaluation_seeds = set(range(test["start_seed"], test["start_seed"] + test["episodes"]))
    demonstration_seeds = set(nominal["behavior_cloning"].get("demonstration_seeds", []))
    if (
        len(training_resets) != len(seeds) * reset_count
        or training_resets & evaluation_seeds
        or demonstration_seeds & evaluation_seeds
    ):
        parser.error("training reset sequences must be independent and evaluation seeds held out")
    base = {
        **plan["training"],
        "algorithm": ALGORITHM,
        "env": ENVIRONMENT,
        "actor_type": nominal["actor_type"],
        "warm_start_checkpoint": str(args.behavior_clone.resolve()),
        "allow_lift_mass_transfer": True,
    }

    args.out.mkdir(parents=True, exist_ok=False)
    save_result(plan, args.out / "study_plan.json")
    jobs = []
    for group in GROUPS:
        mass = (
            {"pot_mass_scale": plan["fixed_mass_scale"]}
            if group == "fixed"
            else {"pot_mass_range": [low, high]}
        )
        for seed in seeds:
            cfg = {**base, "seed": seed, "env_kwargs": {**nominal["env_kwargs"], **mass}}
            jobs.append((group, cfg, args.out / "training" / group / f"seed_{seed}"))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        models = list(pool.map(train_model, jobs))
        evaluations = [
            (
                model["checkpoint"],
                args.out / "evaluation" / f"{model['group']}_seed_{model['seed']}",
                test["mass_scales"],
                test["start_seed"],
                test["episodes"],
            )
            for model in models
        ]
        evaluations.append(
            (
                args.behavior_clone,
                args.out / "evaluation" / "behavior_clone",
                test["mass_scales"],
                test["start_seed"],
                test["episodes"],
            )
        )
        results = list(pool.map(evaluate_checkpoint, evaluations))

    report = {
        "plan": plan,
        "behavior_clone_checkpoint": str(args.behavior_clone),
        "trained_models": models,
        "results": {
            f"{scale:g}": summarize_paired_level(
                scale, models, results[:-1], results[-1][scale][1], plan
            )
            for scale in test["mass_scales"]
        },
    }
    save_result(report, args.out / "summary.json")
    plot_study(report, args.out / "mass_training_comparison.png")
    print(args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
