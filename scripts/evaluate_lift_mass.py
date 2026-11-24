"""Evaluate an existing TwoArmLift policy suite under fixed pot mass interventions."""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import evaluate, save_result


def evaluate_checkpoint(job):
    checkpoint, output_stem, scales, seed, episodes = job
    results = {}
    for scale in scales:
        target = output_stem.parent / f"mass_x{scale:g}" / f"{output_stem.name}.json"
        report = evaluate(
            "two_arm_lift",
            checkpoint=checkpoint,
            seed=seed,
            episodes=episodes,
            pot_mass_scale=scale,
            record_trace=True,
        )
        save_result(report, target)
        results[scale] = target, report
        probability = report["summary"]["stable_dual_lift"]["mean"]
        print(f"{output_stem.name} mass={scale:g} success={probability:.3f}", flush=True)
    return results


def plot_robustness(report, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = report["mass_scales"]
    levels = report["results"]
    first = levels[f"{scales[0]:g}"]
    labels = [*first["stable_dual_lift"]["algorithms"], "behavior_clone"]
    figure, (success, height) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for index, label in enumerate(labels):
        success_points, height_points = [], []
        for scale in scales:
            data = levels[f"{scale:g}"]
            if label == "behavior_clone":
                success_points.append(data["behavior_clone"]["stable_dual_lift"]["mean"])
                height_points.append(data["behavior_clone"]["peak_lift_height_m"]["mean"])
            else:
                success_points.append(data["stable_dual_lift"]["algorithms"][label]["mean"])
                height_points.append(data["peak_lift_height_m"]["algorithms"][label]["mean"])
        style = {"marker": ("o", "s", "^", "D")[index % 4], "fillstyle": "none"}
        success.plot(scales, success_points, label=label, **style)
        height.plot(scales, height_points, **style)
    success.set_ylabel("Stable lift fraction")
    success.legend(ncols=2, fontsize=9)
    height.set_ylabel("Mean peak height (m)")
    height.set_xlabel("Pot mass multiplier")
    height.set_xticks(scales)
    for axis in (success, height):
        axis.grid(alpha=0.25)
    figure.suptitle("TwoArmLift | evaluation-only mass shift")
    figure.tight_layout()
    figure.savefig(out, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--behavior-clone", type=Path, required=True)
    parser.add_argument("--mass-scales", nargs="+", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=62000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scales = sorted(set(args.mass_scales))
    if (
        len(scales) != len(args.mass_scales)
        or not all(math.isfinite(scale) and scale > 0 for scale in scales)
        or args.episodes < 1
        or args.workers < 1
        or args.seed < 0
    ):
        parser.error(
            "use distinct positive mass scales, positive episodes/workers and a nonnegative seed"
        )

    args.out.mkdir(parents=True, exist_ok=False)
    paths = [*args.checkpoints, args.behavior_clone]
    names = [f"online_{index:02d}" for index in range(len(args.checkpoints))] + ["behavior_clone"]
    jobs = [
        (path, args.out / name, scales, args.seed, args.episodes)
        for path, name in zip(paths, names, strict=True)
    ]
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(jobs)),
        mp_context=get_context("spawn"),
    ) as pool:
        runs = list(pool.map(evaluate_checkpoint, jobs))

    report = {
        "mass_scales": scales,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "results": {},
    }
    for scale in scales:
        trained = [run[scale][0] for run in runs[:-1]]
        stable = compare_results(trained, "stable_dual_lift")
        height = compare_results(trained, "peak_lift_height_m")
        _, control = runs[-1][scale]
        base = control["training_initialization"]
        online = stable["training_initialization"]
        if (
            not base
            or not online
            or base["method"] != "behavior_cloning"
            or online["method"] != "behavior_cloning_then_online"
            or base["demonstration_sha256"] != online["demonstration_sha256"]
            or base["demonstration_joint_steps"] != online["demonstration_joint_steps"]
            or control["training_env_kwargs"] != stable["training_env_kwargs"]
            or control["benchmark_id"] != stable["benchmark_id"]
        ):
            raise ValueError(
                "Behavior clone control must match demonstration and environment settings"
            )
        report["results"][f"{scale:g}"] = {
            "stable_dual_lift": stable,
            "peak_lift_height_m": height,
            "behavior_clone": control["summary"],
        }
    save_result(report, args.out / "summary.json")
    plot_robustness(report, args.out / "mass_robustness.png")
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
