"""Evaluate lift start rules with identical frozen actors under a one-arm closure delay."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from statistics import mean

from teamwork.environments.two_arm_lift import LIFT_PHASE_START
from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import evaluate, save_result
from teamwork.evaluation.lift_timing import ARMS, LIFT_TIMING_MODES


def run_case(job):
    checkpoint, index, delay, arm, seed, episodes, out = job
    paths = {}
    for mode in LIFT_TIMING_MODES:
        target = out / f"delay_{delay}" / f"{mode}_model_{index:02d}.json"
        report = evaluate(
            "two_arm_lift",
            checkpoint=checkpoint,
            seed=seed,
            episodes=episodes,
            pot_mass_scale=1.0,
            lift_timing=mode,
            gripper_delay_steps=delay,
            delayed_arm=arm,
            record_trace=True,
        )
        save_result(report, target)
        paths[mode] = target
        print(
            f"model={index} seed={report['training_seed']} delay={delay} mode={mode} "
            f"success={report['summary']['stable_dual_lift']['mean']:.3f}",
            flush=True,
        )
    return delay, paths


def episode_timing(episode):
    start = {}
    early = False
    for step, sample in enumerate(episode["trace"]["samples"][1:], start=1):
        control = sample["control"]
        for arm in ARMS:
            if arm not in start and control["phase_steps"][arm] >= LIFT_PHASE_START:
                start[arm] = step
                early |= not all(control["ready"].values())
    if len(start) == 2:
        skew = abs(start[ARMS[0]] - start[ARMS[1]])
    else:
        skew = None
    return {
        "start_steps": start,
        "entered_lift_before_both_ready": early,
        "command_skew_steps": skew,
        "success": int(episode["metrics"]["stable_dual_lift"]),
    }


def summarize_mode(paths):
    group = compare_results(paths, "stable_dual_lift")
    episodes = [
        episode_timing(row)
        for path in paths
        for row in json.loads(path.read_text(encoding="utf-8"))["episodes"]
    ]
    skews = [
        entry["command_skew_steps"] for entry in episodes if entry["command_skew_steps"] is not None
    ]
    return {
        "comparison": group,
        "n_episodes": len(episodes),
        "successful": sum(entry["success"] for entry in episodes),
        "started_before_both_ready": sum(
            entry["entered_lift_before_both_ready"] for entry in episodes
        ),
        "both_arms_started": len(skews),
        "mean_onset_skew_steps": mean(skews) if skews else None,
        "raw_reports": list(map(str, paths)),
    }


def plot_study(report, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delays = report["delays"]
    seconds = [delay / report["control_hz"] for delay in delays]
    results = report["results"]
    figure, (success, early) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    styles = (
        ("Clock", "#575d67", "o"),
        ("Local grasp", "#a95428", "s"),
        ("Mutual handshake", "#166b81", "D"),
    )
    for mode, (label, color, marker) in zip(LIFT_TIMING_MODES, styles, strict=True):
        modes = [results[str(delay)][mode] for delay in delays]
        success.plot(
            seconds,
            [row["successful"] / row["n_episodes"] for row in modes],
            marker=marker,
            color=color,
            label=label,
            fillstyle="none",
        )
        early.plot(
            seconds,
            [row["started_before_both_ready"] / row["n_episodes"] for row in modes],
            marker=marker,
            color=color,
            fillstyle="none",
        )
    success.set_ylabel("Stable lift fraction")
    success.legend(loc="lower left")
    early.set_ylabel("Lift before both ready")
    early.set_xlabel(f"{report['delayed_arm']} gripper closure delay (s)")
    early.set_xticks(seconds)
    for axis in (success, early):
        axis.set_ylim(-0.04, 1.04)
        axis.grid(alpha=0.25)
    figure.suptitle("TwoArmLift | frozen actors and matched placements")
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 15, 30, 45])
    parser.add_argument("--delay-arm", choices=ARMS, default="arm_1")
    parser.add_argument("--seed", type=int, default=66000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    delays = sorted(set(args.delays))
    if (
        len(delays) != len(args.delays)
        or min(delays) < 0
        or min(args.episodes, args.workers) < 1
        or args.seed < 0
    ):
        parser.error("use distinct nonnegative delays and seed, with positive episodes and workers")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "delays": delays,
        "control_hz": 20,
        "delayed_arm": args.delay_arm,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "checkpoints": list(map(str, args.checkpoints)),
        "results": {},
    }
    save_result(report, args.out / "study_plan.json")
    jobs = [
        (checkpoint, index, delay, args.delay_arm, args.seed, args.episodes, args.out)
        for (index, checkpoint), delay in product(enumerate(args.checkpoints), delays)
    ]
    grouped = {delay: {mode: [] for mode in LIFT_TIMING_MODES} for delay in delays}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for delay, paths in pool.map(run_case, jobs):
            for mode, path in paths.items():
                grouped[delay][mode].append(path)
    for delay, modes in grouped.items():
        summaries = {mode: summarize_mode(paths) for mode, paths in modes.items()}
        if len({row["comparison"]["benchmark_id"] for row in summaries.values()}) != 1:
            raise ValueError("Evaluation conditions differ across timing rules")
        model_hashes = [
            [json.loads(path.read_text(encoding="utf-8"))["model_sha256"] for path in paths]
            for paths in modes.values()
        ]
        if any(hashes != model_hashes[0] for hashes in model_hashes[1:]):
            raise ValueError("Timing modes used different model weights")
        report["results"][str(delay)] = summaries
    save_result(report, args.out / "summary.json")
    plot_study(report, args.out / "timing_vs_delay.png")
    print(args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
