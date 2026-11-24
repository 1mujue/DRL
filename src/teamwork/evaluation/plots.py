"""公共可视化写在这里：训练回报曲线、同场景算法对比图。"""

import csv
import json
from pathlib import Path


def plot_comparison(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["algorithms"])
    values = [result["algorithms"][a]["mean"] for a in names]
    errors = [result["algorithms"][a]["std_across_training_seeds"] or 0 for a in names]
    labels = [f"{a}\n(n={result['algorithms'][a]['n_models']} models)" for a in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.set_ylabel(result["metric"]["name"])
    ax.set_title(f"{result['env']} | std across training seeds; n=1 has no error estimate")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curve(csv_path, out):

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No complete episode in CSV")
    for agent in rows[0]:
        if agent not in {"episode", "steps"}:
            plt.plot(
                [int(r["steps"]) for r in rows],
                [float(r[agent]) for r in rows],
                label=agent,
                alpha=0.7,
            )
    plt.xlabel("Joint environment steps")
    plt.ylabel("Episode return")
    plt.legend()
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def _read_trace(evaluation_path, environment, seed):
    result = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    if result["env"] != environment:
        raise ValueError(f"Trajectory plot requires a {environment} evaluation")
    episode = next((row for row in result["episodes"] if seed is None or row["seed"] == seed), None)
    if episode is None:
        raise ValueError(f"Evaluation does not contain episode seed {seed}")
    if "trace" not in episode:
        raise ValueError("Evaluate with --record-trace to plot the trajectory")
    return result, episode


def plot_lift_trace(evaluation_path, out, *, seed=None):
    """Plot the factual grasp, height, and tilt trace of one lifting episode."""
    import matplotlib

    from ..environments.two_arm_lift import LIFT_PHASE_START
    from .metrics import LIFT_HEIGHT_M, LIFT_TILT_DEGREES

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, episode = _read_trace(evaluation_path, "two_arm_lift", seed)
    trace = episode["trace"]
    records = trace["samples"]
    time = [step / trace["control_hz"] for step in range(len(records))]
    figure, (height, tilt, grasp) = plt.subplots(3, 1, sharex=True, figsize=(8.5, 6))
    height.plot(time, [value["lift_height_m"] for value in records], color="#14647d")
    height.axhline(LIFT_HEIGHT_M, linestyle="--", color="#808080")
    height.set_ylabel("Pot height (m)")
    tilt.plot(time, [value["tilt_degrees"] for value in records], color="#a3541f")
    tilt.axhline(LIFT_TILT_DEGREES, linestyle="--", color="#808080")
    tilt.set_ylabel("Pot tilt (degrees)")
    for index, color in enumerate(("#14647d", "#a3541f")):
        offset = index * 1.3
        grasp.step(
            time,
            [offset + int(value["grasping"][index]) for value in records],
            where="post",
            label=f"Arm {index}",
            color=color,
        )
    grasp.set_yticks((0, 1, 1.3, 2.3), ("Off", "Held", "Off", "Held"))
    grasp.set_ylabel("Handle grasp")
    grasp.set_xlabel("Simulation time (s)")
    grasp.legend(loc="upper left", ncols=2)
    mass = result["benchmark"]["env_kwargs"].get("pot_mass_scale", 1)
    status = "stable lift" if episode["metrics"]["stable_dual_lift"] else "failed lift"
    title = f"{result['algorithm']} | mass ×{mass:g} | seed {episode['seed']} | {status}"
    if "lift_timing" in result:
        mode = result["lift_timing"]["mode"]
        delay = result["benchmark"]["gripper_delay"]
        title = f"{mode} | {delay['arm']} delay {delay['steps'] / trace['control_hz']:g} s | seed {episode['seed']} | {status}"
        for arm, color, style in (("arm_0", "#14647d", "--"), ("arm_1", "#a3541f", ":")):
            first = next(
                (
                    time[step]
                    for step, sample in enumerate(records)
                    if sample.get("control", {}).get("phase_steps", {}).get(arm, -1)
                    >= LIFT_PHASE_START
                ),
                None,
            )
            if first is not None:
                for axis in (height, tilt, grasp):
                    axis.axvline(first, linestyle=style, color=color, alpha=0.75)
                height.text(
                    first,
                    0.96 if arm == "arm_0" else 0.82,
                    f"{arm} lift phase",
                    transform=height.get_xaxis_transform(),
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=color,
                )
    figure.suptitle(title)
    for axis in (height, tilt, grasp):
        axis.grid(alpha=0.2)
    figure.tight_layout()
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)


def plot_handover_trace(evaluation_path, out, *, seed=None):
    """Show grasp order and the scored receiver-only hold from a saved trace."""
    import matplotlib

    from .metrics import HANDOVER_HEIGHT_M, HANDOVER_HOLD_STEPS

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, episode = _read_trace(evaluation_path, "two_arm_handover", seed)
    trace = episode["trace"]
    samples, hz = trace["samples"], trace["control_hz"]
    time = [step / hz for step in range(len(samples))]
    figure, (height, grasp) = plt.subplots(2, 1, sharex=True, figsize=(9, 5), height_ratios=(2, 1))
    height.plot(time, [sample["object_height_m"] for sample in samples], color="#195371")
    height.axhline(HANDOVER_HEIGHT_M, linestyle="--", color="#515d64", label="Required hold height")
    height.set_ylabel("Task height above table (m)")
    for lane, (key, color) in enumerate(
        (("giver_grasp", "#195371"), ("receiver_grasp", "#aa531b"))
    ):
        grasp.fill_between(
            time,
            lane - 0.26,
            lane + 0.26,
            where=[sample[key] for sample in samples],
            step="post",
            color=color,
        )
    grasp.set_yticks((0, 1), ("Giver", "Receiver"))
    grasp.set_ylim(-0.55, 1.55)
    grasp.set_xlabel("Simulation time (s)")
    first = episode["metrics"]["first_handover_step"]
    if first is not None:
        for axis in (height, grasp):
            axis.axvspan(
                (first - HANDOVER_HOLD_STEPS + 1) / hz,
                first / hz,
                alpha=0.20,
                color="#38834b",
                label="Scored receiver-only hold",
            )
    height.legend(loc="upper left")
    for axis in (height, grasp):
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlim(0, time[-1])
    status = f"success at {first / hz:.2f} s" if first is not None else "no completed transfer"
    figure.suptitle(
        f"Two-arm handover  |  {result['algorithm']}  |  seed {episode['seed']}  |  {status}"
    )
    figure.tight_layout()
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
