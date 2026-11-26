"""Plot a recorded two_arm_lift episode."""

from pathlib import Path

from ..trace import read_trace


def plot_lift_trace(evaluation_path, out, *, seed=None):
    """Plot the factual grasp, height, and tilt trace of one lifting episode."""
    import matplotlib

    from ....environments.robotics.two_arm_lift.environment import LIFT_PHASE_START
    from ...metrics.two_arm_lift.rules import LIFT_HEIGHT_M, LIFT_TILT_DEGREES

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, episode = read_trace(evaluation_path, "two_arm_lift", seed)
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
