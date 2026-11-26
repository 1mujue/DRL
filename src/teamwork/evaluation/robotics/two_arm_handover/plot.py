"""Plot a recorded two_arm_handover episode."""

from pathlib import Path

from ..trace import read_trace


def plot_handover_trace(evaluation_path, out, *, seed=None):
    """Show grasp order and the scored receiver-only hold from a saved trace."""
    import matplotlib

    from ...metrics.two_arm_handover.rules import HANDOVER_HEIGHT_M, HANDOVER_HOLD_STEPS

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, episode = read_trace(evaluation_path, "two_arm_handover", seed)
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
