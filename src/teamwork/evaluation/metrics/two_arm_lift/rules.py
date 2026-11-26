"""Fixed per-episode metrics for two_arm_lift."""

from ..base import MEAN_AGENT_RETURN, MetricSuite, mean_return, metric

LIFT_HEIGHT_M = 0.10
LIFT_TILT_DEGREES = 15.0
LIFT_HOLD_STEPS = 10
DEFAULT_ENV_KWARGS = {"horizon": 200, "reward_shaping": True}
DEFINITIONS = (
    MEAN_AGENT_RETURN,
    metric(
        "stable_dual_lift",
        "boolean",
        "higher",
        "双臂握住各自把手、锅底高出桌面 >0.10 米且倾角 ≤15°，连续 ≥10 步",
    ),
    metric("first_stable_lift_step", "step", "lower", "首次完成稳定抬升的步数；失败为 null"),
    metric("dual_grasp_fraction", "fraction", "higher", "两臂同时握住各自把手的步数 / 回合步数"),
    metric("peak_lift_height_m", "m", "higher", "锅底相对桌面的最大高度"),
)


class TwoArmLiftMetrics(MetricSuite):
    def _reset(self):
        self.grasp_steps = 0
        self.hold_steps = 0
        self.first = None
        self.peak = -float("inf")

    def _update(self, snapshot):
        grasp = all(snapshot["grasping"])
        self.grasp_steps += grasp
        height = snapshot["lift_height_m"]
        self.peak = max(self.peak, height)
        stable = grasp and height > LIFT_HEIGHT_M and snapshot["tilt_degrees"] <= LIFT_TILT_DEGREES
        self.hold_steps = self.hold_steps + 1 if stable else 0
        if self.first is None and self.hold_steps >= LIFT_HOLD_STEPS:
            self.first = self.steps

    def finish(self, returns, agents):
        return {
            "mean_agent_return": mean_return(returns),
            "stable_dual_lift": float(self.first is not None),
            "first_stable_lift_step": self.first,
            "dual_grasp_fraction": self.grasp_steps / self.steps,
            "peak_lift_height_m": self.peak,
        }
