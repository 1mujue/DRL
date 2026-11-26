"""Fixed per-episode metrics for two_arm_handover."""

from ..base import MEAN_AGENT_RETURN, MetricSuite, mean_return, metric

HANDOVER_HEIGHT_M = 0.10
HANDOVER_HOLD_STEPS = 10
DEFAULT_ENV_KWARGS = {"horizon": 200, "reward_shaping": True, "scene_seed": 0}
DEFINITIONS = (
    MEAN_AGENT_RETURN,
    metric(
        "handover_success",
        "boolean",
        "higher",
        "先由递出臂握持，再共同抓握，接收臂单独保持高出桌面 >0.10 米至少 10 步",
    ),
    metric("first_handover_step", "step", "lower", "首次完成交接的步数；未完成为 null"),
    metric("mutual_grasp_fraction", "fraction", "higher", "两臂同时抓握的步数 / 回合步数"),
    metric("receiver_grasp_fraction", "fraction", "higher", "接收臂抓握手柄的步数 / 回合步数"),
    metric("peak_object_height_m", "m", "higher", "物体相对桌面的最大任务高度"),
)


class TwoArmHandoverMetrics(MetricSuite):
    def _reset(self):
        self.giver_seen = self.initial["giver_grasp"] and not self.initial["receiver_grasp"]
        self.overlap = False
        self.shared_steps = self.receiver_steps = self.hold_steps = 0
        self.first = None
        self.peak_height = -float("inf")

    def _update(self, snapshot):
        giver, receiver = snapshot["giver_grasp"], snapshot["receiver_grasp"]
        height = snapshot["object_height_m"]
        self.peak_height = max(self.peak_height, height)
        self.shared_steps += giver and receiver
        self.receiver_steps += receiver
        if giver and not receiver:
            self.giver_seen = True
            self.overlap = False
        if giver and receiver and self.giver_seen:
            self.overlap = True
        if not giver and not receiver:
            self.giver_seen = self.overlap = False

        received = self.overlap and receiver and not giver and height > HANDOVER_HEIGHT_M
        self.hold_steps = self.hold_steps + 1 if received else 0
        if self.first is None and self.hold_steps >= HANDOVER_HOLD_STEPS:
            self.first = self.steps

    def finish(self, returns, agents):
        return {
            "mean_agent_return": mean_return(returns),
            "handover_success": float(self.first is not None),
            "first_handover_step": self.first,
            "mutual_grasp_fraction": self.shared_steps / self.steps,
            "receiver_grasp_fraction": self.receiver_steps / self.steps,
            "peak_object_height_m": self.peak_height,
        }
