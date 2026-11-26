"""Fixed per-episode metrics for lbf."""

from ..base import MEAN_AGENT_RETURN, MetricSuite, mean_return, metric

DEFAULT_ENV_KWARGS = {"id": "Foraging-8x8-2p-1f-coop-v3"}
DEFINITIONS = (
    MEAN_AGENT_RETURN,
    metric(
        "food_collected_fraction", "fraction", "higher", "采集的食物格数 / reset 后实际食物格数"
    ),
    metric("success", "boolean", "higher", "回合结束时剩余食物格数为 0"),
    metric("completion_step", "step", "lower", "完成全部食物采集的步数；未完成为 null"),
)


class LBFMetrics(MetricSuite):
    def _reset(self):
        if self.initial["food_remaining"] <= 0:
            raise ValueError("LBF benchmark requires at least one initial food item")

    def finish(self, returns, agents):
        remaining = self.last["food_remaining"]
        return {
            "mean_agent_return": mean_return(returns),
            "food_collected_fraction": 1 - remaining / self.initial["food_remaining"],
            "success": float(remaining == 0),
            "completion_step": self.steps if remaining == 0 else None,
        }
