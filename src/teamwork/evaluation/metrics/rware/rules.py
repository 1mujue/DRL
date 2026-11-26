"""Fixed per-episode metrics for rware."""

from ..base import MEAN_AGENT_RETURN, MetricSuite, mean_return, metric

DEFAULT_ENV_KWARGS = {"id": "rware-tiny-2ag-v2"}
DEFINITIONS = (
    MEAN_AGENT_RETURN,
    metric(
        "deliveries",
        "shelf_delivery/episode",
        "higher",
        "回合内请求货架交付次数，可包含同一货架再次被请求",
    ),
    metric(
        "deliveries_per_100_steps",
        "shelf_delivery/100_steps",
        "higher",
        "100 × 交付次数 / 实际交互步数",
    ),
)


class RWAREMetrics(MetricSuite):
    def _reset(self):
        self.deliveries = 0.0

    def _update(self, snapshot):
        self.deliveries += snapshot["deliveries_this_step"]

    def finish(self, returns, agents):
        return {
            "mean_agent_return": mean_return(returns),
            "deliveries": self.deliveries,
            "deliveries_per_100_steps": 100 * self.deliveries / self.steps,
        }
