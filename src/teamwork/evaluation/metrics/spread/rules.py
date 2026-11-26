"""Fixed per-episode metrics for spread."""

import numpy as np

from ..base import MEAN_AGENT_RETURN, MetricSuite, mean_return, metric

COVERAGE_RADIUS = 0.1
DEFAULT_ENV_KWARGS = {"N": 3, "local_ratio": 0.5, "max_cycles": 50}
DEFINITIONS = (
    MEAN_AGENT_RETURN,
    metric("final_coverage", "fraction", "higher", "最终状态中最近 agent 距离 <0.1 的目标占比"),
    metric("success", "boolean", "higher", "最终状态所有目标均被覆盖；不是途中曾经覆盖"),
    metric("ever_success", "boolean", "higher", "回合任一步全部覆盖，辅助诊断"),
    metric(
        "final_mean_landmark_distance",
        "world_distance",
        "lower",
        "最终各目标到最近 agent 距离的平均",
    ),
    metric("collision_pairs_per_step", "pair/step", "lower", "无序 agent 接触对×步总数 / 回合步数"),
)


def spread_distances(snapshot):
    delta = snapshot["landmarks"][:, None, :] - snapshot["positions"][None, :, :]
    return np.linalg.norm(delta, axis=2).min(axis=1)


class SpreadMetrics(MetricSuite):
    def _reset(self):
        self.contacts = 0
        self.ever_success = False

    def _update(self, snapshot):
        positions, sizes = snapshot["positions"], snapshot["sizes"]
        distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        self.contacts += int(np.triu(distance < sizes[:, None] + sizes[None, :], k=1).sum())
        self.ever_success |= bool(np.all(spread_distances(snapshot) < COVERAGE_RADIUS))

    def finish(self, returns, agents):
        distances = spread_distances(self.last)
        return {
            "mean_agent_return": mean_return(returns),
            "final_coverage": float(np.mean(distances < COVERAGE_RADIUS)),
            "success": float(np.all(distances < COVERAGE_RADIUS)),
            "ever_success": float(self.ever_success),
            "final_mean_landmark_distance": float(distances.mean()),
            "collision_pairs_per_step": self.contacts / self.steps,
        }
