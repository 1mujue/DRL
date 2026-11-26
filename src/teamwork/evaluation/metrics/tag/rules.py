"""Fixed per-episode metrics for tag."""

import numpy as np

from ..base import MetricSuite, metric

BOUNDARY_LIMIT = 1.0
DEFAULT_ENV_KWARGS = {"num_good": 1, "num_adversaries": 3, "num_obstacles": 2, "max_cycles": 50}
DEFINITIONS = (
    metric(
        "predator_return",
        "reward/agent/episode",
        "higher_for_predators",
        "追捕方各 agent 回报平均",
    ),
    metric("prey_return", "reward/agent/episode", "higher_for_prey", "逃避方各 agent 回报平均"),
    metric("capture_success", "boolean", "higher_for_predators", "回合内每只 prey 至少被接触一次"),
    metric(
        "captured_prey_fraction",
        "fraction",
        "higher_for_predators",
        "回合内曾被接触的不同 prey 占比",
    ),
    metric(
        "first_contact_step",
        "step",
        "lower_for_predators",
        "首次跨角色接触的步数；未接触为 null",
    ),
    metric(
        "contact_pairs_per_step",
        "pair/step",
        "higher_for_predators",
        "跨角色接触对×步总数 / 步数",
    ),
    metric("prey_outside_rate", "prey_step_fraction", "lower", "任一坐标绝对值 >1 的 prey×步占比"),
)


class TagMetrics(MetricSuite):
    def _reset(self):
        roles = self.initial["predators"]
        if not roles.any() or roles.all():
            raise ValueError("Tag evaluation requires both predators and prey")
        self.caught = np.zeros(int((~roles).sum()), dtype=bool)
        self.contacts = 0
        self.first = None
        self.outside = 0

    def _update(self, snapshot):
        positions, sizes, roles = snapshot["positions"], snapshot["sizes"], snapshot["predators"]
        distance = np.linalg.norm(positions[roles, None, :] - positions[None, ~roles, :], axis=2)
        contact = distance < sizes[roles, None] + sizes[None, ~roles]
        self.caught |= contact.any(axis=0)
        self.contacts += int(contact.sum())
        if self.first is None and contact.any():
            self.first = self.steps
        self.outside += int(np.any(np.abs(positions[~roles]) > BOUNDARY_LIMIT, axis=1).sum())

    def finish(self, returns, agents):
        roles = self.last["predators"]
        values = np.array([returns[a] for a in agents])
        return {
            "predator_return": float(values[roles].mean()),
            "prey_return": float(values[~roles].mean()),
            "capture_success": float(self.caught.all()),
            "captured_prey_fraction": float(self.caught.mean()),
            "first_contact_step": self.first,
            "contact_pairs_per_step": self.contacts / self.steps,
            "prey_outside_rate": self.outside / (self.steps * len(self.caught)),
        }
