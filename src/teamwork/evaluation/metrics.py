"""固定的场景评估协议：算法不得传入阈值或替换自己的计分函数。

新增场景时增加 MetricSuite、定义和默认设置；变更含义必须递增版本。
原始数据来自 Environment.task_snapshot，策略无法通过此接口获取额外信息。
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

METRICS_VERSION = "1.0.0"
COVERAGE_RADIUS = 0.1
BOUNDARY_LIMIT = 1.0

_DEFAULTS = {
    "spread": {"N": 3, "local_ratio": 0.5, "max_cycles": 50},
    "tag": {"num_good": 1, "num_adversaries": 3, "num_obstacles": 2, "max_cycles": 50},
    "lbf": {"id": "Foraging-8x8-2p-1f-coop-v3"},
    "rware": {"id": "rware-tiny-2ag-v2"},
}


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    direction: str
    meaning: str


def metric(name, unit, direction, meaning):
    return MetricDefinition(name, unit, direction, meaning)


_RETURN = metric("mean_agent_return", "reward/agent/episode", "higher", "各 agent 未折扣回报的平均")
_DEFINITIONS = {
    "spread": (
        _RETURN,
        metric("final_coverage", "fraction", "higher", "最终状态中最近 agent 距离 <0.1 的目标占比"),
        metric("success", "boolean", "higher", "最终状态所有目标均被覆盖；不是途中曾经覆盖"),
        metric("ever_success", "boolean", "higher", "回合任一步全部覆盖，辅助诊断"),
        metric(
            "final_mean_landmark_distance",
            "world_distance",
            "lower",
            "最终各目标到最近 agent 距离的平均",
        ),
        metric(
            "collision_pairs_per_step", "pair/step", "lower", "无序 agent 接触对×步总数 / 回合步数"
        ),
    ),
    "tag": (
        metric(
            "predator_return",
            "reward/agent/episode",
            "higher_for_predators",
            "追捕方各 agent 回报平均",
        ),
        metric("prey_return", "reward/agent/episode", "higher_for_prey", "逃避方各 agent 回报平均"),
        metric(
            "capture_success", "boolean", "higher_for_predators", "回合内每只 prey 至少被接触一次"
        ),
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
        metric(
            "prey_outside_rate", "prey_step_fraction", "lower", "任一坐标绝对值 >1 的 prey×步占比"
        ),
    ),
    "lbf": (
        _RETURN,
        metric(
            "food_collected_fraction", "fraction", "higher", "采集的食物格数 / reset 后实际食物格数"
        ),
        metric("success", "boolean", "higher", "回合结束时剩余食物格数为 0"),
        metric("completion_step", "step", "lower", "完成全部食物采集的步数；未完成为 null"),
    ),
    "rware": (
        _RETURN,
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
    ),
}


class MetricSuite(ABC):
    """每回合创建/重置一个计数器；不要在各算法目录里重复实现这些公式。"""

    def reset(self, snapshot):
        self.initial = snapshot
        self.last = snapshot
        self.steps = 0
        self._reset()

    def _reset(self):
        pass

    def observe(self, snapshot):
        self.steps += 1
        self.last = snapshot
        self._update(snapshot)

    def _update(self, snapshot):
        pass

    @abstractmethod
    def finish(self, returns: dict[str, float], agents: list[str]) -> dict:
        """返回本场景固定指标，不含跨回合平均。"""


def mean_return(returns):
    return float(np.mean(list(returns.values())))


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


_SUITES = {"spread": SpreadMetrics, "tag": TagMetrics, "lbf": LBFMetrics, "rware": RWAREMetrics}


def register_metrics(name, suite, definitions, default_env_kwargs):
    """新环境的协议必须与环境适配器一起提交，不能在每次评估时临时指定。"""
    if name in _SUITES:
        raise ValueError(f"Metrics already registered: {name}; change the protocol version instead")
    _SUITES[name] = suite
    _DEFINITIONS[name] = tuple(definitions)
    _DEFAULTS[name] = deepcopy(default_env_kwargs)


def make_metrics(name) -> MetricSuite:
    return _SUITES[name]()


def protocol_description(name):
    return {
        "environment": name,
        "metrics_version": METRICS_VERSION,
        "default_env_kwargs": deepcopy(_DEFAULTS[name]),
        "metrics": [asdict(d) for d in _DEFINITIONS[name]],
        "coverage_radius": COVERAGE_RADIUS if name == "spread" else None,
        "boundary_limit": BOUNDARY_LIMIT if name == "tag" else None,
        "sampling": "after_each_joint_step; reset state is not a scored step",
    }


def resolved_env_kwargs(name, overrides=None):
    return {**deepcopy(_DEFAULTS[name]), **(overrides or {})}


def summarize(rows):
    """单个模型在测试回合上的均值/样本标准差。null 不当作 0；必须报告有效样本数。"""
    output = {}
    for key in rows[0]["metrics"]:
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        output[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values),
            "episodes": len(rows),
        }
    return output
