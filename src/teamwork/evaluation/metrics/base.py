"""Episode scoring contract and shared aggregation helpers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    direction: str
    meaning: str


def metric(name, unit, direction, meaning):
    return MetricDefinition(name, unit, direction, meaning)


MEAN_AGENT_RETURN = metric(
    "mean_agent_return", "reward/agent/episode", "higher", "各 agent 未折扣回报的平均"
)


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
