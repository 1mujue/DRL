"""环境抽象：沿用 PettingZoo ParallelEnv，只增加任务测量与空间描述。

算法只使用 reset/step/state/空间；task_snapshot 专供评估，不能作为执行策略输入。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pettingzoo import ParallelEnv


@dataclass(frozen=True)
class EnvironmentSpec:
    agents: tuple[str, ...]
    observation_sizes: tuple[int, ...]
    action_counts: tuple[int, ...]
    state_size: int


class Environment(ParallelEnv, ABC):
    @property
    def policy_metadata(self):
        """给规则基线的静态任务信息；不得包含隐藏位置等逐步状态。"""
        return {}

    def describe(self) -> EnvironmentSpec:
        """reset 后调用。当前基线支持固定 agent、向量观测和离散动作。"""
        return EnvironmentSpec(
            tuple(self.possible_agents),
            tuple(self.observation_space(a).shape[0] for a in self.possible_agents),
            tuple(int(self.action_space(a).n) for a in self.possible_agents),
            int(self.state().size),
        )

    @abstractmethod
    def task_snapshot(self) -> dict:
        """给评估器的原始任务事实；此处不决定成功阈值或计算算法得分。"""


def state(env):
    return np.asarray(env.state(), dtype=np.float32).reshape(-1)
