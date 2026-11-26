"""环境抽象：沿用 PettingZoo ParallelEnv，只增加任务测量与空间描述。

算法只使用 reset/step/state/空间；task_snapshot 专供评估，不能作为执行策略输入。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


@dataclass(frozen=True)
class EnvironmentSpec:
    agents: tuple[str, ...]
    observation_sizes: tuple[int, ...]
    action_counts: tuple[int, ...]
    state_size: int


@dataclass(frozen=True)
class ContinuousActionSpec:
    shape: tuple[int, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    dtype: str


@dataclass(frozen=True)
class ContinuousEnvironmentSpec:
    agents: tuple[str, ...]
    observation_sizes: tuple[int, ...]
    action_spaces: tuple[ContinuousActionSpec, ...]
    state_size: int


class Environment(ParallelEnv, ABC):
    @property
    def policy_metadata(self):
        """给规则基线的静态任务信息；不得包含隐藏位置等逐步状态。"""
        return {}

    def describe(self) -> EnvironmentSpec | ContinuousEnvironmentSpec:
        """在 reset 后记录空间；保留已有离散场景的描述格式。"""
        agents = tuple(self.possible_agents)
        observations = tuple(self.observation_space(a).shape[0] for a in agents)
        spaces = tuple(self.action_space(a) for a in agents)
        state_size = state(self).size
        if all(isinstance(space, Discrete) for space in spaces):
            return EnvironmentSpec(
                agents, observations, tuple(int(s.n) for s in spaces), state_size
            )
        if all(isinstance(space, Box) for space in spaces):
            return ContinuousEnvironmentSpec(
                agents,
                observations,
                tuple(
                    ContinuousActionSpec(
                        s.shape,
                        tuple(map(float, s.low.flat)),
                        tuple(map(float, s.high.flat)),
                        str(s.dtype),
                    )
                    for s in spaces
                ),
                state_size,
            )
        raise TypeError("Environment requires homogeneous action-space types: Discrete or Box")

    @abstractmethod
    def task_snapshot(self) -> dict:
        """给评估器的原始任务事实；此处不决定成功阈值或计算算法得分。"""


def state(env):
    return np.asarray(env.state(), dtype=np.float32).reshape(-1)
