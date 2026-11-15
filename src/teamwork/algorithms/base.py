"""算法接入合同：统一调用入口，不强制不同算法使用相同的采样/更新循环。"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Policy(ABC):
    """评估只调用 act；模型内部可使用任意网络，禁止读取环境隐藏状态。"""

    def reset(self):
        """每个测试回合调用一次；循环策略在此清空 hidden state。前馈策略无需实现。"""

    @abstractmethod
    def act(self, observations: dict[str, np.ndarray], deterministic=True) -> dict[str, int]:
        """输入局部观测，返回受控 agent 动作；固定对手动作由公共评估器补全。"""


class Algorithm(ABC):
    """每位成员实现一个或几个子类；算法拥有自己的训练循环。"""

    @abstractmethod
    def train(self, config: dict, out: str | Path) -> Path:
        """按联合环境交互步计预算；保存 config.json、episodes.csv、模型，返回模型路径。"""

    @abstractmethod
    def load_policy(self, checkpoint: str | Path, env, observations: dict) -> Policy:
        """加载执行策略。集中训练信息只能用于训练，不能泄漏到 act 的输入。"""
