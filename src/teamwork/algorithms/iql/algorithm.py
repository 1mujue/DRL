"""IQL 公共算法接口与只访问局部观测的执行策略。"""

import torch

from ..base import Algorithm, Policy
from . import trainer


class IQLPolicy(Policy):
    def __init__(self, models, epsilon):
        self.models = models
        self.epsilon = epsilon

    def act(self, observations, deterministic=True):
        actions = {}
        with torch.no_grad():
            for agent, model in self.models.items():
                values = model(torch.as_tensor(observations[agent], dtype=torch.float32))
                explore = not deterministic and torch.rand(()).item() < self.epsilon
                actions[agent] = (
                    int(torch.randint(values.numel(), ()).item())
                    if explore
                    else int(values.argmax().item())
                )
        return actions


class IQLAlgorithm(Algorithm):
    def train(self, config, out):
        return trainer.train(config, out)

    def load_policy(self, checkpoint, env, observations):
        models, data = trainer.load_policy(checkpoint, env, observations)
        return IQLPolicy(models, data["config"]["eval_epsilon"])
