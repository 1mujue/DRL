"""将已有 PPO 实现接入公共接口；训练细节位于同目录 trainer.py。"""

import torch

from ..base import Algorithm, Policy
from ..names import canonical_algorithm_name
from . import trainer


class PPOPolicy(Policy):
    def __init__(self, models):
        self.models = models

    def act(self, observations, deterministic=True):
        with torch.no_grad():
            actions = {}
            for agent, model in self.models.items():
                observation = torch.as_tensor(observations[agent], dtype=torch.float32)
                distribution = model.distribution(observation)
                action = distribution.mode if deterministic else distribution.sample()
                actions[agent] = model.environment_action(distribution, action)
            return actions


class PPOAlgorithm(Algorithm):
    def __init__(self, name):
        self.name = canonical_algorithm_name(name)
        self.action_kind = "continuous" if self.name in trainer.CONTINUOUS else "discrete"

    def train(self, config, out):
        if canonical_algorithm_name(config["algorithm"]) != self.name:
            raise ValueError("Algorithm configuration does not match selected implementation")
        return trainer.train(config, out)

    def load_policy(self, checkpoint, env, observations):
        models, _ = trainer.load_policy(checkpoint, env, observations, self.name)
        return PPOPolicy(models)
