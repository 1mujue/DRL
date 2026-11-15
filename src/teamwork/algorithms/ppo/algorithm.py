"""将已有 PPO 实现接入公共接口；训练细节位于同目录 trainer.py。"""

import torch

from ..base import Algorithm, Policy
from . import trainer


class PPOPolicy(Policy):
    def __init__(self, models):
        self.models = models

    def act(self, observations, deterministic=True):
        with torch.no_grad():
            actions = {}
            for agent in self.models:
                observation = observations[agent]
                logits = self.models[agent].actor(torch.as_tensor(observation, dtype=torch.float32))
                actions[agent] = (
                    int(logits.argmax())
                    if deterministic
                    else int(torch.distributions.Categorical(logits=logits).sample())
                )
            return actions


class PPOAlgorithm(Algorithm):
    def __init__(self, name):
        self.name = name

    def train(self, config, out):
        if config["algorithm"] != self.name:
            raise ValueError("Algorithm configuration does not match selected implementation")
        return trainer.train(config, out)

    def load_policy(self, checkpoint, env, observations):
        models, _ = trainer.load_policy(checkpoint, env, observations, self.name)
        return PPOPolicy(models)
