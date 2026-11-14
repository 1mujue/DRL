"""MAAC 接入公共接口；训练细节在同目录 trainer.py。"""

import torch

from ..base import Algorithm, Policy
from . import trainer


class MAACPolicy(Policy):
    def __init__(self, actors):
        self.actors = actors

    def act(self, observations, deterministic=True):
        with torch.no_grad():
            actions = {}
            for agent, observation in observations.items():
                logits = self.actors[agent].actor(
                    torch.as_tensor(observation, dtype=torch.float32)
                )
                actions[agent] = (
                    int(logits.argmax())
                    if deterministic
                    else int(torch.distributions.Categorical(logits=logits).sample())
                )
            return actions


class MAACAlgorithm(Algorithm):
    name = "maac"

    def train(self, config, out):
        if config["algorithm"] != self.name:
            raise ValueError("Algorithm configuration does not match selected implementation")
        return trainer.train(config, out)

    def load_policy(self, checkpoint, env, observations):
        actors, data = trainer.load_policy(checkpoint, env, observations)
        if data["config"]["algorithm"] != self.name:
            raise ValueError("Checkpoint algorithm mismatch")
        return MAACPolicy(actors)