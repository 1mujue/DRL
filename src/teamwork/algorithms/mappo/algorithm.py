"""MAPPO 的公共接口；执行策略仅使用 actor 和局部观测。"""

from dataclasses import asdict

import torch

from ..base import Algorithm, Policy
from .networks import MAPPOModel


class MAPPOPolicy(Policy):
    def __init__(self, model):
        self.model = model

    def act(self, observations, deterministic=True):
        with torch.no_grad():
            logits = self.model.local_logits(observations)
            return {
                agent: int(
                    scores.argmax()
                    if deterministic
                    else torch.distributions.Categorical(logits=scores).sample()
                )
                for agent, scores in logits.items()
            }


class MAPPOAlgorithm(Algorithm):
    def train(self, config, out):
        from .trainer import train

        if config["algorithm"] != "mappo_ff":
            raise ValueError("Feedforward MAPPO configuration requires algorithm='mappo_ff'")
        return train(config, out)

    def load_policy(self, checkpoint, env, observations):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data.get("format") != "teamwork.mappo.v1":
            raise ValueError("Unsupported MAPPO checkpoint format")
        spec = env.describe()
        if data["spec"] != asdict(spec):
            raise ValueError("Checkpoint EnvironmentSpec does not match the evaluation environment")
        if set(observations) != set(spec.agents):
            raise ValueError("MAPPO evaluation requires the saved fixed agent population")
        model = MAPPOModel(spec, data["config"])
        model.load_state_dict(data["model"])
        model.eval()
        return MAPPOPolicy(model)
