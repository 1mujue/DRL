"""MAT-Dec adapter for the project's Algorithm / Policy API."""

import torch
from torch.distributions import Categorical

from ..base import Algorithm, Policy
from .trainer import load_checkpoint, train


class MATDecPolicy(Policy):
    def __init__(self, model):
        self.model = model.eval()

    def act(self, observations, deterministic=True):
        """No critic call, joint observation tensor, or previous-agent action."""
        device = next(self.model.parameters()).device
        actions = {}
        with torch.no_grad():
            for agent, observation in observations.items():
                local = torch.as_tensor(observation, dtype=torch.float32, device=device)
                index = self.model.indices[agent]
                if local.shape != (self.model.spec["observation_sizes"][index],):
                    raise ValueError(f"Invalid MAT-Dec policy observation for {agent}")
                logits = self.model.local_logits(agent, local)
                action = logits.argmax() if deterministic else Categorical(logits=logits).sample()
                actions[agent] = int(action.item())
        return actions


class MATDecAlgorithm(Algorithm):
    name = "mat_dec"

    def train(self, config, out):
        if config.get("algorithm") != self.name:
            raise ValueError("Algorithm configuration does not match MAT-Dec")
        return train(config, out)

    def load_policy(self, checkpoint, env, observations):
        model, _ = load_checkpoint(checkpoint, env, observations)
        return MATDecPolicy(model)
