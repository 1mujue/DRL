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
        if not isinstance(observations, dict):
            raise TypeError("MAT-Dec observations must map agent IDs to local vectors")
        parameter = next(self.model.parameters())
        local_observations = {}
        for agent, observation in observations.items():
            if agent not in self.model.indices:
                raise ValueError(f"Unknown MAT-Dec agent: {agent}")
            try:
                local = torch.as_tensor(observation, dtype=parameter.dtype, device=parameter.device)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"Invalid MAT-Dec policy observation for {agent}") from error
            index = self.model.indices[agent]
            if (
                local.shape != (self.model.spec["observation_sizes"][index],)
                or not torch.isfinite(local).all()
            ):
                raise ValueError(f"Invalid MAT-Dec policy observation for {agent}")
            local_observations[agent] = local
        actions = {}
        with torch.no_grad():
            for agent, local in local_observations.items():
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
