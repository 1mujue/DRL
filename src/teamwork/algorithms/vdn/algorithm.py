"""Project adapter for a discrete-action VDN learner."""

import torch

from ..base import Algorithm, Policy
from .trainer import load_checkpoint, train


class VDNPolicy(Policy):
    """Execution policy that uses only each agent's local observation."""

    def __init__(self, networks):
        self.networks = networks

    def act(self, observations, deterministic=True):
        """Greedy evaluation; optional 5% epsilon-greedy exploration."""
        actions = {}
        with torch.no_grad():
            for agent, observation in observations.items():
                q_values = self.networks[agent](torch.as_tensor(observation, dtype=torch.float32))
                if not deterministic and torch.rand(()).item() < 0.05:
                    action = torch.randint(q_values.numel(), ()).item()
                else:
                    action = q_values.argmax().item()
                actions[agent] = int(action)
        return actions


class VDNAlgorithm(Algorithm):
    """VDN with per-agent utility networks and a shared team reward."""

    name = "vdn"

    def train(self, config, out):
        if config.get("algorithm") != self.name:
            raise ValueError("Algorithm configuration does not match VDN")
        return train(config, out)

    def load_policy(self, checkpoint, env, observations):
        networks, _ = load_checkpoint(checkpoint, env, observations)
        return VDNPolicy(networks)


__all__ = ["VDNAlgorithm", "VDNPolicy"]
