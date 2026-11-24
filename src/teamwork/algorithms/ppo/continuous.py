"""Bounded Gaussian actions for continuous PPO."""

import math

from torch.distributions import Normal
from torch.nn.functional import softplus


class SquashedNormal:
    """Store pre-tanh samples so the PPO ratio needs no inverse of saturated actions."""

    def __init__(self, mean, std, center, scale):
        self.base = Normal(mean, std)
        self.center = center
        self.scale = scale

    @property
    def mode(self):
        return self.base.mean

    def sample(self):
        return self.base.sample()

    def rsample(self):
        return self.base.rsample()

    def action(self, latent):
        return self.center + self.scale * latent.tanh()

    def log_prob(self, latent):
        log_jacobian = self.scale.log() + 2 * (math.log(2) - latent - softplus(-2 * latent))
        return (self.base.log_prob(latent) - log_jacobian).sum(dim=-1)

    def entropy(self):
        # One reparameterized sample estimates entropy in the environment's action space.
        return -self.log_prob(self.base.rsample())
