"""Continuous actor with learned pose feedback and a per-control-step bias."""

import torch
from torch import nn


class LiftPoseActor(nn.Module):
    def __init__(self, observation_size, horizon):
        super().__init__()
        self.horizon = horizon
        self.feedback = nn.Linear(6, 7, bias=False)
        self.clock_bias = nn.Embedding(horizon + 1, 7)
        self.residual = nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(), nn.Linear(64, 7))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def time_index(self, observations):
        return (observations[..., -1] * self.horizon).round().long().clamp(0, self.horizon)

    def forward(self, observations):
        errors = observations[..., -7:-1]
        return (
            self.feedback(errors)
            + self.clock_bias(self.time_index(observations))
            + self.residual(observations)
        )

    @torch.no_grad()
    def fit_demonstrations(self, observations, commands):
        """Fit the local feedback and clock terms to demonstrated commands."""
        errors = observations[:, -7:-1].double()
        commands = commands.double()
        steps = self.time_index(observations)
        counts = torch.bincount(steps, minlength=self.horizon + 1).unsqueeze(-1)
        if (counts[:-1] == 0).any():
            raise ValueError("Demonstrations must cover every control step")
        errors_mean = errors.new_zeros((self.horizon + 1, 6)).index_add_(0, steps, errors)
        commands_mean = commands.new_zeros((self.horizon + 1, 7)).index_add_(0, steps, commands)
        errors_mean /= counts.clamp_min(1)
        commands_mean /= counts.clamp_min(1)
        # Within each step, the clock bias cancels. A small ridge term handles quiet axes.
        x = errors - errors_mean[steps]
        y = commands - commands_mean[steps]
        weights = torch.linalg.solve(
            x.T @ x + torch.eye(6, dtype=x.dtype, device=x.device) * 1e-8, x.T @ y
        )
        biases = commands_mean - errors_mean @ weights
        self.feedback.weight.copy_(weights.T)
        self.clock_bias.weight.copy_(biases)
        self.clock_bias.weight[-1].copy_(biases[-2])
        return (self(observations) - commands).square().mean().sqrt().item()
