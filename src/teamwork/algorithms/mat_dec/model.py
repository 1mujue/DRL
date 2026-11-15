"""Project implementation of the MAT decentralized-actor architecture.

The official MAT ``dec_actor=True`` branch uses local MLP actors, while an
unmasked Transformer encoder estimates training values from joint observations.
The critic is never on the execution path. This module uses standard PyTorch
layers rather than importing another repository's training framework.
"""

import math

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical


def _linear(inputs, outputs, gain=0.01):
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _actor(observation_size, actions, width):
    return nn.Sequential(
        nn.LayerNorm(observation_size),
        _linear(observation_size, width, math.sqrt(2)),
        nn.GELU(),
        nn.LayerNorm(width),
        _linear(width, width, math.sqrt(2)),
        nn.GELU(),
        nn.LayerNorm(width),
        _linear(width, actions),
    )


def environment_spec(env, observations):
    """Read spaces and local observations only; never read hidden state."""
    names = list(env.possible_agents)
    if not names or len(set(names)) != len(names) or set(env.agents) != set(names):
        raise ValueError("MAT-Dec requires a fixed, nonempty agent population")
    sizes, actions = [], []
    for agent in names:
        observation_space, action_space = env.observation_space(agent), env.action_space(agent)
        if not isinstance(observation_space, Box) or len(observation_space.shape) != 1:
            raise ValueError("MAT-Dec requires one-dimensional vector observations")
        if not isinstance(action_space, Discrete) or action_space.start != 0:
            raise ValueError("MAT-Dec requires zero-based discrete action spaces")
        observation = np.asarray(observations[agent], dtype=np.float32)
        if observation.size == 0 or observation.shape != observation_space.shape:
            raise ValueError(f"Invalid MAT-Dec observation shape for {agent}")
        if not np.isfinite(observation).all():
            raise ValueError(f"Nonfinite MAT-Dec observation for {agent}")
        sizes.append(int(observation.size))
        actions.append(int(action_space.n))
    return {"agents": names, "observation_sizes": sizes, "action_counts": actions}


class EncoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        # Zero dropout keeps rollout and PPO likelihood evaluation consistent.
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        nn.init.orthogonal_(self.attention.in_proj_weight, gain=0.01)
        nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.orthogonal_(self.attention.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.attention.out_proj.bias)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            _linear(width, width, math.sqrt(2)), nn.GELU(), _linear(width, width)
        )

    def forward(self, tokens):
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_attention(tokens + attended)
        return self.norm_feedforward(tokens + self.feedforward(tokens))


class TransformerCritic(nn.Module):
    def __init__(self, sizes, width, heads, blocks):
        super().__init__()
        # Match the shared observation embedding for homogeneous MAT tasks;
        # separate projections also support the project's heterogeneous spaces.
        self.shared_projection = len(set(sizes)) == 1
        self.projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(size), _linear(size, width, math.sqrt(2)), nn.GELU())
            for size in (sizes[:1] if self.shared_projection else sizes)
        )
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.Sequential(*(EncoderBlock(width, heads) for _ in range(blocks)))
        self.value_head = nn.Sequential(
            _linear(width, width, math.sqrt(2)),
            nn.GELU(),
            nn.LayerNorm(width),
            _linear(width, 1),
        )

    def forward(self, observations):
        tokens = torch.stack(
            [
                self.projections[0 if self.shared_projection else i](obs)
                for i, obs in enumerate(observations)
            ],
            dim=-2,
        )
        return self.value_head(self.blocks(self.input_norm(tokens))).squeeze(-1)


class MATDecModel(nn.Module):
    def __init__(self, spec, embedding_dim=64, num_heads=1, num_blocks=1, share_actor=False):
        super().__init__()
        if embedding_dim < 1 or num_heads < 1 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.spec = spec
        self.names = tuple(spec["agents"])
        self.indices = {agent: i for i, agent in enumerate(self.names)}
        sizes, counts = spec["observation_sizes"], spec["action_counts"]
        if not self.names or len(sizes) != len(self.names) or len(counts) != len(self.names):
            raise ValueError("Invalid MAT-Dec model spaces")
        if share_actor and (len(set(sizes)) != 1 or len(set(counts)) != 1):
            raise ValueError("share_actor requires identical observation and action spaces")
        self.share_actor = share_actor
        self.actors = nn.ModuleList(
            _actor(size, count, embedding_dim)
            for size, count in (list(zip(sizes, counts))[:1] if share_actor else zip(sizes, counts))
        )
        self.critic = TransformerCritic(sizes, embedding_dim, num_heads, num_blocks)

    def local_logits(self, agent, observation):
        index = self.indices[agent]
        return self.actors[0 if self.share_actor else index](observation)

    def values(self, observations):
        return self.critic([observations[agent] for agent in self.names])

    def action_statistics(self, observations, actions):
        log_probs, entropies = [], []
        for index, agent in enumerate(self.names):
            distribution = Categorical(logits=self.local_logits(agent, observations[agent]))
            log_probs.append(distribution.log_prob(actions[..., index]))
            entropies.append(distribution.entropy())
        return torch.stack(log_probs, dim=-1), torch.stack(entropies, dim=-1)


def create_model(env, observations, config, device="cpu"):
    return MATDecModel(
        environment_spec(env, observations),
        embedding_dim=config.get("embedding_dim", 64),
        num_heads=config.get("num_heads", 1),
        num_blocks=config.get("num_blocks", 1),
        share_actor=config.get("share_actor", False),
    ).to(device)
