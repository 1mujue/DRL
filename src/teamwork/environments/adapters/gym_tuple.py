"""Adapt Gymnasium tuple environments to the parallel environment contract."""

from importlib import import_module
from typing import ClassVar

import gymnasium as gym
import numpy as np

from ..base import Environment


class TupleParallel(Environment):
    """Adapt fixed-population Gymnasium tuple environments (LBF / RWARE)."""

    metadata: ClassVar[dict] = {
        "name": "tuple_parallel",
        "is_parallelizable": True,
        "render_modes": [],
    }

    def __init__(self, env):
        self.base = env
        self.possible_agents = [f"agent_{i}" for i in range(len(env.action_space.spaces))]
        self.agents = []
        self.observation_spaces = {
            a: gym.spaces.flatten_space(s)
            for a, s in zip(self.possible_agents, env.observation_space.spaces)
        }
        self.action_spaces = dict(zip(self.possible_agents, env.action_space.spaces))
        self._obs = {}
        self._last_rewards = np.zeros(len(self.possible_agents))

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _flatten(self, observations):
        return {
            a: gym.spaces.flatten(s, o).astype(np.float32)
            for a, s, o in zip(
                self.possible_agents, self.base.observation_space.spaces, observations
            )
        }

    def reset(self, seed=None, options=None):
        observations, _ = self.base.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self._obs = self._flatten(observations)
        self._last_rewards = np.zeros(len(self.possible_agents))
        return self._obs, {a: {} for a in self.agents}

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        obs, rew, terminated, truncated, _ = self.base.step(
            tuple(actions[a] for a in self.possible_agents)
        )
        terminated, truncated = self.episode_end(terminated, truncated)
        self._obs = self._flatten(obs)
        self._last_rewards = np.asarray(rew, dtype=float)
        terms = {a: bool(terminated) for a in self.agents}
        truncs = {a: bool(truncated) for a in self.agents}
        rewards = {a: float(r) for a, r in zip(self.agents, rew)}
        infos = {a: {} for a in self.agents}
        if terminated or truncated:
            self.agents = []
        return self._obs, rewards, terms, truncs, infos

    def state(self):
        # Joint observations, not a claim to expose the simulator's hidden state.
        return np.concatenate([self._obs[a] for a in self.possible_agents])

    def close(self):
        self.base.close()

    @classmethod
    def from_gym(cls, render_mode=None, **kwargs):
        if render_mode is not None:
            raise ValueError("External environment rendering is not yet supported by this adapter")
        import_module(cls.package)
        key = kwargs.pop("id", cls.default_id)
        return cls(gym.make(key, disable_env_checker=True, **cls.gym_options(kwargs)))

    @staticmethod
    def gym_options(kwargs):
        return kwargs

    def episode_end(self, terminated, truncated):
        return terminated, truncated
