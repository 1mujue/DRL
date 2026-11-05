"""LBF/RWARE → 多智能体接口；固定数量 agent 的 Gymnasium tuple 环境。"""

from importlib import import_module
from typing import ClassVar

import gymnasium as gym
import numpy as np

from .base import Environment


class TupleParallel(Environment):
    """Adapt fixed-population Gymnasium tuple environments (LBF / RWARE)."""

    metadata: ClassVar[dict] = {
        "name": "tuple_parallel",
        "is_parallelizable": True,
        "render_modes": [],
    }

    def __init__(self, env, family):
        self.base = env
        self.family = family
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
        # LBF 2.0 reports its internal time limit as termination. Distinguish
        # task completion from timeout so PPO bootstraps the final observation.
        if self.family == "lbf" and terminated:
            task_complete = self.base.unwrapped.field.sum() == 0
            terminated = bool(task_complete)
            truncated = not task_complete
        if self.family == "rware" and terminated:
            base = self.base.unwrapped
            if base.max_steps and base._cur_steps >= base.max_steps:
                terminated, truncated = False, True
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

    def task_snapshot(self):
        base = self.base.unwrapped
        if self.family == "lbf":
            return {"food_remaining": int(np.count_nonzero(base.field))}
        reward_type = base.reward_type.name
        if reward_type == "TWO_STAGE":
            raise ValueError("RWARE metrics v1 require INDIVIDUAL or GLOBAL rewards")
        # 对已锁定的 RWARE 2.0：individual 的总奖励或 global 的平均奖励 = 本步交付数。
        deliveries = (
            self._last_rewards.mean() if reward_type == "GLOBAL" else self._last_rewards.sum()
        )
        return {"deliveries_this_step": float(deliveries), "reward_type": reward_type}


def make_gym_tuple(name, render_mode=None, **kwargs):
    if render_mode is not None:
        raise ValueError("External environment rendering is not yet supported by this adapter")
    import_module("lbforaging" if name == "lbf" else "rware")
    if name == "rware" and isinstance(kwargs.get("reward_type"), str):
        from rware.warehouse import RewardType

        kwargs["reward_type"] = RewardType[kwargs["reward_type"].upper()]
    key = kwargs.pop("id", "Foraging-8x8-2p-1f-coop-v3" if name == "lbf" else "rware-tiny-2ag-v2")
    return TupleParallel(gym.make(key, disable_env_checker=True, **kwargs), name)
