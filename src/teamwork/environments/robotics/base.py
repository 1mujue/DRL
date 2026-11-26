"""Shared parallel environment adapter for two opposed Panda robots in robosuite."""

import numpy as np
from gymnasium.spaces import Box

from ..base import Environment


class TwoPandaEnvironment(Environment):
    task: str
    camera = "frontview"

    def __init__(self, render_mode, *, horizon, reward_shaping, **task_kwargs):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"{self.task} render_mode must be None or rgb_array")
        if type(horizon) is not int or horizon < 1:
            raise ValueError(f"{self.task} horizon must be a positive integer")
        if type(reward_shaping) is not bool:
            raise ValueError(f"{self.task} reward_shaping must be boolean")
        try:
            import robosuite
        except ModuleNotFoundError as exc:
            if exc.name == "robosuite":
                raise ModuleNotFoundError(
                    f"{self.task} requires the robotics extra: pip install -e '.[robotics]'"
                ) from exc
            raise
        self.horizon = horizon
        self._step = 0
        self.render_mode = render_mode
        self.possible_agents = ["arm_0", "arm_1"]
        self.agents = []
        self._env = robosuite.make(
            self.task,
            robots=["Panda", "Panda"],
            env_configuration="opposed",
            initialization_noise=None,
            horizon=horizon,
            control_freq=self.metadata["render_fps"],
            reward_shaping=reward_shaping,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=render_mode == "rgb_array",
            hard_reset=False,
            **task_kwargs,
        )
        self._actions = {
            agent: Box(*(np.asarray(limit, dtype=np.float32) for limit in robot.action_limits))
            for agent, robot in zip(self.possible_agents, self._env.robots, strict=True)
        }
        self._observations = {}
        self._local = {}

    def observation_space(self, agent):
        return self._observations[agent]

    def action_space(self, agent):
        return self._actions[agent]

    def object_observations(self, raw, index):
        raise NotImplementedError

    def local_observations(self, raw):
        # Each arm sees its own proprioception and the task's public object fields.
        return {
            agent: np.concatenate(
                (
                    raw[f"robot{i}_eef_pos"],
                    raw[f"robot{i}_eef_quat_site"],
                    *self.object_observations(raw, i),
                    raw[f"robot{i}_joint_pos_sin"],
                    raw[f"robot{i}_joint_pos_cos"],
                    raw[f"robot{i}_joint_vel"],
                    raw[f"robot{i}_gripper_qpos"],
                    raw[f"robot{i}_gripper_qvel"],
                ),
                dtype=np.float32,
            )
            for i, agent in enumerate(self.possible_agents)
        }

    def _read_observations(self, raw):
        self._local = self.local_observations(raw)
        return {agent: observation.copy() for agent, observation in self._local.items()}

    def reset(self, seed=None, options=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._env.rng = rng
            self._env.placement_initializer.rng = rng
        self._step = 0
        obs = self._read_observations(self._env.reset())
        if not self._observations:
            self._observations = {
                agent: Box(-np.inf, np.inf, shape=observation.shape, dtype=np.float32)
                for agent, observation in obs.items()
            }
        self.agents = self.possible_agents.copy()
        return obs, {agent: {} for agent in self.agents}

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if actions.keys() != self._actions.keys():
            raise ValueError(f"{self.task} needs one action for each arm")
        vectors = []
        for agent in self.possible_agents:
            vector = np.asarray(actions[agent], dtype=np.float32)
            if not self.action_space(agent).contains(vector):
                raise ValueError(f"Invalid continuous action for {agent}")
            vectors.append(vector)
        raw, reward, timeout, info = self._env.step(np.concatenate(vectors))
        self._step += 1
        obs = self._read_observations(raw)
        # robosuite stops at the time limit, even if the lift was never completed.
        rewards = dict.fromkeys(self.agents, float(reward))
        terminated = dict.fromkeys(self.agents, False)
        truncated = dict.fromkeys(self.agents, bool(timeout))
        infos = {agent: dict(info) for agent in self.agents}
        if timeout:
            self.agents = []
        return obs, rewards, terminated, truncated, infos

    def state(self):
        # Joint proprioception and both handles are available only to the training critic.
        return np.concatenate([self._local[agent] for agent in self.possible_agents])

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._env.sim.render(width=480, height=360, camera_name=self.camera)[::-1].copy()

    def close(self):
        self.agents = []
        self._env.close()
