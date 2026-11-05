"""MPE2 的 Spread/Tag 适配；新增 MPE 场景从这里接入。"""

from importlib import import_module

import numpy as np

from .base import Environment


class MPEEnvironment(Environment):
    """MPE2 → 公共环境合同。模拟器内部结构只在适配器中出现。"""

    def __init__(self, env):
        self.base = env
        self.possible_agents = env.possible_agents
        self.metadata = env.metadata

    @property
    def policy_metadata(self):
        # 规则策略只需要不随时间变化的任务信息，不暴露模拟器坐标。
        world = self.unwrapped.world
        return {
            "landmark_count": len(world.landmarks),
            "roles": {a.name: getattr(a, "adversary", False) for a in world.agents},
        }

    @property
    def agents(self):
        return self.base.agents

    @property
    def unwrapped(self):
        return self.base.unwrapped

    def observation_space(self, agent):
        return self.base.observation_space(agent)

    def action_space(self, agent):
        return self.base.action_space(agent)

    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)

    def step(self, actions):
        return self.base.step(actions)

    def state(self):
        return self.base.state()

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()

    def task_snapshot(self):
        world = self.unwrapped.world
        return {
            "positions": np.array([a.state.p_pos for a in world.agents]),
            "sizes": np.array([a.size for a in world.agents]),
            "landmarks": np.array([l.state.p_pos for l in world.landmarks]),
            "predators": np.array([getattr(a, "adversary", False) for a in world.agents]),
        }


def make_mpe(name, render_mode=None, **kwargs):
    module = import_module(f"mpe2.simple_{name}_v3")
    return MPEEnvironment(
        module.parallel_env(render_mode=render_mode, continuous_actions=False, **kwargs)
    )
