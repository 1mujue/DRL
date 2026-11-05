"""QMIX 接入 Algorithm/Policy；评估只加载局部 Q 网络，不使用 mixer 或 state。"""

from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from ..base import Algorithm, Policy
from .networks import LocalQController, pack_observations


class QMIXPolicy(Policy):
    def __init__(self, controller, evaluation_epsilon=0.05):
        self.controller = controller
        self.evaluation_epsilon = evaluation_epsilon
        self.reset()

    def reset(self):
        self.hidden = self.controller.initial_hidden(1)
        self.previous_actions = self.hidden.new_zeros(
            1, len(self.controller.spec.agents), self.controller.max_actions
        )

    def act(self, observations, deterministic=True):
        epsilon = 0.0 if deterministic else self.evaluation_epsilon
        return self.sample_actions(observations, epsilon, np.random)

    @torch.no_grad()
    def sample_actions(self, observations, epsilon, rng):
        packed = torch.as_tensor(
            pack_observations(observations, self.controller.spec),
            device=self.hidden.device,
        ).unsqueeze(0)
        values, self.hidden = self.controller(packed, self.previous_actions, self.hidden)
        actions = self.controller.greedy_actions(values)[0].cpu().numpy().copy()
        for index, count in enumerate(self.controller.spec.action_counts):
            if rng.random() < epsilon:
                actions[index] = rng.choice(count)
        action_tensor = torch.as_tensor(actions, device=self.hidden.device)
        self.previous_actions = F.one_hot(
            action_tensor, self.controller.max_actions
        ).to(self.hidden.dtype).unsqueeze(0)
        return dict(zip(self.controller.spec.agents, map(int, actions)))


class QMIXAlgorithm(Algorithm):
    def train(self, config, out):
        if config["algorithm"] != "qmix":
            raise ValueError("QMIX configuration requires algorithm='qmix'")
        from .trainer import train

        return train(config, out)

    def load_policy(self, checkpoint, env, observations):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data.get("format") != "teamwork.qmix.v1" or data["config"]["algorithm"] != "qmix":
            raise ValueError("Unsupported QMIX checkpoint format or algorithm")
        spec = env.describe()
        if data["spec"] != asdict(spec):
            raise ValueError("QMIX checkpoint EnvironmentSpec does not match the environment")
        pack_observations(observations, spec)
        controller = LocalQController(spec, data["config"])
        controller.load_state_dict(data["controller"])
        controller.eval()
        return QMIXPolicy(controller, data["config"]["evaluation_epsilon"])
