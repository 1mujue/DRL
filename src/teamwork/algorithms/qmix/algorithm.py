"""QMIX 接入 Algorithm/Policy；评估只加载局部 Q 网络，不使用 mixer 或 state。"""

import math
from dataclasses import asdict
from numbers import Real

import numpy as np
import torch
from torch.nn import functional as F

from ...environments.base import EnvironmentSpec
from ..base import Algorithm, Policy
from .networks import LocalQController, pack_observations


def resolve_device(device):
    """训练与独立推理使用同一设备检查，避免写入目录后才发现设备无效。"""
    try:
        result = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("QMIX device must be cpu or an available cuda device") from error
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("QMIX device must be cpu or cuda")
    if result.type == "cuda" and (
        not torch.cuda.is_available()
        or (result.index is not None and result.index >= torch.cuda.device_count())
    ):
        raise ValueError("The requested CUDA device is not available")
    return result


def validate_epsilon(epsilon):
    if (
        isinstance(epsilon, bool) or not isinstance(epsilon, Real)
        or not math.isfinite(epsilon) or not 0 <= epsilon <= 1
    ):
        raise ValueError("epsilon must be finite and lie in [0, 1]")
    return float(epsilon)


class QMIXPolicy(Policy):
    def __init__(self, controller, evaluation_epsilon=0.05):
        self.controller = controller
        self.evaluation_epsilon = validate_epsilon(evaluation_epsilon)
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
        epsilon = validate_epsilon(epsilon)
        packed = torch.as_tensor(
            pack_observations(observations, self.controller.spec),
            device=self.hidden.device,
        ).unsqueeze(0)
        values, self.hidden = self.controller(packed, self.previous_actions, self.hidden)
        actions = self.controller.greedy_actions(values)[0].cpu().numpy().copy()
        if epsilon > 0:
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
        from .trainer import train

        return train(config, out)

    def load_policy(self, checkpoint, env=None, observations=None, *, device="cpu"):
        """仅凭检查点加载局部执行策略；也兼容公共评估器的 env/observations 参数。"""
        device = resolve_device(device)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(data, dict) or data.get("format") != "teamwork.qmix.v1"
            or data.get("config", {}).get("algorithm") != "qmix"
        ):
            raise ValueError("Unsupported QMIX checkpoint format or algorithm")
        try:
            saved = data["spec"]
            spec = EnvironmentSpec(
                tuple(saved["agents"]), tuple(saved["observation_sizes"]),
                tuple(saved["action_counts"]), saved["state_size"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid QMIX checkpoint EnvironmentSpec") from error
        if env is not None and asdict(spec) != asdict(env.describe()):
            raise ValueError("QMIX checkpoint EnvironmentSpec does not match the environment")
        if observations is not None:
            pack_observations(observations, spec)
        controller = LocalQController(spec, data["config"]).to(device)
        controller.load_state_dict(data["controller"])
        controller.eval()
        return QMIXPolicy(controller, data["config"]["evaluation_epsilon"])
