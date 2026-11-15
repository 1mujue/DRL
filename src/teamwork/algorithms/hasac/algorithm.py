"""HASAC 的公共接口；执行策略只使用自己的 actor 与局部观测。"""

from dataclasses import asdict

import torch

from ...environments.base import EnvironmentSpec
from ..base import Algorithm, Policy
from .networks import (
    HASACModel,
    pack_observations,
    resolve_device,
    sample_actions,
    validate_observations,
)


class HASACPolicy(Policy):
    """前馈 categorical 策略，没有跨步状态；reset 只满足公共契约。"""

    def __init__(self, model):
        self.model = model

    def reset(self):
        return None

    def act(self, observations, deterministic=True):
        device = next(self.model.parameters()).device
        packed = pack_observations(observations, self.model.spec, device)
        return sample_actions(self.model, packed, self.model.spec, deterministic)


class HASACAlgorithm(Algorithm):
    def train(self, config, out):
        from .trainer import train

        return train(config, out)

    def load_policy(self, checkpoint, env=None, observations=None, *, device="cpu"):
        """仅凭检查点加载执行策略；公共评估器的三参数调用同样有效。"""
        device = resolve_device(device)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or data.get("format") != "teamwork.hasac.v1":
            raise ValueError("Unsupported HASAC checkpoint format")
        try:
            saved = data["spec"]
            spec = EnvironmentSpec(
                tuple(saved["agents"]),
                tuple(saved["observation_sizes"]),
                tuple(saved["action_counts"]),
                saved["state_size"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid HASAC checkpoint EnvironmentSpec") from error
        if data.get("config", {}).get("algorithm") != "hasac_discrete":
            raise ValueError("Unsupported HASAC checkpoint algorithm")
        if env is not None and asdict(spec) != asdict(env.describe()):
            raise ValueError("HASAC checkpoint EnvironmentSpec does not match the environment")
        if observations is not None:
            validate_observations(observations, spec)
        model = HASACModel(spec, data["config"]).to(device)
        # 只加载执行所需的 actor；目标网络与 critic 属于训练期状态。
        model.actors.load_state_dict(data["actors"])
        model.eval()
        return HASACPolicy(model)
