"""HAPPO 的公共接口；执行策略只使用自己的 actor 与局部观测。"""

from dataclasses import asdict

import torch
from torch.distributions import Categorical

from ...environments.base import EnvironmentSpec
from ..base import Algorithm, Policy
from .networks import HAPPOModel, resolve_device, validate_observations


class HAPPOPolicy(Policy):
    """异构 actor 的确定性/随机执行；资源状态为空，reset 无额外工作。"""

    def __init__(self, model):
        self.model = model

    def act(self, observations, deterministic=True):
        validate_observations(observations, self.model.spec)
        with torch.no_grad():
            logits = self.model.local_logits(observations)
            return {
                agent: int(
                    scores.argmax() if deterministic else Categorical(logits=scores).sample()
                )
                for agent, scores in logits.items()
            }


class HAPPOAlgorithm(Algorithm):
    def train(self, config, out):
        from .trainer import train

        return train(config, out)

    def load_policy(self, checkpoint, env=None, observations=None, *, device="cpu"):
        """可仅凭检查点加载；公共评估器的 (checkpoint, env, observations) 调用同样有效。"""
        device = resolve_device(device)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or data.get("format") != "teamwork.happo.v1":
            raise ValueError("Unsupported HAPPO checkpoint format")
        try:
            saved = data["spec"]
            spec = EnvironmentSpec(
                tuple(saved["agents"]),
                tuple(saved["observation_sizes"]),
                tuple(saved["action_counts"]),
                saved["state_size"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid HAPPO checkpoint EnvironmentSpec") from error
        if data.get("config", {}).get("algorithm") != "happo":
            raise ValueError("Unsupported HAPPO checkpoint algorithm")
        if data["config"].get("share_policy", False):
            raise ValueError("HAPPO checkpoints require share_policy: false")
        if env is not None and asdict(spec) != asdict(env.describe()):
            raise ValueError("HAPPO checkpoint EnvironmentSpec does not match the environment")
        if observations is not None:
            validate_observations(observations, spec)
        model = HAPPOModel(spec, data["config"]).to(device)
        model.load_state_dict(data["model"])
        model.eval()
        return HAPPOPolicy(model)
