"""MAPPO 的公共接口；执行策略仅使用 actor 和局部观测。"""

from dataclasses import asdict

import torch

from ...environments.base import EnvironmentSpec
from ..base import Algorithm, Policy
from .networks import MAPPOModel, resolve_device, validate_observations


class MAPPOPolicy(Policy):
    def __init__(self, model):
        self.model = model

    def act(self, observations, deterministic=True):
        with torch.no_grad():
            logits = self.model.local_logits(observations)
            return {
                agent: int(
                    scores.argmax()
                    if deterministic
                    else torch.distributions.Categorical(logits=scores).sample()
                )
                for agent, scores in logits.items()
            }


class MAPPOAlgorithm(Algorithm):
    def train(self, config, out):
        from .trainer import train

        return train(config, out)

    def load_policy(self, checkpoint, env=None, observations=None, *, device="cpu"):
        """可仅凭检查点加载；旧评估器的 (checkpoint, env, observations) 调用仍有效。"""
        device = resolve_device(device)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or data.get("format") != "teamwork.mappo.v1":
            raise ValueError("Unsupported MAPPO checkpoint format")
        saved_spec = data["spec"]
        spec = EnvironmentSpec(
            tuple(saved_spec["agents"]), tuple(saved_spec["observation_sizes"]),
            tuple(saved_spec["action_counts"]), saved_spec["state_size"],
        )
        if data["config"].get("algorithm", "mappo_ff") != "mappo_ff":
            raise ValueError("Unsupported MAPPO checkpoint algorithm")
        if env is not None and asdict(spec) != asdict(env.describe()):
            raise ValueError("Checkpoint EnvironmentSpec does not match the evaluation environment")
        if observations is not None:
            validate_observations(observations, spec)
        model = MAPPOModel(spec, data["config"]).to(device)
        model.load_state_dict(data["model"])
        model.eval()
        return MAPPOPolicy(model)
