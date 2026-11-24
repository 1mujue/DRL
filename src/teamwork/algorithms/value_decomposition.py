"""QTRAN 和 QPLEX 共用的检查点加载与训练接入。"""

import torch

from ..environments.base import EnvironmentSpec
from .base import Algorithm
from .qmix.algorithm import QMIXPolicy, resolve_device
from .qmix.availability import lbf_action_availability
from .qmix.networks import LocalQController, pack_observations


class ValueDecompositionAlgorithm(Algorithm):
    """复用离散局部 Q 策略。联合价值网络只参与训练。"""

    def train(self, config, out):
        from .qmix.trainer import train

        return train(
            config,
            out,
            algorithm=self.name,
            learner_factory=self.learner_factory,
            algorithm_defaults=self.defaults,
            implementation=self.implementation,
        )

    def load_policy(self, checkpoint, env=None, observations=None, *, device="cpu"):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(data, dict)
            or data.get("format") != f"teamwork.{self.name}.v1"
            or data.get("config", {}).get("algorithm") != self.name
        ):
            raise ValueError(f"Unsupported {self.name} checkpoint")
        saved = data["spec"]
        spec = EnvironmentSpec(
            tuple(saved["agents"]),
            tuple(saved["observation_sizes"]),
            tuple(saved["action_counts"]),
            saved["state_size"],
        )
        if env is not None and env.describe() != spec:
            raise ValueError("Checkpoint EnvironmentSpec does not match the environment")
        if observations is not None:
            pack_observations(observations, spec)
        controller = LocalQController(spec, data["config"]).to(resolve_device(device))
        controller.load_state_dict(data["controller"])
        controller.eval()
        return QMIXPolicy(
            controller,
            data["config"]["evaluation_epsilon"],
            action_availability=lbf_action_availability(data["config"], env, spec),
        )
