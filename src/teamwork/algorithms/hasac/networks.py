"""HASAC 离散适配的网络：每 agent 一个 categorical actor、集中双 soft-Q 与可学温度。

critic 输入 ``[central_state, 联合动作 one-hot]``：这是 HASAC 与"只看状态的共享 critic"
方案的关键差别，也是顺序更新时前序 agent 的动作能够进入价值估计的前提。

目标网络显式注册为子模块，并且**完整写入检查点**；避免出现"目标网络被注册成子模块、
保存时却漏掉，加载时又被静默跳过"这类只在续训时才暴露的坑。
"""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ...environments.base import EnvironmentSpec
from ..mappo.networks import make_mlp
from ..mappo.networks import resolve_device as _resolve_device
from ..mappo.networks import validate_observations as _validate_observations

__all__ = [
    "HASACModel",
    "agent_observation",
    "joint_action_size",
    "pack_observations",
    "resolve_actor_order",
    "resolve_device",
    "sample_actions",
    "validate_observations",
]


def _rename(error):
    return str(error).replace("MAPPO", "HASAC")


def resolve_device(device):
    try:
        return _resolve_device(device)
    except ValueError as error:
        raise ValueError(_rename(error)) from error


def validate_observations(observations, spec, device=None):
    try:
        return _validate_observations(observations, spec, device)
    except ValueError as error:
        raise ValueError(_rename(error)) from error


def joint_action_size(spec: EnvironmentSpec):
    return int(sum(spec.action_counts))


def pack_observations(observations, spec, device=None):
    """按 agent 顺序把各自的局部观测零填充到同一宽度，返回 [N, Omax]。

    填充只发生在尾部；取回时必须按该 agent 的真实维度切片（见 agent_observation）。
    """
    validated = validate_observations(observations, spec, device)
    reference = next(iter(validated.values()))
    packed = torch.zeros(
        (len(spec.agents), max(spec.observation_sizes)),
        dtype=reference.dtype,
        device=reference.device,
    )
    for index, agent in enumerate(spec.agents):
        packed[index, : spec.observation_sizes[index]] = validated[agent]
    return packed


def agent_observation(packed, index, size):
    return packed[..., index, :size]


def resolve_actor_order(spec: EnvironmentSpec, configured):
    """顺序更新的 agent 顺序；缺省按环境顺序，显式值必须是全排列。"""
    names = list(spec.agents)
    if configured is None:
        return names
    if not isinstance(configured, (list, tuple)) or any(
        not isinstance(name, str) for name in configured
    ):
        raise ValueError("HASAC actor_order must be null or a list of agent names")
    order = list(configured)
    if len(order) != len(names) or set(order) != set(names):
        raise ValueError("HASAC actor_order must list every agent exactly once")
    return order


class HASACModel(nn.Module):
    """异构 actor + 双 soft-Q + 每 agent 温度，以及它们的目标网络副本。"""

    def __init__(self, spec: EnvironmentSpec, config: dict):
        super().__init__()
        if (
            not spec.agents
            or len(spec.agents) != len(spec.observation_sizes)
            or len(spec.agents) != len(spec.action_counts)
            or len(set(spec.agents)) != len(spec.agents)
            or min((*spec.observation_sizes, *spec.action_counts, spec.state_size)) < 1
        ):
            raise ValueError("HASAC requires a nonempty fixed vector/discrete EnvironmentSpec")
        self.spec = spec
        self.observation_width = int(max(spec.observation_sizes))
        offsets, running = [], 0
        for count in spec.action_counts:
            offsets.append(running)
            running += int(count)
        self.register_buffer("action_offsets", torch.tensor(offsets, dtype=torch.long))
        actor_gain = config.get("actor_output_gain", 0.01)
        self.actors = nn.ModuleDict(
            {
                agent: make_mlp(observation_size, actions, config, actor_gain)
                for agent, observation_size, actions in zip(
                    spec.agents, spec.observation_sizes, spec.action_counts
                )
            }
        )
        critic_inputs = spec.state_size + joint_action_size(spec)
        self.critics = nn.ModuleList(
            [make_mlp(critic_inputs, 1, config, output_gain=1.0) for _ in range(2)]
        )
        alpha = config.get("alpha_start", 0.2)
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("HASAC alpha_start must be finite and positive")
        self.log_alphas = nn.ParameterDict(
            {agent: nn.Parameter(torch.tensor(math.log(float(alpha)))) for agent in spec.agents}
        )
        # 显式注册并随检查点保存的目标网络；deepcopy 会保留参数值但不共享梯度。
        self.target_actors = deepcopy(self.actors)
        self.target_critics = deepcopy(self.critics)

    def joint_one_hot(self, actions):
        """[..., N] 动作索引 -> [..., sum(action_counts)] 联合动作 one-hot。"""
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.action_offsets.device)
        counts = torch.as_tensor(
            self.spec.action_counts, dtype=torch.long, device=self.action_offsets.device
        )
        if actions.shape[-1] != len(self.spec.agents):
            raise ValueError("HASAC joint action must name every agent exactly once")
        if actions.numel() and bool(((actions < 0) | (actions >= counts)).any()):
            raise ValueError("HASAC joint action contains an index outside its agent's space")
        one_hot = F.one_hot(self.action_offsets + actions, joint_action_size(self.spec)).float()
        # [..., N, 全局动作宽度] -> [..., 全局动作宽度]：各 agent 的偏移互不重叠，
        # 求和得到整支队伍的一个联合动作向量。
        return one_hot.sum(dim=-2)

    def q_values(self, states, actions, targets=False):
        """返回双 Q 的值 [..., 2]；目标网络只在训练期使用。"""
        one_hot = self.joint_one_hot(actions)
        if one_hot.shape[:-1] != states.shape[:-1]:
            raise ValueError("HASAC states and joint actions must share their leading shape")
        inputs = torch.cat((states.to(one_hot.dtype), one_hot), dim=-1)
        networks = self.target_critics if targets else self.critics
        return torch.cat([network(inputs) for network in networks], dim=-1)

    def actor_logits(self, agent, values, targets=False):
        networks = self.target_actors if targets else self.actors
        return networks[agent](values)

    @torch.no_grad()
    def sync_targets(self, tau):
        if isinstance(tau, bool) or not math.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError("HASAC tau must lie in [0, 1]")
        for target, source in (
            (self.target_actors, self.actors),
            (self.target_critics, self.critics),
        ):
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1 - tau).add_(source_parameter, alpha=tau)


@torch.no_grad()
def sample_actions(model, packed, spec, deterministic=False):
    """单步执行动作：packed 为 [N, Omax]，返回 {agent: int}。"""
    result = {}
    for index, agent in enumerate(spec.agents):
        logits = model.actor_logits(
            agent, agent_observation(packed, index, spec.observation_sizes[index])
        )
        if deterministic:
            action = logits.argmax(-1)
        else:
            action = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
        result[agent] = int(action.item())
    return result
