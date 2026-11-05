"""MAPPO 网络与价值归一化，不依赖具体仿真环境。"""

import math

import torch
from torch import nn

from ...environments.base import EnvironmentSpec


class ValueNormalizer(nn.Module):
    """按样本累计 Welford 矩；仅改变训练目标尺度，不是 PopArt。

    critic 输出归一化价值，GAE 使用 denormalize 后的原奖励尺度价值。
    每批 rollout 更新统计一次，PPO 所有 epoch 期间冻结统计。
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("m2", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @property
    def variance(self):
        # 第一批数据之前使用 mean=0、variance=1，避免凭空缩放初始价值。
        return torch.where(
            self.count > 0, self.m2 / self.count.clamp_min(1), torch.ones_like(self.m2)
        ).clamp_min(1e-2)

    @torch.no_grad()
    def update(self, values):
        samples = torch.as_tensor(values, device=self.mean.device, dtype=torch.float64).reshape(-1)
        if samples.numel() == 0 or not torch.isfinite(samples).all():
            raise ValueError("ValueNormalizer requires nonempty finite samples")
        batch_count = samples.numel()
        batch_mean = samples.mean()
        batch_m2 = (samples - batch_mean).square().sum()
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.mean.add_(delta * batch_count / total)
        self.count.copy_(total)

    def normalize(self, values):
        values = torch.as_tensor(values, device=self.mean.device)
        if not values.is_floating_point():
            values = values.float()
        return (values - self.mean.to(values.dtype)) / self.variance.sqrt().to(values.dtype)

    def denormalize(self, values):
        values = torch.as_tensor(values, device=self.mean.device)
        if not values.is_floating_point():
            values = values.float()
        return values * self.variance.sqrt().to(values.dtype) + self.mean.to(values.dtype)


def make_mlp(input_size, output_size, config, output_gain):
    layers = []
    if config.get("feature_normalization", True):
        layers.append(nn.LayerNorm(input_size))
    width = config.get("hidden_size", 64)
    for _ in range(config.get("hidden_layers", 2)):
        layer = nn.Linear(input_size, width)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        input_size = width
    output = nn.Linear(input_size, output_size)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class MAPPOModel(nn.Module):
    """局部 actor 分组共享；critic 输入 [global_state, agent_one_hot]。

    即使个体奖励不同，共享 critic 也能通过 agent identity 区分 V_i。
    actor 不接收 global state 或其他 agent 的观测。
    """

    def __init__(self, spec: EnvironmentSpec, config: dict):
        super().__init__()
        if (
            not spec.agents
            or len(spec.agents) != len(spec.observation_sizes)
            or len(spec.agents) != len(spec.action_counts)
            or len(set(spec.agents)) != len(spec.agents)
            or min((*spec.observation_sizes, *spec.action_counts, spec.state_size)) < 1
        ):
            raise ValueError("MAPPO requires a nonempty fixed vector/discrete EnvironmentSpec")
        self.spec = spec
        self.agent_to_group = {}
        self.actors = nn.ModuleDict()
        signatures = {}
        for i, (agent, observation_size, actions) in enumerate(
            zip(spec.agents, spec.observation_sizes, spec.action_counts)
        ):
            signature = (observation_size, actions) if config.get("share_policy", True) else (i,)
            if signature not in signatures:
                group = f"group_{len(signatures)}"
                signatures[signature] = group
                self.actors[group] = make_mlp(
                    observation_size, actions, config, config.get("actor_output_gain", 0.01)
                )
            self.agent_to_group[agent] = signatures[signature]
        self.critic = make_mlp(spec.state_size + len(spec.agents), 1, config, output_gain=1.0)
        self.register_buffer("agent_ids", torch.eye(len(spec.agents)))

    def local_logits(self, observations):
        """按组批量前向，返回每个 agent 的动作 logits：[action_count]。"""
        result = {}
        device = self.agent_ids.device
        for group, actor in self.actors.items():
            names = [a for a in observations if self.agent_to_group[a] == group]
            if not names:
                continue
            batch = torch.stack(
                [torch.as_tensor(observations[a], dtype=torch.float32, device=device) for a in names]
            )
            logits = actor(batch)
            result.update(zip(names, logits))
        return result

    def critic_inputs(self, states):
        """[S] -> [N,S+N]；[T,S] -> [T,N,S+N]。"""
        states = torch.as_tensor(states, dtype=torch.float32, device=self.agent_ids.device)
        if states.shape[-1] != self.spec.state_size:
            raise ValueError("Global state size does not match the saved EnvironmentSpec")
        n_agents = len(self.spec.agents)
        shape = (*states.shape[:-1], n_agents)
        joint = states.unsqueeze(-2).expand(*shape, self.spec.state_size)
        identities = self.agent_ids.expand(*shape, n_agents)
        return torch.cat((joint, identities), dim=-1)

    def values(self, states):
        return self.critic(self.critic_inputs(states)).squeeze(-1)
