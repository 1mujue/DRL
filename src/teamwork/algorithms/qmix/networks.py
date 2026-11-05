"""QMIX 的局部 Q 网络与单调 mixer；集中状态只进入 mixer。"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...environments.base import EnvironmentSpec


def pack_observations(observations, spec):
    """各 agent 的局部向量补零为 [N,O_max]，不拼接其他 agent 的信息。"""
    if set(observations) != set(spec.agents):
        raise ValueError("QMIX requires the saved fixed agent population")
    packed = np.zeros((len(spec.agents), max(spec.observation_sizes)), dtype=np.float32)
    for index, (agent, width) in enumerate(zip(spec.agents, spec.observation_sizes)):
        local = np.asarray(observations[agent], dtype=np.float32)
        if local.shape != (width,) or not np.isfinite(local).all():
            raise ValueError(f"Invalid local observation for {agent}")
        packed[index, :width] = local
    return packed


class LocalQController(nn.Module):
    """共享局部网络；每个 agent 各自维护 GRU 隐状态和自己的上一步动作。

    输入 [B,N,O_max]，输出 Q：[B,N,A_max]。身份编码是固定信息；
    不同空间用补零与静态动作掩码适配，不让 agent 访问别人的观测。
    """

    def __init__(self, spec: EnvironmentSpec, config):
        super().__init__()
        if (
            not spec.agents or len(set(spec.agents)) != len(spec.agents)
            or len(spec.agents) != len(spec.observation_sizes)
            or len(spec.agents) != len(spec.action_counts)
            or min((*spec.observation_sizes, *spec.action_counts, spec.state_size)) < 1
        ):
            raise ValueError("QMIX requires fixed agents, vector observations and discrete actions")
        self.spec = spec
        self.hidden_size = config["hidden_size"]
        self.use_rnn = config["use_rnn"]
        self.use_last_action = config["use_last_action"]
        self.use_agent_id = config["use_agent_id"]
        self.max_actions = max(spec.action_counts)
        width = max(spec.observation_sizes)
        width += self.max_actions if self.use_last_action else 0
        width += len(spec.agents) if self.use_agent_id else 0
        self.input_layer = nn.Linear(width, self.hidden_size)
        self.core = (
            nn.GRUCell(self.hidden_size, self.hidden_size) if self.use_rnn
            else nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.output_layer = nn.Linear(self.hidden_size, self.max_actions)
        self.register_buffer("agent_ids", torch.eye(len(spec.agents)))
        counts = torch.tensor(spec.action_counts)
        self.register_buffer("action_mask", torch.arange(self.max_actions)[None, :] < counts[:, None])

    def initial_hidden(self, batch_size):
        return self.agent_ids.new_zeros(batch_size, len(self.spec.agents), self.hidden_size)

    def forward(self, observations, previous_actions, hidden):
        parts = [observations]
        if self.use_last_action:
            parts.append(previous_actions)
        if self.use_agent_id:
            parts.append(self.agent_ids.expand(observations.shape[0], -1, -1))
        inputs = torch.cat(parts, dim=-1)
        features = F.relu(self.input_layer(inputs.flatten(0, 1)))
        if self.use_rnn:
            next_hidden = self.core(features, hidden.flatten(0, 1))
        else:
            next_hidden = F.relu(self.core(features))
        values = self.output_layer(next_hidden)
        return (
            values.reshape(observations.shape[0], len(self.spec.agents), self.max_actions),
            next_hidden.reshape_as(hidden),
        )

    def greedy_actions(self, q_values):
        # 只屏蔽由空间尺寸补齐的动作；本接口没有动态 available-actions 输入。
        return q_values.masked_fill(~self.action_mask, -torch.inf).argmax(dim=-1)


class QMixer(nn.Module):
    """state 生成非负混合权重，保证 d Q_tot / d Q_i >= 0。

    agent_values：[...,N]，states：[...,S]，输出联合 Q：[...]。
    ELU 单调，偏置可以任意；执行策略不调用这个网络。
    """

    def __init__(self, n_agents, state_size, mixing_size, hypernet_size):
        super().__init__()
        self.n_agents = n_agents
        self.state_size = state_size
        self.mixing_size = mixing_size

        def hypernetwork(output_size):
            return nn.Sequential(
                nn.Linear(state_size, hypernet_size), nn.ReLU(),
                nn.Linear(hypernet_size, output_size),
            )

        self.first_weights = hypernetwork(n_agents * mixing_size)
        self.final_weights = hypernetwork(mixing_size)
        self.first_bias = nn.Linear(state_size, mixing_size)
        self.state_value = nn.Sequential(
            nn.Linear(state_size, mixing_size), nn.ReLU(), nn.Linear(mixing_size, 1)
        )

    def forward(self, agent_values, states):
        if agent_values.shape[-1] != self.n_agents or states.shape[-1] != self.state_size:
            raise ValueError("QMixer agent/state dimensions do not match its specification")
        if agent_values.shape[:-1] != states.shape[:-1]:
            raise ValueError("QMixer leading batch/time dimensions must agree")
        leading = agent_values.shape[:-1]
        flat_states = states.reshape(-1, self.state_size)
        local = agent_values.reshape(-1, 1, self.n_agents)
        first = self.first_weights(flat_states).abs().reshape(-1, self.n_agents, self.mixing_size)
        bias = self.first_bias(flat_states).unsqueeze(1)
        hidden = F.elu(torch.bmm(local, first) + bias)
        final = self.final_weights(flat_states).abs().reshape(-1, self.mixing_size, 1)
        total = torch.bmm(hidden, final).flatten() + self.state_value(flat_states).flatten()
        return total.reshape(leading)
