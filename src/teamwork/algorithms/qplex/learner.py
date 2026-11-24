"""QPLEX 论文式双层 dueling 与正权重多头混合。"""

import torch
from torch import nn
from torch.nn import functional as F

from ..qmix.learner import ValueDecompositionLearner, masked_td_loss, td_targets, unroll


class DuplexDuelingMixer(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        self.n_agents = len(spec.agents)
        self.n_actions = max(spec.action_counts)
        heads = config["attention_heads"]
        self.heads = heads
        self.gradient_mode = config.get("gradient_mode", "dueling")
        width = config["hypernet_size"]
        joint_action_size = self.n_agents * self.n_actions

        def head(inputs, outputs):
            return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, outputs))

        self.scale = head(spec.state_size, self.n_agents)
        self.bias = head(spec.state_size, self.n_agents)
        self.keys = head(spec.state_size, heads)
        self.agents = head(spec.state_size, heads * self.n_agents)
        self.actions = head(spec.state_size + joint_action_size, heads * self.n_agents)

    def forward(self, local_q, states, actions, action_mask):
        """输入 [...,N,A]、[...,S]、[...,N]。仅从合法动作中取局部最大值。"""
        selected = local_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        local_value = local_q.masked_fill(~action_mask, -torch.inf).max(dim=-1).values
        scale = F.softplus(self.scale(states)) + 1e-6
        advantage = scale * (selected - local_value)

        joint_action = F.one_hot(actions, self.n_actions).flatten(-2).to(states.dtype)
        shape = (*states.shape[:-1], self.heads, self.n_agents)
        keys = F.softplus(self.keys(states)).unsqueeze(-1)
        agent_gate = self.agents(states).reshape(shape).sigmoid()
        action_gate = (
            self.actions(torch.cat((states, joint_action), dim=-1)).reshape(shape).sigmoid()
        )
        importance = (keys * agent_gate * action_gate).sum(dim=-2) + 1e-6
        bias = self.bias(states)
        if self.gradient_mode == "dmaq":
            # 前向可由 Q_max + λA 改写为 Q_chosen + (λ-1)A。
            # DMAQ 对缩放后的 A 停止梯度，λ 的梯度仍由优势项提供。
            value = scale * selected + bias
            return (value + (importance - 1) * advantage.detach()).sum(dim=-1)

        value = scale * local_value + bias
        return (value + importance * advantage).sum(dim=-1)


def successful_action_loss(values, actions, mask, batch):
    """Imitate the actions in successful episodes, weighting collection and travel equally."""
    valid = batch["filled"].bool()
    collected = valid & batch["terminated"].bool() & (batch["rewards"] > 0)
    if not collected.any():
        return values.new_zeros(())
    travel = valid & collected.any(dim=1, keepdim=True) & ~collected
    log_probs = F.log_softmax(values.masked_fill(~mask, -torch.inf), dim=-1)
    nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    loss = nll[collected].mean()
    if travel.any():
        loss = loss + nll[travel].mean()
    return loss


class QPLEXLearner(ValueDecompositionLearner):
    def __init__(self, spec, config, device):
        super().__init__(spec, config, device)
        self.successful_action_loss_weight = config.get("successful_action_loss_weight", 0.0)
        if self.successful_action_loss_weight:
            self.extra_metrics = ("behavior_cloning_loss",)

    def build_mixer(self, spec, config):
        return DuplexDuelingMixer(spec, config)

    def train_batch(self, batch):
        actions, states = batch["actions"], batch["states"]
        values = unroll(self.controller, batch["observations"], actions)
        mask = self.controller.available_action_mask(batch.get("available_actions"))
        current_mask = mask[:, :-1] if mask.ndim > 2 else mask
        next_mask = mask[:, 1:] if mask.ndim > 2 else mask
        total = self.mixer(values[:, :-1], states[:, :-1], actions, current_mask)
        with torch.no_grad():
            target_values = unroll(self.target_controller, batch["observations"], actions)[:, 1:]
            selection = values[:, 1:] if self.config["double_q"] else target_values
            next_actions = self.controller.greedy_actions(selection, next_mask)
            next_total = self.target_mixer(target_values, states[:, 1:], next_actions, next_mask)
            targets = td_targets(
                batch["rewards"], batch["terminated"], next_total, self.config["gamma"]
            )
        loss = masked_td_loss(total, targets, batch["filled"])
        if self.successful_action_loss_weight:
            imitation_loss = successful_action_loss(values[:, :-1], actions, current_mask, batch)
            return self.optimize(
                loss + self.successful_action_loss_weight * imitation_loss,
                total,
                targets,
                batch["filled"],
                behavior_cloning_loss=float(imitation_loss.detach()),
            )
        return self.optimize(loss, total, targets, batch["filled"])
