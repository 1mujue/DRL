"""QTRAN-base 的集中 TD 更新、最优动作等式与采样非最优动作不等式。"""

import torch
from torch import nn
from torch.nn import functional as F

from ..qmix.learner import ValueDecompositionLearner, masked_td_loss, td_targets, unroll


class QTRANMixer(nn.Module):
    """联合 Q 使用全局状态和各 agent 的局部历史-动作编码。"""

    def __init__(self, spec, config):
        super().__init__()
        self.action_count = max(spec.action_counts)
        width = config["mixing_size"]
        self.encode_action = nn.Sequential(
            nn.Linear(config["hidden_size"] + self.action_count, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.joint_q = nn.Sequential(
            nn.Linear(spec.state_size + width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(spec.state_size, width), nn.ReLU(), nn.Linear(width, 1)
        )

    def forward(self, states, histories, actions):
        one_hot = F.one_hot(actions, self.action_count).to(states.dtype)
        encoded = self.encode_action(torch.cat((histories, one_hot), dim=-1)).sum(dim=-2)
        return self.joint_q(torch.cat((states, encoded), dim=-1)).squeeze(-1)


def constraint_losses(local_chosen, local_greedy, joint_chosen, joint_greedy, offset, filled):
    """固定联合 Q 的目标值，避免通过移动目标来满足局部一致性约束。"""
    valid = filled.bool()
    opt_error = local_greedy.sum(dim=-1) - joint_greedy.detach() + offset
    nopt_error = local_chosen.sum(dim=-1) - joint_chosen.detach() + offset
    return opt_error[valid].square().mean(), nopt_error[valid].clamp(max=0).square().mean()


class QTRANLearner(ValueDecompositionLearner):
    extra_metrics = ("td_loss", "opt_loss", "nopt_loss")

    def build_mixer(self, spec, config):
        return QTRANMixer(spec, config)

    def train_batch(self, batch):
        actions, states = batch["actions"], batch["states"]
        values, histories = unroll(
            self.controller, batch["observations"], actions, return_history=True
        )
        availability = batch.get("available_actions")
        greedy = self.controller.greedy_actions(values, availability)
        current = values[:, :-1]
        chosen = current.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        greedy_values = current.gather(-1, greedy[:, :-1].unsqueeze(-1)).squeeze(-1)

        total = self.mixer(states[:, :-1], histories[:, :-1], actions)
        with torch.no_grad():
            target_values, target_histories = unroll(
                self.target_controller, batch["observations"], actions, return_history=True
            )
            next_actions = greedy[:, 1:]
            if not self.config["double_q"]:
                next_available = availability[:, 1:] if availability is not None else None
                next_actions = self.target_controller.greedy_actions(target_values[:, 1:], next_available)
            next_total = self.target_mixer(states[:, 1:], target_histories[:, 1:], next_actions)
            greedy_total = self.mixer(states[:, :-1], histories[:, :-1], greedy[:, :-1])
            targets = td_targets(
                batch["rewards"], batch["terminated"], next_total, self.config["gamma"]
            )
        td_loss = masked_td_loss(total, targets, batch["filled"])
        opt_loss, nopt_loss = constraint_losses(
            chosen,
            greedy_values,
            total,
            greedy_total,
            self.mixer.value(states[:, :-1]).squeeze(-1),
            batch["filled"],
        )
        loss = (
            td_loss
            + self.config["opt_loss_weight"] * opt_loss
            + self.config["nopt_loss_weight"] * nopt_loss
        )
        return self.optimize(
            loss,
            total,
            targets,
            batch["filled"],
            td_loss=float(td_loss.detach()),
            opt_loss=float(opt_loss.detach()),
            nopt_loss=float(nopt_loss.detach()),
        )
