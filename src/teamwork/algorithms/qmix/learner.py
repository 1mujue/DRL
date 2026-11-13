"""QMIX 的离策略 TD 更新：联合回放、Double Q、目标网络与有效步掩码。"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .networks import LocalQController, QMixer


def unroll(controller, observations, actions):
    """从回合起点展开 GRU；observations：[B,T+1,N,O]，Q：[B,T+1,N,A]。"""
    batch_size, time_size, n_agents, _ = observations.shape
    if not controller.use_rnn:
        # 前馈网络没有跨步隐藏依赖，将 B、T 合并为一个 batch，保留局部上一动作。
        previous = observations.new_zeros(batch_size, time_size, n_agents, controller.max_actions)
        if time_size > 1:
            previous[:, 1:] = F.one_hot(
                actions[:, :time_size - 1], controller.max_actions
            ).to(observations.dtype)
        values, _ = controller(
            observations.flatten(0, 1), previous.flatten(0, 1),
            controller.initial_hidden(batch_size * time_size),
        )
        return values.reshape(batch_size, time_size, n_agents, controller.max_actions)
    hidden = controller.initial_hidden(batch_size)
    previous = observations.new_zeros(batch_size, n_agents, controller.max_actions)
    outputs = []
    for time in range(time_size):
        values, hidden = controller(observations[:, time], previous, hidden)
        outputs.append(values)
        if time < actions.shape[1]:
            previous = F.one_hot(actions[:, time], controller.max_actions).to(observations.dtype)
    return torch.stack(outputs, dim=1)


def td_targets(rewards, terminated, next_values, gamma):
    # truncated 不关闭 bootstrap：回放保留截断时的末观测与完整历史。
    return rewards + gamma * (1.0 - terminated) * next_values


def masked_td_loss(predictions, targets, filled):
    """只对实际采集的转移取均值，padding 不影响梯度或统计分母。"""
    valid = filled.bool()
    if not valid.any():
        raise ValueError("A QMIX learner batch must contain valid transitions")
    errors = predictions[valid] - targets[valid]
    if not torch.isfinite(errors).all():
        raise FloatingPointError("Nonfinite QMIX TD error")
    return errors.square().mean()


class QMIXLearner:
    def __init__(self, spec, config, device):
        self.config = config
        self.controller = LocalQController(spec, config).to(device)
        self.mixer = QMixer(
            len(spec.agents), spec.state_size, config["mixing_size"], config["hypernet_size"]
        ).to(device)
        self.target_controller = deepcopy(self.controller).requires_grad_(False).eval()
        self.target_mixer = deepcopy(self.mixer).requires_grad_(False).eval()
        self.parameters = list(self.controller.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(
            self.parameters, lr=config["lr"], alpha=config["rmsprop_alpha"],
            eps=config["optimizer_eps"],
        )
        self.updates = 0

    def sync_targets(self):
        self.target_controller.load_state_dict(self.controller.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train_batch(self, batch):
        all_values = unroll(self.controller, batch["observations"], batch["actions"])
        chosen = all_values[:, :-1].gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        total = self.mixer(chosen, batch["states"][:, :-1])
        with torch.no_grad():
            target_values = unroll(
                self.target_controller, batch["observations"], batch["actions"]
            )[:, 1:]
            if self.config["double_q"]:
                # 在线网络选下一动作，目标网络评价；不能用目标网络同时选和评价。
                greedy = self.controller.greedy_actions(all_values[:, 1:].detach())
            else:
                greedy = self.target_controller.greedy_actions(target_values)
            next_local = target_values.gather(-1, greedy.unsqueeze(-1)).squeeze(-1)
            next_total = self.target_mixer(next_local, batch["states"][:, 1:])
            targets = td_targets(batch["rewards"], batch["terminated"], next_total, self.config["gamma"])
        loss = masked_td_loss(total, targets, batch["filled"])
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.config["max_grad_norm"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite QMIX gradient")
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config["target_update_interval"] == 0:
            self.sync_targets()
        valid = batch["filled"].bool()
        return {
            "loss": float(loss.detach()), "grad_norm": float(grad_norm),
            "td_error_abs": float((total.detach()[valid] - targets[valid]).abs().mean()),
            "q_total_mean": float(total.detach()[valid].mean()),
            "target_mean": float(targets[valid].mean()),
            "valid_transitions": int(valid.sum()),
        }
