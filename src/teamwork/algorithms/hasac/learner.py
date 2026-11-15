"""HASAC 学习器：集中双 soft-Q、逐 agent 顺序软优势更新与可学温度。

离散动作空间下，actor 目标对**自身**动作做精确枚举期望（而非连续版的重参数化采样），
前序 agent 的动作来自刚刚更新过的当前策略采样，后续 agent 的动作沿用回放里的旧动作。
这正是 HASAC 的顺序软优势结构；critic 的 TD 目标则用目标网络的联合动作——包括熵项。
"""

import math

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .networks import HASACModel, agent_observation, resolve_actor_order


class HASACLearner:
    def __init__(self, spec, config, device, actor_order=None):
        self.spec = spec
        self.config = config
        self.device = device
        self.order = resolve_actor_order(
            spec, config.get("actor_order") if actor_order is None else actor_order
        )
        self.model = HASACModel(spec, config).to(device)
        self.actor_optimizers = {
            agent: torch.optim.Adam(
                self.model.actors[agent].parameters(), lr=config["actor_lr"]
            )
            for agent in spec.agents
        }
        self.critic_optimizer = torch.optim.Adam(
            self.model.critics.parameters(), lr=config["critic_lr"]
        )
        self.alpha_optimizers = (
            {
                agent: torch.optim.Adam([self.model.log_alphas[agent]], lr=config["alpha_lr"])
                for agent in spec.agents
            }
            if config["learn_alpha"]
            else {}
        )
        self.target_entropy = {
            agent: config["target_entropy_scale"] * math.log(count)
            for agent, count in zip(spec.agents, spec.action_counts)
        }
        self.updates = 0

    # ------------------------------------------------------------------ helpers
    def alphas(self):
        return torch.stack(
            [self.model.log_alphas[agent].exp() for agent in self.spec.agents]
        )

    def _observation(self, observations, index):
        return agent_observation(
            observations, index, self.spec.observation_sizes[index]
        )

    @torch.no_grad()
    def _sample_joint(self, observations, targets):
        """为每个 agent 采样一个动作，返回 [M, N] 的动作与 [M, N] 的 log 概率。"""
        actions, log_probs = [], []
        for index, agent in enumerate(self.spec.agents):
            logits = self.model.actor_logits(
                agent, self._observation(observations, index), targets=targets
            )
            log_pi = F.log_softmax(logits, dim=-1)
            action = torch.multinomial(log_pi.exp(), 1).squeeze(-1)
            actions.append(action)
            log_probs.append(log_pi.gather(-1, action.unsqueeze(-1)).squeeze(-1))
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)

    @torch.no_grad()
    def _sample_agent(self, agent, observations, index):
        logits = self.model.actor_logits(agent, self._observation(observations, index))
        return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)

    def _flatten(self, batch):
        """把 [B, T, ...] 的回合批次压成只含有效转移的 [M, ...]。"""
        observations = batch["observations"]
        states = batch["states"]
        if observations.dim() != 4 or states.dim() != 3:
            raise ValueError("HASAC expects [B,T+1,N,Omax] observations and [B,T+1,S] states")
        valid = batch["filled"].reshape(-1).bool()
        if not bool(valid.any()):
            raise ValueError("HASAC needs at least one valid transition")

        def take(values):
            return values.reshape(-1, *values.shape[2:])[valid]

        return {
            "observations": take(observations[:, :-1]),
            "next_observations": take(observations[:, 1:]),
            "states": take(states[:, :-1]),
            "next_states": take(states[:, 1:]),
            "actions": take(batch["actions"]),
            "rewards": batch["rewards"].reshape(-1)[valid],
            "terminated": batch["terminated"].reshape(-1)[valid],
        }

    # ------------------------------------------------------------------ update
    def train_batch(self, batch):
        data = self._flatten(batch)
        names = list(self.spec.agents)
        config = self.config
        states, next_states = data["states"], data["next_states"]
        actions, rewards = data["actions"], data["rewards"]
        # ---- critic：软 TD 目标，熵项按 agent 求和 ----
        with torch.no_grad():
            next_actions, next_log_probs = self._sample_joint(data["next_observations"], True)
            joint_target = self.model.q_values(next_states, next_actions, targets=True).min(
                dim=-1
            ).values
            soft = joint_target - (self.alphas() * next_log_probs).sum(dim=-1)
            target = rewards + config["gamma"] * (1.0 - data["terminated"]) * soft
        joint_actions = self.model.q_values(states, actions)
        critic_loss = F.mse_loss(joint_actions, target.unsqueeze(-1).expand_as(joint_actions))
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite HASAC critic loss")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = clip_grad_norm_(
            self.model.critics.parameters(), config["max_grad_norm"]
        )
        if not torch.isfinite(critic_grad):
            raise FloatingPointError("Nonfinite HASAC critic gradient")
        self.critic_optimizer.step()

        # ---- actor：按 order 顺序更新，前序动作取自刚更新过的策略 ----
        boundary = actions.clone()
        actor_losses, entropies, actor_grads = [], [], []
        for index, agent in enumerate(self.order):
            column = names.index(agent)
            for previous in self.order[:index]:
                previous_column = names.index(previous)
                boundary[:, previous_column] = self._sample_agent(
                    previous, data["observations"], previous_column
                )
            count = self.spec.action_counts[column]
            logits = self.model.actor_logits(
                agent, self._observation(data["observations"], column)
            )
            log_pi = F.log_softmax(logits, dim=-1)
            probabilities = log_pi.exp()
            with torch.no_grad():
                candidates = boundary.unsqueeze(1).repeat(1, count, 1)
                candidates[:, :, column] = torch.arange(
                    count, device=boundary.device
                ).unsqueeze(0)
                expanded = states.unsqueeze(1).expand(-1, count, -1)
                q_values = self.model.q_values(
                    expanded.reshape(-1, states.shape[-1]),
                    candidates.reshape(-1, len(names)),
                ).min(dim=-1).values.reshape(len(states), count)
            baseline = (probabilities.detach() * q_values).sum(dim=-1, keepdim=True)
            alpha = self.model.log_alphas[agent].exp().detach()
            actor_loss = (
                probabilities * (alpha * log_pi - (q_values - baseline))
            ).sum(dim=-1).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Nonfinite HASAC actor loss")
            optimizer = self.actor_optimizers[agent]
            optimizer.zero_grad()
            actor_loss.backward()
            grad = clip_grad_norm_(
                self.model.actors[agent].parameters(), config["max_grad_norm"]
            )
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite HASAC actor gradient")
            optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            actor_grads.append(float(grad))
            # ---- 温度：按当前策略的精确熵朝向目标熵 ----
            with torch.no_grad():
                log_pi = F.log_softmax(
                    self.model.actor_logits(
                        agent, self._observation(data["observations"], column)
                    ),
                    dim=-1,
                )
                entropy = -(log_pi.exp() * log_pi).sum(dim=-1).mean()
            entropies.append(float(entropy))
            if agent in self.alpha_optimizers:
                alpha_loss = (
                    self.model.log_alphas[agent]
                    * (entropy - self.target_entropy[agent])
                ).mean()
                alpha_optimizer = self.alpha_optimizers[agent]
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

        self.model.sync_targets(config["tau"])
        self.updates += 1
        with torch.no_grad():
            alphas = self.alphas()
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "entropy": sum(entropies) / len(entropies),
            "alpha_mean": float(alphas.mean()),
            "q_mean": float(joint_actions.mean()),
            "target_mean": float(target.mean()),
            "critic_grad_norm": float(critic_grad),
            "actor_grad_norm": sum(actor_grads) / len(actor_grads),
        }
