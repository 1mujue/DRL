"""MAAC 训练循环：多头注意力 Critic + 离散 SAC 目标。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all


def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_out),
    )


class MAACActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions, hidden)

    def forward(self, obs):
        return self.actor(obs)


class AttentionCritic(nn.Module):
    """单个 agent 的注意力 Critic：Q_i(s, a)，多头注意力聚合所有 agent 的 (obs, action) 编码。"""

    def __init__(self, agent_idx, n_agents, obs_dim, n_actions,
                 embed_dim=64, n_heads=4, hidden=128):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.agent_idx = agent_idx
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + n_actions, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
        )
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs, all_act_oh):
        # all_obs: (B, N, obs_dim), all_act_oh: (B, N, n_actions)
        B, N, _ = all_obs.shape
        x = torch.cat([all_obs, all_act_oh], dim=-1)
        e = self.encoder(x)  # (B, N, embed_dim)

        q = self.W_q(e).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(e).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(e).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q_i = q[:, :, self.agent_idx:self.agent_idx + 1, :]                # (B, H, 1, D)
        attn = torch.softmax(q_i @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)
        c = (attn @ v).squeeze(-2).reshape(B, self.embed_dim)              # (B, embed_dim)
        c = self.W_o(c)

        e_i = e[:, self.agent_idx, :]                                       # (B, embed_dim)
        return self.head(torch.cat([e_i, c], dim=-1)).squeeze(-1)           # (B,)


def _joint_onehot(actions_tensor, n_actions):
    return F.one_hot(actions_tensor.long(), n_actions).float()


def maac_update(actors, critics, target_actors, target_critics,
                opts_actor, opts_critic, batch, cfg):
    names = cfg["_names"]
    n_actions = cfg["_n_actions"]
    gamma = cfg["gamma"]
    tau = cfg["tau"]
    alpha = cfg["alpha"]

    all_obs = batch["all_obs"]            # (B, N, obs_dim)
    all_next_obs = batch["all_next_obs"]  # (B, N, obs_dim)
    acts = batch["acts"]                  # (B, N) int64
    rew = batch["rew"]                    # (B,)
    term = batch["term"]                  # (B,)
    all_act_oh = _joint_onehot(acts, n_actions)

    stats = {}

    # ---- 1) Critic 更新（每个 agent 独立）----
    for i, a in enumerate(names):
        q = critics[a](all_obs, all_act_oh)
        with torch.no_grad():
            next_acts, next_logps = [], []
            for j, b in enumerate(names):
                dist_j = Categorical(logits=target_actors[b](all_next_obs[:, j, :]))
                a_j = dist_j.sample()
                next_acts.append(a_j)
                next_logps.append(dist_j.log_prob(a_j))
            next_act_oh = _joint_onehot(torch.stack(next_acts, dim=1), n_actions)
            next_q = target_critics[a](all_next_obs, next_act_oh)
            target_q = rew + gamma * (1 - term) * (next_q - alpha * next_logps[i])
        loss_c = (q - target_q).square().mean()
        opts_critic[a].zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(critics[a].parameters(), 10.0)
        opts_critic[a].step()
        stats[f"critic_loss_{a}"] = float(loss_c.detach())

    # ---- 2) Actor 更新（每个 agent 独立，离散 SAC 期望形式）----
    for i, a in enumerate(names):
        logits_i = actors[a](all_obs[:, i, :])
        probs_i = torch.softmax(logits_i, dim=-1)
        log_probs_i = torch.log_softmax(logits_i, dim=-1)

        q_values = []
        for u in range(n_actions):
            cf_oh = all_act_oh.clone()
            cf_oh[:, i, :] = 0
            cf_oh[:, i, u] = 1
            q_values.append(critics[a](all_obs, cf_oh))
        q_values = torch.stack(q_values, dim=-1)  # (B, n_actions)

        loss_a = (probs_i * (alpha * log_probs_i - q_values.detach())).sum(-1).mean()
        opts_actor[a].zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actors[a].parameters(), 10.0)
        opts_actor[a].step()
        stats[f"actor_loss_{a}"] = float(loss_a.detach())

    # ---- 3) 软更新 target critic + target actor ----
    with torch.no_grad():
        for a in names:
            for p, tp in zip(critics[a].parameters(), target_critics[a].parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            for p, tp in zip(actors[a].parameters(), target_actors[a].parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    return stats


def train(config, out):
    config = {**config, "env_kwargs": resolved_env_kwargs(config["env"], config.get("env_kwargs"))}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(config["seed"])
    device = torch.device(config.get("device", "cpu"))

    env = make_env(config["env"], **config.get("env_kwargs", {}))
    obs, _ = env.reset(seed=config["seed"])
    names = list(env.possible_agents)
    for i, a in enumerate(names):
        env.action_space(a).seed(config["seed"] + i)

    obs_dim = int(len(obs[names[0]]))
    n_actions = int(env.action_space(names[0]).n)
    n_agents = len(names)

    actor_hidden = int(config.get("actor_hidden", 64))
    critic_hidden = int(config.get("critic_hidden", 128))
    embed_dim = int(config.get("attn_embed_dim", 64))
    n_heads = int(config.get("attn_heads", 4))

    actors = {a: MAACActor(obs_dim, n_actions, actor_hidden).to(device) for a in names}
    critics = {
        a: AttentionCritic(i, n_agents, obs_dim, n_actions, embed_dim, n_heads, critic_hidden).to(device)
        for i, a in enumerate(names)
    }
    target_actors = {a: MAACActor(obs_dim, n_actions, actor_hidden).to(device) for a in names}
    target_critics = {
        a: AttentionCritic(i, n_agents, obs_dim, n_actions, embed_dim, n_heads, critic_hidden).to(device)
        for i, a in enumerate(names)
    }
    for a in names:
        target_actors[a].load_state_dict(actors[a].state_dict())
        target_critics[a].load_state_dict(critics[a].state_dict())

    opts_actor = {a: torch.optim.Adam(actors[a].parameters(), lr=config.get("lr_actor", 3e-4)) for a in names}
    opts_critic = {a: torch.optim.Adam(critics[a].parameters(), lr=config.get("lr_critic", 5e-4)) for a in names}

    cfg_internal = {**config, "_names": names, "_n_actions": n_actions}

    (out / "config.json").write_text(json.dumps(_to_python(config), indent=2), encoding="utf-8")

    episode, steps, ep_seed = 0, 0, config["seed"]
    returns = dict.fromkeys(names, 0.0)
    metrics = (out / "episodes.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics, fieldnames=["episode", "steps"] + names)
    writer.writeheader()

    try:
        while steps < config["total_steps"]:
            buf = {k: [] for k in ("obs", "act", "rew", "next_obs", "term", "end")}
            horizon = min(config.get("rollout_steps", 256), config["total_steps"] - steps)
            for _ in range(horizon):
                acts = {}
                with torch.no_grad():
                    for a in names:
                        local = torch.as_tensor(obs[a], dtype=torch.float32, device=device)
                        acts[a] = int(Categorical(logits=actors[a](local)).sample())
                next_obs, rewards, terminated, truncated, _ = env.step(acts)
                end = all(terminated[a] or truncated[a] for a in names)
                term = all(terminated[a] for a in names)
                reward_scale = cfg_internal.get("reward_scale", 10.0)
                team_r = float(rewards[names[0]]) / reward_scale

                buf["obs"].append({a: np.asarray(obs[a], dtype=np.float32) for a in names})
                buf["act"].append({a: np.int64(acts[a]) for a in names})
                buf["rew"].append(np.float32(team_r))
                buf["next_obs"].append({a: np.asarray(next_obs[a], dtype=np.float32) for a in names})
                buf["term"].append(np.float32(term))
                buf["end"].append(np.float32(end))

                for a in names:
                    returns[a] += float(rewards[a])
                obs = next_obs
                steps += 1
                if end:
                    episode += 1
                    writer.writerow({"episode": episode, "steps": steps, **returns})
                    metrics.flush()
                    returns = dict.fromkeys(names, 0.0)
                    ep_seed += 1
                    obs, _ = env.reset(seed=ep_seed)
                    if steps >= config["total_steps"]:
                        break

            if not buf["rew"]:
                break

            all_obs = torch.as_tensor(
                np.stack([np.stack([b[a] for a in names]) for b in buf["obs"]]),
                dtype=torch.float32, device=device,
            )
            all_next_obs = torch.as_tensor(
                np.stack([np.stack([b[a] for a in names]) for b in buf["next_obs"]]),
                dtype=torch.float32, device=device,
            )
            acts_t = torch.tensor(
                [[b[a] for a in names] for b in buf["act"]],
                dtype=torch.long, device=device,
            )
            batch = {
                "all_obs": all_obs,
                "all_next_obs": all_next_obs,
                "acts": acts_t,
                "rew": torch.tensor(buf["rew"], dtype=torch.float32, device=device),
                "term": torch.tensor(buf["term"], dtype=torch.float32, device=device),
                "end": torch.tensor(buf["end"], dtype=torch.float32, device=device),
            }
            update_epochs = int(cfg_internal.get("update_epochs", 4))
            for _ in range(update_epochs):
                stats = maac_update(
                    actors, critics, target_actors, target_critics,
                    opts_actor, opts_critic, batch, cfg_internal,
                )
            print(f"steps={steps} episodes={episode} {stats}", flush=True)

        torch.save(
            {
                "config": _to_python(config),
                "steps": int(steps),
                "agents": {a: p.state_dict() for a, p in actors.items()},
                "critics": {a: p.state_dict() for a, p in critics.items()},
                "obs_dim": obs_dim,
                "n_actions": n_actions,
            },
            out / "model.pt",
        )
    finally:
        metrics.close()
        env.close()

    return out / "model.pt"


def load_policy(checkpoint, env, observations):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    names = sorted(observations.keys())
    obs_dim = data["obs_dim"]
    n_actions = data["n_actions"]
    hidden = data["config"].get("actor_hidden", 64)
    actors = {a: MAACActor(obs_dim, n_actions, hidden) for a in names}
    for a, m in actors.items():
        m.load_state_dict(data["agents"][a])
        m.eval()
    return actors, data

"""MAAC 训练循环：多头注意力 Critic + 离散 SAC 目标。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all


def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_out),
    )


class MAACActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions, hidden)

    def forward(self, obs):
        return self.actor(obs)


class AttentionCritic(nn.Module):
    """单个 agent 的注意力 Critic：Q_i(s, a)，多头注意力聚合所有 agent 的 (obs, action) 编码。"""

    def __init__(self, agent_idx, n_agents, obs_dim, n_actions,
                 embed_dim=64, n_heads=4, hidden=128):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.agent_idx = agent_idx
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + n_actions, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
        )
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs, all_act_oh):
        # all_obs: (B, N, obs_dim), all_act_oh: (B, N, n_actions)
        B, N, _ = all_obs.shape
        x = torch.cat([all_obs, all_act_oh], dim=-1)
        e = self.encoder(x)  # (B, N, embed_dim)

        q = self.W_q(e).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(e).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(e).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q_i = q[:, :, self.agent_idx:self.agent_idx + 1, :]                # (B, H, 1, D)
        attn = torch.softmax(q_i @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)
        c = (attn @ v).squeeze(-2).reshape(B, self.embed_dim)              # (B, embed_dim)
        c = self.W_o(c)

        e_i = e[:, self.agent_idx, :]                                       # (B, embed_dim)
        return self.head(torch.cat([e_i, c], dim=-1)).squeeze(-1)           # (B,)


def _joint_onehot(actions_tensor, n_actions):
    return F.one_hot(actions_tensor.long(), n_actions).float()


def maac_update(actors, critics, target_actors, target_critics,
                opts_actor, opts_critic, batch, cfg):
    names = cfg["_names"]
    n_actions = cfg["_n_actions"]
    gamma = cfg["gamma"]
    tau = cfg["tau"]
    alpha = cfg["alpha"]

    all_obs = batch["all_obs"]            # (B, N, obs_dim)
    all_next_obs = batch["all_next_obs"]  # (B, N, obs_dim)
    acts = batch["acts"]                  # (B, N) int64
    rew = batch["rew"]                    # (B,)
    term = batch["term"]                  # (B,)
    all_act_oh = _joint_onehot(acts, n_actions)

    stats = {}

    # ---- 1) Critic 更新（每个 agent 独立）----
    for i, a in enumerate(names):
        q = critics[a](all_obs, all_act_oh)
        with torch.no_grad():
            next_acts, next_logps = [], []
            for j, b in enumerate(names):
                dist_j = Categorical(logits=target_actors[b](all_next_obs[:, j, :]))
                a_j = dist_j.sample()
                next_acts.append(a_j)
                next_logps.append(dist_j.log_prob(a_j))
            next_act_oh = _joint_onehot(torch.stack(next_acts, dim=1), n_actions)
            next_q = target_critics[a](all_next_obs, next_act_oh)
            target_q = rew + gamma * (1 - term) * (next_q - alpha * next_logps[i])
        loss_c = (q - target_q).square().mean()
        opts_critic[a].zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(critics[a].parameters(), 10.0)
        opts_critic[a].step()
        stats[f"critic_loss_{a}"] = float(loss_c.detach())

    # ---- 2) Actor 更新（每个 agent 独立，离散 SAC 期望形式）----
    for i, a in enumerate(names):
        logits_i = actors[a](all_obs[:, i, :])
        probs_i = torch.softmax(logits_i, dim=-1)
        log_probs_i = torch.log_softmax(logits_i, dim=-1)

        q_values = []
        for u in range(n_actions):
            cf_oh = all_act_oh.clone()
            cf_oh[:, i, :] = 0
            cf_oh[:, i, u] = 1
            q_values.append(critics[a](all_obs, cf_oh))
        q_values = torch.stack(q_values, dim=-1)  # (B, n_actions)

        loss_a = (probs_i * (alpha * log_probs_i - q_values.detach())).sum(-1).mean()
        opts_actor[a].zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actors[a].parameters(), 10.0)
        opts_actor[a].step()
        stats[f"actor_loss_{a}"] = float(loss_a.detach())

    # ---- 3) 软更新 target critic + target actor ----
    with torch.no_grad():
        for a in names:
            for p, tp in zip(critics[a].parameters(), target_critics[a].parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            for p, tp in zip(actors[a].parameters(), target_actors[a].parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    return stats


def train(config, out):
    config = {**config, "env_kwargs": resolved_env_kwargs(config["env"], config.get("env_kwargs"))}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(config["seed"])
    device = torch.device(config.get("device", "cpu"))

    env = make_env(config["env"], **config.get("env_kwargs", {}))
    obs, _ = env.reset(seed=config["seed"])
    names = list(env.possible_agents)
    for i, a in enumerate(names):
        env.action_space(a).seed(config["seed"] + i)

    obs_dim = int(len(obs[names[0]]))
    n_actions = int(env.action_space(names[0]).n)
    n_agents = len(names)

    actor_hidden = int(config.get("actor_hidden", 64))
    critic_hidden = int(config.get("critic_hidden", 128))
    embed_dim = int(config.get("attn_embed_dim", 64))
    n_heads = int(config.get("attn_heads", 4))

    actors = {a: MAACActor(obs_dim, n_actions, actor_hidden).to(device) for a in names}
    critics = {
        a: AttentionCritic(i, n_agents, obs_dim, n_actions, embed_dim, n_heads, critic_hidden).to(device)
        for i, a in enumerate(names)
    }
    target_actors = {a: MAACActor(obs_dim, n_actions, actor_hidden).to(device) for a in names}
    target_critics = {
        a: AttentionCritic(i, n_agents, obs_dim, n_actions, embed_dim, n_heads, critic_hidden).to(device)
        for i, a in enumerate(names)
    }
    for a in names:
        target_actors[a].load_state_dict(actors[a].state_dict())
        target_critics[a].load_state_dict(critics[a].state_dict())

    opts_actor = {a: torch.optim.Adam(actors[a].parameters(), lr=config.get("lr_actor", 3e-4)) for a in names}
    opts_critic = {a: torch.optim.Adam(critics[a].parameters(), lr=config.get("lr_critic", 5e-4)) for a in names}

    cfg_internal = {**config, "_names": names, "_n_actions": n_actions}

    (out / "config.json").write_text(json.dumps(_to_python(config), indent=2), encoding="utf-8")

    episode, steps, ep_seed = 0, 0, config["seed"]
    returns = dict.fromkeys(names, 0.0)
    metrics = (out / "episodes.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics, fieldnames=["episode", "steps"] + names)
    writer.writeheader()

    try:
        while steps < config["total_steps"]:
            buf = {k: [] for k in ("obs", "act", "rew", "next_obs", "term", "end")}
            horizon = min(config.get("rollout_steps", 256), config["total_steps"] - steps)
            for _ in range(horizon):
                acts = {}
                with torch.no_grad():
                    for a in names:
                        local = torch.as_tensor(obs[a], dtype=torch.float32, device=device)
                        acts[a] = int(Categorical(logits=actors[a](local)).sample())
                next_obs, rewards, terminated, truncated, _ = env.step(acts)
                end = all(terminated[a] or truncated[a] for a in names)
                term = all(terminated[a] for a in names)
                team_r = float(rewards[names[0]])

                buf["obs"].append({a: np.asarray(obs[a], dtype=np.float32) for a in names})
                buf["act"].append({a: np.int64(acts[a]) for a in names})
                buf["rew"].append(np.float32(team_r))
                buf["next_obs"].append({a: np.asarray(next_obs[a], dtype=np.float32) for a in names})
                buf["term"].append(np.float32(term))
                buf["end"].append(np.float32(end))

                for a in names:
                    returns[a] += float(rewards[a])
                obs = next_obs
                steps += 1
                if end:
                    episode += 1
                    writer.writerow({"episode": episode, "steps": steps, **returns})
                    metrics.flush()
                    returns = dict.fromkeys(names, 0.0)
                    ep_seed += 1
                    obs, _ = env.reset(seed=ep_seed)
                    if steps >= config["total_steps"]:
                        break

            if not buf["rew"]:
                break

            all_obs = torch.as_tensor(
                np.stack([np.stack([b[a] for a in names]) for b in buf["obs"]]),
                dtype=torch.float32, device=device,
            )
            all_next_obs = torch.as_tensor(
                np.stack([np.stack([b[a] for a in names]) for b in buf["next_obs"]]),
                dtype=torch.float32, device=device,
            )
            acts_t = torch.tensor(
                [[b[a] for a in names] for b in buf["act"]],
                dtype=torch.long, device=device,
            )
            batch = {
                "all_obs": all_obs,
                "all_next_obs": all_next_obs,
                "acts": acts_t,
                "rew": torch.tensor(buf["rew"], dtype=torch.float32, device=device),
                "term": torch.tensor(buf["term"], dtype=torch.float32, device=device),
                "end": torch.tensor(buf["end"], dtype=torch.float32, device=device),
            }
            stats = maac_update(
                actors, critics, target_actors, target_critics,
                opts_actor, opts_critic, batch, cfg_internal,
            )
            print(f"steps={steps} episodes={episode} {stats}", flush=True)

        torch.save(
            {
                "config": _to_python(config),
                "steps": int(steps),
                "agents": {a: p.state_dict() for a, p in actors.items()},
                "critics": {a: p.state_dict() for a, p in critics.items()},
                "obs_dim": obs_dim,
                "n_actions": n_actions,
            },
            out / "model.pt",
        )
    finally:
        metrics.close()
        env.close()

    return out / "model.pt"


def load_policy(checkpoint, env, observations):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    names = sorted(observations.keys())
    obs_dim = data["obs_dim"]
    n_actions = data["n_actions"]
    hidden = data["config"].get("actor_hidden", 64)
    actors = {a: MAACActor(obs_dim, n_actions, hidden) for a in names}
    for a, m in actors.items():
        m.load_state_dict(data["agents"][a])
        m.eval()
    return actors, data