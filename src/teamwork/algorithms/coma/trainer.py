"""COMA 训练循环：集中式 Critic + 反事实基线 + 分布式 Actor。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all

# ----------------------------- 网络 -----------------------------

def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_out),
    )

def _to_python(obj):
    """递归把 numpy 标量转成 Python 原生类型，让 weights_only=True 能加载。"""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

class COMAActor(nn.Module):
    """暴露 .actor 属性，方便 COMAPolicy 写成与 PPOPolicy 同构的形式。"""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions, hidden)

    def forward(self, obs):
        return self.actor(obs)


class COMACritic(nn.Module):
    """集中式 Q(所有 obs 拼接, 所有动作 one-hot 拼接) -> (B,)。"""

    def __init__(self, n_agents, obs_dim, n_actions, hidden=128):
        super().__init__()
        in_dim = n_agents * obs_dim + n_agents * n_actions
        self.net = mlp(in_dim, 1, hidden)

    def forward(self, all_obs, all_actions_onehot):
        return self.net(torch.cat([all_obs, all_actions_onehot], dim=-1)).squeeze(-1)


def _joint_actions_onehot(acts_dict, names, n_actions):
    return torch.cat(
        [F.one_hot(acts_dict[a].long(), n_actions).float() for a in names], dim=-1
    )


# ----------------------------- 更新 -----------------------------

def coma_update(actors, critic, target_critic, opt_actor, opt_critic, batch, cfg):
    names = cfg["_names"]
    n_actions = cfg["_n_actions"]
    gamma = cfg["gamma"]
    tau = cfg["tau"]

    obs = batch["obs"]            # dict[a -> (H, obs_dim)]
    acts = batch["act"]           # dict[a -> (H,)] int64
    rews = batch["rew"]           # (H,) 团队奖励
    next_obs = batch["next_obs"]
    term = batch["term"]          # (H,) 真正终止

    all_obs = torch.cat([obs[a] for a in names], dim=-1)
    all_next_obs = torch.cat([next_obs[a] for a in names], dim=-1)
    all_act_oh = _joint_actions_onehot(acts, names, n_actions)

    # ---- 1) Critic TD 更新 ----
    q = critic(all_obs, all_act_oh)  # (H,)
    with torch.no_grad():
        next_acts = {}
        for a in names:
            logits = actors[a].actor(next_obs[a])
            next_acts[a] = torch.distributions.Categorical(logits=logits).sample()
        next_oh = _joint_actions_onehot(next_acts, names, n_actions)
        next_q = target_critic(all_next_obs, next_oh)
        target_q = rews + gamma * (1.0 - term) * next_q

    critic_loss = (q - target_q).square().mean()
    opt_critic.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    opt_critic.step()

    # ---- 2) 反事实基线 ----
    # b_a(τ, u^{-a}) = Σ_{u'^a} π_a(u'^a|τ_a) · Q(τ, (u'^a, u^{-a}))
    with torch.no_grad():
        q_current = q.detach()
        baselines = {}
        for a in names:
            cf_qs = []
            for u in range(n_actions):
                cf_acts = {b: acts[b] for b in names}
                cf_acts[a] = torch.full_like(acts[a], u)
                cf_oh = _joint_actions_onehot(cf_acts, names, n_actions)
                cf_qs.append(critic(all_obs, cf_oh))
            cf_qs = torch.stack(cf_qs, dim=-1)                 # (H, n_actions)
            pi_a = F.softmax(actors[a](obs[a]), dim=-1)        # (H, n_actions)
            baselines[a] = (pi_a * cf_qs).sum(-1)              # (H,)

    # ---- 3) Actor 更新（REINFORCE 形式的反事实优势）----
    actor_loss = 0.0
    for a in names:
        logits = actors[a](obs[a])
        log_pi = F.log_softmax(logits, dim=-1).gather(
            -1, acts[a].long().unsqueeze(-1)
        ).squeeze(-1)
        adv = (q_current - baselines[a]).detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        actor_loss = actor_loss - (log_pi * adv).mean()
    opt_actor.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(
        [p for a in names for p in actors[a].parameters()], 10.0
    )
    opt_actor.step()

    # ---- 4) 软更新目标 Critic ----
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), target_critic.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

    return {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "q_mean": float(q.mean().detach()),
    }


# ----------------------------- 主循环 -----------------------------

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

    obs_dim = len(obs[names[0]])
    n_acts = {a: env.action_space(a).n for a in names}
    if len(set(n_acts.values())) != 1:
        raise ValueError("COMA skeleton assumes homogeneous action spaces")
    n_actions = n_acts[names[0]]

    actors = {
        a: COMAActor(obs_dim, n_actions, config.get("actor_hidden", 64)).to(device)
        for a in names
    }
    critic_hidden = config.get("critic_hidden", 128)
    critic = COMACritic(len(names), obs_dim, n_actions, critic_hidden).to(device)
    target_critic = COMACritic(len(names), obs_dim, n_actions, critic_hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())

    opt_actor = torch.optim.Adam(
        [p for a in names for p in actors[a].parameters()],
        lr=config.get("lr_actor", 3e-4),
    )
    opt_critic = torch.optim.Adam(critic.parameters(), lr=config.get("lr_critic", 5e-4))

    cfg_internal = {**config, "_names": names, "_n_actions": n_actions}

    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

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
                    for a, model in actors.items():
                        local = torch.as_tensor(obs[a], dtype=torch.float32, device=device)
                        acts[a] = int(Categorical(logits=model(local)).sample())
                next_obs, rewards, terminated, truncated, _ = env.step(acts)
                end = all(terminated[a] or truncated[a] for a in names)
                term = all(terminated[a] for a in names)
                reward_scale = cfg_internal.get("reward_scale", 10.0)
                team_r = float(rewards[names[0]]) / reward_scale  # 共享奖励假设

                buf["obs"].append({a: np.asarray(obs[a], dtype=np.float32) for a in names})
                buf["act"].append({a: np.int64(acts[a]) for a in names})
                buf["rew"].append(np.float32(team_r))
                buf["next_obs"].append(
                    {a: np.asarray(next_obs[a], dtype=np.float32) for a in names}
                )
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

            batch = {
                "obs": {
                    a: torch.as_tensor(
                        np.stack([b[a] for b in buf["obs"]]), dtype=torch.float32, device=device
                    )
                    for a in names
                },
                "act": {
                    a: torch.tensor([b[a] for b in buf["act"]], dtype=torch.long, device=device)
                    for a in names
                },
                "rew": torch.tensor(buf["rew"], dtype=torch.float32, device=device),
                "next_obs": {
                    a: torch.as_tensor(
                        np.stack([b[a] for b in buf["next_obs"]]), dtype=torch.float32, device=device
                    )
                    for a in names
                },
                "term": torch.tensor(buf["term"], dtype=torch.float32, device=device),
                "end": torch.tensor(buf["end"], dtype=torch.float32, device=device),
            }
            update_epochs = int(cfg_internal.get("update_epochs", 4))
            for _ in range(update_epochs):
                stats = coma_update(
                    actors, critic, target_critic, opt_actor, opt_critic, batch, cfg_internal
                )
            print(f"steps={steps} episodes={episode} {stats}", flush=True)

        torch.save(
            {
                "config": _to_python(config),
                "steps": int(steps),
                "agents": {a: p.state_dict() for a, p in actors.items()},
                "critic": critic.state_dict(),
                "obs_dim": int(obs_dim),
                "n_actions": int(n_actions),
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
    actors = {a: COMAActor(obs_dim, n_actions, hidden) for a in names}
    for a, m in actors.items():
        m.load_state_dict(data["agents"][a])
        m.eval()
    return actors, data