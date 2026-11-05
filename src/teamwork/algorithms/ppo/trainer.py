"""Independent actors; local (IPPO) or centralized (MAPPO-style) per-agent values.

Minimal feedforward categorical PPO baseline. Fixed population, separate agent
networks, no weight sharing, no recurrent policy, no self-play opponent pool.
"""

import csv
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ...environments.base import state
from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all


def mlp(n_in, n_out):
    return nn.Sequential(
        nn.Linear(n_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_out)
    )


class Agent(nn.Module):
    def __init__(self, observation_size, actions, value_size):
        super().__init__()
        self.actor = mlp(observation_size, actions)
        self.critic = mlp(value_size, 1)

    def value(self, value_input):
        return self.critic(value_input).squeeze(-1)


def gae(rewards, values, next_values, terminated, episode_end, gamma, lam):
    """Bootstrap time limits, cut GAE traces at all reset boundaries."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - episode_end[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def create_agents(env, observations, algorithm, device):
    global_size = len(state(env))
    return {
        a: Agent(
            len(observations[a]),
            env.action_space(a).n,
            global_size if algorithm == "mappo" else len(observations[a]),
        ).to(device)
        for a in env.possible_agents
    }


def train(config, out):
    # 环境默认参数统一来自评估协议；不再让训练默认 25 步、评估默认 50 步。
    config = {**config, "env_kwargs": resolved_env_kwargs(config["env"], config.get("env_kwargs"))}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    env = make_env(config["env"], **config.get("env_kwargs", {}))
    obs, _ = env.reset(seed=config["seed"])
    for i, a in enumerate(env.possible_agents):
        env.action_space(a).seed(config["seed"] + i)
    agents = create_agents(env, obs, config["algorithm"], device)
    optimizers = {
        a: torch.optim.Adam(p.parameters(), lr=config.get("lr", 3e-4)) for a, p in agents.items()
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        revision = "uncommitted"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        dirty = None
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "git_commit": revision,
                "git_dirty": dirty,
                "packages": {
                    d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                },
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "steps_unit": "joint environment transitions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    episode, steps, ep_seed = 0, 0, config["seed"]
    returns = dict.fromkeys(agents, 0.0)
    names = list(agents)
    metrics = (out / "episodes.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics, fieldnames=["episode", "steps"] + names)
    writer.writeheader()
    try:
        while steps < config["total_steps"]:
            buffers = {
                a: {
                    k: []
                    for k in ("obs", "vin", "act", "logp", "val", "next", "rew", "term", "end")
                }
                for a in names
            }
            horizon = min(config.get("rollout_steps", 256), config["total_steps"] - steps)
            for _ in range(horizon):
                global_obs = state(env)
                actions = {}
                with torch.no_grad():
                    for a, model in agents.items():
                        local = torch.as_tensor(obs[a], dtype=torch.float32, device=device)
                        vin = (
                            torch.as_tensor(global_obs, device=device)
                            if (config["algorithm"] == "mappo")
                            else local
                        )
                        distribution = Categorical(logits=model.actor(local))
                        action = distribution.sample()
                        b = buffers[a]
                        for key, value in (
                            ("obs", local),
                            ("vin", vin),
                            ("act", action),
                            ("logp", distribution.log_prob(action)),
                            ("val", model.value(vin)),
                        ):
                            b[key].append(value)
                        actions[a] = action.item()
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                next_state = state(env)
                end = all(terminated[a] or truncated[a] for a in names)
                with torch.no_grad():
                    for a, model in agents.items():
                        b = buffers[a]
                        vin = torch.as_tensor(
                            next_state if config["algorithm"] == "mappo" else next_obs[a],
                            dtype=torch.float32,
                            device=device,
                        )
                        b["next"].append(model.value(vin))
                        b["rew"].append(rewards[a])
                        b["term"].append(float(terminated[a]))
                        b["end"].append(float(end))
                        returns[a] += rewards[a]
                obs = next_obs
                steps += 1
                if end:
                    episode += 1
                    writer.writerow({"episode": episode, "steps": steps, **returns})
                    metrics.flush()
                    returns = dict.fromkeys(agents, 0.0)
                    ep_seed += 1
                    obs, _ = env.reset(seed=ep_seed)
            for a, model in agents.items():
                b = {
                    k: torch.stack(v)
                    if isinstance(v[0], torch.Tensor)
                    else torch.tensor(v, dtype=torch.float32, device=device)
                    for k, v in buffers[a].items()
                }
                adv, targets = gae(
                    b["rew"],
                    b["val"],
                    b["next"],
                    b["term"],
                    b["end"],
                    config.get("gamma", 0.99),
                    config.get("gae_lambda", 0.95),
                )
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                for _ in range(config.get("epochs", 4)):
                    for ids in torch.randperm(horizon, device=device).split(
                        config.get("batch_size", 64)
                    ):
                        dist = Categorical(logits=model.actor(b["obs"][ids]))
                        ratio = (dist.log_prob(b["act"][ids]) - b["logp"][ids]).exp()
                        clip = config.get("clip", 0.2)
                        policy_loss = -torch.min(
                            ratio * adv[ids], ratio.clamp(1 - clip, 1 + clip) * adv[ids]
                        ).mean()
                        value_loss = (model.value(b["vin"][ids]) - targets[ids]).square().mean()
                        loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite PPO loss")
                        optimizers[a].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizers[a].step()
            print(f"steps={steps} episodes={episode}", flush=True)
        torch.save(
            {
                "config": config,
                "steps": steps,
                "agents": {a: p.state_dict() for a, p in agents.items()},
            },
            out / "model.pt",
        )
    finally:
        metrics.close()
        env.close()
    return out / "model.pt"


def load_policy(checkpoint, env, observations):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agents = create_agents(env, observations, data["config"]["algorithm"], "cpu")
    for a, model in agents.items():
        model.load_state_dict(data["agents"][a])
        model.eval()
    return agents, data
