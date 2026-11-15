"""独立 DQN：每个 agent 独立网络、回放、目标网络与优化器；无参数共享。"""

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all, write_provenance
from ..training import training_agents, with_opponent_actions

DEFAULTS = {
    "device": "cpu",
    "hidden_size": 64,
    "lr": 0.0005,
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 64,
    "learning_starts": 1000,
    "train_interval": 4,
    "target_update_interval": 200,
    "max_grad_norm": 10.0,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 50000,
    "eval_epsilon": 0.05,
}


class QNetwork(nn.Sequential):
    def __init__(self, observations, actions, hidden_size):
        super().__init__(
            nn.Linear(observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions),
        )


class ReplayBuffer:
    def __init__(self, capacity, observation_size):
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.capacity, self.position, self.size = capacity, 0, 0

    def add(self, observation, action, reward, next_observation, terminated):
        i = self.position
        self.observations[i] = observation
        self.next_observations[i] = next_observation
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        ids = np.random.choice(self.size, batch_size, replace=False)
        return tuple(
            torch.as_tensor(array[ids], device=device)
            for array in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.terminated,
            )
        )


def resolve_config(config):
    config = {**DEFAULTS, **config}
    if config["algorithm"] != "iql":
        raise ValueError("IQL requires algorithm: iql")
    for key in (
        "total_steps",
        "hidden_size",
        "buffer_size",
        "batch_size",
        "train_interval",
        "target_update_interval",
        "epsilon_decay_steps",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["learning_starts"]) is not int or config["learning_starts"] < 0:
        raise ValueError("learning_starts must be a nonnegative integer")
    if config["batch_size"] > config["buffer_size"]:
        raise ValueError("batch_size must not exceed buffer_size")
    for key in ("gamma", "epsilon_start", "epsilon_end", "eval_epsilon"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if config["epsilon_end"] > config["epsilon_start"]:
        raise ValueError("epsilon_end must not exceed epsilon_start")
    for key in ("lr", "max_grad_norm"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    config["env_kwargs"] = resolved_env_kwargs(config["env"], config.get("env_kwargs"))
    return config


def epsilon_at(step, config):
    fraction = min(step / config["epsilon_decay_steps"], 1.0)
    return config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])


def td_targets(rewards, terminated, next_values, gamma):
    # 截断仍 bootstrap；传入的必须是 reset 之前的最终观测。
    return rewards + gamma * (1 - terminated) * next_values


def update(model, target, optimizer, batch, config):
    observations, actions, rewards, next_observations, terminated = batch
    predictions = model(observations).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        targets = td_targets(
            rewards, terminated, target(next_observations).max(dim=1).values, config["gamma"]
        )
    loss = F.smooth_l1_loss(predictions, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite IQL loss")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    optimizer.step()
    return float(loss.item())


def create_agents(env, observations, config):
    return {
        a: QNetwork(len(observations[a]), env.action_space(a).n, config["hidden_size"]).to(
            config["device"]
        )
        for a in training_agents(env, config)
    }


def train(config, out):
    config = resolve_config(config)
    seed_all(config["seed"])
    env = make_env(config["env"], **config["env_kwargs"])
    try:
        observations, _ = env.reset(seed=config["seed"])
        for i, a in enumerate(env.possible_agents):
            env.action_space(a).seed(config["seed"] + i)
        models = create_agents(env, observations, config)
        targets = {a: deepcopy(m).eval().requires_grad_(False) for a, m in models.items()}
        optimizers = {
            a: torch.optim.Adam(m.parameters(), lr=config["lr"]) for a, m in models.items()
        }
        buffers = {a: ReplayBuffer(config["buffer_size"], len(observations[a])) for a in models}
        updates = dict.fromkeys(models, 0)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        write_provenance(out, config["device"])
        returns = dict.fromkeys(env.possible_agents, 0.0)
        episode = 0
        with (
            (out / "episodes.csv").open("w", newline="", encoding="utf-8") as episodes,
            (out / "updates.csv").open("w", newline="", encoding="utf-8") as losses,
        ):
            writer = csv.DictWriter(episodes, fieldnames=["episode", "steps", *returns])
            writer.writeheader()
            loss_writer = csv.DictWriter(losses, fieldnames=["steps", "agent", "loss", "epsilon"])
            loss_writer.writeheader()
            for step in range(1, config["total_steps"] + 1):
                epsilon = epsilon_at(step - 1, config)
                actions = {}
                with torch.no_grad():
                    for a, model in models.items():
                        if np.random.random() < epsilon:
                            actions[a] = int(env.action_space(a).sample())
                        else:
                            obs = torch.as_tensor(
                                observations[a], dtype=torch.float32, device=config["device"]
                            )
                            actions[a] = int(model(obs).argmax().item())
                joint_actions = with_opponent_actions(env, observations, actions, config)
                next_obs, rewards, terminated, truncated, _ = env.step(joint_actions)
                for a in models:
                    buffers[a].add(
                        observations[a], actions[a], rewards[a], next_obs[a], terminated[a]
                    )
                for a in returns:
                    returns[a] += float(rewards[a])
                if step >= config["learning_starts"] and step % config["train_interval"] == 0:
                    for a, model in models.items():
                        if buffers[a].size >= config["batch_size"]:
                            loss = update(
                                model,
                                targets[a],
                                optimizers[a],
                                buffers[a].sample(config["batch_size"], config["device"]),
                                config,
                            )
                            updates[a] += 1
                            loss_writer.writerow(
                                {"steps": step, "agent": a, "loss": loss, "epsilon": epsilon}
                            )
                if step % config["target_update_interval"] == 0:
                    for a in models:
                        targets[a].load_state_dict(models[a].state_dict())
                observations = next_obs
                if all(terminated[a] or truncated[a] for a in env.possible_agents):
                    episode += 1
                    writer.writerow({"episode": episode, "steps": step, **returns})
                    episodes.flush()
                    returns = dict.fromkeys(env.possible_agents, 0.0)
                    observations, _ = env.reset(seed=config["seed"] + episode)
                if step % 1000 == 0 or step == config["total_steps"]:
                    print(f"steps={step} episodes={episode} epsilon={epsilon:.4f}", flush=True)
        torch.save(
            {
                "config": config,
                "steps": config["total_steps"],
                "updates": updates,
                "agents": {a: m.state_dict() for a, m in models.items()},
            },
            out / "model.pt",
        )
        return out / "model.pt"
    finally:
        env.close()


def load_policy(checkpoint, env, observations):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = resolve_config(data["config"])
    models = create_agents(env, observations, {**config, "device": "cpu"})
    if set(models) != set(data["agents"]):
        raise ValueError("Checkpoint agents do not match the training team")
    for a, model in models.items():
        model.load_state_dict(data["agents"][a])
        model.eval()
    return models, data
