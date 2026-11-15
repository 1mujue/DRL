"""A small VDN trainer for the shared project contract.

Per-agent utility networks consume local observations. Their Q-values are
summed during training and optimized against one team reward; execution stays
decentralized.
"""

import csv
import hashlib
import importlib.metadata
import json
import math
import random
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn

from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all

_DEFAULTS = {
    "algorithm": "vdn",
    "device": "cpu",
    "hidden_size": 64,
    "team_reward": "mean",
    "batch_size": 64,
    "warmup_steps": 256,
    "replay_capacity": 50_000,
    "update_frequency": 1,
    "target_update_frequency": 500,
    "log_frequency": 1_000,
    "lr": 3e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 50_000,
}
DEFAULTS = _DEFAULTS


def _mlp(n_in, n_out, hidden_size):
    return nn.Sequential(
        nn.Linear(n_in, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_out),
    )


@dataclass
class Transition:
    observations: dict[str, np.ndarray]
    actions: dict[str, int]
    reward: float
    next_observations: dict[str, np.ndarray]
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.data = []
        self.position = 0
        self.random = random.Random(seed)

    def add(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        return self.random.sample(self.data, size)

    def __len__(self):
        return len(self.data)


def create_q_networks(env, observations, device="cpu", hidden_size=64):
    """Build one local utility network per fixed agent."""
    networks = {}
    for agent in env.possible_agents:
        space, observation_space = env.action_space(agent), env.observation_space(agent)
        if not isinstance(space, Discrete) or space.start != 0:
            raise ValueError("VDN requires zero-based discrete action spaces")
        if not isinstance(observation_space, Box) or len(observation_space.shape) != 1:
            raise ValueError("VDN requires one-dimensional vector observations")
        observation = np.asarray(observations[agent], dtype=np.float32)
        if observation.shape != observation_space.shape or not np.isfinite(observation).all():
            raise ValueError(f"Invalid VDN observation for {agent}")
        networks[agent] = _mlp(len(observation), int(space.n), hidden_size).to(device)
    return networks


def _validate_task(config):
    if config["env"] == "tag":
        raise ValueError("VDN is a cooperative learner and is not supported for Tag")


def _resolve_config(config):
    config = {**_DEFAULTS, **config}
    if config["algorithm"] != "vdn":
        raise ValueError("Algorithm configuration does not match VDN")
    _validate_task(config)
    for key in (
        "total_steps",
        "hidden_size",
        "batch_size",
        "replay_capacity",
        "update_frequency",
        "target_update_frequency",
        "log_frequency",
        "epsilon_decay_steps",
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "warmup_steps"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["replay_capacity"] < config["batch_size"]:
        raise ValueError("replay_capacity must be >= batch_size")
    if config["team_reward"] not in {"mean", "sum"}:
        raise ValueError("team_reward must be 'mean' or 'sum'")
    for key in ("lr", "gamma", "epsilon_start", "epsilon_end"):
        value = float(config[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        config[key] = value
    if config["lr"] <= 0 or not 0 <= config["gamma"] <= 1:
        raise ValueError("lr must be positive and gamma must be in [0, 1]")
    if not 0 <= config["epsilon_end"] <= config["epsilon_start"] <= 1:
        raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
    config["env_kwargs"] = resolved_env_kwargs(config["env"], config.get("env_kwargs"))
    return config


def _team_reward(rewards, agents, mode):
    values = [float(rewards[agent]) for agent in agents]
    if mode == "mean":
        return float(np.mean(values))
    if mode == "sum":
        return float(np.sum(values))
    raise ValueError("team_reward must be 'mean' or 'sum'")


def _epsilon(config, step):
    start = float(config.get("epsilon_start", 1.0))
    end = float(config.get("epsilon_end", 0.05))
    decay = int(config.get("epsilon_decay_steps", 50_000))
    return max(end, start - (start - end) * step / decay)


def _optimize(networks, targets, optimizer, batch, names, gamma, device):
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    terminated = torch.tensor(
        [item.terminated for item in batch], dtype=torch.float32, device=device
    )
    joint_q = torch.zeros(len(batch), dtype=torch.float32, device=device)
    joint_target = torch.zeros(len(batch), dtype=torch.float32, device=device)
    for agent in names:
        observations = torch.as_tensor(
            np.stack([item.observations[agent] for item in batch]),
            dtype=torch.float32,
            device=device,
        )
        next_observations = torch.as_tensor(
            np.stack([item.next_observations[agent] for item in batch]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [item.actions[agent] for item in batch], dtype=torch.int64, device=device
        )
        joint_q = joint_q + networks[agent](observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            joint_target = joint_target + targets[agent](next_observations).max(dim=1).values
    expected = rewards + float(gamma) * (1.0 - terminated) * joint_target
    loss = nn.functional.smooth_l1_loss(joint_q, expected)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite VDN loss")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        [p for network in networks.values() for p in network.parameters()],
        10.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return float(loss.detach().cpu())


def _package_versions():
    versions = {}
    for package in ("torch", "numpy", "pettingzoo", "mpe2", "lbforaging", "rware"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def train(config, out):
    """Train VDN and return ``out/model.pt``."""
    config = _resolve_config(config)
    device = torch.device(config["device"])
    seed_all(config["seed"])
    env = make_env(config["env"], **config["env_kwargs"])
    try:
        return _train(config, Path(out), device, env)
    finally:
        env.close()


def _train(config, out, device, env):
    observations, _ = env.reset(seed=config["seed"])
    names = list(env.possible_agents)
    if not names or set(env.agents) != set(names):
        raise ValueError("VDN requires a fixed, nonempty agent population")
    networks = create_q_networks(env, observations, device, config["hidden_size"])
    targets = create_q_networks(env, observations, device, config["hidden_size"])
    for index, agent in enumerate(names):
        env.action_space(agent).seed(config["seed"] + index)
        targets[agent].load_state_dict(networks[agent].state_dict())
        targets[agent].eval().requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for network in networks.values() for parameter in network.parameters()],
        lr=config["lr"],
    )
    replay = ReplayBuffer(config["replay_capacity"], config["seed"])
    batch_size = config["batch_size"]
    loss_sum, updates, last_loss = 0.0, 0, None
    episode, steps, episode_seed = 0, 0, config["seed"]
    returns = dict.fromkeys(names, 0.0)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "algorithm": "vdn",
                "implementation": "project feedforward VDN; separate per-agent networks",
                "reference": "https://arxiv.org/abs/1706.05296",
                "team_reward": config["team_reward"],
                "reward_formula": f"{config['team_reward']}_i(environment_reward_i)",
                "steps_unit": "joint environment transitions",
                "device": str(device),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                    + Path(__file__).with_name("algorithm.py").read_bytes()
                ).hexdigest(),
                "packages": _package_versions(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (
        (out / "episodes.csv").open("w", newline="", encoding="utf-8") as episodes_file,
        (out / "updates.csv").open("w", newline="", encoding="utf-8") as updates_file,
    ):
        writer = csv.DictWriter(episodes_file, fieldnames=["episode", "steps"] + names)
        writer.writeheader()
        update_writer = csv.DictWriter(
            updates_file, fieldnames=["steps", "update", "loss", "epsilon"]
        )
        update_writer.writeheader()
        while steps < config["total_steps"]:
            actions = {}
            epsilon = _epsilon(config, steps)
            # Snapshot before step: simulators may reuse observation arrays in-place.
            previous = {a: np.asarray(observations[a], dtype=np.float32).copy() for a in names}
            with torch.no_grad():
                for agent in names:
                    if random.random() < epsilon:
                        actions[agent] = int(env.action_space(agent).sample())
                    else:
                        values = networks[agent](
                            torch.as_tensor(observations[agent], dtype=torch.float32, device=device)
                        )
                        actions[agent] = int(values.argmax().item())
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            endings = [terminated[agent] or truncated[agent] for agent in names]
            episode_end = all(endings)
            if (any(endings) and not episode_end) or (
                not episode_end and set(env.agents) != set(names)
            ):
                raise ValueError("VDN requires agents to remain active until a joint episode end")
            if not set(names).issubset(next_observations):
                raise ValueError("VDN requires final observations, including at time limits")
            # A task terminal cuts the TD target; a pure time limit still bootstraps.
            terminal = episode_end and any(terminated[agent] for agent in names)
            replay.add(
                Transition(
                    previous,
                    actions.copy(),
                    _team_reward(rewards, names, config["team_reward"]),
                    {a: np.asarray(next_observations[a], dtype=np.float32).copy() for a in names},
                    terminal,
                )
            )
            for agent in names:
                returns[agent] += float(rewards[agent])
            observations = next_observations
            steps += 1
            if (
                steps >= config["warmup_steps"]
                and len(replay) >= batch_size
                and steps % config["update_frequency"] == 0
            ):
                last_loss = _optimize(
                    networks,
                    targets,
                    optimizer,
                    replay.sample(batch_size),
                    names,
                    config["gamma"],
                    device,
                )
                updates += 1
                loss_sum += last_loss
                update_writer.writerow(
                    {"steps": steps, "update": updates, "loss": last_loss, "epsilon": epsilon}
                )
            if steps % config["target_update_frequency"] == 0:
                for agent in names:
                    targets[agent].load_state_dict(networks[agent].state_dict())
            if episode_end:
                episode += 1
                writer.writerow({"episode": episode, "steps": steps, **returns})
                episodes_file.flush()
                returns = dict.fromkeys(names, 0.0)
                episode_seed += 1
                if steps < config["total_steps"]:
                    observations, _ = env.reset(seed=episode_seed)
            if steps % config["log_frequency"] == 0 or steps == config["total_steps"]:
                updates_file.flush()
                print(
                    f"vdn steps={steps} episodes={episode} updates={updates} "
                    f"epsilon={epsilon:.3f} loss={last_loss}",
                    flush=True,
                )
        summary = {
            "steps": steps,
            "episodes": episode,
            "updates": updates,
            "mean_loss": loss_sum / updates if updates else None,
            "last_loss": last_loss,
            "team_reward": config["team_reward"],
        }
        (out / "vdn_training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        torch.save(
            {
                "config": config,
                "steps": steps,
                "loss": summary["mean_loss"],
                "updates": updates,
                "agents": {agent: networks[agent].state_dict() for agent in names},
            },
            out / "model.pt",
        )
    return out / "model.pt"


def load_checkpoint(checkpoint, env, observations):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data["config"].get("algorithm") != "vdn":
        raise ValueError("Checkpoint does not contain a VDN model")
    if data["config"]["env"] == "tag":
        raise ValueError("VDN does not support adversarial Tag checkpoints")
    if set(data["agents"]) != set(env.possible_agents):
        raise ValueError("VDN checkpoint agents do not match the environment")
    networks = create_q_networks(env, observations, "cpu", data["config"].get("hidden_size", 64))
    for agent, network in networks.items():
        network.load_state_dict(data["agents"][agent])
        network.eval()
    return networks, data


__all__ = ["ReplayBuffer", "create_q_networks", "load_checkpoint", "train"]
