"""On-policy PPO training for MAT-Dec's local actors and Transformer critic."""

import csv
import hashlib
import importlib.metadata
import json
import math
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all
from .model import create_model, environment_spec

REFERENCE_COMMIT = "be3ff49c8264d454c1fe2c41582aa2bfc98498c8"
_DEFAULTS = {
    "algorithm": "mat_dec",
    "seed": 0,
    "device": "cpu",
    "embedding_dim": 64,
    "num_heads": 1,
    "num_blocks": 1,
    "share_actor": False,
    "team_reward": "mean",
    "rollout_steps": 256,
    "batch_size": 64,
    "epochs": 4,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "value_clip": True,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
}
DEFAULTS = _DEFAULTS
_UPDATE_FIELDS = [
    "steps",
    "rollout",
    "update",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
]


def resolve_config(config):
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    config = deepcopy({**_DEFAULTS, **config})
    if config["algorithm"] != "mat_dec":
        raise ValueError("Algorithm configuration does not match MAT-Dec")
    if config["env"] == "tag":
        raise ValueError("MAT-Dec currently supports cooperative teams; Tag is unsupported")
    if config.get("train_team", "all") != "all" or config.get("train_opponent") is not None:
        raise ValueError("MAT-Dec requires the full cooperative team without an opponent")
    for key in (
        "total_steps",
        "embedding_dim",
        "num_heads",
        "num_blocks",
        "rollout_steps",
        "batch_size",
        "epochs",
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        config[key] = int(value)
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], Integral):
        raise TypeError("seed must be a nonnegative integer")
    if not 0 <= config["seed"] <= 2**32 - 1:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    config["seed"] = int(config["seed"])
    if config["embedding_dim"] % config["num_heads"]:
        raise ValueError("embedding_dim must be divisible by num_heads")
    for key in ("share_actor", "value_clip"):
        if not isinstance(config[key], bool):
            raise TypeError(f"{key} must be a boolean")
    if config["team_reward"] not in {"mean", "sum"}:
        raise ValueError("team_reward must be 'mean' or 'sum'")
    for key in (
        "lr",
        "gamma",
        "gae_lambda",
        "clip",
        "entropy_coef",
        "value_coef",
        "max_grad_norm",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], Real):
            raise TypeError(f"{key} must be a finite number")
        config[key] = float(config[key])
        if not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["lr"] <= 0 or config["max_grad_norm"] <= 0:
        raise ValueError("lr and max_grad_norm must be positive")
    if not 0 <= config["gamma"] <= 1 or not 0 <= config["gae_lambda"] <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not 0 < config["clip"] < 1:
        raise ValueError("clip must be in (0, 1)")
    if config["entropy_coef"] < 0 or config["value_coef"] < 0:
        raise ValueError("Loss coefficients must be nonnegative")
    settings = config.get("env_kwargs", {})
    if not isinstance(settings, dict) or any(not isinstance(key, str) for key in settings):
        raise TypeError("env_kwargs must be a dictionary with string keys")
    config["env_kwargs"] = resolved_env_kwargs(config["env"], settings)
    json.dumps(config, allow_nan=False)
    resolve_device(config["device"])
    return config


def resolve_device(value):
    if not isinstance(value, str):
        raise TypeError("MAT-Dec device must be cpu or an available cuda device")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("MAT-Dec device must be cpu or an available cuda device") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("MAT-Dec device must be cpu or an available cuda device")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError("MAT-Dec CUDA device is not available")
    return device


def compute_gae(rewards, values, next_values, terminated, episode_end, gamma, gae_lambda):
    """Keep bootstrap at time limits, and stop traces at every reset boundary.

    Inputs are [time, agent]; terminated and episode_end are [time].
    """
    advantages = torch.empty_like(values)
    trace = torch.zeros_like(values[0])
    for time in range(len(values) - 1, -1, -1):
        delta = rewards[time] + gamma * (1 - terminated[time]) * next_values[time] - values[time]
        trace = delta + gamma * gae_lambda * (1 - episode_end[time]) * trace
        advantages[time] = trace
    return advantages, advantages + values


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip):
    """Clip each agent's categorical ratio, as in MAT's PPO trainer."""
    ratio = (new_log_probs - old_log_probs).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


def _optimize(model, optimizer, observations, batch, config):
    advantages, returns = compute_gae(
        batch["rewards"],
        batch["values"],
        batch["next_values"],
        batch["terminated"],
        batch["episode_end"],
        config["gamma"],
        config["gae_lambda"],
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
    model.train()
    for _ in range(config["epochs"]):
        # Shuffle joint timesteps, keeping all agents together for critic attention.
        order = torch.randperm(len(advantages), device=advantages.device)
        for indices in order.split(config["batch_size"]):
            current = {agent: obs[indices] for agent, obs in observations.items()}
            log_probs, entropies = model.action_statistics(current, batch["actions"][indices])
            old_log_probs = batch["log_probs"][indices]
            policy_loss = clipped_policy_loss(
                log_probs, old_log_probs, advantages[indices], config["clip"]
            )
            values = model.values(current)
            value_error = (values - returns[indices]).square()
            if config["value_clip"]:
                old_values = batch["values"][indices]
                clipped = old_values + (values - old_values).clamp(-config["clip"], config["clip"])
                value_error = torch.maximum(value_error, (clipped - returns[indices]).square())
            value_loss = 0.5 * value_error.mean()
            entropy = entropies.mean()
            loss = (
                policy_loss + config["value_coef"] * value_loss - config["entropy_coef"] * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MAT-Dec PPO loss")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                log_ratio = log_probs - old_log_probs
                ratio = log_ratio.exp()
                statistics = {
                    "loss": float(loss),
                    "policy_loss": float(policy_loss),
                    "value_loss": float(value_loss),
                    "entropy": float(entropy),
                    "approx_kl": float(((ratio - 1) - log_ratio).mean()),
                    "clip_fraction": float(((ratio - 1).abs() > config["clip"]).float().mean()),
                    "grad_norm": float(grad_norm),
                }
            yield statistics


def _local_tensors(observations, names, device):
    # Copy before stepping because a simulator may reuse NumPy observation arrays.
    return {
        agent: torch.tensor(np.asarray(observations[agent]), dtype=torch.float32, device=device)
        for agent in names
    }


def _provenance(config, device):
    versions = {}
    for package in ("torch", "numpy", "pettingzoo", "mpe2", "lbforaging", "rware"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    directory = Path(__file__).parent
    return {
        "algorithm": "mat_dec",
        "implementation": "project MAT-Dec adaptation; local MLP actors and Transformer critic",
        "reference": "https://arxiv.org/abs/2205.14953",
        "reference_code_commit": REFERENCE_COMMIT,
        "execution_information": "own local observation only; no action conditioning",
        "critic_information": "joint local observations during training only",
        "team_reward": config["team_reward"],
        "reward_formula": f"{config['team_reward']}_i(environment_reward_i)",
        "steps_unit": "joint environment transitions",
        "device": str(device),
        "source_sha256": hashlib.sha256(
            b"".join(
                (directory / name).read_bytes()
                for name in ("model.py", "trainer.py", "algorithm.py")
            )
        ).hexdigest(),
        "packages": versions,
    }


def train(config, out):
    config = resolve_config(config)
    device = torch.device(config["device"])
    seed_all(config["seed"])
    env = make_env(config["env"], **config["env_kwargs"])
    try:
        return _train(config, Path(out), device, env)
    finally:
        env.close()


def _train(config, out, device, env):
    observations, _ = env.reset(seed=config["seed"])
    model = create_model(env, observations, config, device)
    names = model.names
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    for index, agent in enumerate(names):
        env.action_space(agent).seed(config["seed"] + index)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(_provenance(config, device), indent=2), encoding="utf-8"
    )
    steps, episodes, rollouts, updates = 0, 0, 0, 0
    episode_seed = config["seed"]
    episode_returns = dict.fromkeys(names, 0.0)
    stats = None
    with (
        (out / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file,
        (out / "updates.csv").open("w", newline="", encoding="utf-8") as update_file,
    ):
        episode_writer = csv.DictWriter(episode_file, fieldnames=["episode", "steps", *names])
        episode_writer.writeheader()
        update_writer = csv.DictWriter(update_file, fieldnames=_UPDATE_FIELDS)
        update_writer.writeheader()
        while steps < config["total_steps"]:
            horizon = min(config["rollout_steps"], config["total_steps"] - steps)
            observation_buffer = {agent: [] for agent in names}
            storage = {
                key: []
                for key in (
                    "actions",
                    "log_probs",
                    "values",
                    "next_values",
                    "rewards",
                    "terminated",
                    "episode_end",
                )
            }
            model.eval()
            for _ in range(horizon):
                local = _local_tensors(observations, names, device)
                with torch.no_grad():
                    values = model.values(local)
                    distributions = [
                        Categorical(logits=model.local_logits(agent, local[agent]))
                        for agent in names
                    ]
                    action_tensor = torch.stack(
                        [distribution.sample() for distribution in distributions]
                    )
                    log_probs = torch.stack(
                        [
                            distribution.log_prob(action_tensor[index])
                            for index, distribution in enumerate(distributions)
                        ]
                    )
                actions = {agent: int(action_tensor[index]) for index, agent in enumerate(names)}
                next_observations, rewards, terminated, truncated, _ = env.step(actions)
                endings = [terminated[agent] or truncated[agent] for agent in names]
                end = all(endings)
                if (any(endings) and not end) or (not end and set(env.agents) != set(names)):
                    raise ValueError("MAT-Dec requires a fixed population with joint episode ends")
                if not set(names).issubset(next_observations):
                    raise ValueError(
                        "MAT-Dec requires final observations, including at time limits"
                    )
                terminal = end and any(terminated[agent] for agent in names)
                with torch.no_grad():
                    next_values = (
                        torch.zeros_like(values)
                        if terminal
                        else model.values(_local_tensors(next_observations, names, device))
                    )
                individual_rewards = [float(rewards[agent]) for agent in names]
                team_reward = float(
                    np.mean(individual_rewards)
                    if config["team_reward"] == "mean"
                    else np.sum(individual_rewards)
                )
                for agent in names:
                    observation_buffer[agent].append(local[agent])
                    episode_returns[agent] += float(rewards[agent])
                for key, value in (
                    ("actions", action_tensor),
                    ("log_probs", log_probs),
                    ("values", values),
                    ("next_values", next_values),
                    ("rewards", team_reward),
                    ("terminated", float(terminal)),
                    ("episode_end", float(end)),
                ):
                    storage[key].append(value)
                observations = next_observations
                steps += 1
                if end:
                    episodes += 1
                    episode_writer.writerow(
                        {"episode": episodes, "steps": steps, **episode_returns}
                    )
                    episode_file.flush()
                    episode_returns = dict.fromkeys(names, 0.0)
                    episode_seed += 1
                    if steps < config["total_steps"]:
                        observations, _ = env.reset(seed=episode_seed)
            batch = {
                key: torch.stack(values)
                if isinstance(values[0], torch.Tensor)
                else torch.tensor(values, dtype=torch.float32, device=device)
                for key, values in storage.items()
            }
            batch["rewards"] = batch["rewards"][:, None].expand(-1, len(names))
            joint_observations = {
                agent: torch.stack(values) for agent, values in observation_buffer.items()
            }
            rollouts += 1
            for stats in _optimize(model, optimizer, joint_observations, batch, config):
                updates += 1
                update_writer.writerow(
                    {"steps": steps, "rollout": rollouts, "update": updates, **stats}
                )
            update_file.flush()
            print(
                f"steps={steps} algorithm=mat_dec episodes={episodes} rollouts={rollouts} "
                f"updates={updates} loss={stats['loss']:.6f}",
                flush=True,
            )
    summary = {
        "steps": steps,
        "episodes": episodes,
        "rollouts": rollouts,
        "updates": updates,
        "team_reward": config["team_reward"],
        "last_update": stats,
    }
    (out / "mat_dec_training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "config": config,
            "steps": steps,
            "updates": updates,
            "rollouts": rollouts,
            "spec": model.spec,
            "model": model.state_dict(),
        },
        out / "model.pt",
    )
    return out / "model.pt"


def load_checkpoint(checkpoint, env, observations):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = resolve_config(data["config"])
    if environment_spec(env, observations) != data["spec"]:
        raise ValueError("MAT-Dec checkpoint spaces or agent order do not match the environment")
    model = create_model(env, observations, config)
    model.load_state_dict(data["model"])
    return model.eval(), data
