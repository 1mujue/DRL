"""单环境前馈 MAPPO：局部 actor、集中 critic、共享参数和独立指标日志。

    仅依赖公共环境接口；不读取 task_snapshot、MPE world 或 Gym 底层对象。
    此实现不包含 RNN、向量并行、自博弈池或精确断点续训。
"""

import csv
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ...environments.base import state
from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from .networks import MAPPOModel, ValueNormalizer, resolve_device, validate_observations

DEFAULTS = {
    "algorithm": "mappo_ff",
    "seed": 0,
    "device": "cpu",
    "rollout_steps": 256,
    "batch_size": 64,
    "epochs": 4,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "value_clip": 0.2,
    "value_loss_coef": 1.0,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "adam_eps": 1e-5,
    "huber_delta": 10.0,
    "use_clipped_value_loss": True,
    "use_huber_loss": True,
    "value_normalization": True,
    "feature_normalization": True,
    "share_policy": True,
    "hidden_size": 64,
    "hidden_layers": 2,
    "actor_output_gain": 0.01,
    "linear_lr_decay": False,
    "checkpoint_interval": 10000,
    "reward_aggregation": "individual",
}


def resolve_config(config):
    if not isinstance(config, dict):
        raise TypeError("MAPPO config must be a dictionary")
    result = {**DEFAULTS, **deepcopy(config)}
    if result["algorithm"] != "mappo_ff":
        raise ValueError("Feedforward MAPPO configuration requires algorithm='mappo_ff'")
    if not isinstance(result.get("env"), str) or not result["env"]:
        raise ValueError("MAPPO config requires an environment name")
    if "total_steps" not in result:
        raise ValueError("MAPPO config requires total_steps")
    if "env_kwargs" in result and not isinstance(result["env_kwargs"], dict):
        raise TypeError("env_kwargs must be a dictionary")
    try:
        result["env_kwargs"] = resolved_env_kwargs(result["env"], result.get("env_kwargs"))
    except KeyError:
        # 自定义已注册环境可显式提供参数，不能隐式冒用其他场景的默认条件。
        result.setdefault("env_kwargs", {})
    for key in ("total_steps", "rollout_steps", "batch_size", "epochs", "hidden_size", "hidden_layers"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int) or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    interval = result["checkpoint_interval"]
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError("checkpoint_interval must be a nonnegative integer (0 disables it)")
    for key in ("gamma", "gae_lambda"):
        if isinstance(result[key], bool) or not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    for key in (
        "lr", "clip", "value_clip", "max_grad_norm", "adam_eps", "huber_delta", "actor_output_gain"
    ):
        if isinstance(result[key], bool) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("value_loss_coef", "entropy_coef"):
        if isinstance(result[key], bool) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("actor_lr", "critic_lr"):
        result.setdefault(key, result["lr"])
        if isinstance(result[key], bool) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in (
        "use_clipped_value_loss", "use_huber_loss", "value_normalization",
        "feature_normalization", "share_policy", "linear_lr_decay"
    ):
        if not isinstance(result[key], bool):
            raise TypeError(f"{key} must be a boolean")
    if result["reward_aggregation"] not in {"individual", "mean", "sum", "shared"}:
        raise ValueError("reward_aggregation must be individual, mean, sum or shared")
    if result["env"] == "tag" and result["reward_aggregation"] != "individual":
        raise ValueError("MAPPO team reward modes cannot combine opposing Tag roles")
    json.dumps(result, allow_nan=False)
    return result


def aggregate_rewards(rewards, agents, mode):
    """训练奖励显式变换，CSV 始终保留环境原始个体回报。"""
    if set(rewards) != set(agents):
        raise ValueError("MAPPO rewards must cover the fixed agent population")
    values = np.asarray([rewards[agent] for agent in agents], dtype=np.float64)
    if values.shape != (len(agents),) or not np.isfinite(values).all():
        raise ValueError("MAPPO requires finite scalar rewards")
    if mode == "individual":
        return values
    if mode == "mean":
        value = values.mean()
    elif mode == "sum":
        value = values.sum()
    elif mode == "shared":
        if not np.allclose(values, values[0], rtol=1e-6, atol=1e-6):
            raise ValueError("shared reward mode requires identical rewards for all agents")
        value = values[0]
    else:
        raise ValueError("Unknown MAPPO reward aggregation")
    return np.full(len(agents), value, dtype=np.float64)


def training_contract(config):
    mode = config["reward_aggregation"]
    return {
        "algorithm": "mappo_ff", "implementation": "mappo_feedforward_v1",
        "reward_aggregation": mode,
        "reward_scope": "individual_agents" if mode == "individual" else "one_cooperative_team",
        "execution": "local_observation", "central_state": "critic_during_training_only",
        "step_unit": "joint_environment_steps", "resume_supported": False,
    }


def environment_state(env, spec):
    values = state(env).copy()
    if values.shape != (spec.state_size,) or not np.isfinite(values).all():
        raise ValueError("MAPPO requires a finite fixed-size centralized state")
    return values


def compute_gae(rewards, values, next_values, terminated, episode_end, gamma, gae_lambda):
    """输入 [T,N]：真正终止不 bootstrap；截断 bootstrap 但切断跨回合递推。"""
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * (1 - episode_end[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def value_loss(
    values, old_values, targets, clip=0.2, huber_delta=10.0,
    use_clipped_value_loss=True, use_huber_loss=True,
):
    """clip 的 current/old 均是网络归一化输出；targets 已按本轮统计归一化。"""
    def error_loss(errors):
        if not use_huber_loss:
            return 0.5 * errors.square()
        absolute = errors.abs()
        return torch.where(
            absolute <= huber_delta,
            0.5 * errors.square(),
            huber_delta * (absolute - 0.5 * huber_delta),
        )

    original = error_loss(targets - values)
    if use_clipped_value_loss:
        clipped = old_values + (values - old_values).clamp(-clip, clip)
        original = torch.maximum(original, error_loss(targets - clipped))
    return original.mean()


def _checkpoint(path, model, normalizer, config, steps):
    data = {
        "format": "teamwork.mappo.v1",
        "implementation": "mappo_feedforward_v1",
        "training_contract": training_contract(config),
        "resume_supported": False,
        "config": config,
        "steps": steps,
        "spec": asdict(model.spec),
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def _update(model, normalizer, actor_optimizers, critic_optimizer, batch, config):
    advantages, targets = compute_gae(
        batch["rewards"], batch["values"], batch["next_values"],
        batch["terminated"], batch["episode_end"], config["gamma"], config["gae_lambda"],
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    if config["value_normalization"]:
        # 官方 ValueNorm 语义：保留 rollout 的 normalized old_values 作 clip 基线。
        # 不能用新统计重算 old_values，而让 current prediction 保持原网络坐标。
        normalizer.update(targets)
        targets = normalizer.normalize(targets)
    totals = dict.fromkeys(
        ("policy_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm"), 0.0
    )
    actor_samples = 0
    names = list(model.spec.agents)
    for group, actor in model.actors.items():
        indices = [i for i, name in enumerate(names) if model.agent_to_group[name] == group]
        observations = torch.stack([batch["observations"][names[i]] for i in indices], dim=1)
        observations = observations.flatten(0, 1)
        actions = batch["actions"][:, indices].flatten()
        old_log_probs = batch["log_probs"][:, indices].flatten()
        group_advantages = advantages[:, indices].flatten()
        optimizer = actor_optimizers[group]
        for _ in range(config["epochs"]):
            permutation = torch.randperm(len(actions), device=actions.device)
            for ids in permutation.split(config["batch_size"]):
                distribution = Categorical(logits=actor(observations[ids]))
                log_ratio = distribution.log_prob(actions[ids]) - old_log_probs[ids]
                ratio = log_ratio.exp()
                surrogate = torch.minimum(
                    ratio * group_advantages[ids],
                    ratio.clamp(1 - config["clip"], 1 + config["clip"]) * group_advantages[ids],
                )
                policy_loss = -surrogate.mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss - config["entropy_coef"] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite MAPPO actor loss")
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), config["max_grad_norm"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite MAPPO actor gradient")
                optimizer.step()
                metrics = {
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "approx_kl": ((ratio - 1) - log_ratio).mean(),
                    "clip_fraction": ((ratio - 1).abs() > config["clip"]).float().mean(),
                    "actor_grad_norm": grad_norm,
                }
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * len(ids)
                actor_samples += len(ids)
    diagnostics = {key: value / actor_samples for key, value in totals.items()}
    inputs = model.critic_inputs(batch["states"]).flatten(0, 1)
    old_values = batch["old_normalized_values"].flatten()
    targets = targets.flatten()
    critic_loss_sum = critic_grad_sum = critic_samples = 0
    for _ in range(config["epochs"]):
        for ids in torch.randperm(len(targets), device=targets.device).split(config["batch_size"]):
            predictions = model.critic(inputs[ids]).squeeze(-1)
            loss = value_loss(
                predictions, old_values[ids], targets[ids], config["value_clip"],
                config["huber_delta"], config["use_clipped_value_loss"], config["use_huber_loss"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MAPPO critic loss")
            critic_optimizer.zero_grad()
            (config["value_loss_coef"] * loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.critic.parameters(), config["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite MAPPO critic gradient")
            critic_optimizer.step()
            critic_loss_sum += float(loss.detach()) * len(ids)
            critic_grad_sum += float(grad_norm) * len(ids)
            critic_samples += len(ids)
    diagnostics["value_loss"] = critic_loss_sum / critic_samples
    diagnostics["critic_grad_norm"] = critic_grad_sum / critic_samples
    diagnostics["learning_rate"] = next(iter(actor_optimizers.values())).param_groups[0]["lr"]
    return diagnostics


def train(config, out):
    config = resolve_config(config)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}")
    device = resolve_device(config["device"])
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    env = make_env(config["env"], **config.get("env_kwargs", {}))
    try:
        observations, _ = env.reset(seed=config["seed"])
        if config["reward_aggregation"] != "individual":
            roles = set(env.policy_metadata.get("roles", {}).values())
            if len(roles) > 1:
                raise ValueError("MAPPO team reward modes cannot combine opposing roles")
        spec = env.describe()
        model = MAPPOModel(spec, config).to(device)
        validate_observations(observations, spec, device)
        environment_state(env, spec)
        if set(env.agents) != set(spec.agents):
            raise ValueError("MAPPO requires the full fixed agent population after reset")
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (out / "training_contract.json").write_text(
            json.dumps(training_contract(config), indent=2), encoding="utf-8"
        )
        return _train_environment(config, out, env, device, observations, model)
    finally:
        env.close()


def _train_environment(config, out, env, device, observations, model):
    spec = model.spec
    names = list(spec.agents)
    normalizer = ValueNormalizer().to(device)
    actor_optimizers = {
        group: torch.optim.Adam(actor.parameters(), lr=config["actor_lr"], eps=config["adam_eps"])
        for group, actor in model.actors.items()
    }
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=config["critic_lr"], eps=config["adam_eps"]
    )
    for i, name in enumerate(names):
        env.action_space(name).seed(config["seed"] + i)
    fields = [
        "update", "steps", "episodes", "policy_loss", "value_loss", "entropy", "approx_kl",
        "clip_fraction", "actor_grad_norm", "critic_grad_norm", "learning_rate",
    ]
    steps = episodes = update = 0
    returns = dict.fromkeys(names, 0.0)
    next_checkpoint = config["checkpoint_interval"]
    with (
        (out / "episodes.csv").open("w", encoding="utf-8", newline="") as episode_file,
        (out / "updates.csv").open("w", encoding="utf-8", newline="") as update_file,
    ):
        episode_writer = csv.DictWriter(episode_file, fieldnames=["episode", "steps", *names])
        update_writer = csv.DictWriter(update_file, fieldnames=fields)
        episode_writer.writeheader()
        update_writer.writeheader()
        while steps < config["total_steps"]:
            if config["linear_lr_decay"]:
                fraction = 1 - steps / config["total_steps"]
                for optimizer in actor_optimizers.values():
                    optimizer.param_groups[0]["lr"] = config["actor_lr"] * fraction
                critic_optimizer.param_groups[0]["lr"] = config["critic_lr"] * fraction
            buffer = {key: [] for key in (
                "states", "actions", "log_probs", "old_normalized_values", "values",
                "next_values", "rewards", "terminated", "episode_end",
            )}
            obs_buffer = {name: [] for name in names}
            horizon = min(config["rollout_steps"], config["total_steps"] - steps)
            if next_checkpoint:
                horizon = min(horizon, next_checkpoint - steps)
            for _ in range(horizon):
                # 环境可能原地复用观测数组，必须在 step 之前冻结 rollout 输入。
                local = {
                    name: values.clone()
                    for name, values in validate_observations(observations, spec, device).items()
                }
                global_state = torch.tensor(
                    environment_state(env, spec), dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    logits = model.local_logits(local)
                    distributions = {name: Categorical(logits=logits[name]) for name in names}
                    actions = torch.stack([distributions[name].sample() for name in names])
                    log_probs = torch.stack([
                        distributions[name].log_prob(actions[i]) for i, name in enumerate(names)
                    ])
                    normalized_values = model.values(global_state)
                    values = (
                        normalizer.denormalize(normalized_values)
                        if config["value_normalization"] else normalized_values
                    )
                next_obs, rewards, terminated, truncated, _ = env.step({
                    name: int(actions[i]) for i, name in enumerate(names)
                })
                if set(terminated) != set(names) or set(truncated) != set(names):
                    raise ValueError("MAPPO requires fixed-population termination dictionaries")
                ends = [terminated[name] or truncated[name] for name in names]
                if any(ends) != all(ends):
                    raise ValueError("MAPPO currently requires synchronous fixed-population episodes")
                ended = all(ends)
                if not ended and (set(next_obs) != set(names) or set(env.agents) != set(names)):
                    raise ValueError("Agent population changed during a MAPPO rollout")
                with torch.no_grad():
                    if all(terminated.values()):
                        # 真终止不读取末状态；截断仍用真实末状态 bootstrap。
                        next_values = torch.zeros_like(values)
                    else:
                        next_predictions = model.values(environment_state(env, spec))
                        next_values = (
                            normalizer.denormalize(next_predictions)
                            if config["value_normalization"] else next_predictions
                        )
                training_rewards = aggregate_rewards(rewards, names, config["reward_aggregation"])
                rows = {
                    "states": global_state,
                    "actions": actions,
                    "log_probs": log_probs,
                    "old_normalized_values": normalized_values,
                    "values": values,
                    "next_values": next_values,
                    "rewards": torch.tensor(
                        training_rewards, dtype=torch.float32, device=device
                    ),
                    "terminated": torch.tensor(
                        [float(terminated[n]) for n in names], dtype=torch.float32, device=device
                    ),
                    "episode_end": torch.tensor(
                        [float(end) for end in ends], dtype=torch.float32, device=device
                    ),
                }
                for key, value in rows.items():
                    buffer[key].append(value)
                for name in names:
                    obs_buffer[name].append(local[name])
                    returns[name] += float(rewards[name])
                steps += 1
                observations = next_obs
                if ended:
                    episodes += 1
                    episode_writer.writerow({"episode": episodes, "steps": steps, **returns})
                    episode_file.flush()
                    returns = dict.fromkeys(names, 0.0)
                    observations, _ = env.reset(seed=config["seed"] + episodes)
            batch = {key: torch.stack(values) for key, values in buffer.items()}
            batch["observations"] = {name: torch.stack(values) for name, values in obs_buffer.items()}
            diagnostics = _update(model, normalizer, actor_optimizers, critic_optimizer, batch, config)
            update += 1
            update_writer.writerow({"update": update, "steps": steps, "episodes": episodes, **diagnostics})
            update_file.flush()
            if next_checkpoint and steps >= next_checkpoint:
                _checkpoint(
                    out / "checkpoints" / f"step_{steps:09d}.pt", model, normalizer, config, steps
                )
                next_checkpoint = (steps // config["checkpoint_interval"] + 1) * config["checkpoint_interval"]
            print(
                f"steps={steps} episodes={episodes} policy_loss={diagnostics['policy_loss']:.5f} "
                f"value_loss={diagnostics['value_loss']:.5f}", flush=True,
            )
    _checkpoint(out / "model.pt", model, normalizer, config, steps)
    return out / "model.pt"
