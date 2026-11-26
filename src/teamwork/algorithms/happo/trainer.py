"""HAPPO：异构 actor + 集中 critic + 顺序（Gauss-Seidel）更新。

与 MAPPO 共用同一套 critic、价值归一化与 GAE，差别只在 actor 的更新方式：
按 ``actor_order`` 逐个 agent 更新，且每个 agent 的代理目标都要乘上
"已更新 agent 在 **当前参数** 下的重要性比率"——HAPPO 的顺序单调改进保证
就来自这个乘积累积。因此前序比率必须在每次参数更新之后重新计算，
不能用更新前的旧值，否则第一个 minibatch 里它恰好等于 1，顺序更新会退化成同时更新。

本实现不包含：论文的 M^{i} 优势分解、顺序更新的信赖域线搜索、RNN 和向量化环境。
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
from ..mappo.trainer import compute_gae, value_loss
from ..training import validate_training_scope
from .networks import (
    HAPPOModel,
    ValueNormalizer,
    actor_for,
    resolve_actor_order,
    resolve_device,
    validate_observations,
)

DEFAULTS = {
    "algorithm": "happo",
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
    "share_policy": False,
    "actor_order": None,
    "hidden_size": 64,
    "hidden_layers": 2,
    "actor_output_gain": 0.01,
    "checkpoint_interval": 10000,
    "reward_aggregation": "mean",
}


def resolve_config(config):
    if not isinstance(config, dict):
        raise TypeError("HAPPO config must be a dictionary")
    result = {**DEFAULTS, **deepcopy(config)}
    if result["algorithm"] != "happo":
        raise ValueError("HAPPO configuration requires algorithm='happo'")
    if not isinstance(result.get("env"), str) or not result["env"]:
        raise ValueError("HAPPO config requires an environment name")
    if "total_steps" not in result:
        raise ValueError("HAPPO config requires total_steps")
    if "env_kwargs" in result and not isinstance(result["env_kwargs"], dict):
        raise TypeError("env_kwargs must be a dictionary")
    try:
        result["env_kwargs"] = resolved_env_kwargs(result["env"], result.get("env_kwargs"))
    except KeyError:
        # 自定义已注册环境可以没有公共评估协议，但必须显式给出自己的参数。
        result.setdefault("env_kwargs", {})
    for key in (
        "total_steps",
        "rollout_steps",
        "batch_size",
        "epochs",
        "hidden_size",
        "hidden_layers",
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or result["seed"] < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    interval = result["checkpoint_interval"]
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError("checkpoint_interval must be a nonnegative integer (0 disables it)")
    for key in ("gamma", "gae_lambda"):
        if (
            isinstance(result[key], bool)
            or not math.isfinite(result[key])
            or not 0 <= result[key] <= 1
        ):
            raise ValueError(f"{key} must lie in [0, 1]")
    for key in (
        "lr",
        "clip",
        "value_clip",
        "max_grad_norm",
        "adam_eps",
        "huber_delta",
        "actor_output_gain",
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
        "use_clipped_value_loss",
        "use_huber_loss",
        "value_normalization",
        "feature_normalization",
        "share_policy",
    ):
        if not isinstance(result[key], bool):
            raise TypeError(f"{key} must be a boolean")
    if result["share_policy"]:
        # HAPPO 的前提是异构 actor；参数共享会让顺序更新失去定义。
        raise ValueError("HAPPO requires heterogeneous actors; set share_policy: false")
    order = result["actor_order"]
    if order is not None and (
        not isinstance(order, (list, tuple)) or any(not isinstance(name, str) for name in order)
    ):
        raise ValueError("HAPPO actor_order must be null or a list of agent names")
    if result["reward_aggregation"] not in {"individual", "mean", "sum", "shared"}:
        raise ValueError("reward_aggregation must be individual, mean, sum or shared")
    if result["env"] == "tag" and result["reward_aggregation"] != "individual":
        raise ValueError("HAPPO team reward modes cannot combine opposing Tag roles")
    json.dumps(result, allow_nan=False)
    return result


def aggregate_rewards(rewards, agents, mode):
    """训练奖励显式变换；episodes.csv 始终保留环境原始个体回报。"""
    if set(rewards) != set(agents):
        raise ValueError("HAPPO rewards must cover the fixed agent population")
    values = np.asarray([rewards[agent] for agent in agents], dtype=np.float64)
    if values.shape != (len(agents),) or not np.isfinite(values).all():
        raise ValueError("HAPPO requires finite scalar rewards")
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
        raise ValueError("Unknown HAPPO reward aggregation")
    return np.full(len(agents), value, dtype=np.float64)


def training_contract(config, actor_order):
    mode = config["reward_aggregation"]
    return {
        "algorithm": "happo",
        "implementation": "happo_heterogeneous_sequential_v1",
        "reward_aggregation": mode,
        "reward_scope": "individual_agents" if mode == "individual" else "one_cooperative_team",
        "execution": "local_observation_heterogeneous_actor",
        "central_state": "critic_during_training_only",
        "shared_components": "mappo_actor_critic_normalizer_gae",
        "update": "sequential_gauss_seidel",
        "actor_order": list(actor_order),
        "advantage": "centralized_critic_gae_normalized_over_all_agents",
        "clip_target": "cumulative_product_of_importance_ratios",
        "mac_advantage_decomposition": False,
        "prefix_ratio_recomputed_after_each_update": True,
        "known_deviations": [
            "clips the accumulated ratio product instead of a per-agent surrogate",
            "no M^{i} advantage decomposition beyond the ratio product",
            "no trust-region line search over actor orders",
        ],
        "resume_supported": False,
    }


def environment_state(env, spec):
    values = state(env).copy()
    if values.shape != (spec.state_size,) or not np.isfinite(values).all():
        raise ValueError("HAPPO requires a finite fixed-size centralized state")
    return values


def agent_logits(model, agent, observations, ids):
    """某个 agent 在自己那份局部观测上的 logits；HAPPO 不共享 actor 参数。"""
    return actor_for(model, agent)(observations[agent][ids])


def prefix_ratio(model, order, index, batch, ids):
    """已更新 agent 在**当前参数**下的比率乘积（已 detach）；index 为 0 时返回 None。

    这个函数是 HAPPO 正确性的核心：它在每个 agent 更新前被重新调用，
    因此前序 agent 的比率来自它们刚刚更新后的策略，而不是 rollout 时的旧策略。
    """
    if index <= 0:
        return None
    names = list(model.spec.agents)
    product = None
    with torch.no_grad():
        for name in order[:index]:
            column = names.index(name)
            distribution = Categorical(logits=agent_logits(model, name, batch["observations"], ids))
            ratio = (
                distribution.log_prob(batch["actions"][ids, column])
                - batch["log_probs"][ids, column]
            ).exp()
            product = ratio if product is None else product * ratio
    return product


def _sequential_actor_update(model, actor_optimizers, batch, advantages, order, config):
    names = list(model.spec.agents)
    horizon = advantages.shape[0]
    clip = config["clip"]
    keys = ("policy_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm")
    totals = dict.fromkeys(keys, 0.0)
    deviation_sum = deviation_weight = samples = 0
    for _ in range(config["epochs"]):
        for ids in torch.randperm(horizon, device=advantages.device).split(config["batch_size"]):
            for index, name in enumerate(order):
                column = names.index(name)
                distribution = Categorical(
                    logits=agent_logits(model, name, batch["observations"], ids)
                )
                log_ratio = (
                    distribution.log_prob(batch["actions"][ids, column])
                    - batch["log_probs"][ids, column]
                )
                own_ratio = log_ratio.exp()
                prefix = prefix_ratio(model, order, index, batch, ids)
                cumulative = own_ratio if prefix is None else prefix * own_ratio
                advantage = advantages[ids, column]
                surrogate = torch.minimum(
                    cumulative * advantage,
                    cumulative.clamp(1 - clip, 1 + clip) * advantage,
                )
                policy_loss = -surrogate.mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss - config["entropy_coef"] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite HAPPO actor loss")
                optimizer = actor_optimizers[name]
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    actor_for(model, name).parameters(), config["max_grad_norm"]
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite HAPPO actor gradient")
                optimizer.step()
                metrics = {
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "approx_kl": ((own_ratio - 1) - log_ratio).mean(),
                    "clip_fraction": ((cumulative - 1).abs() > clip).float().mean(),
                    "actor_grad_norm": grad_norm,
                }
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * len(ids)
                samples += len(ids)
                if prefix is not None:
                    deviation_sum += float((prefix - 1).abs().mean()) * len(ids)
                    deviation_weight += len(ids)
    diagnostics = {key: value / samples for key, value in totals.items()}
    # 该诊断大于 0 才说明顺序更新真的生效；恒为 0 意味着退化成了同时更新。
    diagnostics["prefix_ratio_deviation"] = (
        deviation_sum / deviation_weight if deviation_weight else 0.0
    )
    return diagnostics


def _update(model, normalizer, actor_optimizers, critic_optimizer, batch, order, config):
    advantages, targets = compute_gae(
        batch["rewards"],
        batch["values"],
        batch["next_values"],
        batch["terminated"],
        batch["episode_end"],
        config["gamma"],
        config["gae_lambda"],
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    if config["value_normalization"]:
        normalizer.update(targets)
        targets = normalizer.normalize(targets)
    diagnostics = _sequential_actor_update(
        model, actor_optimizers, batch, advantages, order, config
    )
    inputs = model.critic_inputs(batch["states"]).flatten(0, 1)
    old_values = batch["old_normalized_values"].flatten()
    targets = targets.flatten()
    critic_loss_sum = critic_grad_sum = critic_samples = 0
    for _ in range(config["epochs"]):
        for ids in torch.randperm(len(targets), device=targets.device).split(config["batch_size"]):
            predictions = model.critic(inputs[ids]).squeeze(-1)
            loss = value_loss(
                predictions,
                old_values[ids],
                targets[ids],
                config["value_clip"],
                config["huber_delta"],
                config["use_clipped_value_loss"],
                config["use_huber_loss"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite HAPPO critic loss")
            critic_optimizer.zero_grad()
            (config["value_loss_coef"] * loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.critic.parameters(), config["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite HAPPO critic gradient")
            critic_optimizer.step()
            critic_loss_sum += float(loss.detach()) * len(ids)
            critic_grad_sum += float(grad_norm) * len(ids)
            critic_samples += len(ids)
    diagnostics["value_loss"] = critic_loss_sum / critic_samples
    diagnostics["critic_grad_norm"] = critic_grad_sum / critic_samples
    diagnostics["learning_rate"] = next(iter(actor_optimizers.values())).param_groups[0]["lr"]
    return diagnostics


def _checkpoint(path, model, normalizer, config, steps, actor_order):
    data = {
        "format": "teamwork.happo.v1",
        "implementation": "happo_heterogeneous_sequential_v1",
        "training_contract": training_contract(config, actor_order),
        "resume_supported": False,
        "config": config,
        "steps": steps,
        "actor_order": list(actor_order),
        "spec": asdict(model.spec),
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def train(config, out):
    config = resolve_config(config)
    validate_training_scope(config)
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
                raise ValueError("HAPPO team reward modes cannot combine opposing roles")
        spec = env.describe()
        model = HAPPOModel(spec, config).to(device)
        validate_observations(observations, spec, device)
        environment_state(env, spec)
        if set(env.agents) != set(spec.agents):
            raise ValueError("HAPPO requires the full fixed agent population after reset")
        actor_order = resolve_actor_order(spec, config["actor_order"])
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (out / "training_contract.json").write_text(
            json.dumps(training_contract(config, actor_order), indent=2), encoding="utf-8"
        )
        return _train_environment(env, observations, model, spec, config, out, device, actor_order)
    finally:
        env.close()


def _train_environment(env, observations, model, spec, config, out, device, actor_order):
    names = list(spec.agents)
    normalizer = ValueNormalizer().to(device)
    actor_optimizers = {
        name: torch.optim.Adam(
            actor_for(model, name).parameters(), lr=config["actor_lr"], eps=config["adam_eps"]
        )
        for name in names
    }
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=config["critic_lr"], eps=config["adam_eps"]
    )
    for i, name in enumerate(names):
        env.action_space(name).seed(config["seed"] + i)
    fields = [
        "update",
        "steps",
        "episodes",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "prefix_ratio_deviation",
        "learning_rate",
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
            buffer = {
                key: []
                for key in (
                    "states",
                    "actions",
                    "log_probs",
                    "old_normalized_values",
                    "values",
                    "next_values",
                    "rewards",
                    "terminated",
                    "episode_end",
                )
            }
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
                    log_probs = torch.stack(
                        [distributions[name].log_prob(actions[i]) for i, name in enumerate(names)]
                    )
                    normalized_values = model.values(global_state)
                    values = (
                        normalizer.denormalize(normalized_values)
                        if config["value_normalization"]
                        else normalized_values
                    )
                next_obs, rewards, terminated, truncated, _ = env.step(
                    {name: int(actions[i]) for i, name in enumerate(names)}
                )
                if set(terminated) != set(names) or set(truncated) != set(names):
                    raise ValueError("HAPPO requires fixed-population termination dictionaries")
                ends = [terminated[name] or truncated[name] for name in names]
                if any(ends) != all(ends):
                    raise ValueError(
                        "HAPPO currently requires synchronous fixed-population episodes"
                    )
                ended = all(ends)
                if not ended and (set(next_obs) != set(names) or set(env.agents) != set(names)):
                    raise ValueError("Agent population changed during a HAPPO rollout")
                with torch.no_grad():
                    if all(terminated.values()):
                        # 真终止不读取末状态；截断仍用真实末状态 bootstrap。
                        next_values = torch.zeros_like(values)
                    else:
                        next_predictions = model.values(environment_state(env, spec))
                        next_values = (
                            normalizer.denormalize(next_predictions)
                            if config["value_normalization"]
                            else next_predictions
                        )
                training_rewards = aggregate_rewards(rewards, names, config["reward_aggregation"])
                rows = {
                    "states": global_state,
                    "actions": actions,
                    "log_probs": log_probs,
                    "old_normalized_values": normalized_values,
                    "values": values,
                    "next_values": next_values,
                    "rewards": torch.tensor(training_rewards, dtype=torch.float32, device=device),
                    "terminated": torch.tensor(
                        [float(terminated[name]) for name in names],
                        dtype=torch.float32,
                        device=device,
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
            batch["observations"] = {
                name: torch.stack(values) for name, values in obs_buffer.items()
            }
            diagnostics = _update(
                model,
                normalizer,
                actor_optimizers,
                critic_optimizer,
                batch,
                actor_order,
                config,
            )
            update += 1
            update_writer.writerow(
                {"update": update, "steps": steps, "episodes": episodes, **diagnostics}
            )
            update_file.flush()
            if next_checkpoint and steps >= next_checkpoint:
                _checkpoint(
                    out / "checkpoints" / f"step_{steps:09d}.pt",
                    model,
                    normalizer,
                    config,
                    steps,
                    actor_order,
                )
                next_checkpoint = (steps // config["checkpoint_interval"] + 1) * config[
                    "checkpoint_interval"
                ]
            print(
                f"steps={steps} episodes={episodes} "
                f"policy_loss={diagnostics['policy_loss']:.5f} "
                f"value_loss={diagnostics['value_loss']:.5f} "
                f"prefix_dev={diagnostics['prefix_ratio_deviation']:.2e}",
                flush=True,
            )
    _checkpoint(out / "model.pt", model, normalizer, config, steps, actor_order)
    return out / "model.pt"
