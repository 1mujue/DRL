"""HASAC 离散适配的独立训练循环；公共 runner 仍负责场景默认值与 training.json。

采集不依赖 epsilon 探索，动作直接来自 categorical 策略采样；回放按回合保存，
与 QMIX 共用同一套 EpisodeReplay（保留末状态、区分终止与截断、padding 不计损失）。
这里只支持单一合作团队：Tag 需要固定的分阵营训练封装，当前不提供。
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

from ...environments.base import state
from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ..qmix.replay import EpisodeReplay
from .learner import HASACLearner
from .networks import (
    pack_observations,
    resolve_device,
    sample_actions,
    validate_observations,
)

DEFAULTS = {
    "algorithm": "hasac_discrete",
    "policy_family": "categorical",
    "seed": 0,
    "device": "cpu",
    "gamma": 0.99,
    "tau": 0.005,
    "hidden_size": 64,
    "hidden_layers": 2,
    "feature_normalization": True,
    "actor_output_gain": 0.01,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "alpha_start": 0.2,
    "learn_alpha": True,
    "target_entropy_scale": 0.98,
    "max_grad_norm": 10.0,
    "replay_capacity": 256,
    "batch_size": 16,
    "learning_starts_episodes": 16,
    "updates_per_episode": 1,
    "reward_aggregation": "mean",
    "checkpoint_interval": 10000,
    "actor_order": None,
}


def resolve_config(config):
    if not isinstance(config, dict):
        raise TypeError("HASAC config must be a dictionary")
    result = {**DEFAULTS, **deepcopy(config)}
    if result["algorithm"] != "hasac_discrete":
        raise ValueError("HASAC configuration requires algorithm='hasac_discrete'")
    if result["policy_family"] != "categorical":
        # 离散适配只实现 categorical 策略，不冒充论文的连续高斯版本。
        raise ValueError("HASAC supports policy_family='categorical' only")
    if not isinstance(result.get("env"), str) or not result["env"]:
        raise ValueError("HASAC config requires an environment name")
    if "total_steps" not in result:
        raise ValueError("HASAC config requires total_steps")
    if result["env"] == "tag":
        raise ValueError("HASAC requires one cooperative team; Tag needs a fixed-team wrapper")
    if "env_kwargs" in result and not isinstance(result["env_kwargs"], dict):
        raise TypeError("env_kwargs must be a dictionary")
    try:
        result["env_kwargs"] = resolved_env_kwargs(result["env"], result.get("env_kwargs"))
    except KeyError:
        result.setdefault("env_kwargs", {})
    for key in (
        "total_steps",
        "hidden_size",
        "hidden_layers",
        "replay_capacity",
        "batch_size",
        "learning_starts_episodes",
        "updates_per_episode",
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "checkpoint_interval"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if max(result["batch_size"], result["learning_starts_episodes"]) > result["replay_capacity"]:
        raise ValueError("Replay capacity must accommodate batch_size and learning_starts_episodes")
    if (
        isinstance(result["gamma"], bool)
        or not math.isfinite(result["gamma"])
        or not 0 <= result["gamma"] <= 1
    ):
        raise ValueError("gamma must lie in [0, 1]")
    if (
        isinstance(result["tau"], bool)
        or not math.isfinite(result["tau"])
        or not 0 < result["tau"] <= 1
    ):
        raise ValueError("tau must lie in (0, 1]")
    for key in (
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "alpha_start",
        "target_entropy_scale",
        "max_grad_norm",
        "actor_output_gain",
    ):
        value = result[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("learn_alpha", "feature_normalization"):
        if not isinstance(result[key], bool):
            raise TypeError(f"{key} must be a boolean")
    order = result["actor_order"]
    if order is not None and (
        not isinstance(order, (list, tuple)) or any(not isinstance(name, str) for name in order)
    ):
        raise ValueError("HASAC actor_order must be null or a list of agent names")
    if result["reward_aggregation"] not in {"mean", "sum", "shared"}:
        raise ValueError("reward_aggregation must be mean, sum or shared")
    json.dumps(result, allow_nan=False)
    return result


def aggregate_reward(rewards, agents, mode):
    """只构造学习器使用的团队奖励；episodes.csv 仍记录原始个体回报。"""
    if set(rewards) != set(agents):
        raise ValueError("HASAC rewards must cover the fixed team")
    values = np.asarray([rewards[agent] for agent in agents], dtype=np.float64)
    if values.shape != (len(agents),) or not np.isfinite(values).all():
        raise ValueError("HASAC requires finite scalar rewards")
    if mode == "mean":
        return float(values.mean())
    if mode == "sum":
        return float(values.sum())
    if mode == "shared":
        if not np.allclose(values, values[0], rtol=1e-6, atol=1e-6):
            raise ValueError("shared reward mode requires identical rewards for all agents")
        return float(values[0])
    raise ValueError("Unknown HASAC reward aggregation")


def training_contract(config, actor_order):
    return {
        "algorithm": "hasac_discrete",
        "implementation": "hasac_discrete_sequential_soft_q_v1",
        "policy_family": "categorical",
        "reward_aggregation": config["reward_aggregation"],
        "reward_scope": "one_cooperative_team",
        "execution": "local_observation_heterogeneous_actor",
        "central_state": "joint_soft_q_critic_during_training_only",
        "critic_input": "central_state_plus_joint_action_one_hot",
        "update": "sequential_soft_advantage",
        "actor_order": list(actor_order),
        "own_action_expectation": "exact_enumeration_over_own_discrete_action",
        "prefix_actions": "sampled_from_updated_current_policy",
        "suffix_actions": "replay_old_actions",
        "temperature": {
            "learned": config["learn_alpha"],
            "alpha_start": config["alpha_start"],
            "target_entropy_scale": config["target_entropy_scale"],
        },
        "known_deviations": [
            "categorical discrete policy replaces the paper's continuous Gaussian actor",
            "no reparameterised sampling; exact discrete expectation over the agent's own action",
            "no heterogeneous-agent soft advantage decomposition beyond the sequential update",
        ],
        "resume_supported": False,
    }


def environment_state(env, spec):
    values = state(env).copy()
    if values.shape != (spec.state_size,) or not np.isfinite(values).all():
        raise ValueError("HASAC requires a finite fixed-size centralized state")
    return values


def save_checkpoint(path, learner, config, spec, steps, actor_order):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 目标网络与在线网络分开存放：推理只加载 actors，续训需要的内容也都在文件里。
    data = {
        "format": "teamwork.hasac.v1",
        "implementation": "hasac_discrete_sequential_soft_q_v1",
        "training_contract": training_contract(config, actor_order),
        "resume_supported": False,
        "config": config,
        "steps": steps,
        "updates": learner.updates,
        "actor_order": list(actor_order),
        "spec": asdict(spec),
        "actors": learner.model.actors.state_dict(),
        "critics": learner.model.critics.state_dict(),
        "log_alphas": learner.model.log_alphas.state_dict(),
        "target_actors": learner.model.target_actors.state_dict(),
        "target_critics": learner.model.target_critics.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


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
        roles = set(env.policy_metadata.get("roles", {}).values())
        if len(roles) > 1:
            raise ValueError("HASAC cannot combine opposing roles in one cooperative team")
        spec = env.describe()
        validate_observations(observations, spec, device)
        environment_state(env, spec)
        learner = HASACLearner(spec, config, device)
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (out / "training_contract.json").write_text(
            json.dumps(training_contract(config, learner.order), indent=2), encoding="utf-8"
        )
        return collect_and_train(env, observations, learner, spec, config, out, device)
    finally:
        env.close()


def collect_and_train(env, observations, learner, spec, config, out, device):
    replay = EpisodeReplay(spec, config["replay_capacity"], config["seed"] + 1)
    for index, agent in enumerate(spec.agents):
        env.action_space(agent).seed(config["seed"] + index)
    steps = episodes = 0
    next_checkpoint = config["checkpoint_interval"]
    fields = [
        "update",
        "steps",
        "episodes",
        "replay_episodes",
        "critic_loss",
        "actor_loss",
        "entropy",
        "alpha_mean",
        "q_mean",
        "target_mean",
        "critic_grad_norm",
        "actor_grad_norm",
    ]
    with (
        (out / "episodes.csv").open("w", encoding="utf-8", newline="") as episode_file,
        (out / "updates.csv").open("w", encoding="utf-8", newline="") as update_file,
    ):
        episode_writer = csv.DictWriter(
            episode_file, fieldnames=["episode", "steps", *spec.agents]
        )
        update_writer = csv.DictWriter(update_file, fieldnames=fields)
        episode_writer.writeheader()
        update_writer.writeheader()
        while steps < config["total_steps"]:
            trajectory = {
                "observations": [
                    pack_observations(observations, spec, device).cpu().numpy()
                ],
                "states": [environment_state(env, spec)],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "truncated": [],
            }
            returns = dict.fromkeys(spec.agents, 0.0)
            while steps < config["total_steps"]:
                packed = pack_observations(observations, spec, device)
                actions = sample_actions(learner.model, packed, spec)
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                if set(terminated) != set(spec.agents) or set(truncated) != set(spec.agents):
                    raise ValueError("HASAC requires fixed-population termination dictionaries")
                if len(set(terminated.values())) != 1 or len(set(truncated.values())) != 1:
                    raise ValueError("HASAC currently requires synchronous team episode endings")
                terminal, time_limit = all(terminated.values()), all(truncated.values())
                ended = terminal or time_limit
                if not ended and set(env.agents) != set(spec.agents):
                    raise ValueError("HASAC agent population changed during collection")
                if terminal:
                    # 真终止不 bootstrap，末状态只是占位；截断必须保留真实末状态。
                    next_packed = np.zeros_like(trajectory["observations"][0])
                    next_state = np.zeros(spec.state_size, np.float32)
                else:
                    next_packed = pack_observations(next_obs, spec, device).cpu().numpy()
                    next_state = environment_state(env, spec)
                team_reward = aggregate_reward(rewards, spec.agents, config["reward_aggregation"])
                steps += 1
                cutoff = steps == config["total_steps"]
                trajectory["observations"].append(next_packed)
                trajectory["states"].append(next_state)
                trajectory["actions"].append([actions[agent] for agent in spec.agents])
                trajectory["rewards"].append(team_reward)
                trajectory["terminated"].append(float(terminal))
                trajectory["truncated"].append(float(time_limit or (cutoff and not terminal)))
                for agent in spec.agents:
                    returns[agent] += float(rewards[agent])
                observations = next_obs
                if ended or cutoff:
                    replay.add({key: np.asarray(values) for key, values in trajectory.items()})
                    if ended:
                        episodes += 1
                        episode_writer.writerow({"episode": episodes, "steps": steps, **returns})
                        episode_file.flush()
                    if len(replay) >= max(
                        config["batch_size"], config["learning_starts_episodes"]
                    ):
                        for _ in range(config["updates_per_episode"]):
                            metrics = learner.train_batch(
                                replay.sample(config["batch_size"], device)
                            )
                            update_writer.writerow(
                                {
                                    "update": learner.updates,
                                    "steps": steps,
                                    "episodes": episodes,
                                    "replay_episodes": len(replay),
                                    **metrics,
                                }
                            )
                            update_file.flush()
                    if next_checkpoint and steps >= next_checkpoint:
                        save_checkpoint(
                            out / "checkpoints" / f"step_{steps:09d}.pt",
                            learner,
                            config,
                            spec,
                            steps,
                            learner.order,
                        )
                        next_checkpoint = (
                            steps // config["checkpoint_interval"] + 1
                        ) * config["checkpoint_interval"]
                    print(
                        f"steps={steps} episodes={episodes} updates={learner.updates} "
                        f"replay={len(replay)}",
                        flush=True,
                    )
                    if not cutoff:
                        observations, _ = env.reset(seed=config["seed"] + episodes)
                    break
    checkpoint = out / "model.pt"
    save_checkpoint(checkpoint, learner, config, spec, steps, learner.order)
    return checkpoint
