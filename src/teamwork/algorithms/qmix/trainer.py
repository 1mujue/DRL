"""QMIX 独立训练循环；公共 runner 仍负责默认场景参数与 training.json。

合作团队使用显式配置的共享奖励。回放保存回合前缀和真实截断末状态，
GRU 从回合起点展开；这里不提供动态动作掩码、对抗团队训练或精确续训。
"""

import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...environments.base import state
from ...environments.registry import make_env
from .algorithm import QMIXPolicy
from .learner import QMIXLearner
from .networks import pack_observations
from .replay import EpisodeReplay

DEFAULTS = {
    "device": "cpu", "gamma": 0.99, "lr": 0.0005,
    "hidden_size": 64, "mixing_size": 32, "hypernet_size": 64,
    "use_rnn": True, "use_last_action": True, "use_agent_id": True,
    "double_q": True, "replay_capacity": 256, "batch_size": 16,
    "learning_starts_episodes": 16, "updates_per_episode": 1,
    "target_update_interval": 200, "max_grad_norm": 10.0,
    "rmsprop_alpha": 0.99, "optimizer_eps": 0.00001,
    "epsilon_start": 1.0, "epsilon_finish": 0.05, "epsilon_anneal_steps": 50000,
    "evaluation_epsilon": 0.05, "reward_aggregation": "mean", "checkpoint_interval": 10000,
}


def resolve_config(config):
    result = {**DEFAULTS, **config}
    if result.get("algorithm") != "qmix":
        raise ValueError("QMIX configuration requires algorithm='qmix'")
    if result["env"] == "tag":
        raise ValueError("QMIX requires one cooperative team; Tag needs a fixed-team wrapper")
    if not isinstance(result.get("env_kwargs"), dict) or not result["env_kwargs"]:
        raise ValueError("Provide resolved env_kwargs or train through experiments.run_training")
    for key in (
        "total_steps", "hidden_size", "mixing_size", "hypernet_size", "replay_capacity",
        "batch_size", "learning_starts_episodes", "updates_per_episode",
        "target_update_interval", "epsilon_anneal_steps",
    ):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "checkpoint_interval"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if max(result["batch_size"], result["learning_starts_episodes"]) > result["replay_capacity"]:
        raise ValueError("Replay capacity must accommodate batch_size and learning_starts_episodes")
    for key in ("lr", "max_grad_norm", "optimizer_eps"):
        if isinstance(result[key], bool) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("gamma", "epsilon_start", "epsilon_finish", "evaluation_epsilon", "rmsprop_alpha"):
        if isinstance(result[key], bool) or not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    if result["rmsprop_alpha"] == 1 or result["epsilon_finish"] > result["epsilon_start"]:
        raise ValueError("Require rmsprop_alpha < 1 and epsilon_finish <= epsilon_start")
    for key in ("use_rnn", "use_last_action", "use_agent_id", "double_q"):
        if not isinstance(result[key], bool):
            raise TypeError(f"{key} must be a boolean")
    if result["reward_aggregation"] not in {"mean", "sum", "shared"}:
        raise ValueError("reward_aggregation must be mean, sum or shared")
    json.dumps(result, allow_nan=False)
    return result


def aggregate_reward(rewards, agents, mode):
    """仅构造 learner 的团队奖励，episodes.csv 仍记录原始个体回报。"""
    if set(rewards) != set(agents):
        raise ValueError("QMIX rewards must cover the fixed team")
    values = np.asarray([rewards[agent] for agent in agents], dtype=np.float64)
    if values.shape != (len(agents),) or not np.isfinite(values).all():
        raise ValueError("QMIX requires finite scalar rewards")
    if mode == "mean":
        return float(values.mean())
    if mode == "sum":
        return float(values.sum())
    if mode == "shared":
        if not np.allclose(values, values[0], rtol=1e-6, atol=1e-6):
            raise ValueError("shared reward mode requires identical rewards for all agents")
        return float(values[0])
    raise ValueError("Unknown QMIX reward aggregation")


def epsilon_at(steps, config):
    fraction = min(steps / config["epsilon_anneal_steps"], 1.0)
    return config["epsilon_start"] + fraction * (config["epsilon_finish"] - config["epsilon_start"])


def training_contract(config):
    return {
        "algorithm": "qmix", "implementation": "qmix_local_q_v1",
        "reward_aggregation": config["reward_aggregation"],
        "reward_scope": "one_cooperative_team",
        "execution": "own_observation_own_previous_action_static_agent_id",
        "central_state": "mixer_during_training_only",
        "target_update_unit": "learner_updates", "replay_unit": "episodes",
        "use_rnn": config["use_rnn"], "resume_supported": False,
    }


def save_checkpoint(path, learner, config, spec, steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "teamwork.qmix.v1", "config": config, "steps": steps,
        "updates": learner.updates, "spec": asdict(spec),
        "controller": learner.controller.state_dict(), "mixer": learner.mixer.state_dict(),
        "training_contract": training_contract(config), "resume_supported": False,
    }, temporary)
    temporary.replace(path)


def environment_state(env, spec):
    values = state(env).copy()
    if values.shape != (spec.state_size,) or not np.isfinite(values).all():
        raise ValueError("QMIX requires a finite fixed-size centralized state")
    return values


def train(config, out):
    config = resolve_config(config)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}")
    device = torch.device(config["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("QMIX device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    env = make_env(config["env"], **config["env_kwargs"])
    try:
        observations, _ = env.reset(seed=config["seed"])
        roles = set(env.policy_metadata.get("roles", {}).values())
        if len(roles) > 1:
            raise ValueError("QMIX cannot combine opposing roles in one cooperative team")
        spec = env.describe()
        learner = QMIXLearner(spec, config, device)
        pack_observations(observations, spec)
        environment_state(env, spec)
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (out / "training_contract.json").write_text(
            json.dumps(training_contract(config), indent=2), encoding="utf-8"
        )
        return collect_and_train(env, observations, learner, spec, config, out, device)
    finally:
        env.close()


def collect_and_train(env, observations, learner, spec, config, out, device):
    replay = EpisodeReplay(spec, config["replay_capacity"], config["seed"] + 1)
    rng = np.random.default_rng(config["seed"])
    policy = QMIXPolicy(learner.controller, config["evaluation_epsilon"])
    for index, agent in enumerate(spec.agents):
        env.action_space(agent).seed(config["seed"] + index)
    steps = episodes = 0
    updates_fields = [
        "update", "steps", "episodes", "epsilon", "replay_episodes", "loss", "grad_norm",
        "td_error_abs", "q_total_mean", "target_mean", "valid_transitions",
    ]
    with (
        (out / "episodes.csv").open("w", encoding="utf-8", newline="") as episode_file,
        (out / "updates.csv").open("w", encoding="utf-8", newline="") as update_file,
    ):
        episode_writer = csv.DictWriter(episode_file, fieldnames=["episode", "steps", *spec.agents])
        update_writer = csv.DictWriter(update_file, fieldnames=updates_fields)
        episode_writer.writeheader()
        update_writer.writeheader()
        while steps < config["total_steps"]:
            policy.reset()
            trajectory = {
                "observations": [pack_observations(observations, spec)],
                "states": [environment_state(env, spec)], "actions": [], "rewards": [],
                "terminated": [], "truncated": [],
            }
            returns = dict.fromkeys(spec.agents, 0.0)
            while steps < config["total_steps"]:
                epsilon = epsilon_at(steps, config)
                actions = policy.sample_actions(observations, epsilon, rng)
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                if set(terminated) != set(spec.agents) or set(truncated) != set(spec.agents):
                    raise ValueError("QMIX requires fixed-population termination dictionaries")
                if len(set(terminated.values())) != 1 or len(set(truncated.values())) != 1:
                    raise ValueError("QMIX currently requires synchronous team episode endings")
                terminal, time_limit = all(terminated.values()), all(truncated.values())
                ended = terminal or time_limit
                if not ended and set(env.agents) != set(spec.agents):
                    raise ValueError("QMIX agent population changed during collection")
                if terminal:
                    # 真终止后无需 bootstrap，可以使用占位末状态；截断必须保留真实末状态。
                    next_packed = np.zeros_like(trajectory["observations"][0])
                    next_state = np.zeros(spec.state_size, np.float32)
                else:
                    next_packed = pack_observations(next_obs, spec)
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
                    if len(replay) >= max(config["batch_size"], config["learning_starts_episodes"]):
                        for _ in range(config["updates_per_episode"]):
                            metrics = learner.train_batch(replay.sample(config["batch_size"], device))
                            update_writer.writerow({
                                "update": learner.updates, "steps": steps, "episodes": episodes,
                                "epsilon": epsilon, "replay_episodes": len(replay), **metrics,
                            })
                            update_file.flush()
                if config["checkpoint_interval"] and steps % config["checkpoint_interval"] == 0:
                    save_checkpoint(
                        out / "checkpoints" / f"step_{steps:09d}.pt", learner, config, spec, steps
                    )
                if ended or cutoff:
                    print(
                        f"steps={steps} episodes={episodes} updates={learner.updates} "
                        f"replay={len(replay)} epsilon={epsilon:.4f}", flush=True,
                    )
                    if not cutoff:
                        observations, _ = env.reset(seed=config["seed"] + episodes)
                    break
    checkpoint = out / "model.pt"
    save_checkpoint(checkpoint, learner, config, spec, steps)
    return checkpoint
