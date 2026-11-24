"""Independent actors; local (IPPO) or centralized (MAPPO-style) per-agent values.

Feedforward PPO baselines for categorical and bounded continuous actions. Fixed population, separate agent
networks, no weight sharing, no recurrent policy, no self-play opponent pool.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn
from torch.distributions import Categorical

from ...environments.base import state
from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all, write_provenance
from ..mat_dec.model import TransformerCritic
from ..training import training_agents, with_opponent_actions
from .continuous import SquashedNormal
from .lift_actor import LiftPoseActor

HAPPO = "happo_continuous"
MAT_DEC = "mat_dec_continuous"
CONTINUOUS = frozenset({"ippo_continuous", "mappo_continuous", HAPPO, MAT_DEC})
CENTRALIZED = frozenset({"mappo", "mappo_continuous", HAPPO, MAT_DEC})

DEFAULTS = {
    "device": "cpu",
    "rollout_steps": 256,
    "batch_size": 64,
    "epochs": 4,
    "lr": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "entropy_coef": 0.01,
}


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

    def distribution(self, observation):
        return Categorical(logits=self.actor(observation))

    def environment_action(self, distribution, action):
        return action.item()


class ContinuousActor(nn.Module):
    def __init__(self, observation_size, action_space, *, stochastic=True):
        if (
            not isinstance(action_space, Box)
            or len(action_space.shape) != 1
            or not np.isfinite(action_space.low).all()
            or not np.isfinite(action_space.high).all()
            or np.any(action_space.high <= action_space.low)
        ):
            raise ValueError("Continuous PPO requires a flat, finite, nonempty Box action space")
        dimensions = action_space.shape[0]
        if dimensions < 1:
            raise ValueError("Continuous PPO requires a nonempty Box action space")
        super().__init__()
        self.actor = mlp(observation_size, dimensions)
        if stochastic:
            self.log_std = nn.Parameter(torch.full((dimensions,), -0.5))
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_center", (high + low) / 2, persistent=False)
        self.register_buffer("action_scale", (high - low) / 2, persistent=False)

    def distribution(self, observation):
        return SquashedNormal(
            self.actor(observation),
            self.log_std.clamp(-5, 2).exp(),
            self.action_center,
            self.action_scale,
        )

    def bounded(self, latent):
        return self.action_center + self.action_scale * latent.tanh()

    def clip(self, actions):
        return actions.clamp(
            self.action_center - self.action_scale, self.action_center + self.action_scale
        )

    def environment_action(self, distribution, action):
        return distribution.action(action).cpu().numpy()


class ContinuousAgent(ContinuousActor):
    def __init__(self, observation_size, action_space, value_size):
        super().__init__(observation_size, action_space)
        self.critic = mlp(value_size, 1)

    def value(self, value_input):
        return self.critic(value_input).squeeze(-1)


class TokenValue(nn.Module):
    def __init__(self, encoder, sizes, index):
        super().__init__()
        self.encoder = encoder
        self.sizes = sizes
        self.index = index

    def values(self, joint_observations):
        return self.encoder(joint_observations.split(self.sizes, dim=-1))

    def forward(self, joint_observations):
        return self.values(joint_observations)[..., self.index : self.index + 1]


def configure_actor(model, env, observation_size, actor_type):
    if actor_type in {"lift_pose", "handover_pose"}:
        name = "two_arm_lift" if actor_type == "lift_pose" else "two_arm_handover"
        if (
            not isinstance(model, ContinuousActor)
            or not getattr(env, "pose_feedback", False)
            or env.metadata.get("name") != name
        ):
            raise ValueError(f"{actor_type} requires continuous control with {name} pose_feedback")
        phases = env.horizon if actor_type == "lift_pose" else env.pose_phases
        model.actor = LiftPoseActor(observation_size, phases)
    elif actor_type != "mlp":
        raise ValueError(f"Unknown actor type: {actor_type}")
    return model


def gae(rewards, values, next_values, terminated, episode_end, gamma, lam):
    """Bootstrap time limits, cut GAE traces at all reset boundaries."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - episode_end[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def create_agents(env, observations, algorithm, device, names=None, *, actor_type="mlp"):
    names = env.possible_agents if names is None else names
    if algorithm == MAT_DEC:
        sizes = tuple(len(observations[name]) for name in names)
        global_size = sum(sizes)
    else:
        global_size = len(state(env)) if algorithm in CENTRALIZED else None
    agents = {}
    for agent in names:
        observation_size = len(observations[agent])
        value_size = global_size if global_size is not None else observation_size
        action_space = env.action_space(agent)
        model = (
            ContinuousAgent(observation_size, action_space, value_size)
            if algorithm in CONTINUOUS
            else Agent(observation_size, action_space.n, value_size)
        )
        agents[agent] = configure_actor(model, env, observation_size, actor_type).to(device)
    if algorithm == HAPPO:
        critic = next(iter(agents.values())).critic
        for agent in agents.values():
            agent.critic = critic
    elif algorithm == MAT_DEC:
        encoder = TransformerCritic(sizes, width=64, heads=1, blocks=1).to(device)
        for index, agent in enumerate(agents.values()):
            agent.critic = TokenValue(encoder, sizes, index)
    return agents


def _update_happo(agents, optimizers, critic_optimizer, buffers, config, device):
    names = list(agents)
    shared = buffers[names[0]]
    training_rewards = torch.as_tensor(
        np.mean([buffers[name]["rew"] for name in names], axis=0),
        dtype=torch.float32,
        device=device,
    )
    advantages, targets = gae(
        training_rewards,
        torch.stack(shared["val"]),
        torch.stack(shared["next"]),
        torch.as_tensor(shared["term"], device=device),
        torch.as_tensor(shared["end"], device=device),
        config["gamma"],
        config["gae_lambda"],
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    joint_states = torch.stack(shared["vin"])
    horizon = len(targets)
    actor_order = torch.randperm(len(names)).tolist()
    log_correction = torch.zeros_like(advantages)
    for index in actor_order:
        name = names[index]
        model = agents[name]
        observations = torch.stack(buffers[name]["obs"])
        actions = torch.stack(buffers[name]["act"])
        old_log_probs = torch.stack(buffers[name]["logp"])
        # Previous agents' completed updates only reweight the advantages.
        weighted_advantage = advantages * log_correction.exp()
        for _ in range(config["epochs"]):
            for ids in torch.randperm(horizon, device=device).split(config["batch_size"]):
                distribution = model.distribution(observations[ids])
                ratio = (distribution.log_prob(actions[ids]) - old_log_probs[ids]).exp()
                clip = config["clip"]
                objective = torch.minimum(
                    ratio * weighted_advantage[ids],
                    ratio.clamp(1 - clip, 1 + clip) * weighted_advantage[ids],
                )
                loss = -objective.mean() - config["entropy_coef"] * distribution.entropy().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite HAPPO actor loss")
                opt = optimizers[name]
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*model.actor.parameters(), model.log_std], 0.5)
                opt.step()
        with torch.no_grad():
            log_correction += model.distribution(observations).log_prob(actions) - old_log_probs
    critic = agents[names[0]]
    for _ in range(config["epochs"]):
        for ids in torch.randperm(horizon, device=device).split(config["batch_size"]):
            loss = (critic.value(joint_states[ids]) - targets[ids]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite HAPPO critic loss")
            critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.critic.parameters(), 0.5)
            critic_optimizer.step()


def _joint_value_input(env, observations, algorithm):
    if algorithm == MAT_DEC:
        # The attention encoder receives only joint observations, as in discrete MAT-Dec.
        return np.concatenate([observations[name] for name in env.possible_agents])
    if algorithm in CENTRALIZED:
        return state(env)
    return None


def _update_token_value(critic, optimizer, buffers, targets, config, device):
    inputs = torch.stack(next(iter(buffers.values()))["vin"])
    targets = torch.stack(list(targets.values()), dim=-1)
    for _ in range(config["epochs"]):
        for ids in torch.randperm(len(inputs), device=device).split(config["batch_size"]):
            loss = (critic.values(inputs[ids]) - targets[ids]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MAT-Dec critic loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()


def initialize_from_checkpoint(agents, config):
    path = Path(config["warm_start_checkpoint"])
    data = torch.load(path, map_location="cpu", weights_only=True)
    origin = data["config"]
    for key in ("algorithm", "env"):
        if origin[key] != config[key]:
            raise ValueError(f"Warm-start checkpoint differs in {key}")
    origin_settings = resolved_env_kwargs(origin["env"], origin.get("env_kwargs"))
    target_settings = config["env_kwargs"]
    if config.get("allow_lift_mass_transfer") is True and config["env"] == "two_arm_lift":
        mass_parameters = {"pot_mass_scale", "pot_mass_range"}
        origin_settings = {
            key: value for key, value in origin_settings.items() if key not in mass_parameters
        }
        target_settings = {
            key: value for key, value in target_settings.items() if key not in mass_parameters
        }
    if origin_settings != target_settings:
        raise ValueError("Warm-start checkpoint differs in env_kwargs")
    if "behavior_cloning" in origin:
        if config.get("behavior_cloning", origin["behavior_cloning"]) != origin["behavior_cloning"]:
            raise ValueError("Warm-start checkpoint has different behavior-cloning provenance")
        config["behavior_cloning"] = origin["behavior_cloning"]
    if origin.get("actor_type", "mlp") != config.get("actor_type", "mlp"):
        raise ValueError("Warm-start checkpoint uses a different actor type")
    if data["agents"].keys() != agents.keys():
        raise ValueError("Warm-start checkpoint uses different agents")
    for name, model in agents.items():
        model.load_state_dict(data["agents"][name])
    config["warm_start_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def train(config, out):
    # 环境默认参数统一来自评估协议；不再让训练默认 25 步、评估默认 50 步。
    config = {
        **DEFAULTS,
        **config,
        "env_kwargs": resolved_env_kwargs(config["env"], config.get("env_kwargs")),
    }
    seed_all(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    env = make_env(config["env"], **config.get("env_kwargs", {}))
    out = Path(out)
    try:
        obs, _ = env.reset(seed=config["seed"])
        for i, a in enumerate(env.possible_agents):
            env.action_space(a).seed(config["seed"] + i)
        names = training_agents(env, config)
        agents = create_agents(
            env,
            obs,
            config["algorithm"],
            device,
            names,
            actor_type=config.get("actor_type", "mlp"),
        )
        if config.get("warm_start_checkpoint"):
            initialize_from_checkpoint(agents, config)
        out.mkdir(parents=True, exist_ok=False)
    except Exception:
        env.close()
        raise
    sequential = config["algorithm"] == HAPPO
    attention = config["algorithm"] == MAT_DEC
    shared_critic = sequential or attention
    optimizers = {
        a: torch.optim.Adam(
            [*model.actor.parameters(), model.log_std] if shared_critic else model.parameters(),
            lr=config["lr"],
        )
        for a, model in agents.items()
    }
    if shared_critic:
        critic_optimizer = torch.optim.Adam(
            next(iter(agents.values())).critic.parameters(), lr=config["lr"]
        )
    updates = 0
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_provenance(out, device)
    centralized = config["algorithm"] in CENTRALIZED
    episode, steps, ep_seed = 0, 0, config["seed"]
    returns = dict.fromkeys(env.possible_agents, 0.0)
    names = list(agents)
    metrics = (out / "episodes.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics, fieldnames=["episode", "steps"] + list(returns))
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
                global_obs = _joint_value_input(env, obs, config["algorithm"])
                actions = {}
                with torch.no_grad():
                    for a, model in agents.items():
                        local = torch.as_tensor(obs[a], dtype=torch.float32, device=device)
                        vin = torch.as_tensor(global_obs, device=device) if centralized else local
                        distribution = model.distribution(local)
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
                        actions[a] = model.environment_action(distribution, action)
                actions = with_opponent_actions(env, obs, actions, config)
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                next_state = _joint_value_input(env, next_obs, config["algorithm"])
                end = all(terminated[a] or truncated[a] for a in env.possible_agents)
                with torch.no_grad():
                    for a, model in agents.items():
                        b = buffers[a]
                        vin = torch.as_tensor(
                            next_state if centralized else next_obs[a],
                            dtype=torch.float32,
                            device=device,
                        )
                        b["next"].append(model.value(vin))
                        b["rew"].append(rewards[a])
                        b["term"].append(float(terminated[a]))
                        b["end"].append(float(end))
                for a in returns:
                    returns[a] += rewards[a]
                obs = next_obs
                steps += 1
                if end:
                    episode += 1
                    writer.writerow({"episode": episode, "steps": steps, **returns})
                    metrics.flush()
                    returns = dict.fromkeys(env.possible_agents, 0.0)
                    ep_seed += 1
                    obs, _ = env.reset(seed=ep_seed)
            if sequential:
                _update_happo(agents, optimizers, critic_optimizer, buffers, config, device)
                updates += 1
                print(f"steps={steps} episodes={episode} updates={updates}", flush=True)
                continue
            value_targets = {}
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
                if attention:
                    value_targets[a] = targets
                for _ in range(config.get("epochs", 4)):
                    for ids in torch.randperm(horizon, device=device).split(
                        config.get("batch_size", 64)
                    ):
                        dist = model.distribution(b["obs"][ids])
                        ratio = (dist.log_prob(b["act"][ids]) - b["logp"][ids]).exp()
                        clip = config.get("clip", 0.2)
                        policy_loss = -torch.min(
                            ratio * adv[ids], ratio.clamp(1 - clip, 1 + clip) * adv[ids]
                        ).mean()
                        loss = policy_loss
                        if not attention:
                            value_loss = (model.value(b["vin"][ids]) - targets[ids]).square().mean()
                            loss = loss + 0.5 * value_loss
                        loss = loss - config["entropy_coef"] * dist.entropy().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite PPO loss")
                        optimizers[a].zero_grad()
                        loss.backward()
                        parameters = (
                            [*model.actor.parameters(), model.log_std]
                            if attention
                            else model.parameters()
                        )
                        nn.utils.clip_grad_norm_(parameters, 0.5)
                        optimizers[a].step()
            if attention:
                _update_token_value(
                    agents[names[0]].critic,
                    critic_optimizer,
                    buffers,
                    value_targets,
                    config,
                    device,
                )
            updates += 1
            print(f"steps={steps} episodes={episode} updates={updates}", flush=True)
        torch.save(
            {
                "config": config,
                "steps": steps,
                "updates": updates,
                "agents": {a: p.state_dict() for a, p in agents.items()},
            },
            out / "model.pt",
        )
    finally:
        metrics.close()
        env.close()
    return out / "model.pt"


def load_policy(checkpoint, env, observations, expected_algorithm=None):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if expected_algorithm is not None and data["config"]["algorithm"] != expected_algorithm:
        raise ValueError("Checkpoint algorithm mismatch")
    agents = create_agents(
        env,
        observations,
        data["config"]["algorithm"],
        "cpu",
        training_agents(env, data["config"]),
        actor_type=data["config"].get("actor_type", "mlp"),
    )
    for a, model in agents.items():
        model.load_state_dict(data["agents"][a])
        model.eval()
    return agents, data
