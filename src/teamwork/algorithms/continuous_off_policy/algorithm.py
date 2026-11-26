"""Continuous DDPG, TD3 and SAC with local or centralized critics."""

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...environments.base import state
from ...environments.registry import make_env
from ...evaluation.metrics import resolved_env_kwargs
from ...experiments.runner import seed_all, write_provenance
from ..base import Algorithm, Policy
from ..ppo.trainer import ContinuousActor, configure_actor, initialize_from_checkpoint, mlp


@dataclass(frozen=True)
class Variant:
    method: str
    centralized: bool

    @property
    def soft(self):
        return self.method in {"sac", "hasac"}

    @property
    def twin(self):
        return self.method != "ddpg"

    @property
    def delayed(self):
        return self.method == "td3"

    @property
    def sequential(self):
        return self.method == "hasac"


VARIANTS = {
    "iddpg_continuous": Variant("ddpg", False),
    "maddpg_continuous": Variant("ddpg", True),
    "itd3_continuous": Variant("td3", False),
    "matd3_continuous": Variant("td3", True),
    "isac_continuous": Variant("sac", False),
    "masac_continuous": Variant("sac", True),
    "hasac_continuous": Variant("hasac", True),
}

DEFAULTS = {
    "device": "cpu",
    "actor_lr": 0.000001,
    "critic_lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "replay_capacity": 20000,
    "update_every": 8,
    "policy_delay": 2,
    "exploration_noise": 0.02,
    "target_noise": 0.02,
    "target_noise_clip": 0.05,
    "temperature": 0.005,
}


def build_actors(env, observations, config, device):
    stochastic = VARIANTS[config["algorithm"]].soft
    models = {}
    for agent in env.possible_agents:
        obs_size = len(observations[agent])
        actor = ContinuousActor(obs_size, env.action_space(agent), stochastic=stochastic)
        models[agent] = configure_actor(actor, env, obs_size, config.get("actor_type", "mlp")).to(
            device
        )
    return models


class OffPolicyPolicy(Policy):
    def __init__(self, actors, stochastic):
        self.actors = actors
        self.stochastic = stochastic

    @torch.no_grad()
    def act(self, observations, deterministic=True):
        actions = {}
        for agent, actor in self.actors.items():
            local = torch.as_tensor(observations[agent], dtype=torch.float32)
            if self.stochastic and not deterministic:
                distribution = actor.distribution(local)
                action = distribution.action(distribution.sample())
            else:
                action = actor.bounded(actor.actor(local))
            actions[agent] = action.numpy()
        return actions


class Critics(nn.Module):
    def __init__(self, input_size, twin):
        super().__init__()
        self.values = nn.ModuleList(mlp(input_size, 1) for _ in range(2 if twin else 1))

    def forward(self, features):
        return torch.cat([network(features) for network in self.values], dim=-1)


@dataclass(frozen=True)
class Transition:
    observations: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    next_observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    terminated: dict[str, bool]
    state: np.ndarray
    next_state: np.ndarray


class Replay:
    def __init__(self, capacity, seed):
        self.data = []
        self.capacity = capacity
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def append(self, transition):
        if len(self.data) == self.capacity:
            self.data[self.cursor] = transition
        else:
            self.data.append(transition)
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, size, agents, device):
        rows = [self.data[i] for i in self.rng.choice(len(self.data), size, replace=False)]

        def stack(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

        return Transition(
            {name: stack([row.observations[name] for row in rows]) for name in agents},
            {name: stack([row.actions[name] for row in rows]) for name in agents},
            {name: stack([row.next_observations[name] for row in rows]) for name in agents},
            {name: stack([row.rewards[name] for row in rows]) for name in agents},
            {name: stack([row.terminated[name] for row in rows]) for name in agents},
            stack([row.state for row in rows]),
            stack([row.next_state for row in rows]),
        )


class Learner:
    def __init__(self, actors, observations, state_size, config, device):
        self.actors = actors
        self.config = config
        self.variant = VARIANTS[config["algorithm"]]
        self.names = tuple(actors)
        critic_names = self.names[:1] if self.variant.sequential else self.names
        joint_size = state_size + sum(actor.action_scale.numel() for actor in actors.values())
        self.critics = {
            name: Critics(
                joint_size
                if self.variant.centralized
                else len(observations[name]) + actors[name].action_scale.numel(),
                self.variant.twin,
            ).to(device)
            for name in critic_names
        }
        self.target_critics = {
            name: deepcopy(q).requires_grad_(False) for name, q in self.critics.items()
        }
        # SAC evaluates next actions with the current stochastic policy.
        self.target_actors = (
            actors
            if self.variant.soft
            else {name: deepcopy(actor).requires_grad_(False) for name, actor in actors.items()}
        )
        self.actor_optimizers = {
            name: torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
            for name, actor in actors.items()
        }
        self.critic_optimizers = {
            name: torch.optim.Adam(critic.parameters(), lr=config["critic_lr"])
            for name, critic in self.critics.items()
        }
        self.device = device
        self.critic_updates = 0
        self.actor_updates = 0

    def _features(self, name, observations, actions, global_state):
        if self.variant.centralized:
            return torch.cat((global_state, *(actions[agent] for agent in self.names)), dim=-1)
        return torch.cat((observations[name], actions[name]), dim=-1)

    def _sample(self, actor, observation, *, target=False):
        if self.variant.soft:
            distribution = actor.distribution(observation)
            latent = distribution.rsample()
            return distribution.action(latent), distribution.log_prob(latent)
        action = actor.bounded(actor.actor(observation))
        if target and self.variant.delayed:
            std, bound = self.config["target_noise"], self.config["target_noise_clip"]
            noise = (torch.randn_like(action) * std).clamp(-bound, bound)
            action = actor.clip(action + noise * actor.action_scale)
        return action, None

    @torch.no_grad()
    def explore(self, observations):
        actions = {}
        for name, actor in self.actors.items():
            obs = torch.as_tensor(observations[name], dtype=torch.float32, device=self.device)
            action, _ = self._sample(actor, obs)
            if not self.variant.soft:
                noise = torch.randn_like(action) * self.config["exploration_noise"]
                action = actor.clip(action + noise * actor.action_scale)
            actions[name] = action.cpu().numpy()
        return actions

    @torch.no_grad()
    def _targets(self, batch):
        drawn = {
            name: self._sample(actor, batch.next_observations[name], target=True)
            for name, actor in self.target_actors.items()
        }
        actions = {name: pair[0] for name, pair in drawn.items()}
        values = {}
        for name, critic in self.target_critics.items():
            inputs = self._features(name, batch.next_observations, actions, batch.next_state)
            target = critic(inputs).min(dim=-1).values
            if self.variant.soft:
                entropy_agents = self.names if self.variant.centralized else (name,)
                target -= self.config["temperature"] * sum(drawn[a][1] for a in entropy_agents)
            rewards = batch.rewards[name]
            if self.variant.sequential:
                rewards = torch.stack(list(batch.rewards.values())).mean(dim=0)
            # Time limits are truncations, so the final observation still bootstraps.
            values[name] = rewards + self.config["gamma"] * (1 - batch.terminated[name]) * target
        return values

    def update(self, batch):
        targets = self._targets(batch)
        for name, critic in self.critics.items():
            inputs = self._features(name, batch.observations, batch.actions, batch.state)
            loss = (critic(inputs) - targets[name].unsqueeze(-1)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite off-policy critic loss")
            optimizer = self.critic_optimizers[name]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.critic_updates += 1
        delay = self.config["policy_delay"] if self.variant.delayed else 1
        if self.critic_updates % delay:
            return
        self._update_actors(batch)
        self.actor_updates += 1
        self._sync_targets()

    def _update_actors(self, batch):
        with torch.no_grad():
            other_actions = {
                name: self._sample(actor, batch.observations[name])[0]
                for name, actor in self.actors.items()
            }
        order = self.names
        if self.variant.sequential:
            order = tuple(self.names[i] for i in torch.randperm(len(self.names)).tolist())
        for name in order:
            actor = self.actors[name]
            own_action, log_prob = self._sample(actor, batch.observations[name])
            actions = {**other_actions, name: own_action}
            inputs = self._features(name, batch.observations, actions, batch.state)
            critic_name = self.names[0] if self.variant.sequential else name
            critic = self.critics[critic_name]
            critic.requires_grad_(False)
            try:
                values = critic(inputs)
                if self.variant.soft:
                    loss = (
                        self.config["temperature"] * log_prob - values.min(dim=-1).values
                    ).mean()
                else:
                    loss = -values[:, 0].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite off-policy actor loss")
                optimizer = self.actor_optimizers[name]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            finally:
                critic.requires_grad_(True)
            if self.variant.sequential:
                # The next actor conditions on this actor's freshly updated distribution.
                with torch.no_grad():
                    other_actions[name] = self._sample(actor, batch.observations[name])[0]

    @torch.no_grad()
    def _sync_targets(self):
        pairs = [(self.critics, self.target_critics)]
        if not self.variant.soft:
            pairs.append((self.actors, self.target_actors))
        for sources, targets in pairs:
            for name, target in targets.items():
                for current, delayed in zip(
                    sources[name].parameters(), target.parameters(), strict=True
                ):
                    delayed.lerp_(current, self.config["tau"])


def _reset_for_training(env, seed, resets):
    observation, infos = env.reset(seed=seed)
    if any(infos.values()):
        resets.append({"seed": seed, "infos": infos})
    return observation


class ContinuousOffPolicyAlgorithm(Algorithm):
    action_kind = "continuous"

    def __init__(self, name):
        self.name = name

    def train(self, config, out):
        if config["algorithm"] != self.name:
            raise ValueError("Algorithm configuration does not match selected implementation")
        config = {**DEFAULTS, **config}
        config["env_kwargs"] = resolved_env_kwargs(config["env"], config.get("env_kwargs"))
        for key in ("total_steps", "batch_size", "replay_capacity", "update_every", "policy_delay"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["batch_size"] > config["replay_capacity"]:
            raise ValueError("batch_size exceeds replay_capacity")
        if config.get("train_team", "all") != "all" or config.get("train_opponent") is not None:
            raise ValueError("Continuous off-policy training requires the full team")
        seed_all(config["seed"])
        env = make_env(config["env"], **config["env_kwargs"])
        try:
            return self._train(env, config, Path(out))
        finally:
            env.close()

    def _train(self, env, config, out):
        device = torch.device(config["device"])
        resets = []
        observations = _reset_for_training(env, config["seed"], resets)
        actors = build_actors(env, observations, config, device)
        if config.get("warm_start_checkpoint"):
            initialize_from_checkpoint(actors, config)
        learner = Learner(actors, observations, len(state(env)), config, device)
        replay = Replay(config["replay_capacity"], config["seed"])
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        write_provenance(out, device)
        returns = dict.fromkeys(actors, 0.0)
        episode = 0
        with (out / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "steps", *actors])
            writer.writeheader()
            for steps in range(1, config["total_steps"] + 1):
                before = state(env).copy()
                actions = learner.explore(observations)
                following, rewards, terminated, truncated, _ = env.step(actions)
                replay.append(
                    Transition(
                        observations,
                        actions,
                        following,
                        rewards,
                        terminated,
                        before,
                        state(env).copy(),
                    )
                )
                if len(replay.data) >= config["batch_size"] and steps % config["update_every"] == 0:
                    learner.update(replay.sample(config["batch_size"], actors, device))
                for name in actors:
                    returns[name] += rewards[name]
                observations = following
                ended = [terminated[name] or truncated[name] for name in actors]
                if any(ended):
                    if not all(ended):
                        raise ValueError(
                            "Continuous training requires synchronous episode boundaries"
                        )
                    episode += 1
                    writer.writerow({"episode": episode, "steps": steps, **returns})
                    if steps < config["total_steps"]:
                        observations = _reset_for_training(env, config["seed"] + episode, resets)
                    returns = dict.fromkeys(actors, 0.0)
            print(
                f"steps={steps} episodes={episode} actor_updates={learner.actor_updates} "
                f"critic_updates={learner.critic_updates}",
                flush=True,
            )
        if resets:
            (out / "episode_resets.json").write_text(
                json.dumps(resets, indent=2) + "\n", encoding="utf-8"
            )
        checkpoint = out / "model.pt"
        torch.save(
            {
                "config": config,
                "steps": steps,
                "updates": learner.actor_updates,
                "critic_updates": learner.critic_updates,
                "agents": {name: actor.state_dict() for name, actor in actors.items()},
            },
            checkpoint,
        )
        return checkpoint

    def load_policy(self, checkpoint, env, observations):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["config"]["algorithm"] != self.name:
            raise ValueError("Checkpoint algorithm mismatch")
        actors = build_actors(env, observations, data["config"], "cpu")
        for name, actor in actors.items():
            actor.load_state_dict(data["agents"][name])
            actor.eval()
        return OffPolicyPolicy(actors, VARIANTS[self.name].soft)
