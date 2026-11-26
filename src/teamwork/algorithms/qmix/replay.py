"""按回合回放：保留完整历史、末状态、真实终止与截断，padding 不计损失。"""

from collections import deque

import numpy as np
import torch


class EpisodeReplay:
    def __init__(
        self,
        spec,
        capacity,
        seed,
        *,
        use_action_availability=False,
        positive_replay_capacity=0,
        positive_replay_samples=0,
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Replay capacity must be a positive episode count")
        self.spec = spec
        self.positive_episodes = deque(maxlen=positive_replay_capacity)
        self.positive_replay_samples = positive_replay_samples
        self.use_action_availability = use_action_availability
        self.episodes = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.episodes)

    def add(self, episode):
        length = len(episode["actions"])
        n_agents = len(self.spec.agents)
        shapes = {
            "observations": (length + 1, n_agents, max(self.spec.observation_sizes)),
            "states": (length + 1, self.spec.state_size),
            "actions": (length, n_agents),
            "rewards": (length,),
            "terminated": (length,),
            "truncated": (length,),
        }
        if length < 1 or any(np.shape(episode[key]) != shape for key, shape in shapes.items()):
            raise ValueError("Replay requires nonempty episodes with a final observation/state")
        if any(not np.isfinite(episode[key]).all() for key in shapes):
            raise ValueError("Replay episode contains nonfinite values")
        actions = np.asarray(episode["actions"])
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("Replay actions must be integer indices")
        if np.any(actions < 0) or np.any(actions >= np.asarray(self.spec.action_counts)):
            raise ValueError("Replay contains an action outside the agent's space")
        for key in ("terminated", "truncated"):
            flags = np.asarray(episode[key])
            if not np.isin(flags, (0, 1)).all() or flags[:-1].any():
                raise ValueError("A replay episode cannot cross a reset boundary")
        copied = {}
        for key in shapes:
            dtype = np.int64 if key == "actions" else np.float32
            copied[key] = np.array(episode[key], dtype=dtype, copy=True)
        if self.use_action_availability:
            availability = np.asarray(episode["available_actions"])
            shape = (length + 1, n_agents, max(self.spec.action_counts))
            if availability.shape != shape or availability.dtype != np.bool_:
                raise ValueError(
                    "Replay availability requires a boolean mask for every observation"
                )
            in_space = np.arange(shape[-1])[None, :] < np.asarray(self.spec.action_counts)[:, None]
            picked = np.take_along_axis(availability[:-1], actions[..., None], axis=-1)
            if (
                (availability & ~in_space).any()
                or not picked.all()
                or not availability.any(axis=-1).all()
            ):
                raise ValueError("Replay contains missing or invalid available actions")
            copied["available_actions"] = availability.copy()
        self.episodes.append(copied)
        if (
            self.positive_episodes.maxlen
            and copied["terminated"][-1]
            and (copied["rewards"] > 0).any()
        ):
            self.positive_episodes.append(copied)

    def sample(self, batch_size, device="cpu"):
        if batch_size < 1 or batch_size > len(self):
            raise ValueError("Not enough replay episodes for this batch")
        positive_count = min(
            self.positive_replay_samples,
            len(self.positive_episodes),
            batch_size - 1,
        )
        chosen = []
        recent = self.episodes
        if positive_count:
            indices = self.rng.choice(
                len(self.positive_episodes), size=positive_count, replace=False
            )
            chosen = [self.positive_episodes[index] for index in indices]
            selected = {id(episode) for episode in chosen}
            recent = [episode for episode in self.episodes if id(episode) not in selected]
        indices = self.rng.choice(len(recent), size=batch_size - positive_count, replace=False)
        chosen.extend(recent[index] for index in indices)
        max_length = max(len(episode["actions"]) for episode in chosen)
        n_agents = len(self.spec.agents)
        result = {
            "observations": np.zeros(
                (batch_size, max_length + 1, n_agents, max(self.spec.observation_sizes)), np.float32
            ),
            "states": np.zeros((batch_size, max_length + 1, self.spec.state_size), np.float32),
            "actions": np.zeros((batch_size, max_length, n_agents), np.int64),
            "rewards": np.zeros((batch_size, max_length), np.float32),
            "terminated": np.zeros((batch_size, max_length), np.float32),
            "truncated": np.zeros((batch_size, max_length), np.float32),
            "filled": np.zeros((batch_size, max_length), np.float32),
        }
        if self.use_action_availability:
            shape = (batch_size, max_length + 1, n_agents, max(self.spec.action_counts))
            availability = np.zeros(shape, dtype=np.bool_)
            # 填充步无损失，但仍需保留一个有限的局部最大 Q。
            availability[..., 0] = True
            result["available_actions"] = availability
        for index, episode in enumerate(chosen):
            length = len(episode["actions"])
            for key, values in episode.items():
                result[key][index, : len(values)] = values
            result["filled"][index, :length] = 1.0
        return {key: torch.as_tensor(value, device=device) for key, value in result.items()}
