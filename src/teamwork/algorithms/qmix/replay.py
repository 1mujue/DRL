"""按回合回放：保留完整历史、末状态、真实终止与截断，padding 不计损失。"""

from collections import deque

import numpy as np
import torch


class EpisodeReplay:
    def __init__(self, spec, capacity, seed):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Replay capacity must be a positive episode count")
        self.spec = spec
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
            "actions": (length, n_agents), "rewards": (length,),
            "terminated": (length,), "truncated": (length,),
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
        self.episodes.append(copied)

    def sample(self, batch_size, device="cpu"):
        if batch_size < 1 or batch_size > len(self):
            raise ValueError("Not enough replay episodes for this batch")
        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        chosen = [self.episodes[index] for index in indices]
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
        for index, episode in enumerate(chosen):
            length = len(episode["actions"])
            for key, values in episode.items():
                result[key][index, :len(values)] = values
            result["filled"][index, :length] = 1.0
        return {key: torch.as_tensor(value, device=device) for key, value in result.items()}
