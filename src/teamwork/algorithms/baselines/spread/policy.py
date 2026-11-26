"""Move toward the closest landmark in the agent's observations."""

import numpy as np

from ..mpe import cardinal_action


def spread_actions(env, observations):
    n_landmarks = env.policy_metadata["landmark_count"]
    actions = {}
    for agent in env.agents:
        obs = observations[agent]
        candidates = obs[4 : 4 + 2 * n_landmarks].reshape(-1, 2)
        delta = candidates[np.argmin(np.linalg.norm(candidates, axis=1))]
        actions[agent] = cardinal_action(delta)
    return actions
