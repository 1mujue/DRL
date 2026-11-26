"""Local pursuit and evasion rules for Simple Tag."""

import numpy as np

from ..mpe import cardinal_action


def tag_actions(env, observations):
    metadata = env.policy_metadata
    n_landmarks = metadata["landmark_count"]
    roles = metadata["roles"]
    actions = {}
    for agent in env.agents:
        obs = observations[agent]
        others = [a for a in env.possible_agents if a != agent]
        positions = obs[4 + 2 * n_landmarks : 4 + 2 * n_landmarks + 2 * len(others)]
        candidates = np.array(
            [p for a, p in zip(others, positions.reshape(-1, 2)) if roles[a] != roles[agent]]
        )
        delta = candidates[np.argmin(np.linalg.norm(candidates, axis=1))]
        if not roles[agent]:
            delta = -delta
            # Inward motion near the boundary, using own observed position.
            if np.max(np.abs(obs[2:4])) > 0.9:
                delta = -obs[2:4]
        actions[agent] = cardinal_action(delta)
    return actions
