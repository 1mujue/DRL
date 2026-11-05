"""随机/规则对照策略；只使用局部观测和静态任务信息。"""

import numpy as np


def baseline_actions(env, observations, policy="random"):
    if policy == "random":
        return {a: env.action_space(a).sample() for a in env.agents}
    # Rule uses local observations only. No privileged simulator coordinates.
    if policy != "rule" or not env.metadata.get("name", "").startswith("simple_"):
        raise ValueError("Rule baseline currently supports MPE Spread/Tag only")
    name = env.metadata["name"]
    n_landmarks = env.policy_metadata["landmark_count"]
    roles = env.policy_metadata["roles"]
    result = {}
    for agent in env.agents:
        obs = observations[agent]
        if "spread" in name:
            candidates = obs[4 : 4 + 2 * n_landmarks].reshape(-1, 2)
            delta = candidates[np.argmin(np.linalg.norm(candidates, axis=1))]
        else:
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
        axis = int(np.argmax(np.abs(delta)))
        result[agent] = (2 if delta[0] > 0 else 1) if axis == 0 else (4 if delta[1] > 0 else 3)
    return result
