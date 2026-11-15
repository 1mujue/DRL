"""可选的 LBF 当前动作可用性，按环境的 agent 顺序返回。"""

import numpy as np


def lbf_action_availability(config, env, spec):
    if not config.get("mask_unavailable_actions", False):
        return None
    if getattr(env, "family", None) != "lbf":
        raise ValueError("mask_unavailable_actions requires the LBF environment adapter")
    base = env.base.unwrapped
    shape = (len(spec.agents), max(spec.action_counts))

    def available_actions():
        # LBF 在每次 reset 或 step 后更新可执行动作，包括原地等待。
        available = np.zeros(shape, dtype=np.bool_)
        for index, player in enumerate(base.players):
            available[index, [action.value for action in base._valid_actions[player]]] = True
        if not available.any(axis=-1).all():
            raise ValueError("Each LBF agent must have at least one available action")
        return available

    return available_actions
