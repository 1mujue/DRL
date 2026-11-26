"""训练阵营校验，以及 PPO/IQL 的阵营选择与固定对手动作。"""

from .baselines import baseline_actions
from .names import CENTRALIZED_PPO, canonical_algorithm_name

FIXED_OPPONENT_ALGORITHMS = frozenset({"ippo", "iql", CENTRALIZED_PPO})


def validate_training_scope(config, *, require_fixed_opponent=False):
    """在启动训练前检查所选算法是否实现请求中的阵营模式。"""
    team = config.get("train_team", "all")
    opponent = config.get("train_opponent")
    joint_training = team == "all" and opponent is None
    if config["env"] != "tag":
        if not joint_training:
            raise ValueError("Fixed-opponent training is supported only for Tag")
        return
    if joint_training and not require_fixed_opponent:
        return

    algorithm = canonical_algorithm_name(config["algorithm"])
    if algorithm not in FIXED_OPPONENT_ALGORITHMS:
        raise ValueError(
            f"{algorithm} does not support fixed-opponent Tag training. "
            "Choose a cooperative environment or a supported algorithm."
        )
    if team not in ("predators", "prey") or opponent not in ("rule", "random"):
        raise ValueError(
            "Fixed-opponent training requires Tag, train_team predators/prey "
            "and train_opponent random/rule"
        )


def training_agents(env, config):
    validate_training_scope(config)
    team = config.get("train_team", "all")
    if team == "all":
        return list(env.possible_agents)
    roles = env.policy_metadata["roles"]
    return [a for a in env.possible_agents if roles[a] == (team == "predators")]


def with_opponent_actions(env, observations, actions, config):
    if config.get("train_opponent"):
        return {**baseline_actions(env, observations, config["train_opponent"]), **actions}
    return actions
