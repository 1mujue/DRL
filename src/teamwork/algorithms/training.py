"""IPPO/IQL 的训练阵营选择与固定对手动作。"""

from .baselines import baseline_actions


def training_agents(env, config):
    team = config.get("train_team", "all")
    opponent = config.get("train_opponent")
    if team == "all" and opponent is None:
        return list(env.possible_agents)
    if (
        config["env"] != "tag"
        or team not in {"predators", "prey"}
        or opponent not in {"random", "rule"}
    ):
        raise ValueError(
            "Fixed-opponent training requires Tag, train_team predators/prey "
            "and train_opponent random/rule"
        )
    roles = env.policy_metadata["roles"]
    return [a for a in env.possible_agents if roles[a] == (team == "predators")]


def with_opponent_actions(env, observations, actions, config):
    if config.get("train_opponent"):
        return {**baseline_actions(env, observations, config["train_opponent"]), **actions}
    return actions
