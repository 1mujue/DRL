"""Canonical PPO baseline names and compatibility with old experiment identifiers."""

CENTRALIZED_PPO = "centralized_ppo"
CENTRALIZED_PPO_CONTINUOUS = "centralized_ppo_continuous"

_LEGACY_NAMES = {
    "mappo": CENTRALIZED_PPO,
    "mappo_continuous": CENTRALIZED_PPO_CONTINUOUS,
}


def canonical_algorithm_name(name):
    return _LEGACY_NAMES.get(name, name)
