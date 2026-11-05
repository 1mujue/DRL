"""新增算法：实现 Algorithm，然后在此注册；CLI 和评估无需增加 if 分支。"""

from .base import Algorithm, Policy
from .ppo.algorithm import PPOAlgorithm


def _make_mappo():
    # 独立方法按公共接口接入，不改变现有 PPO 注册项。
    from .mappo.algorithm import MAPPOAlgorithm

    return MAPPOAlgorithm()


_FACTORIES = {
    "ippo": lambda: PPOAlgorithm("ippo"),
    "mappo": lambda: PPOAlgorithm("mappo"),  # 保留旧配置名；只是集中 critic PPO 基线。
    "mappo_ff": _make_mappo,
}


def register_algorithm(name, factory):
    if name in _FACTORIES:
        raise ValueError(f"Algorithm already registered: {name}")
    _FACTORIES[name] = factory


def available_algorithms():
    return tuple(_FACTORIES)


def make_algorithm(name) -> Algorithm:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown algorithm {name!r}; available: {available_algorithms()}")
    algorithm = _FACTORIES[name]()
    if not isinstance(algorithm, Algorithm):
        raise TypeError("Registered factory must return an Algorithm implementation")
    return algorithm


__all__ = ["Algorithm", "Policy", "available_algorithms", "make_algorithm", "register_algorithm"]
