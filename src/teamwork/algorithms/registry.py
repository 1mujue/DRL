"""新增算法：实现 Algorithm，然后在此注册；CLI 和评估无需增加 if 分支。"""

from .base import Algorithm, Policy
from .coma.algorithm import COMAAlgorithm
from .continuous_off_policy import VARIANTS, ContinuousOffPolicyAlgorithm
from .happo.algorithm import HAPPOAlgorithm
from .hasac.algorithm import HASACAlgorithm
from .iql.algorithm import IQLAlgorithm
from .maac.algorithm import MAACAlgorithm
from .mappo.algorithm import MAPPOAlgorithm
from .mat_dec.algorithm import MATDecAlgorithm
from .names import (
    CENTRALIZED_PPO,
    CENTRALIZED_PPO_CONTINUOUS,
    canonical_algorithm_name,
)
from .ppo.algorithm import PPOAlgorithm
from .qmix.algorithm import QMIXAlgorithm
from .qplex.algorithm import QPLEXAlgorithm
from .qtran.algorithm import QTRANAlgorithm
from .vdn.algorithm import VDNAlgorithm

_FACTORIES = {
    "iql": IQLAlgorithm,
    "ippo_continuous": lambda: PPOAlgorithm("ippo_continuous"),
    CENTRALIZED_PPO_CONTINUOUS: lambda: PPOAlgorithm(CENTRALIZED_PPO_CONTINUOUS),
    "happo_continuous": lambda: PPOAlgorithm("happo_continuous"),
    "mat_dec_continuous": lambda: PPOAlgorithm("mat_dec_continuous"),
    "ippo": lambda: PPOAlgorithm("ippo"),
    CENTRALIZED_PPO: lambda: PPOAlgorithm(CENTRALIZED_PPO),
    "coma": lambda: COMAAlgorithm(),
    "maac": lambda: MAACAlgorithm(),
    "vdn": VDNAlgorithm,
    "mat_dec": MATDecAlgorithm,
}


def register_algorithm(name, factory):
    if name in _FACTORIES or canonical_algorithm_name(name) != name:
        raise ValueError(f"Algorithm already registered: {name}")
    _FACTORIES[name] = factory


def available_algorithms():
    return tuple(_FACTORIES)


def make_algorithm(name) -> Algorithm:
    name = canonical_algorithm_name(name)
    if name not in _FACTORIES:
        raise ValueError(f"Unknown algorithm {name!r}; available: {available_algorithms()}")
    algorithm = _FACTORIES[name]()
    if not isinstance(algorithm, Algorithm):
        raise TypeError("Registered factory must return an Algorithm implementation")
    return algorithm


for algorithm_name in VARIANTS:
    register_algorithm(
        algorithm_name, lambda name=algorithm_name: ContinuousOffPolicyAlgorithm(name)
    )

register_algorithm("mappo_ff", MAPPOAlgorithm)
register_algorithm("qmix", QMIXAlgorithm)
register_algorithm("happo", HAPPOAlgorithm)
register_algorithm("hasac_discrete", HASACAlgorithm)
register_algorithm("qtran", QTRANAlgorithm)
register_algorithm("qplex", QPLEXAlgorithm)


__all__ = ["Algorithm", "Policy", "available_algorithms", "make_algorithm", "register_algorithm"]
