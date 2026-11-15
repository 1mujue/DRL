"""HAPPO：异构 actor + 集中 critic + 顺序更新，通过公共 Algorithm / Policy 接口接入。"""

from .algorithm import HAPPOAlgorithm, HAPPOPolicy

__all__ = ["HAPPOAlgorithm", "HAPPOPolicy"]
