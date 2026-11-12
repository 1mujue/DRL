"""前馈 MAPPO：通过公共 Algorithm / Policy 接口接入。"""

from .algorithm import MAPPOAlgorithm, MAPPOPolicy

__all__ = ["MAPPOAlgorithm", "MAPPOPolicy"]
