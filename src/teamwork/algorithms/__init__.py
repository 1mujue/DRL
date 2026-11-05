"""新增算法在本目录创建自己的子目录；公共接口和注册入口如下。"""

from .base import Algorithm, Policy
from .registry import available_algorithms, make_algorithm, register_algorithm

__all__ = ["Algorithm", "Policy", "available_algorithms", "make_algorithm", "register_algorithm"]
