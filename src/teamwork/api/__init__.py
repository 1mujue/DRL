"""后端接入入口：训练任务、模型推理、统一评估与结果查询。"""

from .contracts import EvaluationRequest, TrainingRequest
from .errors import AlgorithmAPIError
from .policy import PolicySession
from .service import AlgorithmAPI

__all__ = [
    "AlgorithmAPI",
    "AlgorithmAPIError",
    "EvaluationRequest",
    "PolicySession",
    "TrainingRequest",
]
