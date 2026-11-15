"""QPLEX 注册入口。"""

from typing import ClassVar

from ..value_decomposition import ValueDecompositionAlgorithm
from .learner import QPLEXLearner


class QPLEXAlgorithm(ValueDecompositionAlgorithm):
    name = "qplex"
    implementation = "qplex_duplex_dueling_v1"
    learner_factory = QPLEXLearner
    defaults: ClassVar[dict[str, int]] = {"attention_heads": 4}
