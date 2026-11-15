"""QTRAN-base 注册入口。"""

from ..value_decomposition import ValueDecompositionAlgorithm
from .learner import QTRANLearner


class QTRANAlgorithm(ValueDecompositionAlgorithm):
    name = "qtran"
    implementation = "qtran_base_v1"
    learner_factory = QTRANLearner
    defaults = {"opt_loss_weight": 1.0, "nopt_loss_weight": 1.0}
