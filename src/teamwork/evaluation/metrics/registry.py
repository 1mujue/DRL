"""Registration and lookup for versioned scene scoring protocols."""

from copy import deepcopy
from dataclasses import asdict

from .base import MetricSuite
from .lbf import rules as lbf
from .rware import rules as rware
from .spread import rules as spread
from .tag import rules as tag
from .two_arm_handover import rules as two_arm_handover
from .two_arm_lift import rules as two_arm_lift

METRICS_VERSION = "1.0.0"

_SUITES = {}
_DEFINITIONS = {}
_DEFAULTS = {}


def register_metrics(name, suite, definitions, default_env_kwargs):
    """新环境的协议必须与环境适配器一起提交，不能在每次评估时临时指定。"""
    if name in _SUITES:
        raise ValueError(f"Metrics already registered: {name}; change the protocol version instead")
    _SUITES[name] = suite
    _DEFINITIONS[name] = tuple(definitions)
    _DEFAULTS[name] = deepcopy(default_env_kwargs)


register_metrics("spread", spread.SpreadMetrics, spread.DEFINITIONS, spread.DEFAULT_ENV_KWARGS)
register_metrics("tag", tag.TagMetrics, tag.DEFINITIONS, tag.DEFAULT_ENV_KWARGS)
register_metrics("lbf", lbf.LBFMetrics, lbf.DEFINITIONS, lbf.DEFAULT_ENV_KWARGS)
register_metrics("rware", rware.RWAREMetrics, rware.DEFINITIONS, rware.DEFAULT_ENV_KWARGS)
register_metrics(
    "two_arm_lift",
    two_arm_lift.TwoArmLiftMetrics,
    two_arm_lift.DEFINITIONS,
    two_arm_lift.DEFAULT_ENV_KWARGS,
)
register_metrics(
    "two_arm_handover",
    two_arm_handover.TwoArmHandoverMetrics,
    two_arm_handover.DEFINITIONS,
    two_arm_handover.DEFAULT_ENV_KWARGS,
)


def make_metrics(name) -> MetricSuite:
    return _SUITES[name]()


def protocol_description(name):
    return {
        "environment": name,
        "metrics_version": METRICS_VERSION,
        "default_env_kwargs": deepcopy(_DEFAULTS[name]),
        "metrics": [asdict(d) for d in _DEFINITIONS[name]],
        "coverage_radius": spread.COVERAGE_RADIUS if name == "spread" else None,
        "boundary_limit": tag.BOUNDARY_LIMIT if name == "tag" else None,
        "sampling": "after_each_joint_step; reset state is not a scored step",
    }


def resolved_env_kwargs(name, overrides=None):
    return {**deepcopy(_DEFAULTS[name]), **(overrides or {})}
