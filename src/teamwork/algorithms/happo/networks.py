"""HAPPO 的网络组件：复用 MAPPO 的 actor/critic、价值归一化与执行期校验。

HAPPO 与 MAPPO 的结构差异只有一个：actor 更新顺序。因此这里不复制网络代码，
两者共用同一套 MAPPOModel / ValueNormalizer / GAE，使“顺序更新 vs 同时更新”
成为唯一变量，而不是被不同的网络实现或优势估计口径混淆。
"""

from ...environments.base import EnvironmentSpec
from ..mappo.networks import MAPPOModel as HAPPOModel
from ..mappo.networks import ValueNormalizer, make_mlp
from ..mappo.networks import resolve_device as _resolve_device
from ..mappo.networks import validate_observations as _validate_observations

__all__ = [
    "HAPPOModel",
    "ValueNormalizer",
    "actor_for",
    "make_mlp",
    "resolve_actor_order",
    "resolve_device",
    "validate_observations",
]


def _rename(error):
    """复用模块的报错里带有 MAPPO 字样，改成 HAPPO 以免排错时误判来源。"""
    return str(error).replace("MAPPO", "HAPPO")


def resolve_device(device):
    try:
        return _resolve_device(device)
    except ValueError as error:
        raise ValueError(_rename(error)) from error


def validate_observations(observations, spec, device=None):
    try:
        return _validate_observations(observations, spec, device)
    except ValueError as error:
        raise ValueError(_rename(error)) from error


def actor_for(model, agent):
    """HAPPO 要求异构 actor；share_policy 必须为 false，因此每个 agent 一组参数。"""
    return model.actors[model.agent_to_group[agent]]


def resolve_actor_order(spec: EnvironmentSpec, configured):
    """缺省按环境 agent 顺序；显式顺序必须是全体 agent 的一个排列。"""
    names = list(spec.agents)
    if configured is None:
        return names
    if not isinstance(configured, (list, tuple)) or any(
        not isinstance(name, str) for name in configured
    ):
        raise ValueError("HAPPO actor_order must be null or a list of agent names")
    order = list(configured)
    if len(order) != len(names) or set(order) != set(names):
        raise ValueError("HAPPO actor_order must list every agent exactly once")
    return order
