"""新增环境：实现适配器，再在这里注册工厂；指标在 evaluation/metrics.py 定义。"""

from functools import partial

from .base import Environment
from .gym_tuple import make_gym_tuple
from .mpe import make_mpe

_BUILDERS = {
    "spread": partial(make_mpe, "spread"),
    "tag": partial(make_mpe, "tag"),
    "lbf": partial(make_gym_tuple, "lbf"),
    "rware": partial(make_gym_tuple, "rware"),
}


def register_environment(name, builder):
    if name in _BUILDERS:
        raise ValueError(f"Environment already registered: {name}")
    _BUILDERS[name] = builder


def available_environments():
    return tuple(_BUILDERS)


def make_env(name, render_mode=None, **kwargs) -> Environment:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown environment {name!r}; available: {available_environments()}")
    env = _BUILDERS[name](render_mode=render_mode, **kwargs)
    if not isinstance(env, Environment):
        raise TypeError("Environment builder must return an Environment adapter")
    return env
