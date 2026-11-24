"""注册环境工厂、动作类型和评估所需的依赖信息。"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from .base import Environment
from .gym_tuple import make_gym_tuple
from .mpe import make_mpe
from .two_arm_handover import TwoArmHandover
from .two_arm_lift import TwoArmLift


@dataclass(frozen=True)
class EnvironmentRegistration:
    builder: Callable[..., Environment]
    action_kind: str
    packages: tuple[str, ...]
    benchmark_sources: tuple[str, ...]


_BUILDERS: dict[str, EnvironmentRegistration] = {}


def register_environment(
    name, builder, *, action_kind="discrete", packages=(), benchmark_sources=()
):
    if name in _BUILDERS:
        raise ValueError(f"Environment already registered: {name}")
    if action_kind not in {"discrete", "continuous"}:
        raise ValueError("action_kind must be discrete or continuous")
    _BUILDERS[name] = EnvironmentRegistration(
        builder, action_kind, tuple(packages), tuple(benchmark_sources)
    )


def available_environments(*, action_kind=None):
    return tuple(
        name
        for name, registration in _BUILDERS.items()
        if action_kind is None or registration.action_kind == action_kind
    )


def environment_registration(name):
    try:
        return _BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown environment {name!r}; available: {available_environments()}"
        ) from None


def make_env(name, render_mode=None, **kwargs) -> Environment:
    env = environment_registration(name).builder(render_mode=render_mode, **kwargs)
    if not isinstance(env, Environment):
        raise TypeError("Environment builder must return an Environment adapter")
    return env


register_environment("spread", partial(make_mpe, "spread"), packages=("mpe2",))
register_environment("tag", partial(make_mpe, "tag"), packages=("mpe2",))
register_environment("lbf", partial(make_gym_tuple, "lbf"), packages=("lbforaging",))
register_environment("rware", partial(make_gym_tuple, "rware"), packages=("rware",))
register_environment(
    "two_arm_lift",
    TwoArmLift,
    action_kind="continuous",
    packages=("robosuite", "mujoco"),
    benchmark_sources=(
        "environments/two_panda.py",
        "environments/two_arm_lift.py",
        "environments/lift_coordinates.py",
        "algorithms/two_arm_lift_policy.py",
        "evaluation/lift_timing.py",
    ),
)

register_environment(
    "two_arm_handover",
    TwoArmHandover,
    action_kind="continuous",
    packages=("robosuite", "mujoco"),
    benchmark_sources=(
        "environments/two_panda.py",
        "environments/two_arm_handover.py",
        "environments/handover_reference.py",
        "environments/lift_coordinates.py",
        "algorithms/two_arm_handover_policy.py",
    ),
)
