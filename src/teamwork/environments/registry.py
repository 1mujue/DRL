"""注册环境工厂、动作类型和评估所需的依赖信息。"""

from collections.abc import Callable
from dataclasses import dataclass

from .base import Environment
from .lbf.environment import LBFEnvironment
from .robotics.two_arm_handover.environment import TwoArmHandover
from .robotics.two_arm_lift.environment import TwoArmLift
from .rware.environment import RWAREEnvironment
from .spread.environment import make_spread
from .tag.environment import make_tag


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


register_environment(
    "spread",
    make_spread,
    packages=("mpe2",),
    benchmark_sources=(
        "environments/adapters/mpe.py",
        "algorithms/baselines/mpe.py",
        "environments/spread/environment.py",
        "algorithms/baselines/spread/policy.py",
    ),
)
register_environment(
    "tag",
    make_tag,
    packages=("mpe2",),
    benchmark_sources=(
        "environments/adapters/mpe.py",
        "algorithms/baselines/mpe.py",
        "environments/tag/environment.py",
        "algorithms/baselines/tag/policy.py",
    ),
)
register_environment(
    "lbf",
    LBFEnvironment.from_gym,
    packages=("lbforaging",),
    benchmark_sources=("environments/adapters/gym_tuple.py", "environments/lbf/environment.py"),
)
register_environment(
    "rware",
    RWAREEnvironment.from_gym,
    packages=("rware",),
    benchmark_sources=("environments/adapters/gym_tuple.py", "environments/rware/environment.py"),
)
register_environment(
    "two_arm_lift",
    TwoArmLift,
    action_kind="continuous",
    packages=("robosuite", "mujoco"),
    benchmark_sources=(
        "environments/robotics/base.py",
        "environments/robotics/two_arm_lift/environment.py",
        "environments/robotics/two_arm_lift/reference.py",
        "environments/robotics/pose.py",
        "algorithms/baselines/two_arm_lift/policy.py",
        "evaluation/robotics/two_arm_lift/timing.py",
    ),
)

register_environment(
    "two_arm_handover",
    TwoArmHandover,
    action_kind="continuous",
    packages=("robosuite", "mujoco"),
    benchmark_sources=(
        "environments/robotics/base.py",
        "environments/robotics/two_arm_handover/environment.py",
        "environments/robotics/two_arm_handover/reference.py",
        "environments/robotics/pose.py",
        "algorithms/baselines/two_arm_handover/policy.py",
    ),
)
