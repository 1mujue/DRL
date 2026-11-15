"""SDK 请求合同；在创建任务前拒绝无效参数。"""

import json
import math
from dataclasses import asdict, dataclass, field, fields

import torch

from ..environments.registry import available_environments, make_env
from ..evaluation.metrics import resolved_env_kwargs
from .catalog import ALGORITHMS, INTEGER_PARAMETERS
from .errors import invalid


def positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise invalid(name, f"{name} must be an integer >= {minimum}")
    return value


def seed_value(value):
    positive_integer(value, "seed", 0)
    if value > 2**32 - 1:
        raise invalid("seed", "seed must fit an unsigned 32-bit integer")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise invalid(name, f"{name} must be a boolean")
    return value


def json_object(value, name):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise invalid(name, f"{name} must be an object with string keys")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise invalid(name, f"{name} must contain finite JSON values") from exc


class Request:
    @classmethod
    def from_dict(cls, value):
        value = json_object(value, "request")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise invalid("request", f"Unknown fields: {sorted(unknown)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise invalid("request", str(exc)) from exc

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainingRequest(Request):
    algorithm: str
    env: str = "spread"
    steps: int = 100000
    seed: int = 0
    device: str = "cpu"
    parameters: dict = field(default_factory=dict)
    env_kwargs: dict = field(default_factory=dict)
    team: str = "all"
    opponent: str | None = None

    def config(self):
        if not isinstance(self.algorithm, str) or self.algorithm not in ALGORITHMS:
            raise invalid("algorithm", "Choose ippo or iql")
        if not isinstance(self.env, str) or self.env not in available_environments():
            raise invalid("env", f"Unknown environment: {self.env!r}")
        positive_integer(self.steps, "steps")
        seed_value(self.seed)
        if self.device not in ("cpu", "cuda"):
            raise invalid("device", "Choose cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise invalid("device", "CUDA is not available on this host")
        if self.env == "tag":
            if self.team not in ("predators", "prey") or self.opponent not in ("rule", "random"):
                raise invalid("team", "Tag requires an explicit team and fixed opponent")
        elif self.team != "all" or self.opponent is not None:
            raise invalid("team", "Team/opponent selection is supported only for Tag")
        parameters = json_object(self.parameters, "parameters")
        defaults = ALGORITHMS[self.algorithm]["defaults"]
        unknown = set(parameters) - (set(defaults) - {"device"})
        if unknown:
            raise invalid("parameters", f"Unknown algorithm parameters: {sorted(unknown)}")
        parameters = {**defaults, **parameters}
        for name, value in parameters.items():
            if name == "device":
                continue
            if name in INTEGER_PARAMETERS[self.algorithm]:
                positive_integer(value, name, 0 if name == "learning_starts" else 1)
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise invalid(name, f"{name} must be a finite number")
            if name in ("gamma", "gae_lambda", "epsilon_start", "epsilon_end", "eval_epsilon"):
                if not 0 <= value <= 1:
                    raise invalid(name, f"{name} must be in [0, 1]")
            elif value <= 0:
                raise invalid(name, f"{name} must be positive")
        if self.algorithm == "iql":
            if parameters["batch_size"] > parameters["buffer_size"]:
                raise invalid("batch_size", "batch_size must not exceed buffer_size")
            if parameters["epsilon_end"] > parameters["epsilon_start"]:
                raise invalid("epsilon_end", "epsilon_end must not exceed epsilon_start")
        settings = resolved_env_kwargs(self.env, json_object(self.env_kwargs, "env_kwargs"))
        return {
            **parameters,
            "algorithm": self.algorithm,
            "env": self.env,
            "total_steps": self.steps,
            "seed": self.seed,
            "device": self.device,
            "env_kwargs": settings,
            "train_team": self.team,
            "train_opponent": self.opponent,
        }

    def validate(self):
        config = self.config()
        env = None
        try:
            env = make_env(self.env, **config["env_kwargs"])
            env.reset(seed=self.seed)
            spec = asdict(env.describe())
        except Exception as exc:
            raise invalid("env_kwargs", f"Environment initialization failed: {exc}") from exc
        finally:
            if env is not None:
                env.close()
        return config, spec


@dataclass(frozen=True)
class EvaluationRequest(Request):
    episodes: int = 10
    seed: int = 10000

    def validate(self):
        positive_integer(self.episodes, "episodes")
        seed_value(self.seed)
        return self
