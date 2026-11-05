from pathlib import Path

import numpy as np
import pytest

from teamwork.algorithms import Algorithm, Policy, make_algorithm, register_algorithm
from teamwork.environments.base import Environment
from teamwork.environments.registry import make_env


class ExamplePolicy(Policy):
    def act(self, observations, deterministic=True):
        return dict.fromkeys(observations, 0)


class ExampleAlgorithm(Algorithm):
    def train(self, config, out):
        return Path(out) / "model.pt"

    def load_policy(self, checkpoint, env, observations):
        return ExamplePolicy()


def test_new_algorithm_uses_registry_without_editing_evaluator(monkeypatch):
    from teamwork.algorithms import registry

    monkeypatch.setattr(registry, "_FACTORIES", dict(registry._FACTORIES))
    register_algorithm("example_for_test_only", ExampleAlgorithm)
    algorithm = make_algorithm("example_for_test_only")
    policy = algorithm.load_policy(None, None, {"a": np.zeros(2)})
    assert policy.act({"a": np.zeros(2)}) == {"a": 0}
    with pytest.raises(ValueError, match="already registered"):
        register_algorithm("ippo", ExampleAlgorithm)


def test_environment_exposes_spaces_and_evaluation_snapshot():
    env = make_env("tag")
    try:
        assert isinstance(env, Environment)
        env.reset(seed=0)
        spec = env.describe()
        assert spec.observation_sizes == (16, 16, 16, 14)
        assert spec.action_counts == (5, 5, 5, 5)
        assert len(env.task_snapshot()["positions"]) == 4
    finally:
        env.close()
