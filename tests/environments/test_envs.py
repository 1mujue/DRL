import importlib.util

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from teamwork.algorithms.baselines import baseline_actions
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate


@pytest.mark.parametrize("name", ["spread", "tag", "lbf", "rware"])
def test_parallel_contract(name):
    if (
        name in {"lbf", "rware"}
        and importlib.util.find_spec("lbforaging" if name == "lbf" else "rware") is None
    ):
        pytest.skip("optional environment not installed")
    env = make_env(name)
    try:
        parallel_api_test(env, num_cycles=60)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["spread", "tag"])
def test_seed_and_local_rules(name):
    env = make_env(name)
    try:
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        for a in first:
            np.testing.assert_array_equal(first[a], second[a])
        actions = baseline_actions(env, second, "rule")
        assert all(env.action_space(a).contains(action) for a, action in actions.items())
    finally:
        env.close()


def test_random_evaluation_reproducible():
    a = evaluate("tag", episodes=2, seed=77)
    b = evaluate("tag", episodes=2, seed=77)
    assert a == b


@pytest.mark.parametrize("name", ["lbf", "rware"])
def test_external_timeout_is_truncation(name):
    if importlib.util.find_spec("lbforaging" if name == "lbf" else "rware") is None:
        pytest.skip("optional environment not installed")
    env = make_env(name)
    try:
        env.reset(seed=3)
        if name == "lbf":
            env.base.unwrapped._max_episode_steps = 1
        else:
            env.base.unwrapped.max_steps = 1
        _, _, terms, truncs, _ = env.step({a: 0 for a in env.agents})
        assert all(truncs.values())
        assert not any(terms.values())
        assert env.agents == []
    finally:
        env.close()
