"""TwoArmLift's real simulator integration tests are optional in the base install."""

import numpy as np
import pytest
import torch

from teamwork.environments.base import ContinuousEnvironmentSpec
from teamwork.environments.registry import make_env
from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import evaluate, save_result
from teamwork.evaluation.metrics import LIFT_HOLD_STEPS, make_metrics
from teamwork.experiments.runner import run_training


def test_dual_lift_requires_a_continuous_hold_by_both_arms():
    suite = make_metrics("two_arm_lift")
    goal = {"lift_height_m": 0.11, "tilt_degrees": 15.0, "grasping": (True, True)}
    suite.reset(goal)
    returns = {"arm_0": 0.0, "arm_1": 0.0}
    for _ in range(LIFT_HOLD_STEPS - 1):
        suite.observe(goal)
    assert not suite.finish(returns, tuple(returns))["stable_dual_lift"]

    # Height alone, excess tilt, and one-arm grasp cannot complete the hold.
    suite.observe({**goal, "lift_height_m": 0.10})
    suite.observe({**goal, "grasping": (True, False)})
    suite.observe({**goal, "tilt_degrees": 15.01})
    for _ in range(LIFT_HOLD_STEPS):
        suite.observe(goal)
    result = suite.finish(returns, tuple(returns))
    assert result["stable_dual_lift"] == 1.0
    assert result["first_stable_lift_step"] == 2 * LIFT_HOLD_STEPS + 2


@pytest.mark.parametrize(
    "algorithm,name,kind",
    [("qplex", "two_arm_lift", "discrete"), ("ippo_continuous", "spread", "continuous")],
)
def test_incompatible_action_kind_fails_before_creating_run(tmp_path, algorithm, name, kind):
    out = tmp_path / "training"
    with pytest.raises(ValueError, match=f"requires {kind} actions"):
        run_training({"algorithm": algorithm, "env": name, "total_steps": 1, "seed": 0}, out)
    assert not out.exists()


def test_two_arm_adapter_seed_action_limits_and_time_limit():
    pytest.importorskip("robosuite")
    env = make_env("two_arm_lift", horizon=3)
    try:
        first, _ = env.reset(seed=7)
        repeat, _ = env.reset(seed=7)
        another, _ = env.reset(seed=8)
        assert isinstance(env.describe(), ContinuousEnvironmentSpec)
        for arm, observation in first.items():
            np.testing.assert_array_equal(repeat[arm], observation)
            assert not np.array_equal(another[arm], observation)
            assert env.observation_space(arm).contains(observation)
        actions = {
            arm: np.zeros(env.action_space(arm).shape, dtype=np.float32) for arm in env.agents
        }
        with pytest.raises(ValueError, match="one action for each arm"):
            env.step({"arm_0": actions["arm_0"]})
        with pytest.raises(ValueError, match="Invalid continuous action"):
            env.step({**actions, "arm_0": np.full(actions["arm_0"].shape, np.nan)})
        for step in range(3):
            obs, rewards, terminated, truncated, _ = env.step(actions)
            assert obs.keys() == rewards.keys() == actions.keys()
            assert rewards["arm_0"] == rewards["arm_1"]
            assert not any(terminated.values())
            assert all(truncated.values()) == (step == 2)
        assert not env.agents
        assert env.task_snapshot()["grasping"] == (False, False)
    finally:
        env.close()


def test_continuous_ppo_checkpoint_evaluates_under_same_benchmark(tmp_path):
    pytest.importorskip("robosuite")
    paths = []
    for algorithm, critic_size in [("ippo_continuous", 1), ("centralized_ppo_continuous", 2)]:
        config = {
            "algorithm": algorithm,
            "env": "two_arm_lift",
            "env_kwargs": {"horizon": 4},
            "total_steps": 8,
            "rollout_steps": 4,
            "batch_size": 4,
            "epochs": 1,
            "seed": 3,
        }
        checkpoint = run_training(config, tmp_path / algorithm)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = data["agents"]["arm_0"]
        assert data["steps"] == 8
        assert "log_std" in weights
        assert (
            weights["critic.0.weight"].shape[1] == critic_size * weights["actor.0.weight"].shape[1]
        )
        result = evaluate("two_arm_lift", checkpoint=checkpoint, episodes=1, seed=11)
        assert result["episodes"][0]["steps"] == 4
        path = tmp_path / f"{algorithm}.json"
        save_result(result, path)
        paths.append(path)

    report = compare_results(paths, "stable_dual_lift")
    assert set(report["algorithms"]) == {"ippo_continuous", "centralized_ppo_continuous"}


def test_scripted_positive_control_requires_both_working_arms():
    pytest.importorskip("robosuite")
    cooperative = evaluate("two_arm_lift", policy="scripted_lift", episodes=1, seed=10000)
    one_arm = evaluate("two_arm_lift", policy="scripted_single_arm", episodes=1, seed=10000)
    assert cooperative["benchmark_id"] == one_arm["benchmark_id"]
    assert cooperative["summary"]["stable_dual_lift"]["mean"] == 1.0
    assert one_arm["summary"]["stable_dual_lift"]["mean"] == 0.0


def test_mass_override_preserves_spaces_and_seeded_initial_pose():
    pytest.importorskip("robosuite")
    nominal = make_env("two_arm_lift", pose_feedback=True)
    heavy = make_env("two_arm_lift", pose_feedback=True, pot_mass_scale=2.5)
    try:
        first, _ = nominal.reset(seed=45)
        changed, _ = heavy.reset(seed=45)
        assert heavy.describe() == nominal.describe()
        for name in first:
            np.testing.assert_array_equal(first[name], changed[name])
        base = nominal._env
        model = heavy._env
        assert model.sim.model.body_mass[model.pot_body_id] == pytest.approx(
            2.5 * base.sim.model.body_mass[base.pot_body_id]
        )
        np.testing.assert_allclose(
            model.sim.model.body_inertia[model.pot_body_id],
            2.5 * base.sim.model.body_inertia[base.pot_body_id],
        )
        heavy.reset(seed=45)
        assert model.sim.model.body_mass[model.pot_body_id] == pytest.approx(
            2.5 * base.sim.model.body_mass[base.pot_body_id]
        )
    finally:
        nominal.close()
        heavy.close()


def test_randomized_mass_uses_independent_seeded_stream_and_absolute_scaling():
    pytest.importorskip("robosuite")
    nominal = make_env("two_arm_lift", pose_feedback=True)
    randomized = make_env("two_arm_lift", pose_feedback=True, pot_mass_range=[3.0, 6.5])
    draws = []
    try:
        for seed in (1, 2, 1):
            reference, _ = nominal.reset(seed=seed)
            observations, infos = randomized.reset(seed=seed)
            for arm in reference:
                np.testing.assert_array_equal(reference[arm], observations[arm])
            scale = infos["arm_0"]["pot_mass_scale"]
            assert scale == infos["arm_1"]["pot_mass_scale"]
            assert 3.0 <= scale <= 6.5
            draws.append(scale)
            sim_model = randomized._env.sim.model
            root = randomized._env.pot_body_id
            nominal_mass = nominal._env.sim.model.body_mass[nominal._env.pot_body_id]
            assert sim_model.body_mass[root] == pytest.approx(nominal_mass * scale)
        assert draws[0] == draws[2]
        assert draws[0] != draws[1]
    finally:
        nominal.close()
        randomized.close()


def test_fixed_evaluation_overrides_randomized_checkpoint_mass(tmp_path):
    pytest.importorskip("robosuite")
    config = {
        "algorithm": "hasac_continuous",
        "env": "two_arm_lift",
        "env_kwargs": {"horizon": 3, "pot_mass_range": [3.0, 6.5]},
        "total_steps": 6,
        "batch_size": 3,
        "update_every": 3,
        "seed": 0,
    }
    checkpoint = run_training(config, tmp_path / "trained")
    result = evaluate("two_arm_lift", episodes=1, checkpoint=checkpoint, pot_mass_scale=7.0)
    assert result["training_env_kwargs"]["pot_mass_range"] == [3.0, 6.5]
    assert "pot_mass_range" not in result["benchmark"]["env_kwargs"]
    assert result["benchmark"]["env_kwargs"]["pot_mass_scale"] == 7.0
