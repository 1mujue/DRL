import importlib.util

import pytest
import torch

from teamwork.algorithms import available_algorithms, make_algorithm
from teamwork.algorithms.ppo.trainer import create_agents, gae, load_policy, seed_all, train
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate
from teamwork.experiments.runner import run_training


def test_time_limit_bootstrap_and_trace_cut():
    adv, returns = gae(
        torch.tensor([1.0, 100.0]),
        torch.zeros(2),
        torch.tensor([5.0, 8.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        0.9,
        0.95,
    )
    torch.testing.assert_close(adv, torch.tensor([5.5, 100.0]))
    torch.testing.assert_close(returns, adv)


@pytest.mark.parametrize(
    "algorithm,environment", [("ippo", "spread"), ("centralized_ppo", "spread"), ("ippo", "tag")]
)
def test_train_save_load_evaluate(tmp_path, algorithm, environment):
    config = {
        "env": environment,
        "env_kwargs": {"max_cycles": 5},
        "algorithm": algorithm,
        "seed": 3,
        "total_steps": 16,
        "rollout_steps": 8,
        "epochs": 1,
        "batch_size": 8,
    }
    seed_all(3)
    initial_env = make_env(environment, max_cycles=5)
    try:
        initial_obs, _ = initial_env.reset(seed=3)
        initial = create_agents(initial_env, initial_obs, algorithm, "cpu")
    finally:
        initial_env.close()
    checkpoint = run_training(config, tmp_path / "run")
    assert (checkpoint.parent / "training.json").is_file()
    env = make_env(environment, max_cycles=5)
    try:
        obs, _ = env.reset(seed=0)
        models, data = load_policy(checkpoint, env, obs)
        assert data["steps"] == 16
        assert all(torch.isfinite(p).all() for m in models.values() for p in m.parameters())
        assert any(
            not torch.equal(p, q)
            for a in models
            for p, q in zip(models[a].parameters(), initial[a].parameters())
        )
    finally:
        env.close()
    result = evaluate(environment, episodes=1, checkpoint=checkpoint, env_kwargs={"max_cycles": 5})
    assert result["episodes"][0]["steps"] == 5


def test_legacy_centralized_ppo_name_preserves_fixed_opponent_checkpoints(tmp_path):
    canonical = "centralized_ppo"
    assert canonical in available_algorithms()
    assert "mappo" not in available_algorithms()
    assert make_algorithm("mappo").name == canonical
    assert make_algorithm("mappo_ff").__class__ is not type(make_algorithm(canonical))

    config = {
        "algorithm": "mappo",
        "env": "tag",
        "env_kwargs": {"max_cycles": 5},
        "seed": 0,
        "total_steps": 8,
        "epochs": 1,
        "train_team": "predators",
        "train_opponent": "rule",
    }
    checkpoint = run_training(config, tmp_path / "canonical")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert data["config"]["algorithm"] == canonical

    data["config"]["algorithm"] = "mappo"
    legacy = tmp_path / "legacy.pt"
    torch.save(data, legacy)
    options = {"episodes": 1, "team": "predators", "opponent": "rule"}
    legacy_eval = evaluate("tag", checkpoint=legacy, **options)
    current_eval = evaluate("tag", checkpoint=checkpoint, **options)
    assert legacy_eval["algorithm"] == current_eval["algorithm"] == canonical
    assert legacy_eval["summary"] == current_eval["summary"]

    warmed = run_training(
        {**config, "algorithm": canonical, "warm_start_checkpoint": str(legacy)},
        tmp_path / "warm_start",
    )
    assert torch.load(warmed, map_location="cpu", weights_only=True)["config"]["warm_start_sha256"]


@pytest.mark.parametrize("environment", ["lbf", "rware"])
def test_external_training(tmp_path, environment):
    if importlib.util.find_spec("lbforaging" if environment == "lbf" else "rware") is None:
        pytest.skip("optional environment not installed")
    config = {
        "env": environment,
        "algorithm": "ippo",
        "seed": 0,
        "total_steps": 8,
        "rollout_steps": 8,
        "epochs": 1,
        "batch_size": 8,
    }
    checkpoint = train(config, tmp_path / environment)
    assert checkpoint.is_file()
