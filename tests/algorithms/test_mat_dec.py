import csv
import importlib.util
import json

import numpy as np
import pytest
import torch

from teamwork.algorithms import make_algorithm
from teamwork.algorithms.mat_dec.algorithm import MATDecPolicy
from teamwork.algorithms.mat_dec.model import MATDecModel, create_model
from teamwork.algorithms.mat_dec.trainer import clipped_policy_loss, compute_gae, train
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate
from teamwork.experiments.runner import run_training, seed_all


def _spec():
    return {"agents": ["a", "b"], "observation_sizes": [3, 3], "action_counts": [2, 2]}


def test_mat_dec_executes_locally_and_never_calls_the_critic(monkeypatch):
    seed_all(0)
    model = MATDecModel(_spec(), embedding_dim=16, num_heads=2)
    policy = MATDecPolicy(model)
    observations = {"a": np.array([0.1, 0.5, -0.2]), "b": np.array([0.0, -1.0, 2.0])}

    def forbidden(*args, **kwargs):
        raise AssertionError("The execution policy must not call the centralized critic")

    monkeypatch.setattr(model.critic, "forward", forbidden)
    actions = policy.act(observations)
    assert policy.act({"a": observations["a"]}) == {"a": actions["a"]}
    observations["b"] *= -100
    changed_actions = policy.act(observations)
    assert changed_actions["a"] == actions["a"]
    assert policy.act(dict(reversed(list(observations.items())))) == changed_actions
    assert set(policy.act(observations, deterministic=False)) == {"a", "b"}
    assert all(0 <= value < 2 for value in actions.values())


def test_mat_dec_transformer_critic_uses_joint_observations_without_actor_gradients():
    seed_all(0)
    model = MATDecModel(_spec(), embedding_dim=16, num_heads=2).eval()
    observations = {"a": torch.tensor([[0.1, 0.5, -0.2]]), "b": torch.tensor([[0.0, -1.0, 2.0]])}
    changed = {**observations, "b": torch.tensor([[5.0, 2.0, -3.0]])}
    original_values = model.values(observations)
    changed_values = model.values(changed)
    assert abs(float((original_values[0, 0] - changed_values[0, 0]).detach())) > 1e-8
    torch.testing.assert_close(
        model.local_logits("a", observations["a"]), model.local_logits("a", changed["a"])
    )
    original_values.sum().backward()
    assert all(parameter.grad is None for actor in model.actors for parameter in actor.parameters())
    assert any(parameter.grad is not None for parameter in model.critic.parameters())


@pytest.mark.parametrize("share_actor", [False, True])
def test_mat_dec_likelihood_is_unchanged_between_rollout_and_ppo(share_actor):
    seed_all(0)
    model = MATDecModel(_spec(), embedding_dim=16, num_heads=2, share_actor=share_actor)
    observations = {agent: torch.randn(5, 3) for agent in model.names}
    actions = torch.randint(2, (5, 2))
    model.eval()
    with torch.no_grad():
        old_log_probs, _ = model.action_statistics(observations, actions)
    model.train()
    current_log_probs, _ = model.action_statistics(observations, actions)
    torch.testing.assert_close((current_log_probs - old_log_probs).exp(), torch.ones(5, 2))


def test_mat_dec_supports_heterogeneous_spaces_with_separate_actors():
    spec = {"agents": ["a", "b"], "observation_sizes": [3, 5], "action_counts": [2, 4]}
    model = MATDecModel(spec, embedding_dim=16, num_heads=2)
    observations = {"a": torch.randn(4, 3), "b": torch.randn(4, 5)}
    assert model.values(observations).shape == (4, 2)
    assert model.local_logits("b", observations["b"]).shape == (4, 4)
    with pytest.raises(ValueError, match="identical"):
        MATDecModel(spec, share_actor=True)


def test_mat_dec_gae_bootstraps_time_limits_but_cuts_reset_traces():
    advantages, returns = compute_gae(
        torch.tensor([[1.0, 2.0], [100.0, 200.0]]),
        torch.zeros(2, 2),
        torch.tensor([[5.0, 6.0], [8.0, 9.0]]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        0.9,
        0.95,
    )
    torch.testing.assert_close(advantages, torch.tensor([[5.5, 7.4], [100.0, 200.0]]))
    torch.testing.assert_close(returns, advantages)


def test_mat_dec_clips_each_agents_ppo_ratio():
    probabilities = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    loss = clipped_policy_loss(
        probabilities.log(),
        torch.full((2, 2), 0.5).log(),
        torch.tensor([[1.0, -1.0], [2.0, -2.0]]),
        0.2,
    )
    # Clipped signed surrogates are [1.2, -0.8, 1.6, -2.4].
    assert float(loss) == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("task_terminal", [False, True])
def test_mat_dec_collection_distinguishes_task_completion_from_timeout(
    monkeypatch, tmp_path, task_terminal
):
    from teamwork.algorithms.mat_dec import trainer

    env = make_env("spread", max_cycles=2)
    step = env.step
    original_optimize = trainer._optimize
    captured = []

    def end_episode(actions):
        obs, rewards, terminated, truncated, infos = step(actions)
        if not env.agents and task_terminal:
            terminated = dict.fromkeys(env.possible_agents, True)
            truncated = dict.fromkeys(env.possible_agents, False)
        return obs, rewards, terminated, truncated, infos

    def optimize(model, optimizer, observations, batch, config):
        captured.append(batch)
        yield from original_optimize(model, optimizer, observations, batch, config)

    monkeypatch.setattr(env, "step", end_episode)
    monkeypatch.setattr(trainer, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(trainer, "_optimize", optimize)
    train(
        {
            "env": "spread",
            "seed": 0,
            "total_steps": 4,
            "rollout_steps": 4,
            "epochs": 1,
            "batch_size": 4,
            "embedding_dim": 16,
        },
        tmp_path / "boundaries",
    )
    assert len(captured) == 1
    batch = captured[0]
    torch.testing.assert_close(batch["episode_end"], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(
        batch["terminated"], torch.tensor([0.0, float(task_terminal), 0.0, float(task_terminal)])
    )
    if task_terminal:
        torch.testing.assert_close(batch["next_values"][[1, 3]], torch.zeros(2, 3))
    else:
        assert batch["next_values"][[1, 3]].abs().max() > 1e-8
    torch.testing.assert_close(batch["rewards"][:, 0], batch["rewards"][:, 1])


def test_mat_dec_trains_updates_saves_loads_and_reproduces(tmp_path, monkeypatch):
    from teamwork.algorithms.mat_dec import trainer

    config = {
        "env": "spread",
        "algorithm": "mat_dec",
        "seed": 3,
        "total_steps": 19,
        "env_kwargs": {"max_cycles": 5},
        "embedding_dim": 16,
        "num_heads": 2,
        "rollout_steps": 8,
        "epochs": 2,
        "batch_size": 4,
    }
    seed_all(config["seed"])
    env = make_env("spread", max_cycles=5)
    try:
        observations, _ = env.reset(seed=3)
        initial = create_model(env, observations, config)
    finally:
        env.close()

    def training_env(*args, **kwargs):
        environment = make_env(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("MAT-Dec should not read hidden simulator state or snapshots")

        monkeypatch.setattr(environment, "state", forbidden)
        monkeypatch.setattr(environment, "task_snapshot", forbidden)
        return environment

    monkeypatch.setattr(trainer, "make_env", training_env)
    checkpoint = run_training(config, tmp_path / "mat_dec")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert data["steps"] == 19
    assert data["rollouts"] == 3
    assert data["updates"] == 10
    assert data["config"]["team_reward"] == "mean"
    assert all(torch.isfinite(parameter).all() for parameter in data["model"].values())
    for prefix in ["actors.0.", "actors.1.", "actors.2.", "critic."]:
        assert any(
            not torch.equal(value, data["model"][key])
            for key, value in initial.state_dict().items()
            if key.startswith(prefix)
        )
    for name in ("config.json", "provenance.json", "episodes.csv", "training.json"):
        assert (checkpoint.parent / name).is_file()
    with (checkpoint.parent / "updates.csv").open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 10
    summary = json.loads((checkpoint.parent / "mat_dec_training.json").read_text(encoding="utf-8"))
    assert summary["updates"] == 10
    result = evaluate("spread", checkpoint=checkpoint, episodes=2, env_kwargs={"max_cycles": 5})
    assert result["algorithm"] == "mat_dec"
    assert all(episode["steps"] == 5 for episode in result["episodes"])

    env = make_env("spread", max_cycles=5)
    try:
        observations, _ = env.reset(seed=0)
        policy = make_algorithm("mat_dec").load_policy(checkpoint, env, observations)
        for key, value in policy.model.state_dict().items():
            torch.testing.assert_close(value, data["model"][key], atol=0, rtol=0)
        assert all(env.action_space(a).contains(v) for a, v in policy.act(observations).items())
    finally:
        env.close()
    duplicate = run_training(config, tmp_path / "same_seed")
    duplicate_data = torch.load(duplicate, map_location="cpu", weights_only=True)
    for key, value in data["model"].items():
        torch.testing.assert_close(value, duplicate_data["model"][key], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        run_training(config, checkpoint.parent)


@pytest.mark.parametrize(
    "override",
    [
        {"env": "tag"},
        {"total_steps": 0},
        {"num_heads": 3},
        {"rollout_steps": 0},
        {"epochs": 0},
        {"gamma": float("nan")},
        {"share_actor": "false"},
        {"team_reward": "individual"},
        {"clip": 1.0},
        {"lr": True},
        {"lr": "0.0003"},
        {"seed": 2**32},
        {"device": "meta"},
        {"device": "cuda:9999"},
        {"env_kwargs": []},
        {"env_kwargs": {1: 2}},
        {"metadata": {"invalid": object()}},
        {"metadata": {"invalid": float("nan")}},
        {"train_team": "predators"},
        {"train_opponent": "rule"},
    ],
)
def test_mat_dec_rejects_invalid_configs_without_creating_experiment(tmp_path, override):
    out = tmp_path / "invalid"
    with pytest.raises((ValueError, TypeError)):
        train({"env": "spread", "seed": 0, "total_steps": 8, **override}, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "invalid",
    [
        {"b": [float("nan"), 0.0, 0.0]},
        {"b": [float("inf"), 0.0, 0.0]},
        {"b": [[0.0, 0.0, 0.0]]},
        {"b": [0.0]},
        {"b": ["invalid", 0.0, 0.0]},
        {"unknown": [0.0, 0.0, 0.0]},
    ],
)
def test_mat_dec_validates_all_local_inputs_before_sampling(invalid):
    policy = MATDecPolicy(MATDecModel(_spec(), embedding_dim=8))
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="observation|agent"):
        policy.act({"a": [0.0, 0.0, 0.0], **invalid}, deterministic=False)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("environment,package", [("lbf", "lbforaging"), ("rware", "rware")])
def test_mat_dec_external_cooperative_environments(tmp_path, environment, package):
    if importlib.util.find_spec(package) is None:
        pytest.skip("optional environment not installed")
    checkpoint = run_training(
        {
            "env": environment,
            "algorithm": "mat_dec",
            "seed": 0,
            "total_steps": 8,
            "rollout_steps": 4,
            "epochs": 1,
            "batch_size": 4,
            "embedding_dim": 16,
        },
        tmp_path / environment,
    )
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert data["updates"] == 2
    assert evaluate(environment, checkpoint=checkpoint, episodes=1)["algorithm"] == "mat_dec"
