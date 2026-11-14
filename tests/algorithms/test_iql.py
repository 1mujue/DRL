import csv
import importlib.util
import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from teamwork.algorithms.iql.trainer import (
    QNetwork,
    ReplayBuffer,
    create_agents,
    epsilon_at,
    load_policy,
    resolve_config,
    td_targets,
    update,
)
from teamwork.algorithms.registry import make_algorithm
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate
from teamwork.experiments.runner import run_training, seed_all


def small_config(environment="spread", **overrides):
    return {
        "algorithm": "iql",
        "env": environment,
        "seed": 3,
        "total_steps": 24,
        "batch_size": 4,
        "buffer_size": 12,
        "hidden_size": 16,
        "learning_starts": 4,
        "train_interval": 2,
        "target_update_interval": 4,
        **overrides,
    }


def test_td_target_bootstraps_time_limits_but_not_termination():
    targets = td_targets(
        torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([5.0, 5.0]), 0.9
    )
    torch.testing.assert_close(targets, torch.tensor([5.5, 1.0]))


def test_replay_retains_final_observation_and_evicts_old_transitions():
    buffer = ReplayBuffer(2, 1)
    observation = np.array([0.0], dtype=np.float32)
    final = np.array([9.0], dtype=np.float32)
    buffer.add(observation, 0, 1.0, final, False)
    observation[:] = -1
    final[:] = -1
    assert buffer.next_observations[0, 0] == 9
    for i in (1, 2):
        buffer.add([i], 0, i, [i + 1], i == 2)
    observations, actions, rewards, next_obs, terminated = buffer.sample(2, "cpu")
    assert set(observations.flatten().tolist()) == {1.0, 2.0}
    assert actions.dtype == torch.int64
    torch.testing.assert_close(next_obs, observations + 1)
    torch.testing.assert_close(terminated, (rewards == 2).float())


def test_epsilon_anneals_and_stays_at_floor():
    config = resolve_config(small_config(epsilon_decay_steps=10))
    assert epsilon_at(0, config) == 1.0
    assert epsilon_at(5, config) == pytest.approx(0.525)
    assert epsilon_at(10, config) == pytest.approx(0.05)
    assert epsilon_at(100, config) == pytest.approx(0.05)


def test_dqn_learns_rewarding_action_and_keeps_target_frozen():
    seed_all(5)
    model = QNetwork(1, 2, 8)
    target = deepcopy(model)
    initial_target = deepcopy(target.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    actions = torch.tensor([0, 1] * 8)
    observations = torch.zeros(16, 1)
    batch = (observations, actions, actions.float(), observations, torch.ones(16))
    config = resolve_config(small_config())
    for _ in range(40):
        update(model, target, optimizer, batch, config)
    with torch.no_grad():
        values = model(torch.zeros(1))
    assert values.argmax().item() == 1
    torch.testing.assert_close(values, torch.tensor([0.0, 1.0]), atol=0.15, rtol=0)
    for name, parameter in target.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, initial_target[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "override",
    [
        {"batch_size": 13},
        {"train_interval": 0},
        {"learning_starts": -1},
        {"gamma": 1.1},
        {"epsilon_end": 1.1},
        {"lr": float("nan")},
        {"algorithm": "ippo"},
    ],
)
def test_invalid_configuration_fails_before_training(override):
    with pytest.raises(ValueError):
        resolve_config(small_config(**override))


@pytest.mark.parametrize("environment", ["spread", "tag", "lbf", "rware"])
def test_iql_updates_saves_loads_and_evaluates_all_environments(tmp_path, environment):
    if environment in {"lbf", "rware"} and importlib.util.find_spec(
        "lbforaging" if environment == "lbf" else "rware"
    ) is None:
        pytest.skip("optional environment not installed")
    config = resolve_config(small_config(environment))
    seed_all(config["seed"])
    env = make_env(environment, **config["env_kwargs"])
    try:
        obs, _ = env.reset(seed=config["seed"])
        initial = create_agents(env, obs, config)
    finally:
        env.close()
    checkpoint = run_training(config, tmp_path / environment)
    env = make_env(environment, **config["env_kwargs"])
    try:
        obs, _ = env.reset(seed=0)
        models, data = load_policy(checkpoint, env, obs)
        assert data["steps"] == config["total_steps"]
        assert all(n > 0 for n in data["updates"].values())
        for a, model in models.items():
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert any(
                not torch.equal(p, q) for p, q in zip(model.parameters(), initial[a].parameters())
            )
        policy = make_algorithm("iql").load_policy(checkpoint, env, obs)
        for deterministic in (True, False):
            assert all(
                env.action_space(a).contains(action)
                for a, action in policy.act(obs, deterministic).items()
            )
        with pytest.raises(ValueError, match="algorithm"):
            make_algorithm("ippo").load_policy(checkpoint, env, obs)
    finally:
        env.close()
    result = evaluate(environment, episodes=1, checkpoint=checkpoint)
    assert result["algorithm"] == "iql"
    assert result["training_steps"] == config["total_steps"]
    assert (checkpoint.parent / "provenance.json").is_file()
    assert json.loads((checkpoint.parent / "training.json").read_text())["algorithm"] == "iql"
    with (checkpoint.parent / "updates.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == sum(data["updates"].values())
    with pytest.raises(FileExistsError):
        run_training(config, checkpoint.parent)


def test_same_seed_reproduces_parameters(tmp_path):
    checkpoints = [run_training(small_config(), tmp_path / str(i)) for i in range(2)]
    first, second = [torch.load(p, weights_only=True)["agents"] for p in checkpoints]
    for agent in first:
        for key in first[agent]:
            torch.testing.assert_close(first[agent][key], second[agent][key], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["ippo", "iql"])
@pytest.mark.parametrize("team,opponent", [("predators", "rule"), ("prey", "random")])
def test_fixed_opponent_training_and_evaluation(tmp_path, algorithm, team, opponent):
    config = small_config(
        "tag",
        algorithm=algorithm,
        env_kwargs={"max_cycles": 4},
        train_team=team,
        train_opponent=opponent,
        rollout_steps=8,
        epochs=1,
    )
    checkpoint = run_training(config, tmp_path / "run")
    data = torch.load(checkpoint, weights_only=True)
    assert len(data["agents"]) == (3 if team == "predators" else 1)
    assert all(a.startswith("adversary_") == (team == "predators") for a in data["agents"])
    result = evaluate("tag", checkpoint=checkpoint, episodes=2, team=team, opponent=opponent)
    assert result["comparison_allowed"]
    assert result["training_setup"] == {"team": team, "opponent": opponent}
    assert all(len(row["returns"]) == 4 for row in result["episodes"])
    with pytest.raises(ValueError, match="team and opponent"):
        evaluate("tag", checkpoint=checkpoint, episodes=1)


@pytest.mark.parametrize("algorithm", ["ippo", "iql"])
@pytest.mark.parametrize(
    "environment,team,opponent",
    [
        ("spread", "predators", "rule"),
        ("tag", "predators", None),
        ("tag", "all", "rule"),
    ],
)
def test_invalid_training_team_is_rejected(tmp_path, algorithm, environment, team, opponent):
    with pytest.raises(ValueError, match="Fixed-opponent training"):
        run_training(
            small_config(
                environment, algorithm=algorithm, train_team=team, train_opponent=opponent
            ),
            tmp_path / "run",
        )
