import csv
import importlib.util
import json

import numpy as np
import pytest
import torch
from torch import nn

from teamwork.algorithms.registry import available_algorithms, make_algorithm
from teamwork.algorithms.vdn.algorithm import VDNPolicy
from teamwork.algorithms.vdn.trainer import (
    Transition,
    _optimize,
    _team_reward,
    create_q_networks,
    train,
)
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate
from teamwork.experiments.runner import run_training, seed_all


def test_vdn_is_registered_and_aggregates_team_rewards():
    assert "vdn" in available_algorithms()
    assert type(make_algorithm("vdn")).__name__ == "VDNAlgorithm"
    assert _team_reward({"a": 2.0, "b": 4.0}, ["a", "b"], "mean") == 3.0
    assert _team_reward({"a": 2.0, "b": 4.0}, ["a", "b"], "sum") == 6.0


def test_vdn_networks_use_local_observation_sizes():
    env = make_env("spread", max_cycles=2)
    try:
        observations, _ = env.reset(seed=0)
        networks = create_q_networks(env, observations)
        for agent, network in networks.items():
            assert network[0].in_features == len(observations[agent])
            assert network[-1].out_features == env.action_space(agent).n
    finally:
        env.close()


def test_vdn_joint_td_target_sums_utilities_and_bootstraps_only_nonterminals():
    def constant_q(values):
        network = nn.Linear(1, len(values))
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(torch.tensor(values))
        return network

    networks = {"a": constant_q([1.0, 9.0]), "b": constant_q([2.0, 8.0])}
    targets = {"a": constant_q([4.0, 0.0]), "b": constant_q([0.0, 5.0])}
    optimizer = torch.optim.SGD(
        [p for network in networks.values() for p in network.parameters()], lr=0.01
    )
    observations = {agent: np.zeros(1, dtype=np.float32) for agent in networks}
    batch = [
        Transition(observations, {"a": 0, "b": 0}, 2.0, observations, terminated)
        for terminated in (False, True)
    ]
    loss = _optimize(networks, targets, optimizer, batch, list(networks), 0.5, "cpu")
    # Q_tot=1+2=3; max target Q_tot=4+5=9. Targets are 6.5 and 2.
    # Smooth L1 errors 3.5 and -1 give losses 3 and 0.5, averaged to 1.75.
    assert loss == pytest.approx(1.75)
    assert all(
        parameter.grad is None for model in targets.values() for parameter in model.parameters()
    )


def test_vdn_learns_a_small_additive_cooperative_payoff():
    seed_all(0)
    networks = {agent: nn.Linear(1, 2) for agent in ("a", "b")}
    targets = {agent: nn.Linear(1, 2) for agent in networks}
    observations = {agent: np.zeros(1, dtype=np.float32) for agent in networks}
    batch = [
        Transition(observations, {"a": a, "b": b}, float(a + 2 * (1 - b)), observations, True)
        for a in range(2)
        for b in range(2)
    ]
    optimizer = torch.optim.Adam(
        [p for network in networks.values() for p in network.parameters()], lr=0.05
    )
    for _ in range(200):
        loss = _optimize(networks, targets, optimizer, batch, list(networks), 0.99, "cpu")
    assert loss < 0.001
    assert VDNPolicy(networks).act(observations) == {"a": 1, "b": 0}


@pytest.mark.parametrize("task_terminal", [False, True])
def test_vdn_collection_keeps_final_observation_and_distinguishes_timeouts(
    monkeypatch, tmp_path, task_terminal
):
    from teamwork.algorithms.vdn import trainer

    env = make_env("spread", max_cycles=2)
    step = env.step
    final_observations, transitions = [], []

    def end_episode(actions):
        obs, rewards, terminated, truncated, infos = step(actions)
        if not env.agents:
            final_observations.append({a: v.copy() for a, v in obs.items()})
            if task_terminal:
                terminated = dict.fromkeys(env.possible_agents, True)
                truncated = dict.fromkeys(env.possible_agents, False)
        return obs, rewards, terminated, truncated, infos

    add = trainer.ReplayBuffer.add

    def record(buffer, transition):
        transitions.append(transition)
        add(buffer, transition)

    monkeypatch.setattr(env, "step", end_episode)
    monkeypatch.setattr(trainer, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(trainer.ReplayBuffer, "add", record)
    train(
        {"env": "spread", "seed": 0, "total_steps": 4, "batch_size": 2, "warmup_steps": 2},
        tmp_path / "boundaries",
    )
    assert [transition.terminated for transition in transitions] == [
        False,
        task_terminal,
        False,
        task_terminal,
    ]
    for transition, final in zip(transitions[1::2], final_observations):
        for agent in final:
            np.testing.assert_array_equal(transition.next_observations[agent], final[agent])


def test_vdn_trains_saves_loads_and_evaluates(tmp_path):
    config = {
        "env": "spread",
        "algorithm": "vdn",
        "seed": 3,
        "total_steps": 16,
        "env_kwargs": {"max_cycles": 5},
        "batch_size": 4,
        "replay_capacity": 4,
        "warmup_steps": 8,  # Pretraining collection may exceed replay capacity.
        "hidden_size": 32,
        "target_update_frequency": 2,
        "epsilon_decay_steps": 8,
    }
    seed_all(config["seed"])
    env = make_env("spread", max_cycles=5)
    try:
        observations, _ = env.reset(seed=config["seed"])
        initial = create_q_networks(env, observations, hidden_size=32)
    finally:
        env.close()
    checkpoint = run_training(config, tmp_path / "vdn")
    assert checkpoint.is_file()
    assert (checkpoint.parent / "config.json").is_file()
    assert (checkpoint.parent / "episodes.csv").is_file()
    assert (checkpoint.parent / "training.json").is_file()
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert data["steps"] == 16
    assert data["loss"] is not None
    assert data["updates"] == 9
    assert all(torch.isfinite(p).all() for model in data["agents"].values() for p in model.values())
    assert all(
        any(
            not torch.equal(parameter, data["agents"][agent][key])
            for key, parameter in model.state_dict().items()
        )
        for agent, model in initial.items()
    )
    with (checkpoint.parent / "updates.csv").open(encoding="utf-8") as stream:
        updates = list(csv.DictReader(stream))
    assert len(updates) == 9
    assert int(updates[0]["steps"]) == 8
    summary = json.loads((checkpoint.parent / "vdn_training.json").read_text(encoding="utf-8"))
    assert summary["updates"] == 9
    assert summary["team_reward"] == "mean"
    env = make_env("spread", max_cycles=5)
    try:
        observations, _ = env.reset(seed=0)
        policy = make_algorithm("vdn").load_policy(checkpoint, env, observations)
        # Agent a's action is unaffected by another agent's observation.
        agent = env.possible_agents[0]
        assert policy.act({agent: observations[agent]})[agent] == policy.act(observations)[agent]
        assert all(env.action_space(a).contains(v) for a, v in policy.act(observations).items())
    finally:
        env.close()
    result = evaluate("spread", episodes=1, checkpoint=checkpoint, env_kwargs={"max_cycles": 5})
    assert result["algorithm"] == "vdn"
    assert result["episodes"][0]["steps"] == 5
    duplicate = run_training(config, tmp_path / "same_seed")
    duplicate_data = torch.load(duplicate, map_location="cpu", weights_only=True)
    for agent, model in data["agents"].items():
        for key, tensor in model.items():
            torch.testing.assert_close(tensor, duplicate_data["agents"][agent][key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        run_training(config, checkpoint.parent)


def test_vdn_rejects_adversarial_tag(tmp_path):
    with pytest.raises(ValueError, match="cooperative learner"):
        train(
            {"env": "tag", "algorithm": "vdn", "seed": 0, "total_steps": 1},
            tmp_path / "tag",
        )


@pytest.mark.parametrize(
    "override",
    [
        {"total_steps": 0},
        {"batch_size": 0},
        {"replay_capacity": 2},
        {"epsilon_start": 1.1},
        {"epsilon_end": -0.1},
        {"update_frequency": 0},
        {"gamma": float("nan")},
        {"team_reward": "individual"},
    ],
)
def test_vdn_rejects_invalid_configs_without_creating_an_experiment(tmp_path, override):
    config = {"env": "spread", "seed": 0, "total_steps": 8, "batch_size": 4, **override}
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        train(config, out)
    assert not out.exists()


@pytest.mark.parametrize("environment,package", [("lbf", "lbforaging"), ("rware", "rware")])
def test_vdn_external_cooperative_environments(tmp_path, environment, package):
    if importlib.util.find_spec(package) is None:
        pytest.skip("optional environment not installed")
    checkpoint = run_training(
        {
            "env": environment,
            "algorithm": "vdn",
            "seed": 0,
            "total_steps": 8,
            "batch_size": 4,
            "warmup_steps": 4,
        },
        tmp_path / environment,
    )
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert data["updates"] == 5
    result = evaluate(environment, episodes=1, checkpoint=checkpoint)
    assert result["algorithm"] == "vdn"
