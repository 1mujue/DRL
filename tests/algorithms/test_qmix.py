"""QMIX 的单调性、回放时序、TD 数值和公共接入合同；不证明任务已收敛。"""

import csv
import importlib.util
import itertools
import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from teamwork.algorithms import make_algorithm
from teamwork.algorithms.qmix.algorithm import QMIXPolicy
from teamwork.algorithms.qmix.learner import QMIXLearner, masked_td_loss, td_targets, unroll
from teamwork.algorithms.qmix.networks import LocalQController, QMixer, pack_observations
from teamwork.algorithms.qmix.replay import EpisodeReplay
from teamwork.algorithms.qmix.trainer import aggregate_reward, epsilon_at, resolve_config
from teamwork.environments.base import EnvironmentSpec
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate
from teamwork.evaluation.metrics import resolved_env_kwargs
from teamwork.experiments.runner import run_training


def config_for(environment="spread", **overrides):
    return resolve_config({
        "algorithm": "qmix", "env": environment,
        "env_kwargs": resolved_env_kwargs(environment), "seed": 11, "total_steps": 12,
        "hidden_size": 8, "mixing_size": 4, "hypernet_size": 8,
        "batch_size": 2, "learning_starts_episodes": 2, "replay_capacity": 4,
        "target_update_interval": 2, "checkpoint_interval": 5, **overrides,
    })


def episode_for(spec, length, *, terminal=False, truncated=False):
    observations = np.full(
        (length + 1, len(spec.agents), max(spec.observation_sizes)), 0.25, np.float32
    )
    states = np.repeat(np.arange(length + 1, dtype=np.float32)[:, None], spec.state_size, axis=1)
    flags = np.zeros(length, np.float32)
    return {
        "observations": observations, "states": states,
        "actions": np.zeros((length, len(spec.agents)), np.int64),
        "rewards": np.ones(length, np.float32),
        "terminated": np.array([*flags[:-1], float(terminal)], np.float32),
        "truncated": np.array([*flags[:-1], float(truncated)], np.float32),
    }


def test_mixer_is_monotonic_and_local_greedy_maximizes_joint_value():
    torch.manual_seed(3)
    mixer = QMixer(2, 4, 8, 16)
    local = torch.randn(4, 3, 2, requires_grad=True)
    states = torch.randn(4, 3, 4)
    total = mixer(local, states)
    total.sum().backward()
    assert torch.isfinite(local.grad).all() and (local.grad >= -1e-7).all()
    assert (local.grad > 0).any()
    with torch.no_grad():
        assert (mixer(local + 0.1, states) >= total - 1e-6).all()
        # 枚举联合动作，验证分散贪心与集中联合最大值的一致性。
        values = [torch.tensor([-2.0, 3.0, 1.0]), torch.tensor([4.0, -1.0])]
        combinations = list(itertools.product(range(3), range(2)))
        joint = torch.stack([torch.stack([values[0][a], values[1][b]]) for a, b in combinations])
        state = torch.tensor([1.0, -1.0, 0.5, 0.0])
        totals = mixer(joint, state.expand(len(joint), -1))
        greedy = torch.stack([value.max() for value in values])
        torch.testing.assert_close(mixer(greedy, state), totals.max())


@pytest.mark.parametrize("use_rnn", [True, False])
def test_local_controller_keeps_agents_separate_and_masks_padded_actions(use_rnn):
    spec = EnvironmentSpec(("a", "b"), (2, 3), (2, 3), 4)
    config = config_for(use_rnn=use_rnn)
    controller = LocalQController(spec, config)
    observations = {"a": np.array([1, -1], np.float32), "b": np.array([2, 3, 4], np.float32)}
    packed = pack_observations(observations, spec)
    assert packed.shape == (2, 3) and packed[0, 2] == 0
    previous = torch.zeros(1, 2, 3)
    first, _ = controller(torch.tensor(packed)[None], previous, controller.initial_hidden(1))
    packed[1] += 100
    changed, _ = controller(torch.tensor(packed)[None], previous, controller.initial_hidden(1))
    torch.testing.assert_close(first[:, 0], changed[:, 0])
    impossible = torch.tensor([[[1.0, 2.0, 1000.0], [1.0, 2.0, 3.0]]])
    assert controller.greedy_actions(impossible).tolist() == [[1, 2]]
    policy = QMIXPolicy(controller)
    rng = np.random.default_rng(4)
    for _ in range(20):
        actions = policy.sample_actions(observations, epsilon=1.0, rng=rng)
        assert 0 <= actions["a"] < 2 and 0 <= actions["b"] < 3


def test_recurrent_policy_and_replay_unroll_have_the_same_action_history():
    torch.manual_seed(5)
    spec = EnvironmentSpec(("a", "b"), (2, 2), (3, 3), 4)
    controller = LocalQController(spec, config_for())
    policy = QMIXPolicy(controller)
    observations = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 10
    captured = []
    hook = controller.register_forward_hook(lambda _module, _inputs, output: captured.append(output[0]))
    try:
        actions = []
        for time in range(4):
            local = {agent: observations[0, time, i].numpy() for i, agent in enumerate(spec.agents)}
            chosen = policy.act(local)
            actions.append([chosen[agent] for agent in spec.agents])
    finally:
        hook.remove()
    assert policy.hidden.abs().sum() > 0
    assert policy.previous_actions.sum().item() == 2
    batch_values = unroll(controller, observations, torch.tensor([actions[:-1]]))
    torch.testing.assert_close(batch_values, torch.stack(captured, dim=1))
    policy.reset()
    assert not policy.hidden.any() and not policy.previous_actions.any()
    initial = {agent: observations[0, 0, i].numpy() for i, agent in enumerate(spec.agents)}
    assert policy.act(initial) == dict(zip(spec.agents, actions[0]))


def test_episode_replay_preserves_final_states_masks_padding_and_copies_data():
    spec = EnvironmentSpec(("a", "b"), (2, 2), (2, 2), 4)
    replay = EpisodeReplay(spec, capacity=2, seed=2)
    first = episode_for(spec, 1, terminal=True)
    replay.add(first)
    first["observations"][:] = 999
    replay.add(episode_for(spec, 3, truncated=True))
    batch = replay.sample(2)
    assert batch["observations"].shape == (2, 4, 2, 2)
    assert batch["filled"].sum() == 4
    assert batch["observations"].max() < 999
    lengths = batch["filled"].sum(dim=1).tolist()
    short, long = lengths.index(1), lengths.index(3)
    assert batch["terminated"][short, 0] == 1
    assert batch["truncated"][long, 2] == 1 and not batch["terminated"][long].any()
    torch.testing.assert_close(batch["states"][long, 3], torch.full((4,), 3.0))
    assert not batch["filled"][short, 1:].any()
    replay.add(episode_for(spec, 2, truncated=True))
    assert len(replay) == 2
    assert sorted(replay.sample(2)["filled"].sum(dim=1).tolist()) == [2, 3]


def test_replay_rejects_reset_crossings_and_missing_final_observation():
    spec = EnvironmentSpec(("a",), (2,), (2,), 3)
    replay = EpisodeReplay(spec, 2, seed=0)
    crossed = episode_for(spec, 3)
    crossed["terminated"][0] = 1
    with pytest.raises(ValueError, match="reset boundary"):
        replay.add(crossed)
    incomplete = episode_for(spec, 2)
    incomplete["observations"] = incomplete["observations"][:-1]
    with pytest.raises(ValueError, match="final observation"):
        replay.add(incomplete)
    with pytest.raises(ValueError, match="Not enough"):
        replay.sample(1)


def test_td_targets_bootstrap_time_limits_and_padding_does_not_change_loss():
    target = td_targets(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]),
                        torch.tensor([10.0, 10.0]), gamma=0.9)
    torch.testing.assert_close(target, torch.tensor([1.0, 10.0]))
    predictions = torch.tensor([[2.0, 4.0, 100.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 2.0, -1000000.0]])
    loss = masked_td_loss(predictions, targets, torch.tensor([[1.0, 1.0, 0.0]]))
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[1.0, 2.0, 0.0]]))


def fixed_value_learner(*, double_q=True, target_interval=200):
    spec = EnvironmentSpec(("a",), (1,), (2,), 1)
    config = config_for(mixing_size=1, use_rnn=False, double_q=double_q,
                        target_update_interval=target_interval)
    learner = QMIXLearner(spec, config, "cpu")
    with torch.no_grad():
        for module in (learner.controller, learner.target_controller, learner.mixer, learner.target_mixer):
            for parameter in module.parameters():
                parameter.zero_()
        learner.controller.output_layer.bias.copy_(torch.tensor([1.0, 2.0]))
        learner.target_controller.output_layer.bias.copy_(torch.tensor([10.0, 3.0]))
        # 正值区域中使 mixer 等于恒等映射，得到可手算的联合 TD 目标。
        for mixer in (learner.mixer, learner.target_mixer):
            mixer.first_weights[-1].bias.fill_(1.0)
            mixer.final_weights[-1].bias.fill_(1.0)
    batch = {
        "observations": torch.zeros(1, 2, 1, 1), "states": torch.zeros(1, 2, 1),
        "actions": torch.zeros(1, 1, 1, dtype=torch.long), "rewards": torch.zeros(1, 1),
        "terminated": torch.zeros(1, 1), "truncated": torch.ones(1, 1),
        "filled": torch.ones(1, 1),
    }
    return learner, batch


@pytest.mark.parametrize("double_q,expected", [(True, 2.97), (False, 9.9)])
def test_double_q_uses_online_selection_and_frozen_target_evaluation(double_q, expected):
    learner, batch = fixed_value_learner(double_q=double_q)
    before = deepcopy(learner.target_controller.state_dict())
    metrics = learner.train_batch(batch)
    assert metrics["target_mean"] == pytest.approx(expected)
    assert metrics["loss"] == pytest.approx((1 - expected) ** 2)
    for name, value in before.items():
        torch.testing.assert_close(learner.target_controller.state_dict()[name], value)
    assert all(parameter.grad is None for parameter in learner.target_controller.parameters())
    assert all(parameter.grad is None for parameter in learner.target_mixer.parameters())


def test_terminal_does_not_bootstrap_and_target_sync_copies_both_networks():
    learner, batch = fixed_value_learner(target_interval=1)
    batch["terminated"].fill_(1)
    metrics = learner.train_batch(batch)
    assert metrics["target_mean"] == 0
    for online, target in ((learner.controller, learner.target_controller),
                           (learner.mixer, learner.target_mixer)):
        for name, value in online.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])


def test_team_reward_modes_and_epsilon_schedule_are_explicit():
    agents = ("a", "b")
    assert aggregate_reward({"a": 1, "b": 3}, agents, "mean") == 2
    assert aggregate_reward({"a": 1, "b": 3}, agents, "sum") == 4
    assert aggregate_reward({"a": 2, "b": 2}, agents, "shared") == 2
    with pytest.raises(ValueError, match="identical rewards"):
        aggregate_reward({"a": 1, "b": 3}, agents, "shared")
    config = config_for()
    assert epsilon_at(0, config) == 1.0
    assert epsilon_at(config["epsilon_anneal_steps"], config) == pytest.approx(0.05)
    assert epsilon_at(2 * config["epsilon_anneal_steps"], config) == pytest.approx(0.05)


@pytest.mark.parametrize("environment", ["spread", "lbf", "rware"])
def test_cooperative_training_updates_saves_loads_and_evaluates(tmp_path, monkeypatch, environment):
    if environment != "spread" and importlib.util.find_spec(
        "lbforaging" if environment == "lbf" else "rware"
    ) is None:
        pytest.skip("optional cooperative environment not installed")
    settings = resolved_env_kwargs(environment)
    setting = {"spread": "max_cycles", "lbf": "max_episode_steps", "rware": "max_steps"}[environment]
    settings[setting] = 3
    config = config_for(environment, env_kwargs=settings)
    env = make_env(environment, **settings)
    try:
        observations, _ = env.reset(seed=config["seed"])
        torch.manual_seed(config["seed"])
        initial = QMIXLearner(env.describe(), config, "cpu")
        checkpoint = run_training(config, tmp_path / environment)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["steps"] == config["total_steps"] and saved["updates"] > 0
        assert saved["training_contract"]["reward_aggregation"] == "mean"
        assert saved["resume_supported"] is False
        for key, module in (("controller", initial.controller), ("mixer", initial.mixer)):
            assert any(not torch.equal(saved[key][name], value)
                       for name, value in module.named_parameters())
            assert all(torch.isfinite(value).all() for value in saved[key].values())
        for step in (5, 10):
            periodic = checkpoint.parent / "checkpoints" / f"step_{step:09d}.pt"
            assert torch.load(periodic, map_location="cpu", weights_only=True)["steps"] == step
        report = json.loads((checkpoint.parent / "training.json").read_text(encoding="utf-8"))
        assert report["requested_joint_steps"] == 12
        with (checkpoint.parent / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert {"episode", "steps", *env.possible_agents} <= rows[0].keys()
        with (checkpoint.parent / "updates.csv").open(encoding="utf-8", newline="") as stream:
            updates = list(csv.DictReader(stream))
        assert len(updates) == saved["updates"]
        assert all(np.isfinite(float(row["loss"])) for row in updates)
        algorithm = make_algorithm("qmix")
        loaded = algorithm.load_policy(checkpoint, env, observations)
        again = algorithm.load_policy(checkpoint, env, observations)
        # 加载之后，执行不再读取集中状态或评估快照。
        def forbidden():
            raise AssertionError("Execution policy accessed hidden environment information")

        monkeypatch.setattr(env, "state", forbidden)
        monkeypatch.setattr(env, "task_snapshot", forbidden)
        actions = loaded.act(observations)
        assert again.act(observations) == actions
        loaded.act(observations)
        loaded.reset()
        assert loaded.act(observations) == actions
        assert all(env.action_space(agent).contains(action) for agent, action in actions.items())
        original = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            algorithm.train(config, checkpoint.parent)
        assert checkpoint.read_bytes() == original
    finally:
        env.close()
    evaluation = evaluate(environment, episodes=2, checkpoint=checkpoint, seed=10000)
    assert evaluation["training_steps"] == 12 and len(evaluation["episodes"]) == 2


@pytest.mark.parametrize("environment,env_kwargs", [("tag", {"max_cycles": 5}), ("spread", {})])
def test_rejects_opposing_teams_and_unresolved_environment_without_writing(tmp_path, environment, env_kwargs):
    config = {"algorithm": "qmix", "env": environment, "env_kwargs": env_kwargs,
              "seed": 0, "total_steps": 8}
    with pytest.raises(ValueError, match="cooperative team|env_kwargs"):
        make_algorithm("qmix").train(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
