"""MAPPO 的数值、执行信息和检查点合同检查；短程训练不证明收敛。"""

import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from teamwork.algorithms import make_algorithm
from teamwork.algorithms.mappo import MAPPOAlgorithm, MAPPOPolicy
from teamwork.algorithms.mappo.networks import MAPPOModel, ValueNormalizer
from teamwork.algorithms.mappo.trainer import (
    aggregate_rewards,
    compute_gae,
    resolve_config,
    value_loss,
)
from teamwork.environments.base import Environment, EnvironmentSpec
from teamwork.environments.registry import make_env


def test_value_normalizer_combines_batches_and_round_trips():
    # 总体均值 2.5、总体方差 1.25；分批更新不能改变样本权重。
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    whole, split = ValueNormalizer(), ValueNormalizer()
    whole.update(values)
    split.update(values[:1])
    split.update(values[1:])
    expected = (values - 2.5) / math.sqrt(1.25)
    torch.testing.assert_close(whole.normalize(values), expected)
    torch.testing.assert_close(split.normalize(values), expected)
    outside_batch = torch.tensor([-5.0, 0.0, 20.0])
    torch.testing.assert_close(
        split.denormalize(split.normalize(outside_batch)), outside_batch
    )
    restored = ValueNormalizer()
    restored.load_state_dict(split.state_dict())
    torch.testing.assert_close(restored.normalize(outside_batch), split.normalize(outside_batch))


def test_joint_gae_bootstraps_truncation_and_cuts_each_agent_trace():
    # agent 0 在第二步真正终止；agent 1 每步都在截断边界，仍保留末状态 bootstrap。
    advantages, returns = compute_gae(
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
        torch.tensor([[4.0, 40.0], [5.0, 50.0]]),
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        gamma=0.9,
        gae_lambda=0.5,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.7, 16.0], [-2.0, 25.0]]))
    torch.testing.assert_close(returns, torch.tensor([[3.7, 46.0], [2.0, 65.0]]))


@pytest.mark.parametrize("use_huber, expected", [(False, 2.77), (True, 1.8)])
def test_value_clip_uses_pessimistic_loss(use_huber, expected):
    # clipped 预测为 [1.2, 0.8]，需要逐样本取较大损失，再求均值。
    loss = value_loss(
        torch.tensor([3.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([4.0, -1.0]),
        clip=0.2,
        huber_delta=1.0,
        use_clipped_value_loss=True,
        use_huber_loss=use_huber,
    )
    torch.testing.assert_close(loss, torch.tensor(expected))
    unclipped = value_loss(
        torch.tensor([3.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([4.0, -1.0]),
        use_clipped_value_loss=False,
        use_huber_loss=use_huber,
    )
    torch.testing.assert_close(unclipped, torch.tensor(0.5))


def test_policy_sharing_respects_spaces_and_can_be_disabled():
    spec = EnvironmentSpec(("a", "b", "c"), (3, 3, 4), (2, 2, 3), 8)
    shared = MAPPOModel(spec, {"share_policy": True})
    assert shared.agent_to_group["a"] == shared.agent_to_group["b"]
    assert shared.agent_to_group["a"] != shared.agent_to_group["c"]
    independent = MAPPOModel(spec, {"share_policy": False})
    assert len(set(independent.agent_to_group.values())) == 3


def test_actor_local_observation_and_critic_agent_identity():
    spec = EnvironmentSpec(("a", "b"), (3, 3), (2, 2), 4)
    model = MAPPOModel(spec, {"share_policy": True})
    first = {"a": np.array([1.0, 0.0, -1.0], dtype=np.float32), "b": np.zeros(3)}
    changed = {**first, "b": np.array([100.0, -300.0, 10.0], dtype=np.float32)}
    torch.testing.assert_close(model.local_logits(first)["a"], model.local_logits(changed)["a"])
    critic_inputs = []
    hook = model.critic.register_forward_pre_hook(
        lambda _module, inputs: critic_inputs.append(inputs[0].detach().clone())
    )
    try:
        state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert model.values(state).shape == (2,)
    finally:
        hook.remove()
    assert len(critic_inputs) == 1
    torch.testing.assert_close(critic_inputs[0][:, :4], state.expand(2, -1))
    torch.testing.assert_close(critic_inputs[0][:, 4:], torch.eye(2))


def test_training_updates_actor_keeps_clip_coordinates_and_saves_checkpoints(tmp_path, monkeypatch):
    from teamwork.algorithms.mappo import trainer

    config = {
        "env": "spread",
        "env_kwargs": {"max_cycles": 5},
        "algorithm": "mappo_ff",
        "seed": 19,
        "total_steps": 16,
        "rollout_steps": 8,
        "epochs": 1,
        "batch_size": 24,
        "hidden_size": 16,
        "hidden_layers": 1,
        "share_policy": True,
        "feature_normalization": False,
        "value_normalization": True,
        "checkpoint_interval": 8,
    }
    env = make_env("spread", max_cycles=5)
    try:
        observations, _ = env.reset(seed=config["seed"])
        torch.manual_seed(config["seed"])
        initial = MAPPOModel(env.describe(), config)
        initial_actor = {key: value.clone() for key, value in initial.state_dict().items()
                         if key.startswith("actors.")}
        first_critic_call = []
        original_loss = trainer.value_loss

        def inspect_first_update(values, old_values, targets, *args, **kwargs):
            if not first_critic_call:
                # ValueNorm 已更新统计，但未更新权重：旧值不能从 raw units 误映射到新坐标。
                torch.testing.assert_close(values, old_values, atol=1e-6, rtol=1e-5)
                first_critic_call.append(True)
            return original_loss(values, old_values, targets, *args, **kwargs)

        monkeypatch.setattr(trainer, "value_loss", inspect_first_update)
        algorithm = make_algorithm("mappo_ff")
        checkpoint = algorithm.train(config, tmp_path / "run")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["steps"] == 16
        assert saved["format"] == "teamwork.mappo.v1"
        assert first_critic_call
        assert any(not torch.equal(saved["model"][key], value)
                   for key, value in initial_actor.items())
        assert all(torch.isfinite(value).all() for value in saved["model"].values())
        for step in (8, 16):
            periodic = checkpoint.parent / "checkpoints" / f"step_{step:09d}.pt"
            assert torch.load(periodic, map_location="cpu", weights_only=True)["steps"] == step
        with (checkpoint.parent / "updates.csv").open(encoding="utf-8", newline="") as stream:
            updates = list(csv.DictReader(stream))
        assert [int(row["steps"]) for row in updates] == [8, 16]
        for row in updates:
            for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
                assert math.isfinite(float(row[name]))
        loaded = algorithm.load_policy(checkpoint, env, observations)
        before = loaded.act(observations)
        loaded_again = algorithm.load_policy(checkpoint, env, observations)
        assert loaded_again.act(observations) == before
        assert all(env.action_space(agent).contains(action) for agent, action in before.items())
        original_bytes = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            algorithm.train(config, checkpoint.parent)
        assert checkpoint.read_bytes() == original_bytes
    finally:
        env.close()


def test_checkpoint_schedule_splits_rollout_without_exceeding_budget(tmp_path):
    # interval 与 rollout 不整除时，也应在约定的联合步数保存模型。
    config = {
        "env": "spread",
        "env_kwargs": {"max_cycles": 5},
        "algorithm": "mappo_ff",
        "seed": 23,
        "total_steps": 21,
        "rollout_steps": 8,
        "epochs": 1,
        "batch_size": 64,
        "hidden_size": 16,
        "hidden_layers": 1,
        "checkpoint_interval": 10,
    }
    final = make_algorithm("mappo_ff").train(config, tmp_path / "nondivisible")
    assert torch.load(final, map_location="cpu", weights_only=True)["steps"] == 21
    checkpoints = sorted((final.parent / "checkpoints").glob("step_*.pt"))
    assert [int(path.stem.removeprefix("step_")) for path in checkpoints] == [10, 20]
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["steps"] == int(path.stem.removeprefix("step_"))


@pytest.mark.parametrize("env_kwargs", [None, {}, {"max_cycles": 7}])
def test_direct_training_resolves_protocol_conditions_without_mutating_config(env_kwargs):
    # API 自己采用公共协议默认值，不要求调用者先经过 CLI/runner。
    config = {
        "env": "spread", "total_steps": 8,
    }
    if env_kwargs is not None:
        config["env_kwargs"] = env_kwargs
    before = deepcopy(config)
    resolved = resolve_config(config)
    assert config == before
    assert resolved["algorithm"] == "mappo_ff" and resolved["seed"] == 0
    assert resolved["env_kwargs"] == {
        "N": 3, "local_ratio": 0.5, "max_cycles": 7 if env_kwargs else 50,
    }


def test_yaml_api_and_simulator_free_loading_are_legacy_compatible(tmp_path):
    source = Path(__file__).resolve().parents[2] / "configs" / "algorithms" / "mappo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.update(total_steps=5, rollout_steps=5, epochs=1, hidden_size=8, hidden_layers=1,
                  checkpoint_interval=0)
    before = deepcopy(config)
    checkpoint = MAPPOAlgorithm().train(config, tmp_path / "yaml-api")
    assert config == before
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["config"]["env_kwargs"]["max_cycles"] == 50
    policy = MAPPOAlgorithm().load_policy(checkpoint)
    assert isinstance(policy, MAPPOPolicy)
    spec = policy.model.spec
    observations = {
        agent: np.linspace(-1, 1, size, dtype=np.float32)
        for agent, size in zip(spec.agents, spec.observation_sizes)
    }
    expected = policy.act(observations)
    assert policy.model.agent_ids.device.type == "cpu"
    # 旧 v1 检查点没有 reward_aggregation，仍可推理。
    saved["config"].pop("reward_aggregation", None)
    saved["training_contract"] = {"reward": "environment", "execution": "local_observation"}
    legacy = checkpoint.parent / "legacy.pt"
    torch.save(saved, legacy)
    assert MAPPOAlgorithm().load_policy(legacy, observations=observations).act(observations) == expected
    assert all(0 <= expected[a] < n for a, n in zip(spec.agents, spec.action_counts))
    mismatched = make_env("spread", N=2)
    try:
        mismatched.reset(seed=0)
        with pytest.raises(ValueError, match="EnvironmentSpec"):
            MAPPOAlgorithm().load_policy(checkpoint, mismatched)
    finally:
        mismatched.close()
    for invalid in ({}, {**observations, spec.agents[0]: np.zeros((1, spec.observation_sizes[0]))}):
        with pytest.raises(ValueError, match="observation"):
            MAPPOAlgorithm().load_policy(checkpoint, observations=invalid)


@pytest.mark.parametrize("problem", ["missing", "extra", "rank", "size", "nan", "inf"])
def test_policy_rejects_invalid_observations(problem):
    spec = EnvironmentSpec(("a", "b"), (3, 3), (2, 2), 4)
    policy = MAPPOPolicy(MAPPOModel(spec, {"hidden_size": 8, "hidden_layers": 1}))
    observations = {"a": np.zeros(3), "b": np.ones(3)}
    if problem == "missing":
        del observations["b"]
    elif problem == "extra":
        observations["c"] = np.zeros(3)
    elif problem == "rank":
        observations["a"] = np.zeros((1, 3))
    elif problem == "size":
        observations["a"] = np.zeros(4)
    else:
        observations["a"][0] = float(problem)
    with pytest.raises(ValueError, match="observation"):
        policy.act(observations)


@pytest.mark.parametrize("mode, expected", [
    ("individual", [1.0, 3.0]), ("mean", [2.0, 2.0]), ("sum", [4.0, 4.0]),
])
def test_reward_transform_math(mode, expected):
    np.testing.assert_allclose(aggregate_rewards({"a": 1.0, "b": 3.0}, ("a", "b"), mode), expected)
    np.testing.assert_allclose(
        aggregate_rewards({"a": 4.0, "b": 4.0}, ("a", "b"), "shared"), [4.0, 4.0]
    )
    with pytest.raises(ValueError, match="identical"):
        aggregate_rewards({"a": 1.0, "b": 3.0}, ("a", "b"), "shared")


class _BoundaryEnvironment(Environment):
    """真实终止后 state 无效；时间限制则保留真实末状态。"""

    def __init__(self, terminal=True, invalid=None, mixed_roles=False):
        self.possible_agents = ["a", "b"]
        self.terminal = terminal
        self.invalid = invalid
        self.mixed_roles = mixed_roles
        self.closed = False
        self.done = False
        self.agents = []
        self.final_state_reads = 0

    @property
    def policy_metadata(self):
        return {"roles": {"a": False, "b": self.mixed_roles}}

    def observation_space(self, agent):
        return gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    def action_space(self, agent):
        return gym.spaces.Discrete(2)

    def describe(self):
        if self.invalid == "spec":
            return EnvironmentSpec(("a", "b"), (3, 3), (0, 2), 4)
        return EnvironmentSpec(("a", "b"), (3, 3), (2, 2), 4)

    def reset(self, seed=None, options=None):
        if self.invalid == "reset":
            raise ValueError("Invalid reset")
        self.done = False
        self.agents = self.possible_agents[:]
        observations = {a: np.zeros(3, np.float32) for a in self.agents}
        if self.invalid == "observation":
            observations["a"][0] = np.nan
        return observations, {}

    def state(self):
        if self.done:
            if self.terminal:
                raise AssertionError("State unavailable after true termination")
            self.final_state_reads += 1
            return np.full(4, 42, np.float32)
        if self.invalid == "state_shape":
            return np.zeros(5, np.float32)
        return np.full(4, np.nan if self.invalid == "state" else 0, np.float32)

    def step(self, actions):
        self.done = True
        self.agents = []
        terms = dict.fromkeys(self.possible_agents, self.terminal)
        truncs = dict.fromkeys(self.possible_agents, not self.terminal)
        observations = {} if self.terminal else {a: np.ones(3) for a in self.possible_agents}
        return observations, {"a": 1.0, "b": 3.0}, terms, truncs, {}

    def task_snapshot(self):
        raise AssertionError("Training must not read evaluation snapshots")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("terminal", [True, False])
def test_episode_boundary_bootstrap_and_raw_csv_reward_contract(tmp_path, monkeypatch, terminal):
    from teamwork.algorithms.mappo import trainer

    env = _BoundaryEnvironment(terminal=terminal)
    monkeypatch.setattr(trainer, "make_env", lambda *args, **kwargs: env)
    checked = []
    original_update = trainer._update

    def inspect_update(model, normalizer, actor_optimizers, critic_optimizer, batch, config):
        torch.testing.assert_close(batch["rewards"], torch.full((2, 2), 2.0))
        if terminal:
            torch.testing.assert_close(batch["next_values"], torch.zeros(2, 2))
        else:
            expected = model.values(np.full(4, 42, np.float32)).detach().expand(2, -1)
            torch.testing.assert_close(batch["next_values"], expected)
        checked.append(True)
        return original_update(model, normalizer, actor_optimizers, critic_optimizer, batch, config)

    monkeypatch.setattr(trainer, "_update", inspect_update)
    config = {
        "env": "spread", "total_steps": 2, "rollout_steps": 2, "epochs": 1,
        "hidden_size": 8, "hidden_layers": 1, "reward_aggregation": "mean",
        "checkpoint_interval": 0,
    }
    before = deepcopy(config)
    checkpoint = MAPPOAlgorithm().train(config, tmp_path / f"boundary-{terminal}")
    assert config == before and checked and env.closed
    assert env.final_state_reads == (0 if terminal else 2)
    with (checkpoint.parent / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    assert [(float(row["a"]), float(row["b"])) for row in episodes] == [(1.0, 3.0)] * 2
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = json.loads((checkpoint.parent / "training_contract.json").read_text(encoding="utf-8"))
    assert contract == saved["training_contract"]
    assert contract["reward_aggregation"] == "mean"
    assert contract["reward_scope"] == "one_cooperative_team"


@pytest.mark.parametrize("invalid", [
    "reset", "spec", "observation", "state", "state_shape", "mixed_roles",
])
def test_invalid_environment_setup_closes_without_creating_run(tmp_path, monkeypatch, invalid):
    from teamwork.algorithms.mappo import trainer

    env = _BoundaryEnvironment(invalid=invalid, mixed_roles=invalid == "mixed_roles")
    monkeypatch.setattr(trainer, "make_env", lambda *args, **kwargs: env)
    out = tmp_path / invalid
    with pytest.raises(ValueError):
        MAPPOAlgorithm().train({"env": "spread", "total_steps": 1, "reward_aggregation": "mean"}, out)
    assert env.closed and not out.exists()


@pytest.mark.parametrize("overrides", [
    {"algorithm": "qmix"}, {"env_kwargs": None}, {"env_kwargs": []},
    {"extra": {1, 2}}, {"extra": float("nan")}, {"reward_aggregation": "mean", "env": "tag"},
])
def test_invalid_config_fails_before_artifacts(tmp_path, overrides):
    out = tmp_path / "invalid-config"
    with pytest.raises((TypeError, ValueError)):
        MAPPOAlgorithm().train({"env": "spread", "total_steps": 1, **overrides}, out)
    assert not out.exists()


def test_missing_budget_reports_a_configuration_error(tmp_path):
    out = tmp_path / "missing-budget"
    with pytest.raises(ValueError, match="total_steps"):
        MAPPOAlgorithm().train({"env": "spread"}, out)
    assert not out.exists()


def test_non_dictionary_checkpoint_reports_a_format_error(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save([], checkpoint)
    with pytest.raises(ValueError, match="checkpoint format"):
        MAPPOAlgorithm().load_policy(checkpoint)


def test_collection_freezes_reused_observation_arrays_before_step(tmp_path, monkeypatch):
    from teamwork.algorithms.mappo import trainer

    class MutableEnvironment(_BoundaryEnvironment):
        def reset(self, seed=None, options=None):
            self.done = False
            self.agents = self.possible_agents[:]
            self.counter = 0
            if not hasattr(self, "observations"):
                self.observations = {a: np.zeros(3, np.float32) for a in self.possible_agents}
            for values in self.observations.values():
                values.fill(0)
            return self.observations, {}

        def step(self, actions):
            self.counter += 1
            for values in self.observations.values():
                values.fill(self.counter)
            self.done = self.counter == 2
            self.agents = [] if self.done else self.possible_agents[:]
            return (
                {} if self.done else self.observations, {"a": 1.0, "b": 3.0},
                dict.fromkeys(self.possible_agents, self.done),
                dict.fromkeys(self.possible_agents, False), {},
            )

    env = MutableEnvironment()
    monkeypatch.setattr(trainer, "make_env", lambda *args, **kwargs: env)
    original_update = trainer._update
    checked = []

    def inspect_update(model, normalizer, actor_optimizers, critic_optimizer, batch, config):
        for agent in model.spec.agents:
            torch.testing.assert_close(
                batch["observations"][agent], torch.tensor([[0.0] * 3, [1.0] * 3])
            )
        for step in range(2):
            logits = model.local_logits({a: values[step] for a, values in batch["observations"].items()})
            for index, agent in enumerate(model.spec.agents):
                distribution = torch.distributions.Categorical(logits=logits[agent])
                torch.testing.assert_close(
                    distribution.log_prob(batch["actions"][step, index]),
                    batch["log_probs"][step, index],
                )
        checked.append(True)
        return original_update(model, normalizer, actor_optimizers, critic_optimizer, batch, config)

    monkeypatch.setattr(trainer, "_update", inspect_update)
    MAPPOAlgorithm().train({
        "env": "spread", "total_steps": 2, "rollout_steps": 2, "epochs": 1,
        "hidden_size": 8, "hidden_layers": 1, "checkpoint_interval": 0,
        "feature_normalization": False,
    }, tmp_path / "mutable-observations")
    assert checked and env.closed
