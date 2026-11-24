"""HAPPO 的顺序更新语义、异构 actor、检查点与公共 API。

除显式标注的端到端用例外，其余用例都不创建仿真环境。
"""

import json
import math

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from teamwork.algorithms import make_algorithm
from teamwork.algorithms.happo import trainer
from teamwork.algorithms.happo.networks import (
    HAPPOModel,
    ValueNormalizer,
    actor_for,
    resolve_actor_order,
)
from teamwork.environments.base import EnvironmentSpec

SPEC = EnvironmentSpec(("a0", "a1"), (4, 4), (3, 3), 5)
ORDER = ["a0", "a1"]

SPREAD_CONFIG = {
    "algorithm": "happo",
    "env": "spread",
    "seed": 0,
    "device": "cpu",
    "total_steps": 32,
    "rollout_steps": 16,
    "batch_size": 8,
    "epochs": 1,
    "reward_aggregation": "mean",
}


def make_config(**overrides):
    config = {
        "algorithm": "happo",
        "env": "spread",
        "total_steps": 32,
        "rollout_steps": 8,
        "batch_size": 4,
        "epochs": 1,
        "seed": 0,
        "device": "cpu",
    }
    config.update(overrides)
    return trainer.resolve_config(config)


def make_model(config=None):
    torch.manual_seed(0)
    return HAPPOModel(SPEC, config or make_config())


def make_batch(model, horizon=6, seed=0):
    """观测与动作随机，但 log_probs 取自模型当前策略，因此初始重要性比率恰为 1。"""
    generator = torch.Generator().manual_seed(seed)
    observations = {
        agent: torch.randn(horizon, size, generator=generator)
        for agent, size in zip(SPEC.agents, SPEC.observation_sizes)
    }
    actions = torch.stack(
        [torch.randint(0, count, (horizon,), generator=generator) for count in SPEC.action_counts],
        dim=1,
    )
    with torch.no_grad():
        log_probs = torch.stack(
            [
                Categorical(logits=actor_for(model, agent)(observations[agent])).log_prob(
                    actions[:, index]
                )
                for index, agent in enumerate(SPEC.agents)
            ],
            dim=1,
        )
    zeros = torch.zeros(horizon, len(SPEC.agents))
    return {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "states": torch.randn(horizon, SPEC.state_size, generator=generator),
        # 优势必须非零，否则顺序更新的乘积只能作用在零梯度上，测不出顺序差异。
        "values": torch.randn(horizon, len(SPEC.agents), generator=generator),
        "next_values": torch.randn(horizon, len(SPEC.agents), generator=generator),
        "rewards": torch.randn(horizon, len(SPEC.agents), generator=generator),
        "terminated": zeros.clone(),
        "episode_end": zeros.clone(),
        "old_normalized_values": zeros.clone(),
    }


def make_update_state(model, config):
    normalizer = ValueNormalizer()
    actor_optimizers = {
        agent: torch.optim.Adam(
            actor_for(model, agent).parameters(),
            lr=config["actor_lr"],
            eps=config["adam_eps"],
        )
        for agent in SPEC.agents
    }
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=config["critic_lr"], eps=config["adam_eps"]
    )
    return normalizer, actor_optimizers, critic_optimizer


def test_registered():
    assert type(make_algorithm("happo")).__name__ == "HAPPOAlgorithm"


def test_heterogeneous_actors_are_one_per_agent():
    model = make_model()
    first, second = (actor_for(model, agent) for agent in SPEC.agents)
    assert first is not second
    assert {id(parameter) for parameter in first.parameters()}.isdisjoint(
        {id(parameter) for parameter in second.parameters()}
    )


def test_critic_input_carries_agent_identity_but_actor_does_not():
    model = make_model()
    inputs = model.critic_inputs(torch.zeros(1, SPEC.state_size))
    assert inputs.shape == (1, len(SPEC.agents), SPEC.state_size + len(SPEC.agents))
    assert not torch.allclose(inputs[0, 0], inputs[0, 1])


def test_config_requires_heterogeneous_actors():
    with pytest.raises(ValueError, match="share_policy"):
        make_config(share_policy=True)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"algorithm": "mappo_ff"}, "algorithm"),
        ({"gamma": 1.5}, "gamma"),
        ({"rollout_steps": 0}, "rollout_steps"),
        ({"batch_size": 0.5}, "batch_size"),
        ({"epochs": -1}, "epochs"),
        ({"checkpoint_interval": -1}, "checkpoint_interval"),
        ({"clip": 0.0}, "clip"),
        ({"entropy_coef": -0.1}, "entropy_coef"),
        ({"reward_aggregation": "bogus"}, "reward_aggregation"),
        ({"env": "tag", "reward_aggregation": "mean"}, "Tag"),
        ({"actor_order": "a0"}, "actor_order"),
        ({"share_policy": "no"}, "share_policy"),
    ],
)
def test_config_validation_rejects_invalid_combinations(overrides, message):
    with pytest.raises((ValueError, TypeError), match=message):
        make_config(**overrides)


def test_actor_order_must_cover_the_whole_population():
    assert resolve_actor_order(SPEC, None) == ["a0", "a1"]
    assert resolve_actor_order(SPEC, ["a1", "a0"]) == ["a1", "a0"]
    for invalid in (["a0"], ["a0", "a0"], ["a0", "zz"]):
        with pytest.raises(ValueError, match="actor_order"):
            resolve_actor_order(SPEC, invalid)


def test_prefix_ratio_is_one_before_updates_and_follows_new_parameters():
    """守住 HAPPO 的核心：前序比率必须来自更新后的参数，而不是增量前的旧值。"""
    model = make_model()
    batch = make_batch(model)
    ids = torch.arange(3)
    assert trainer.prefix_ratio(model, ORDER, 0, batch, ids) is None

    fresh = trainer.prefix_ratio(model, ORDER, 1, batch, ids)
    assert torch.allclose(fresh, torch.ones_like(fresh))

    with torch.no_grad():
        for parameter in actor_for(model, "a0").parameters():
            parameter.add_(0.4)
    moved = trainer.prefix_ratio(model, ORDER, 1, batch, ids)
    assert not torch.allclose(moved, torch.ones_like(moved))


def test_sequential_update_reports_a_live_prefix_ratio():
    """prefix_ratio_deviation 恒为 0 就说明顺序更新退化成了同时更新。"""
    config = make_config(actor_lr=0.05, epochs=1, batch_size=4)
    model = make_model(config)
    normalizer, actor_optimizers, critic_optimizer = make_update_state(model, config)
    batch = make_batch(model)
    diagnostics = trainer._update(
        model, normalizer, actor_optimizers, critic_optimizer, batch, ORDER, config
    )
    assert diagnostics["prefix_ratio_deviation"] > 0
    assert math.isfinite(diagnostics["policy_loss"])
    assert math.isfinite(diagnostics["value_loss"])


def test_update_order_changes_the_outcome():
    config = make_config(actor_lr=0.05)
    base = make_model(config)
    reference = {key: value.clone() for key, value in base.state_dict().items()}
    parameters = []
    for order in (ORDER, list(reversed(ORDER))):
        model = make_model(config)
        model.load_state_dict(reference)
        normalizer, actor_optimizers, critic_optimizer = make_update_state(model, config)
        batch = make_batch(model)
        trainer._update(model, normalizer, actor_optimizers, critic_optimizer, batch, order, config)
        parameters.append(
            torch.cat([value.detach().flatten() for value in actor_for(model, "a1").parameters()])
        )
    assert not torch.allclose(parameters[0], parameters[1])


def test_checkpoint_round_trip_without_an_environment(tmp_path):
    run = tmp_path / "run"
    checkpoint = make_algorithm("happo").train(SPREAD_CONFIG, run)
    algorithm = make_algorithm("happo")
    policy = algorithm.load_policy(checkpoint, device="cpu")

    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = data["spec"]
    observations = {
        agent: np.zeros(size, dtype=np.float32)
        for agent, size in zip(spec["agents"], spec["observation_sizes"])
    }
    policy.reset()
    actions = policy.act(observations, deterministic=True)
    assert set(actions) == set(spec["agents"])
    for agent, action in actions.items():
        assert isinstance(action, int)
        assert 0 <= action < dict(zip(spec["agents"], spec["action_counts"]))[agent]

    with pytest.raises(ValueError, match="HAPPO"):
        policy.act(dict.fromkeys(spec["agents"], np.zeros(3)), deterministic=True)


def test_load_policy_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "teamwork.qmix.v1", "config": {"algorithm": "qmix"}}, path)
    with pytest.raises(ValueError, match="format"):
        make_algorithm("happo").load_policy(path)


def test_train_writes_the_public_artifacts(tmp_path):
    run = tmp_path / "run"
    checkpoint = make_algorithm("happo").train(SPREAD_CONFIG, run)
    for name in (
        "config.json",
        "episodes.csv",
        "updates.csv",
        "training_contract.json",
        "model.pt",
    ):
        assert (run / name).is_file(), name
    assert checkpoint == run / "model.pt"

    contract = json.loads((run / "training_contract.json").read_text(encoding="utf-8"))
    assert contract["algorithm"] == "happo"
    assert contract["reward_aggregation"] == "mean"
    assert contract["update"] == "sequential_gauss_seidel"
    assert contract["prefix_ratio_recomputed_after_each_update"] is True
    assert contract["mac_advantage_decomposition"] is False
    assert contract["actor_order"] == ["agent_0", "agent_1", "agent_2"]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["steps"] == SPREAD_CONFIG["total_steps"]
    assert saved["format"] == "teamwork.happo.v1"
    updates = (run / "updates.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(updates) > 1

    with pytest.raises(FileExistsError):
        make_algorithm("happo").train(SPREAD_CONFIG, run)
