"""HASAC 离散适配的学习器、联合动作编码、温度、检查点与公共 API。

除显式标注的端到端用例外，其余用例都不创建仿真环境。
"""

import math

import numpy as np
import pytest
import torch

from teamwork.algorithms import make_algorithm
from teamwork.algorithms.hasac import trainer
from teamwork.algorithms.hasac.learner import HASACLearner
from teamwork.algorithms.hasac.networks import HASACModel, pack_observations
from teamwork.algorithms.qmix.replay import EpisodeReplay
from teamwork.environments.base import EnvironmentSpec

SPEC = EnvironmentSpec(("a0", "a1"), (4, 4), (3, 3), 5)
HETERO = EnvironmentSpec(("a0", "a1"), (4, 3), (3, 2), 6)
DEVICE = torch.device("cpu")

SPREAD_CONFIG = {
    "algorithm": "hasac_discrete",
    "env": "spread",
    "seed": 0,
    "device": "cpu",
    "total_steps": 120,
    "replay_capacity": 8,
    "batch_size": 2,
    "learning_starts_episodes": 2,
    "updates_per_episode": 1,
    "checkpoint_interval": 0,
}


def make_config(**overrides):
    config = {
        "algorithm": "hasac_discrete",
        "env": "spread",
        "total_steps": 64,
        "device": "cpu",
        "seed": 0,
        "replay_capacity": 16,
        "batch_size": 4,
        "learning_starts_episodes": 4,
    }
    config.update(overrides)
    return trainer.resolve_config(config)


def build_batch(spec=SPEC, lengths=(5, 4, 3, 2), seed=0):
    """回合长度不同，因此批次里必然存在被 padding 的位置。"""
    rng = np.random.default_rng(seed)
    replay = EpisodeReplay(spec, capacity=len(lengths), seed=seed)
    for length in lengths:
        actions = np.stack(
            [rng.integers(0, count, size=length) for count in spec.action_counts], axis=1
        ).astype(np.int64)
        replay.add(
            {
                "observations": rng.normal(
                    size=(length + 1, len(spec.agents), max(spec.observation_sizes))
                ).astype(np.float32),
                "states": rng.normal(size=(length + 1, spec.state_size)).astype(np.float32),
                "actions": actions,
                "rewards": rng.normal(size=length).astype(np.float32),
                "terminated": np.zeros(length, dtype=np.float32),
                "truncated": np.concatenate(
                    [np.zeros(length - 1, dtype=np.float32), np.ones(1, dtype=np.float32)]
                ),
            }
        )
    return replay.sample(len(lengths))


def make_learner(config=None, spec=SPEC):
    torch.manual_seed(0)
    return HASACLearner(spec, config or make_config(), DEVICE)


def test_registered():
    assert type(make_algorithm("hasac_discrete")).__name__ == "HASACAlgorithm"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"algorithm": "happo"}, "algorithm"),
        ({"policy_family": "gaussian"}, "policy_family"),
        ({"env": "tag"}, "Tag"),
        ({"gamma": 1.5}, "gamma"),
        ({"tau": 0.0}, "tau"),
        ({"alpha_start": 0.0}, "alpha_start"),
        ({"learn_alpha": "yes"}, "learn_alpha"),
        ({"reward_aggregation": "individual"}, "reward_aggregation"),
        ({"replay_capacity": 2, "batch_size": 4}, "Replay capacity"),
        ({"checkpoint_interval": -1}, "checkpoint_interval"),
        ({"updates_per_episode": 0}, "updates_per_episode"),
    ],
)
def test_config_validation_rejects_invalid_combinations(overrides, message):
    with pytest.raises((ValueError, TypeError), match=message):
        make_config(**overrides)


def test_joint_action_encoding_uses_disjoint_offsets():
    model = HASACModel(HETERO, make_config())
    encoded = model.joint_one_hot(torch.tensor([[0, 1], [2, 1]]))
    assert encoded.shape == (2, 5)  # 3 + 2 个动作槽位
    assert encoded[0].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert encoded[1].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]


def test_joint_action_rejects_out_of_range_indices():
    model = HASACModel(HETERO, make_config())
    with pytest.raises(ValueError, match="space"):
        model.joint_one_hot(torch.tensor([[3, 0]]))
    with pytest.raises(ValueError, match="every agent"):
        model.joint_one_hot(torch.tensor([[0]]))


def test_twin_soft_q_has_two_independent_heads():
    model = HASACModel(SPEC, make_config())
    states = torch.zeros(4, SPEC.state_size)
    actions = torch.zeros(4, len(SPEC.agents), dtype=torch.long)
    values = model.q_values(states, actions)
    assert values.shape == (4, 2)
    assert not torch.allclose(values[:, 0], values[:, 1])
    with pytest.raises(ValueError, match="leading shape"):
        model.q_values(states, torch.zeros(3, len(SPEC.agents), dtype=torch.long))


def test_pack_observations_pads_only_trailing_entries():
    observations = {
        "a0": np.arange(4, dtype=np.float32),
        "a1": np.arange(3, dtype=np.float32),
    }
    packed = pack_observations(observations, HETERO)
    assert packed.shape == (2, 4)
    assert torch.equal(packed[1, :3], torch.arange(3, dtype=torch.float32))
    assert float(packed[1, 3]) == 0.0


def test_train_batch_updates_every_network_and_reports_finite_metrics():
    learner = make_learner()
    before = {name: value.detach().clone() for name, value in learner.model.named_parameters()}
    metrics = learner.train_batch(build_batch())
    after = dict(learner.model.named_parameters())
    for name, value in before.items():
        if name.startswith("target_"):
            continue
        assert not torch.allclose(value, after[name]), name
    # 目标网络按 tau 软同步，单次更新的变化量可能小于 float32 分辨率，故只要求整体有变化。
    assert any(
        not torch.equal(before[name], after[name]) for name in before if name.startswith("target_")
    )
    for key in (
        "critic_loss",
        "actor_loss",
        "entropy",
        "alpha_mean",
        "q_mean",
        "target_mean",
        "critic_grad_norm",
        "actor_grad_norm",
    ):
        assert math.isfinite(metrics[key]), key
    assert learner.updates == 1


def test_padding_does_not_change_the_update():
    """被 padding 的转移不得进入损失。

    只污染 reward/terminated：观测与状态里被 padding 的槽位仍可能作为"截断回合的
    末观测"参与 bootstrap，那是有意保留的语义，不属于 padding 泄漏。
    """
    batch = build_batch()
    masked = batch["filled"] == 0
    assert bool(masked.any())
    poisoned = {key: value.clone() for key, value in batch.items()}
    poisoned["rewards"][masked] = 99.0
    poisoned["terminated"][masked] = 1.0

    torch.manual_seed(1)
    clean = make_learner().train_batch(batch)
    torch.manual_seed(1)
    dirty = make_learner().train_batch(poisoned)
    assert clean == pytest.approx(dirty, rel=1e-6)


def test_soft_sync_with_full_rate_copies_the_online_networks():
    learner = make_learner()
    learner.model.sync_targets(1.0)
    for target, source in (
        (learner.model.target_actors, learner.model.actors),
        (learner.model.target_critics, learner.model.critics),
    ):
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            assert torch.equal(target_parameter, source_parameter)


def test_temperature_moves_toward_the_target_entropy():
    """初始化策略接近均匀、熵高于目标熵，温度应当下调。"""
    learner = make_learner()
    before = learner.model.log_alphas["a0"].exp().item()
    learner.train_batch(build_batch())
    after = learner.model.log_alphas["a0"].exp().item()
    assert after < before


def test_learn_alpha_false_keeps_the_temperature():
    learner = make_learner(make_config(learn_alpha=False))
    assert learner.alpha_optimizers == {}
    before = learner.model.log_alphas["a0"].exp().item()
    learner.train_batch(build_batch())
    assert learner.model.log_alphas["a0"].exp().item() == before


def test_actor_order_changes_which_actions_the_prefix_uses():
    batch = build_batch()

    def run(order):
        learner = make_learner(make_config(actor_order=order, actor_lr=0.5))
        torch.manual_seed(2)
        learner.train_batch(batch)
        return torch.cat(
            [value.detach().flatten() for value in learner.model.actors["a0"].parameters()]
        )

    first, second = run(["a0", "a1"]), run(["a1", "a0"])
    assert not torch.allclose(first, second)


def test_checkpoint_round_trip_without_an_environment(tmp_path):
    run = tmp_path / "run"
    checkpoint = make_algorithm("hasac_discrete").train(SPREAD_CONFIG, run)
    policy = make_algorithm("hasac_discrete").load_policy(checkpoint, device="cpu")

    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert {"actors", "critics", "log_alphas", "target_actors", "target_critics"} <= set(data)
    saved_actors = data["actors"]
    for key, value in policy.model.actors.state_dict().items():
        assert torch.equal(value, saved_actors[key]), key

    spec = data["spec"]
    observations = {
        agent: np.zeros(size, dtype=np.float32)
        for agent, size in zip(spec["agents"], spec["observation_sizes"])
    }
    policy.reset()
    actions = policy.act(observations, deterministic=True)
    assert set(actions) == set(spec["agents"])
    assert actions == policy.act(observations, deterministic=True)
    for agent, action in actions.items():
        assert isinstance(action, int)
        assert 0 <= action < dict(zip(spec["agents"], spec["action_counts"]))[agent]

    with pytest.raises(ValueError, match="HASAC"):
        policy.act(dict.fromkeys(spec["agents"], np.zeros(3)), deterministic=True)


def test_load_policy_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "teamwork.happo.v1", "config": {"algorithm": "happo"}}, path)
    with pytest.raises(ValueError, match="format"):
        make_algorithm("hasac_discrete").load_policy(path)


def test_train_writes_the_public_artifacts_and_learns(tmp_path):
    run = tmp_path / "run"
    checkpoint = make_algorithm("hasac_discrete").train(SPREAD_CONFIG, run)
    for name in (
        "config.json",
        "episodes.csv",
        "updates.csv",
        "training_contract.json",
        "model.pt",
    ):
        assert (run / name).is_file(), name
    assert checkpoint == run / "model.pt"

    contract = trainer.training_contract(
        trainer.resolve_config(SPREAD_CONFIG), ["agent_0", "agent_1", "agent_2"]
    )
    saved_contract = torch.load(checkpoint, map_location="cpu", weights_only=True)[
        "training_contract"
    ]
    assert saved_contract == contract
    assert saved_contract["algorithm"] == "hasac_discrete"
    assert saved_contract["critic_input"] == "central_state_plus_joint_action_one_hot"
    assert saved_contract["update"] == "sequential_soft_advantage"
    assert saved_contract["reward_aggregation"] == "mean"

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["steps"] == SPREAD_CONFIG["total_steps"]
    assert saved["updates"] >= 1
    updates = (run / "updates.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(updates) >= 2

    with pytest.raises(FileExistsError):
        make_algorithm("hasac_discrete").train(SPREAD_CONFIG, run)
