"""MAPPO 的数值、执行信息和检查点合同检查；短程训练不证明收敛。"""

import csv
import math

import numpy as np
import pytest
import torch

from teamwork.algorithms import make_algorithm
from teamwork.algorithms.mappo.networks import MAPPOModel, ValueNormalizer
from teamwork.algorithms.mappo.trainer import compute_gae, value_loss
from teamwork.environments.base import EnvironmentSpec
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


@pytest.mark.parametrize("env_kwargs", [None, {}])
def test_direct_training_requires_resolved_environment_conditions(tmp_path, env_kwargs):
    # 禁止直接训练隐式采用环境原生 horizon，而评估又采用公共协议的不同 horizon。
    config = {
        "env": "spread", "algorithm": "mappo_ff", "seed": 0, "total_steps": 8,
    }
    if env_kwargs is not None:
        config["env_kwargs"] = env_kwargs
    out = tmp_path / "unresolved"
    with pytest.raises(ValueError, match="env_kwargs"):
        make_algorithm("mappo_ff").train(config, out)
    assert not out.exists()
