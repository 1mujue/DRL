"""Fixed-input checks for QTRAN-base objectives and QPLEX gradient modes."""

import csv
import math

import numpy as np
import pytest
import torch

from teamwork.algorithms.qmix.replay import EpisodeReplay
from teamwork.algorithms.qmix.trainer import resolve_config, training_contract
from teamwork.algorithms.qplex.algorithm import QPLEXAlgorithm
from teamwork.algorithms.qplex.learner import DuplexDuelingMixer
from teamwork.algorithms.qtran.algorithm import QTRANAlgorithm
from teamwork.algorithms.qtran.learner import QTRANLearner, constraint_losses
from teamwork.algorithms.registry import make_algorithm
from teamwork.environments.base import EnvironmentSpec
from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import _algorithm_variant, evaluate, save_result


def test_qplex_gradient_modes_match_forward_values_and_route_gradients():
    spec = EnvironmentSpec(("a", "b"), (1, 1), (3, 3), 1)
    states = torch.zeros(1, 1, dtype=torch.float64)
    actions = torch.tensor([[1, 0]])
    available = torch.tensor([[True, True, False], [True, True, True]])
    scale = math.log(2) + 1e-6
    importance = math.log(2) / 4 + 1e-6
    for mode in ("dueling", "dmaq"):
        mixer = DuplexDuelingMixer(
            spec,
            {
                "attention_heads": 1,
                "hypernet_size": 2,
                "gradient_mode": mode,
            },
        ).double()
        with torch.no_grad():
            for parameter in mixer.parameters():
                parameter.zero_()
        local_q = torch.tensor(
            [[[4.0, 1.0, 100.0], [2.0, 5.0, 8.0]]], dtype=torch.float64, requires_grad=True
        )
        joint = mixer(local_q, states, actions, available)
        # 合法局部最大和为 12，实际动作的局部 Q 和为 3。
        torch.testing.assert_close(joint, states.new_tensor([scale * (12 - 9 * importance)]))

        q_grad, weight_grad = torch.autograd.grad(joint.sum(), (local_q, mixer.keys[-1].bias))
        if mode == "dueling":
            selected_grad = scale * importance
            max_grad = scale * (1 - importance)
        else:
            selected_grad = scale
            max_grad = 0.0
        torch.testing.assert_close(
            q_grad,
            local_q.new_tensor([[[max_grad, selected_grad, 0], [selected_grad, 0, max_grad]]]),
        )
        torch.testing.assert_close(weight_grad, states.new_tensor([-9 * scale / 8]))


def test_qplex_gradient_mode_is_validated_and_recorded():
    def resolve(**overrides):
        config = {"env": "spread", "total_steps": 1, **overrides}
        return resolve_config(config, algorithm="qplex", algorithm_defaults=QPLEXAlgorithm.defaults)

    assert resolve()["gradient_mode"] == "dueling"
    assert training_contract(resolve(gradient_mode="dmaq"))["gradient_mode"] == "dmaq"
    with pytest.raises(ValueError, match="gradient_mode"):
        resolve(gradient_mode="unknown")


def test_qtran_constraints_mask_padding_and_detach_joint_values():
    local_chosen = torch.tensor([[1.0, 2.0], [1.0, 2.0], [0.0, 0.0]], requires_grad=True)
    local_greedy = torch.tensor([[4.0, 2.0], [1.0, 2.0], [50.0, 50.0]], requires_grad=True)
    joint_chosen = torch.tensor([5.0, 2.0, 99.0], requires_grad=True)
    joint_greedy = torch.tensor([5.0, 5.0, 0.0], requires_grad=True)
    offset = torch.zeros(3, requires_grad=True)
    filled = torch.tensor([True, True, False])

    opt, nopt = constraint_losses(
        local_chosen,
        local_greedy,
        joint_chosen,
        joint_greedy,
        offset,
        filled,
    )
    torch.testing.assert_close(opt, torch.tensor(2.5))
    torch.testing.assert_close(nopt, torch.tensor(2.0))
    (opt + nopt).backward()

    assert joint_chosen.grad is None
    assert joint_greedy.grad is None
    torch.testing.assert_close(local_chosen.grad, torch.tensor([[-2.0, -2.0], [0, 0], [0, 0]]))
    torch.testing.assert_close(local_greedy.grad, torch.tensor([[1.0, 1.0], [-2, -2], [0, 0]]))
    torch.testing.assert_close(offset.grad, torch.tensor([-1.0, -2.0, 0.0]))


def test_qtran_combines_td_bootstrap_and_weighted_constraints():
    spec = EnvironmentSpec(("a", "b"), (1, 1), (2, 2), 1)
    config = resolve_config(
        {
            "env": "spread",
            "total_steps": 3,
            "gamma": 0.5,
            "opt_loss_weight": 0.25,
            "nopt_loss_weight": 0.5,
        },
        algorithm="qtran",
        algorithm_defaults=QTRANAlgorithm.defaults,
    )
    learner = QTRANLearner(spec, config, "cpu")
    with torch.no_grad():
        for parameter in learner.parameters:
            parameter.zero_()
        learner.controller.output_layer.bias.copy_(torch.tensor([0.0, 1.0]))
        learner.mixer.joint_q[-1].bias.fill_(1.0)
        learner.sync_targets()
        learner.target_mixer.joint_q[-1].bias.fill_(4.0)

    batch = {
        "observations": torch.zeros(2, 3, 2, 1),
        "states": torch.zeros(2, 3, 1),
        "actions": torch.zeros(2, 2, 2, dtype=torch.long),
        "rewards": torch.tensor([[1.0, 100.0], [2.0, 100.0]]),
        "terminated": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "truncated": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "filled": torch.tensor([[True, False], [True, False]]),
    }
    metrics = learner.train_batch(batch)

    # 截断步保留 bootstrap 得到 3，终止步仅使用奖励 2，padding 不计入损失。
    assert metrics["valid_transitions"] == 2
    assert metrics["target_mean"] == pytest.approx(2.5)
    assert metrics["td_loss"] == pytest.approx(2.5)
    assert metrics["opt_loss"] == pytest.approx(1.0)
    assert metrics["nopt_loss"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(3.25)
    assert learner.updates == 1


def test_positive_replay_retains_success_and_excludes_duplicate_and_timeout():
    spec = EnvironmentSpec(("a",), (1,), (1,), 1)
    replay = EpisodeReplay(
        spec, capacity=2, seed=0, positive_replay_capacity=2, positive_replay_samples=1
    )

    def add(marker, reward=0.0, terminal=False):
        replay.add(
            {
                "observations": np.full((2, 1, 1), marker, dtype=np.float32),
                "states": np.full((2, 1), marker, dtype=np.float32),
                "actions": np.zeros((1, 1), dtype=np.int64),
                "rewards": np.array([reward]),
                "terminated": np.array([terminal]),
                "truncated": np.array([not terminal]),
            }
        )

    add(1)
    add(2, reward=1.0, terminal=True)
    assert set(replay.sample(2)["states"][:, 0, 0].tolist()) == {1, 2}
    add(3, reward=1.0)
    add(4)
    assert len(replay.positive_episodes) == 1
    markers = replay.sample(2)["states"][:, 0, 0].tolist()
    assert markers[0] == 2
    assert markers[1] in {3, 4}


def test_compare_rejects_mixing_qplex_variants_across_training_seeds(tmp_path):
    paths = []
    for seed, mode in enumerate(("dueling", "dmaq")):
        path = tmp_path / f"seed_{seed}.json"
        save_result(
            {
                "schema_version": 2,
                "benchmark_id": "same",
                "comparison_allowed": True,
                "training_steps": 100000,
                "training_seed": seed,
                "env": "lbf",
                "algorithm": "qplex",
                "algorithm_variant": _algorithm_variant(
                    {"algorithm": "qplex", "gradient_mode": mode}
                ),
                "benchmark": {"protocol": {"metrics": [{"name": "success"}]}},
            },
            path,
        )
        paths.append(path)
    with pytest.raises(ValueError, match="Algorithm variants differ"):
        compare_results(paths, "success")


def test_guided_lbf_training_saves_local_execution_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("lbforaging")
    algorithm = make_algorithm("qplex")
    config = {
        "env": "lbf",
        "total_steps": 60,
        "seed": 0,
        "hidden_size": 8,
        "hypernet_size": 8,
        "batch_size": 2,
        "learning_starts_episodes": 2,
        "mask_unavailable_actions": True,
        "lbf_demo_every": 1,
        "successful_action_loss_weight": 1.0,
    }
    checkpoint = algorithm.train(config, tmp_path / "guided")
    with (checkpoint.parent / "updates.csv").open(encoding="utf-8") as log:
        assert any(float(update["behavior_cloning_loss"]) > 0 for update in csv.DictReader(log))

    def unexpected_demo(env):
        raise AssertionError("Evaluation must use the learned local policy")

    monkeypatch.setattr("teamwork.algorithms.qplex.lbf_teacher.demo_actions", unexpected_demo)
    result = evaluate("lbf", checkpoint=checkpoint, episodes=1, seed=10000)
    assert result["training_steps"] == 60
    assert result["algorithm_variant"]["lbf_demo_every"] == 1
