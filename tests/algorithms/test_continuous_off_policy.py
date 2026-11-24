import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from teamwork.algorithms.continuous_off_policy import DEFAULTS, VARIANTS, Learner, Transition
from teamwork.algorithms.ppo.trainer import ContinuousActor


@pytest.mark.parametrize("algorithm", sorted(VARIANTS))
def test_continuous_updates_use_correct_critic_inputs_and_policy_delay(algorithm):
    torch.manual_seed(5)
    names = ("left", "right")
    obs_sizes = {"left": 3, "right": 4}
    action_sizes = {"left": 2, "right": 1}
    variant = VARIANTS[algorithm]
    actors = {
        name: ContinuousActor(
            obs_sizes[name],
            Box(-1, 1, shape=(action_sizes[name],), dtype=np.float32),
            stochastic=variant.soft,
        )
        for name in names
    }
    obs = {name: torch.randn(8, obs_sizes[name]) for name in names}
    actions = {name: torch.zeros(8, action_sizes[name]) for name in names}
    states = torch.randn(8, 7)
    rewards = dict.fromkeys(names, torch.full((8,), 0.3))
    terminals = dict.fromkeys(names, torch.tensor([1.0, 0.0] * 4))
    batch = Transition(obs, actions, obs, rewards, terminals, states, states)
    config = {**DEFAULTS, "algorithm": algorithm, "actor_lr": 0.001}
    learner = Learner(
        actors, {name: obs[name][0] for name in names}, states.shape[-1], config, "cpu"
    )
    assert len(learner.critics) == (1 if variant.sequential else 2)
    for name, critic in learner.critics.items():
        size = 10 if variant.centralized else obs_sizes[name] + action_sizes[name]
        assert critic.values[0][0].in_features == size
        assert len(critic.values) == (2 if variant.twin else 1)
    for target in learner._targets(batch).values():
        torch.testing.assert_close(target[::2], rewards["left"][::2])

    initial = {name: model.actor[0].weight.detach().clone() for name, model in actors.items()}
    learner.update(batch)
    assert learner.critic_updates == 1
    assert learner.actor_updates == (0 if variant.delayed else 1)
    if variant.delayed:
        for name, model in actors.items():
            torch.testing.assert_close(model.actor[0].weight, initial[name])
    learner.update(batch)
    assert learner.critic_updates == 2
    assert learner.actor_updates == (1 if variant.delayed else 2)
    assert all(
        not torch.equal(model.actor[0].weight, initial[name]) for name, model in actors.items()
    )
