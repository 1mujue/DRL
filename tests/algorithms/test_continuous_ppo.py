import torch
from torch.distributions import Normal, TanhTransform

from teamwork.algorithms.ppo.continuous import SquashedNormal


def test_gaussian_log_prob_includes_tanh_and_affine_jacobian():
    mean = torch.tensor([[0.2, -0.4], [0.3, -0.1]], requires_grad=True)
    std = torch.tensor([0.5, 1.2])
    center = torch.tensor([0.0, 1.0])
    scale = torch.tensor([1.0, 2.0])
    latent = torch.tensor([[0.4, -0.9], [-2.4, 1.1]])
    distribution = SquashedNormal(mean, std, center, scale)
    transform = TanhTransform()
    expected = (
        Normal(mean, std).log_prob(latent)
        - transform.log_abs_det_jacobian(latent, transform(latent))
        - scale.log()
    ).sum(dim=-1)
    torch.testing.assert_close(distribution.log_prob(latent), expected)
    action = distribution.action(latent)
    assert torch.all(action < center + scale)
    assert torch.all(action > center - scale)
    loss = distribution.log_prob(torch.full_like(latent, 30)).mean() + distribution.entropy().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(mean.grad).all()


def test_lift_actor_learns_feedback_and_clock_from_demonstrated_commands():
    from teamwork.algorithms.ppo.lift_actor import LiftPoseActor

    generator = torch.Generator().manual_seed(0)
    horizon = 5
    steps = torch.arange(horizon).repeat_interleave(16)
    errors = torch.randn(len(steps), 6, generator=generator)
    phase = torch.randn(horizon, 7, generator=generator)
    feedback = torch.randn(6, 7, generator=generator)
    observations = torch.cat((errors, steps[:, None] / horizon), dim=-1)
    commands = errors @ feedback + phase[steps]
    actor = LiftPoseActor(observation_size=7, horizon=horizon)
    assert actor.fit_demonstrations(observations, commands) < 1e-6
    test_errors = torch.randn(horizon, 6, generator=generator)
    unseen = torch.cat((test_errors, torch.arange(horizon)[:, None] / horizon), dim=-1)
    torch.testing.assert_close(actor(unseen), test_errors @ feedback + phase)


def test_lift_mass_transfer_requires_opt_in_and_keeps_other_settings(tmp_path):
    import pytest

    from teamwork.algorithms.ppo.trainer import initialize_from_checkpoint

    actor = torch.nn.Linear(1, 1)
    origin = {
        "algorithm": "hasac_continuous",
        "env": "two_arm_lift",
        "env_kwargs": {"horizon": 200, "reward_shaping": True, "pose_feedback": True},
        "actor_type": "lift_pose",
        "behavior_cloning": {"method": "behavior_cloning"},
    }
    checkpoint = tmp_path / "demo.pt"
    torch.save({"config": origin, "agents": {"a": actor.state_dict()}}, checkpoint)
    target = {
        **origin,
        "warm_start_checkpoint": str(checkpoint),
        "env_kwargs": {**origin["env_kwargs"], "pot_mass_range": [3.0, 6.5]},
    }
    with pytest.raises(ValueError, match="env_kwargs"):
        initialize_from_checkpoint({"a": actor}, target)
    target["allow_lift_mass_transfer"] = True
    initialize_from_checkpoint({"a": actor}, target)
    assert target["behavior_cloning"] == origin["behavior_cloning"]
    target["env_kwargs"]["horizon"] = 199
    with pytest.raises(ValueError, match="env_kwargs"):
        initialize_from_checkpoint({"a": actor}, target)
