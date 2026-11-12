import torch

from teamwork.algorithms.maac import trainer as maac_trainer
from teamwork.algorithms.registry import available_algorithms, make_algorithm


def test_registered():
    assert "maac" in available_algorithms()
    assert make_algorithm("maac").__class__.__name__ == "MAACAlgorithm"


def test_attention_critic_shape():
    n_agents, obs_dim, n_actions, B = 3, 6, 4, 8
    critic = maac_trainer.AttentionCritic(0, n_agents, obs_dim, n_actions,
                                          embed_dim=16, n_heads=4, hidden=32)
    all_obs = torch.randn(B, n_agents, obs_dim)
    all_act_oh = torch.zeros(B, n_agents, n_actions)
    all_act_oh.scatter_(-1, torch.randint(0, n_actions, (B, n_agents, 1)), 1.0)
    q = critic(all_obs, all_act_oh)
    assert q.shape == (B,)


def test_update_step_runs():
    names = ["a0", "a1", "a2"]
    obs_dim, n_actions, B = 6, 4, 16
    actors = {a: maac_trainer.MAACActor(obs_dim, n_actions, 16) for a in names}
    critics = {
        a: maac_trainer.AttentionCritic(i, len(names), obs_dim, n_actions, 16, 4, 32)
        for i, a in enumerate(names)
    }
    target_actors = {a: maac_trainer.MAACActor(obs_dim, n_actions, 16) for a in names}
    target_critics = {
        a: maac_trainer.AttentionCritic(i, len(names), obs_dim, n_actions, 16, 4, 32)
        for i, a in enumerate(names)
    }
    for a in names:
        target_actors[a].load_state_dict(actors[a].state_dict())
        target_critics[a].load_state_dict(critics[a].state_dict())
    opts_actor = {a: torch.optim.Adam(actors[a].parameters(), lr=1e-3) for a in names}
    opts_critic = {a: torch.optim.Adam(critics[a].parameters(), lr=1e-3) for a in names}

    # 注意这里的键名与 maac/trainer.py 里的 maac_update 严格对齐
    batch = {
        "all_obs": torch.randn(B, len(names), obs_dim),
        "all_next_obs": torch.randn(B, len(names), obs_dim),
        "acts": torch.randint(0, n_actions, (B, len(names))),
        "rew": torch.randn(B),
        "term": torch.zeros(B),
    }
    cfg = {"_names": names, "_n_actions": n_actions,
           "gamma": 0.99, "tau": 0.005, "alpha": 0.5}
           
    stats = maac_trainer.maac_update(
        actors, critics, target_actors, target_critics,
        opts_actor, opts_critic, batch, cfg,
    )
    assert any("critic_loss" in k for k in stats)
    assert any("actor_loss" in k for k in stats)