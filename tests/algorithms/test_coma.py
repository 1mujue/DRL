import numpy as np
import torch

from teamwork.algorithms.coma import trainer as coma_trainer
from teamwork.algorithms.registry import available_algorithms, make_algorithm


def test_registered():
    assert "coma" in available_algorithms()
    assert make_algorithm("coma").__class__.__name__ == "COMAAlgorithm"


def test_update_step_runs():
    agents = ["a0", "a1"]
    obs_dim, n_actions, B = 6, 3, 8
    actors = {a: coma_trainer.COMAActor(obs_dim, n_actions) for a in agents}
    critic = coma_trainer.COMACritic(len(agents), obs_dim, n_actions)
    target = coma_trainer.COMACritic(len(agents), obs_dim, n_actions)
    target.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam([p for a in agents for p in actors[a].parameters()], lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # 注意这里的键名：act, rew, term 需要与 trainer.py 里的 coma_update 严格对齐
    batch = {
        "obs": {a: torch.randn(B, obs_dim) for a in agents},
        "act": {a: torch.randint(0, n_actions, (B,)) for a in agents},
        "rew": torch.randn(B),
        "next_obs": {a: torch.randn(B, obs_dim) for a in agents},
        "term": torch.zeros(B),
    }
    # 注意这里的 _names 和 _n_actions
    cfg = {"_names": agents, "_n_actions": n_actions, "gamma": 0.99, "tau": 0.005}
    
    stats = coma_trainer.coma_update(actors, critic, target, opt_a, opt_c, batch, cfg)
    assert set(stats) == {"critic_loss", "actor_loss", "q_mean"}


def test_save_load_roundtrip(tmp_path):
    agents = ["a0", "a1"]
    obs_dim, n_actions = 6, 3
    actors = {a: coma_trainer.COMAActor(obs_dim, n_actions) for a in agents}
    critic = coma_trainer.COMACritic(len(agents), obs_dim, n_actions)
    
    # 注意这里 agents 存的是 state_dict 字典，而不是列表，以匹配 load_policy
    ckpt = {
        "config": {"algorithm": "coma", "actor_hidden": 64},
        "obs_dim": obs_dim, 
        "n_actions": n_actions,
        "agents": {a: actors[a].state_dict() for a in agents},
        "critic": critic.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(ckpt, path)
    
    loaded, data = coma_trainer.load_policy(path, None, {a: np.zeros(obs_dim) for a in agents})
    assert data["config"]["algorithm"] == "coma"
    assert set(loaded) == set(agents)