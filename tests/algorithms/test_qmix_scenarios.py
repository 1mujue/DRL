"""QMIX 合作场景的直接 API 验收；短程验证不代表任务已学会。"""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from teamwork.algorithms import make_algorithm
from teamwork.environments.registry import make_env

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "algorithms" / "qmix"


def test_rware_config_accounts_for_long_episodes():
    config = yaml.safe_load((CONFIGS / "rware.yaml").read_text(encoding="utf-8"))
    assert config["env_kwargs"]["max_steps"] == 500
    # 验证给定预算确实允许预热后同步目标网络，而非直到结束都用初始目标。
    episodes = config["total_steps"] // config["env_kwargs"]["max_steps"]
    updates = (episodes - config["learning_starts_episodes"] + 1) * config["updates_per_episode"]
    assert updates // config["target_update_interval"] >= 2
    assert config["use_rnn"] is True


@pytest.mark.parametrize("scenario", ["spread", "lbf", "lbf_partial", "rware"])
def test_qmix_scenario_trains_and_deploys_through_public_api(tmp_path, scenario):
    config = yaml.safe_load((CONFIGS / f"{scenario}.yaml").read_text(encoding="utf-8"))
    family = config["env"]
    package = {"lbf": "lbforaging", "rware": "rware"}.get(family)
    if package and importlib.util.find_spec(package) is None:
        pytest.skip(f"optional environment {package} not installed")
    horizon = {"spread": "max_cycles", "lbf": "max_episode_steps", "rware": "max_steps"}[family]
    config["env_kwargs"][horizon] = 3
    config.update(total_steps=7, batch_size=2, learning_starts_episodes=2,
                  hidden_size=8, mixing_size=4, hypernet_size=8, checkpoint_interval=0)
    algorithm = make_algorithm(config["algorithm"])
    checkpoint = algorithm.train(config, tmp_path / scenario)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["steps"] == 7 and saved["updates"] > 0
    assert saved["config"]["env_kwargs"] == config["env_kwargs"]
    assert saved["training_contract"]["reward_aggregation"] == "mean"
    env = make_env(family, **config["env_kwargs"])
    try:
        observations, _ = env.reset(seed=10000)
        deployed = algorithm.load_policy(checkpoint)
        checked = algorithm.load_policy(checkpoint, env, observations)
        actions = deployed.act(observations)
        assert checked.act(observations) == actions
        assert all(env.action_space(agent).contains(action) for agent, action in actions.items())
        deployed.act(observations)
        deployed.reset()
        assert deployed.act(observations) == actions
    finally:
        env.close()
