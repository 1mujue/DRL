"""MAPPO 场景配置的可比条件和实际 API 链路，不检验训练收敛。"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from teamwork.algorithms import make_algorithm
from teamwork.environments.registry import make_env

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "algorithms"
SCENARIOS = ("spread", "lbf", "lbf_partial", "rware")


def load_config(method, scenario):
    return yaml.safe_load((CONFIGS / method / f"{scenario}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_paired_scenarios_keep_the_same_task_reward_and_budget(scenario):
    mappo, qmix = (load_config(method, scenario) for method in ("mappo", "qmix"))
    for field in (
        "env", "env_kwargs", "seed", "total_steps", "reward_aggregation", "gamma",
        "checkpoint_interval",
    ):
        assert mappo[field] == qmix[field], field
    assert mappo["algorithm"] == "mappo_ff" and qmix["algorithm"] == "qmix"
    assert mappo["reward_aggregation"] == "mean"
    assert mappo["total_steps"] == 100000


def test_lbf_visibility_variant_changes_only_the_information_condition():
    for method in ("mappo", "qmix"):
        full, partial = (load_config(method, name) for name in ("lbf", "lbf_partial"))
        assert full["env_kwargs"]["sight"] == 8
        assert partial["env_kwargs"]["sight"] == 2
        partial["env_kwargs"]["sight"] = 8
        assert full == partial


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_mappo_scenario_trains_and_deploys_through_public_api(tmp_path, scenario):
    config = load_config("mappo", scenario)
    family = config["env"]
    package = {"lbf": "lbforaging", "rware": "rware"}.get(family)
    if package and importlib.util.find_spec(package) is None:
        pytest.skip(f"optional environment {package} not installed")
    # 只缩短测试回合和预算；场景原配置的视野、agent、奖励条件仍保留。
    horizon = {"spread": "max_cycles", "lbf": "max_episode_steps", "rware": "max_steps"}[family]
    config["env_kwargs"][horizon] = 3
    config.update(total_steps=7, rollout_steps=4, epochs=1, hidden_size=8,
                  batch_size=16, checkpoint_interval=0)
    algorithm = make_algorithm(config["algorithm"])
    checkpoint = algorithm.train(config, tmp_path / scenario)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["steps"] == 7
    assert saved["config"]["env_kwargs"] == config["env_kwargs"]
    contract = json.loads((checkpoint.parent / "training_contract.json").read_text("utf-8"))
    assert contract["reward_aggregation"] == "mean"
    env = make_env(family, **config["env_kwargs"])
    try:
        observations, _ = env.reset(seed=10000)
        deployed = algorithm.load_policy(checkpoint)
        checked = algorithm.load_policy(checkpoint, env, observations)
        actions = deployed.act(observations)
        assert checked.act(observations) == actions
        assert all(env.action_space(agent).contains(action) for agent, action in actions.items())
    finally:
        env.close()
