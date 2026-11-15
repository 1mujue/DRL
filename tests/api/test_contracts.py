import math

import pytest

from teamwork.api import AlgorithmAPIError, EvaluationRequest, TrainingRequest
from teamwork.api.catalog import catalog


@pytest.mark.parametrize(
    "change",
    [
        {"algorithm": "unknown"},
        {"env": "unknown"},
        {"steps": 0},
        {"steps": True},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.2},
        {"device": "mps"},
        {"parameters": []},
        {"parameters": {"learing_rate": 1}},
        {"parameters": {"lr": 0}},
        {"parameters": {"gamma": 1.1}},
        {"parameters": {"lr": math.nan}},
        {"parameters": {"lr": True}},
        {"parameters": {"buffer_size": 2.1}},
        {"parameters": {"epsilon_start": 0.1, "epsilon_end": 0.5}},
        {"parameters": {"buffer_size": 2, "batch_size": 3}},
        {"env": "tag"},
        {"team": "predators", "opponent": "rule"},
        {"env_kwargs": []},
        {"env_kwargs": {"max_cycles": math.inf}},
    ],
)
def test_bad_training_request_fails_with_serializable_error(change):
    with pytest.raises(AlgorithmAPIError) as caught:
        TrainingRequest.from_dict({"algorithm": "iql", **change}).validate()
    error = caught.value.to_dict()
    assert error["code"] == "INVALID_REQUEST"
    assert error["details"]["field"]


@pytest.mark.parametrize("payload", [{}, {"algorithm": "iql", "step": 1}, []])
def test_missing_and_unknown_request_fields(payload):
    with pytest.raises(AlgorithmAPIError, match="fields|argument|object"):
        TrainingRequest.from_dict(payload)


def test_environment_options_are_validated_before_submitting():
    with pytest.raises(AlgorithmAPIError, match="initialization failed"):
        TrainingRequest("ippo", env_kwargs={"nonexistent_option": True}).validate()


@pytest.mark.parametrize("payload", [{"episodes": 0}, {"episodes": True}, {"seed": -1}])
def test_bad_evaluation_request(payload):
    with pytest.raises(AlgorithmAPIError):
        EvaluationRequest.from_dict(payload).validate()


def test_catalog_and_requests_return_independent_data():
    info = catalog()
    assert {a["id"] for a in info["algorithms"]} == {
        "ippo", "iql", "happo", "hasac_discrete", "vdn", "mat_dec"
    }
    assert {e["id"] for e in info["environments"]} == {"spread", "tag", "lbf", "rware"}
    info["algorithms"][0]["defaults"]["lr"] = -1
    assert catalog()["algorithms"][0]["defaults"]["lr"] > 0
    request = TrainingRequest("ippo", parameters={"epochs": 2})
    config, spec = request.validate()
    assert config["epochs"] == 2
    assert config["rollout_steps"] == 256
    assert len(spec["agents"]) == 3
    assert request.to_dict()["parameters"] == {"epochs": 2}


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        (
            "vdn",
            {"batch_size": 2, "replay_capacity": 8, "warmup_steps": 0},
        ),
        (
            "mat_dec",
            {"embedding_dim": 8, "num_heads": 2, "rollout_steps": 4, "batch_size": 2},
        ),
    ],
)
def test_new_algorithm_training_requests_resolve_defaults(algorithm, parameters):
    config, spec = TrainingRequest(
        algorithm, steps=8, parameters=parameters, env_kwargs={"max_cycles": 2}
    ).validate()
    assert config["algorithm"] == algorithm
    assert len(spec["agents"]) == 3


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        ("vdn", {"batch_size": 8, "replay_capacity": 4}),
        ("vdn", {"epsilon_start": 0.1, "epsilon_end": 0.2}),
        ("mat_dec", {"embedding_dim": 7, "num_heads": 2}),
        ("mat_dec", {"entropy_coef": -0.1}),
        ("mat_dec", {"clip": 1.0}),
        ("mat_dec", {"clip": 2.0}),
    ],
)
def test_new_algorithm_invalid_parameter_relations_fail_before_training(algorithm, parameters):
    with pytest.raises(AlgorithmAPIError):
        TrainingRequest(algorithm, parameters=parameters).config()


@pytest.mark.parametrize("algorithm", ["vdn", "mat_dec"])
def test_cooperative_algorithms_reject_tag_before_task_creation(algorithm):
    with pytest.raises(AlgorithmAPIError, match="cooperative"):
        TrainingRequest(algorithm, env="tag", team="predators", opponent="rule").config()
