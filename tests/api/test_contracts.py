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
    assert {a["id"] for a in info["algorithms"]} == {"ippo", "iql"}
    assert {e["id"] for e in info["environments"]} == {"spread", "tag", "lbf", "rware"}
    info["algorithms"][0]["defaults"]["lr"] = -1
    assert catalog()["algorithms"][0]["defaults"]["lr"] > 0
    request = TrainingRequest("ippo", parameters={"epochs": 2})
    config, spec = request.validate()
    assert config["epochs"] == 2
    assert config["rollout_steps"] == 256
    assert len(spec["agents"]) == 3
    assert request.to_dict()["parameters"] == {"epochs": 2}
