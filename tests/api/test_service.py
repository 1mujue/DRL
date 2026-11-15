import importlib.util
import json
import sys

import pytest
import torch

from teamwork.api import AlgorithmAPI, AlgorithmAPIError, EvaluationRequest, TrainingRequest
from teamwork.environments.registry import make_env


def request(algorithm="iql", environment="spread", **changes):
    parameters = (
        {
            "learning_starts": 4,
            "batch_size": 4,
            "buffer_size": 64,
            "train_interval": 2,
            "target_update_interval": 8,
        }
        if algorithm == "iql"
        else {"rollout_steps": 16, "batch_size": 8, "epochs": 1}
    )
    tag = {"team": "predators", "opponent": "rule"} if environment == "tag" else {}
    return TrainingRequest.from_dict(
        {
            "algorithm": algorithm,
            "env": environment,
            "steps": 64,
            "parameters": parameters,
            **tag,
            **changes,
        }
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    api = AlgorithmAPI(tmp_path_factory.mktemp("sdk"))
    records = {}
    try:
        for environment in ("spread", "tag", "lbf", "rware"):
            if (
                environment in {"lbf", "rware"}
                and importlib.util.find_spec("lbforaging" if environment == "lbf" else "rware")
                is None
            ):
                continue
            for algorithm in ("ippo", "iql"):
                records[algorithm, environment] = api.train(
                    request(algorithm, environment), timeout=60
                )
        yield api, records
    finally:
        api.close()


@pytest.mark.parametrize("environment", ["spread", "tag", "lbf", "rware"])
@pytest.mark.parametrize("algorithm", ["ippo", "iql"])
def test_train_load_predict_batch_evaluate_and_artifacts(trained, algorithm, environment):
    api, records = trained
    if (algorithm, environment) not in records:
        pytest.skip("optional environment not installed")
    record = records[algorithm, environment]
    run_id = record["run_id"]
    assert record["status"] == "succeeded"
    assert record["progress"]["logged_steps"] == 64
    assert record["progress"]["fraction"] == 1
    assert record["return_code"] == 0
    artifacts = api.artifacts(run_id)
    assert {"checkpoint", "config", "training", "provenance", "log"} <= artifacts.keys()
    assert all(item["bytes"] > 0 for item in artifacts.values())
    env = make_env(environment, **record["config"]["env_kwargs"])
    try:
        obs, _ = env.reset(seed=42)
        with api.open_policy(run_id, seed=7) as policy:
            expected = 3 if environment == "tag" else len(env.possible_agents)
            assert len(policy.agents) == expected
            actions = policy.predict(obs)
            assert all(env.action_space(a).contains(v) for a, v in actions.items())
            assert policy.predict_batch([obs, obs]) == [actions, actions]
            policy.reset()
            assert api.predict(run_id, obs) == actions
        with pytest.raises(AlgorithmAPIError, match="closed"):
            policy.predict(obs)
    finally:
        env.close()
    evaluation = api.evaluate(run_id, EvaluationRequest(episodes=1))
    assert len(evaluation["result"]["episodes"]) == 1
    assert evaluation["result"]["comparison_allowed"]
    assert api.evaluation(run_id, evaluation["evaluation_id"]) == evaluation["result"]
    assert api.status(run_id)["evaluations"][-1]["evaluation_id"] == evaluation["evaluation_id"]
    if algorithm == "iql":
        data = torch.load(artifacts["checkpoint"]["path"], weights_only=True)
        assert all(n > 0 for n in data["updates"].values())


def test_prediction_validation_and_randomness_isolation(trained):
    api, records = trained
    record = records["ippo", "spread"]
    env = make_env("spread", **record["config"]["env_kwargs"])
    try:
        obs, _ = env.reset(seed=0)
        before_loading = torch.get_rng_state().clone()
        with api.open_policy(record["run_id"], seed=5) as policy:
            torch.testing.assert_close(torch.get_rng_state(), before_loading, rtol=0, atol=0)
            for bad in (
                {},
                {"unknown": []},
                {**obs, "agent_0": [float("nan")]},
                {**obs, "agent_0": [[1]]},
                {**obs, 1: []},
            ):
                with pytest.raises(AlgorithmAPIError, match="agents|values|vector"):
                    policy.predict(bad)
            with pytest.raises(AlgorithmAPIError, match="boolean"):
                policy.predict(obs, deterministic=1)
            with pytest.raises(AlgorithmAPIError, match="nonempty"):
                policy.predict_batch([])
            state = torch.get_rng_state().clone()
            first = policy.predict_batch([obs] * 10, deterministic=False)
            torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
            policy.reset(seed=5)
            assert policy.predict_batch([obs] * 10, deterministic=False) == first
            before = policy.rng_state.clone()
            with pytest.raises(AlgorithmAPIError):
                policy.predict_batch([obs, {}], deterministic=False)
            torch.testing.assert_close(policy.rng_state, before, rtol=0, atol=0)
    finally:
        env.close()


def test_comparison_export_and_read_completed_history(trained, tmp_path):
    api, records = trained
    refs = [api.evaluate(records[a, "spread"]["run_id"], {"episodes": 2}) for a in ("ippo", "iql")]
    result = api.compare(refs, "final_coverage")
    assert set(result["algorithms"]) == {"ippo", "iql"}
    with pytest.raises(AlgorithmAPIError, match="Duplicate training seed"):
        api.compare([refs[0], refs[0]], "success")
    with pytest.raises(AlgorithmAPIError, match="Unknown metric"):
        api.compare(refs, "no_metric")
    different = api.evaluate(refs[0]["run_id"], {"episodes": 3})
    with pytest.raises(AlgorithmAPIError, match="Incomparable"):
        api.compare([refs[0], different], "success")
    export = tmp_path / "handoff.json"
    api.export_manifest(refs[0]["run_id"], export)
    assert json.loads(export.read_text())["run"]["run_id"] == refs[0]["run_id"]
    with AlgorithmAPI(api.store.root) as reopened:
        assert len(reopened.list_runs()) == len(records)
        assert reopened.status(refs[0]["run_id"])["status"] == "succeeded"


def test_bad_ids_and_missing_artifacts_are_structured(trained):
    api, records = trained
    with pytest.raises(AlgorithmAPIError) as caught:
        api.status("../secret")
    assert caught.value.code == "INVALID_REQUEST"
    with pytest.raises(AlgorithmAPIError) as caught:
        api.status("run_" + "0" * 32)
    assert caught.value.code == "RUN_NOT_FOUND"
    run_id = records["ippo", "spread"]["run_id"]
    with pytest.raises(AlgorithmAPIError) as caught:
        api.store.artifact(run_id, "updates")
    assert caught.value.code == "ARTIFACT_NOT_FOUND"
    with pytest.raises(AlgorithmAPIError):
        api.evaluation(run_id, "../outside")


def test_timeout_capacity_cancel_and_context_cleanup(tmp_path):
    with AlgorithmAPI(tmp_path) as api:
        record = api.submit_training(request(steps=10000000))
        run_id = record["run_id"]
        with pytest.raises(AlgorithmAPIError) as caught:
            api.wait(run_id, timeout=0)
        assert caught.value.code == "TIMEOUT"
        with pytest.raises(AlgorithmAPIError) as caught:
            api.submit_training(request())
        assert caught.value.code == "BUSY"
        assert len(api.list_runs()) == 1
        with pytest.raises(AlgorithmAPIError) as caught:
            api.open_policy(run_id)
        assert caught.value.code == "RUN_NOT_READY"
        cancelled = api.cancel(run_id)
        assert cancelled["status"] == "cancelled"
        assert cancelled["return_code"] is not None
        assert api.cancel(run_id)["status"] == "cancelled"
        new = api.submit_training(request(steps=10000000))
    assert api.tasks.store.read(new["run_id"])["status"] == "cancelled"
    with pytest.raises(AlgorithmAPIError) as caught:
        api.submit_training(request())
    assert caught.value.code == "API_CLOSED"


def test_subprocess_failure_preserves_logs_and_error(tmp_path, monkeypatch):
    from teamwork.api import tasks

    original = tasks.subprocess.Popen

    def fail(command, **kwargs):
        return original(
            [sys.executable, "-c", "print('worker failed'); raise SystemExit(7)"], **kwargs
        )

    monkeypatch.setattr(tasks.subprocess, "Popen", fail)
    with AlgorithmAPI(tmp_path) as api:
        record = api.submit_training(request())
        record = api.wait(record["run_id"], timeout=10)
        assert record["status"] == "failed"
        assert record["return_code"] == 7
        assert "worker failed" in api.logs(record["run_id"])
        assert record["error"]["code"] == "TRAINING_FAILED"
        with pytest.raises(AlgorithmAPIError) as caught:
            api.train(request(), timeout=10)
        assert caught.value.code == "TRAINING_FAILED"


def test_validation_creates_no_run_and_does_not_reseed_caller(tmp_path):
    with AlgorithmAPI(tmp_path) as api:
        torch.manual_seed(33)
        rng = torch.get_rng_state().clone()
        config = api.validate_training(request())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        assert config["config"]["algorithm"] == "iql"
        with pytest.raises(AlgorithmAPIError):
            api.submit_training({"algorithm": "iql", "parameters": {"typo": 3}})
        assert api.list_runs() == []


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_bad_training_timeout_creates_no_task(tmp_path, timeout):
    with AlgorithmAPI(tmp_path) as api:
        with pytest.raises(AlgorithmAPIError):
            api.train(request(), timeout=timeout)
        assert api.list_runs() == []


def test_process_start_failure_is_recorded(tmp_path, monkeypatch):
    from teamwork.api import tasks

    def cannot_start(*args, **kwargs):
        raise OSError("process start rejected")

    monkeypatch.setattr(tasks.subprocess, "Popen", cannot_start)
    with AlgorithmAPI(tmp_path) as api:
        with pytest.raises(AlgorithmAPIError) as caught:
            api.submit_training(request())
        assert caught.value.code == "START_FAILED"
        records = api.list_runs()
        assert len(records) == 1
        assert records[0]["status"] == "failed"
        assert records[0]["error"]["code"] == "START_FAILED"


def test_evaluation_timeout_kills_process_without_registering_result(trained):
    api, records = trained
    run_id = records["ippo", "spread"]["run_id"]
    before = len(api.status(run_id)["evaluations"])
    with pytest.raises(AlgorithmAPIError) as caught:
        api.evaluate(run_id, {"episodes": 10000}, timeout=0.001)
    assert caught.value.code == "TIMEOUT"
    assert len(api.status(run_id)["evaluations"]) == before
