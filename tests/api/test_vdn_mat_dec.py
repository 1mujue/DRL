"""Regression coverage for VDN and MAT-Dec through the public SDK."""

import pytest
import torch

from teamwork.api import AlgorithmAPI, AlgorithmAPIError, TrainingRequest
from teamwork.environments.registry import make_env


@pytest.mark.parametrize(
    "algorithm,parameters,expected_updates",
    [
        ("vdn", {"batch_size": 4, "warmup_steps": 4, "hidden_size": 8}, 5),
        (
            "mat_dec",
            {"rollout_steps": 4, "batch_size": 4, "epochs": 1, "embedding_dim": 8},
            2,
        ),
    ],
)
def test_cooperative_algorithms_through_sdk(tmp_path, algorithm, parameters, expected_updates):
    with AlgorithmAPI(tmp_path) as api:
        record = api.train(
            TrainingRequest(
                algorithm,
                steps=8,
                parameters=parameters,
                env_kwargs={"max_cycles": 2},
            ),
            timeout=60,
        )
        run_id = record["run_id"]
        assert record["status"] == "succeeded"
        artifacts = api.artifacts(run_id)
        data = torch.load(artifacts["checkpoint"]["path"], weights_only=True)
        assert data["updates"] == expected_updates
        env = make_env("spread", max_cycles=2)
        try:
            observations, _ = env.reset(seed=0)
            with api.open_policy(run_id, seed=7) as policy:
                actions = policy.predict(observations)
                assert all(env.action_space(a).contains(v) for a, v in actions.items())
                assert policy.predict_batch([observations, observations]) == [actions, actions]
        finally:
            env.close()
        result = api.evaluate(run_id, {"episodes": 1})["result"]
        assert result["algorithm"] == algorithm
        assert result["episodes"][0]["steps"] == 2

        # Feed real trainer output through the running-task parser. Completed tasks
        # override parsed steps, which would conceal an incompatible log format.
        pending_id = api.store.create({"total_steps": 16}, {}, "progress-test")
        api.store.update(pending_id, status="running")
        (api.store.path(pending_id) / "train.log").write_text(api.logs(run_id), encoding="utf-8")
        progress = api.status(pending_id)["progress"]
        assert progress == {"logged_steps": 8, "requested_steps": 16, "fraction": 0.5}


@pytest.mark.parametrize("clip", [1.0, 2.0])
def test_mat_dec_invalid_clip_creates_no_task(tmp_path, clip):
    with AlgorithmAPI(tmp_path) as api:
        with pytest.raises(AlgorithmAPIError):
            api.submit_training(TrainingRequest("mat_dec", parameters={"clip": clip}))
        assert api.list_runs() == []
