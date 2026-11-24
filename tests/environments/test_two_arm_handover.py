import numpy as np
import pytest

from teamwork.environments.base import ContinuousEnvironmentSpec
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import evaluate


def test_handover_uses_stable_seed_and_records_scope_snapshot():
    pytest.importorskip("robosuite")
    with_model = make_env("two_arm_handover", scene_seed=7, horizon=3)
    try:
        initial, _ = with_model.reset(seed=2)
        with_model.step({arm: with_model.action_space(arm).sample() for arm in with_model.agents})
        repeated, _ = with_model.reset(seed=2)
        for arm in initial:
            np.testing.assert_array_equal(initial[arm], repeated[arm])
        assert isinstance(with_model.describe(), ContinuousEnvironmentSpec)
    finally:
        with_model.close()
    report = evaluate("two_arm_handover", env_kwargs={"horizon": 3}, episodes=1, record_trace=True)
    row = report["episodes"][0]
    assert row["steps"] == 3
    assert len(row["trace"]["samples"]) == 4
    assert row["trace"]["control_hz"] == 20
    assert "handover_success" in report["summary"]


def test_scripted_handover_records_finite_successful_transfer(tmp_path):
    pytest.importorskip("robosuite")
    from teamwork.evaluation.evaluator import save_result
    from teamwork.evaluation.metrics import HANDOVER_HEIGHT_M, HANDOVER_HOLD_STEPS

    report = evaluate(
        "two_arm_handover",
        policy="scripted_handover",
        env_kwargs={"horizon": 400, "scene_seed": 0, "pose_feedback": True},
        seed=67001,
        episodes=1,
        record_trace=True,
    )
    row = report["episodes"][0]
    assert row["metrics"]["handover_success"] == 1
    first = row["metrics"]["first_handover_step"]
    samples = row["trace"]["samples"]
    giver = next(
        t
        for t, sample in enumerate(samples)
        if sample["giver_grasp"] and not sample["receiver_grasp"]
    )
    overlap = next(
        t for t, sample in enumerate(samples) if sample["giver_grasp"] and sample["receiver_grasp"]
    )
    hold_start = first - HANDOVER_HOLD_STEPS + 1
    assert 0 < giver < overlap < hold_start
    assert all(
        sample["receiver_grasp"]
        and not sample["giver_grasp"]
        and sample["object_height_m"] > HANDOVER_HEIGHT_M
        for sample in samples[hold_start : first + 1]
    )
    save_result(report, tmp_path / "handover.json")
