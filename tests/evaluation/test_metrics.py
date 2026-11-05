"""用可手算的场景验证指标含义，避免各算法产生不同的成功率口径。"""

import importlib.util
from copy import deepcopy

import numpy as np
import pytest

from teamwork.evaluation.compare import compare_results
from teamwork.evaluation.evaluator import evaluate, save_result
from teamwork.evaluation.metrics import (
    LBFMetrics,
    RWAREMetrics,
    SpreadMetrics,
    TagMetrics,
    summarize,
)


def snapshot(positions, landmarks=(), predators=None, sizes=None):
    return {
        "positions": np.array(positions, dtype=float),
        "landmarks": np.array(landmarks, dtype=float),
        "sizes": np.array(sizes if sizes is not None else [0.05] * len(positions)),
        "predators": np.array(predators if predators is not None else [False] * len(positions)),
    }


def test_spread_final_success_differs_from_ever_success():
    suite = SpreadMetrics()
    covered = snapshot([[0, 0], [1, 0]], [[0, 0], [1, 0]])
    suite.reset(covered)
    suite.observe(covered)
    suite.observe(snapshot([[0, 0], [2, 0]], [[0, 0], [1, 0]]))
    result = suite.finish({"a": 2, "b": 4}, ["a", "b"])
    assert result["success"] == 0
    assert result["ever_success"] == 1
    assert result["final_coverage"] == 0.5
    assert result["final_mean_landmark_distance"] == 0.5
    assert result["mean_agent_return"] == 3


def test_spread_contact_is_unordered_pair_per_step_and_radius_is_strict():
    suite = SpreadMetrics()
    s = snapshot([[0, 0], [0, 0]], [[0.1, 0]])
    suite.reset(s)
    suite.observe(s)
    suite.observe(s)
    result = suite.finish({"a": 0, "b": 0}, ["a", "b"])
    assert result["collision_pairs_per_step"] == 1
    assert result["final_coverage"] == 0


def test_tag_counts_distinct_prey_across_steps_and_roles_separately():
    suite = TagMetrics()
    first = snapshot([[0, 0], [0.01, 0], [5, 0]], predators=[True, False, False])
    suite.reset(first)
    suite.observe(first)
    suite.observe(snapshot([[5, 0], [0, 0], [5.01, 0]], predators=[True, False, False]))
    result = suite.finish({"p": 10, "a": -2, "b": -4}, ["p", "a", "b"])
    assert result["capture_success"] == 1
    assert result["captured_prey_fraction"] == 1
    assert result["first_contact_step"] == 1
    assert result["contact_pairs_per_step"] == 1
    assert result["prey_outside_rate"] == 0.5
    assert result["predator_return"] == 10
    assert result["prey_return"] == -3


def test_failure_times_are_null_and_not_zero():
    suite = LBFMetrics()
    suite.reset({"food_remaining": 3})
    suite.observe({"food_remaining": 1})
    result = suite.finish({"a": 0}, ["a"])
    assert result["food_collected_fraction"] == pytest.approx(2 / 3)
    assert result["completion_step"] is None
    summary = summarize([{"metrics": result}])
    assert summary["completion_step"]["mean"] is None
    assert summary["completion_step"]["n"] == 0
    suite.observe({"food_remaining": 0})
    assert suite.finish({"a": 0}, ["a"])["completion_step"] == 2


def test_rware_throughput_uses_actual_steps():
    suite = RWAREMetrics()
    suite.reset({"deliveries_this_step": 0})
    suite.observe({"deliveries_this_step": 2})
    suite.observe({"deliveries_this_step": 1})
    result = suite.finish({"a": 3}, ["a"])
    assert result["deliveries"] == 3
    assert result["deliveries_per_100_steps"] == 150


def write_results(tmp_path, *results):
    paths = []
    for i, result in enumerate(results):
        path = tmp_path / f"result{i}.json"
        save_result(result, path)
        paths.append(path)
    return paths


def test_compare_checks_protocol_and_independent_training_seeds(tmp_path):
    a = evaluate("spread", episodes=2, env_kwargs={"max_cycles": 2})
    b = deepcopy(a)
    a.update(algorithm="test_algorithm", training_seed=0, training_steps=100)
    b.update(algorithm="test_algorithm", training_seed=1, training_steps=100)
    a["summary"]["success"]["mean"] = 0
    b["summary"]["success"]["mean"] = 1
    report = compare_results(write_results(tmp_path, a, b), "success")
    assert report["algorithms"]["test_algorithm"]["mean"] == 0.5
    assert report["algorithms"]["test_algorithm"]["n_models"] == 2
    assert report["algorithms"]["test_algorithm"]["std_across_training_seeds"] == pytest.approx(
        2**-0.5
    )
    with pytest.raises(ValueError, match="Duplicate training seed"):
        compare_results(write_results(tmp_path, a, a), "success")
    c = evaluate("spread", episodes=2, env_kwargs={"max_cycles": 3})
    with pytest.raises(ValueError, match="Incomparable"):
        compare_results(write_results(tmp_path, a, c), "success")
    b["training_steps"] = 200
    with pytest.raises(ValueError, match="budgets"):
        compare_results(write_results(tmp_path, a, b), "success")


def test_tag_comparison_requires_fixed_opponent(tmp_path):
    joint = evaluate("tag", episodes=1, env_kwargs={"max_cycles": 2})
    with pytest.raises(ValueError, match="fixed opponent"):
        compare_results(write_results(tmp_path, joint), "capture_success")
    fixed = evaluate(
        "tag", episodes=1, env_kwargs={"max_cycles": 2}, team="predators", opponent="rule"
    )
    assert compare_results(write_results(tmp_path, fixed), "capture_success")["env"] == "tag"


@pytest.mark.parametrize("name", ["spread", "tag", "lbf", "rware"])
def test_all_environments_have_complete_fixed_protocol(name):
    if (
        name in {"lbf", "rware"}
        and importlib.util.find_spec("lbforaging" if name == "lbf" else "rware") is None
    ):
        pytest.skip("optional environment not installed")
    result = evaluate(name, episodes=1)
    assert result["schema_version"] == 2
    assert set(result["summary"]) == {d["name"] for d in result["benchmark"]["protocol"]["metrics"]}
    assert len(result["benchmark_id"]) == 64
