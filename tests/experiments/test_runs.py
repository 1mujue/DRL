import json
from pathlib import Path

import pytest

from teamwork.algorithms.base import Algorithm
from teamwork.algorithms.registry import register_algorithm
from teamwork.experiments.runner import run_training
from teamwork.experiments.runs import allocate_run_dir, atomic_write_json, list_runs, reserve_run


class FixtureAlgorithm(Algorithm):
    def __init__(self, callback):
        self.callback = callback

    def train(self, config, out):
        return self.callback(config, Path(out))

    def load_policy(self, checkpoint, env, observations):
        raise NotImplementedError


@pytest.fixture
def install_algorithm(monkeypatch):
    from teamwork.algorithms import registry

    monkeypatch.setattr(registry, "_FACTORIES", dict(registry._FACTORIES))

    def install(callback):
        register_algorithm("run_fixture", lambda: FixtureAlgorithm(callback))
        return {"algorithm": "run_fixture", "env": "spread", "seed": 2, "total_steps": 8}

    return install


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_runner_preserves_contract_and_tracks_running_and_completed(tmp_path, install_algorithm):
    out = tmp_path / "trial"

    def train(config, target):
        assert target == out
        assert not target.exists()  # 不要求第三方算法接受已创建的输出目录。
        running = list_runs(tmp_path, status="running")
        assert len(running) == 1
        assert running[0]["path"] == str(out)
        target.mkdir()
        atomic_write_json(target / "config.json", {**config, "algorithm_resolved": True})
        atomic_write_json(target / "provenance.json", {"algorithm_owned": True})
        checkpoint = target / "model.pt"
        checkpoint.write_bytes(b"fixture model")
        return checkpoint

    config = install_algorithm(train)
    checkpoint = run_training(config, out)
    assert checkpoint == out / "model.pt"
    assert "env_kwargs" not in config
    assert read_json(out / "config.json")["algorithm_resolved"]
    assert read_json(out / "provenance.json") == {"algorithm_owned": True}
    assert read_json(out / "run.json")["status"] == "completed"
    report = read_json(out / "training.json")
    assert report["training_seed"] == 2
    assert report["requested_joint_steps"] == 8
    assert report["wall_seconds"] >= 0
    completed = list_runs(tmp_path, status="completed", algorithm="run_fixture", env="spread")
    assert len(completed) == 1
    assert completed[0]["checkpoint"] == str(checkpoint)
    assert not list_runs(tmp_path, status="failed")


def test_runner_allocates_unique_structured_directories(tmp_path, install_algorithm):
    def train(config, target):
        target.mkdir()
        checkpoint = target / "model.pt"
        checkpoint.write_bytes(b"fixture")
        return checkpoint

    config = install_algorithm(train)
    first = run_training(config, runs_root=tmp_path)
    second = run_training(config, runs_root=tmp_path)
    expected_parent = tmp_path / "run_fixture" / "train" / "spread"
    assert first.parent.parent == expected_parent
    assert second.parent.parent == expected_parent
    assert first != second
    assert "-s2-" in first.parent.name
    assert len(list_runs(tmp_path)) == 2


@pytest.mark.parametrize("create_output", [False, True])
def test_failed_run_survives_errors_and_cannot_be_reused(
    tmp_path, install_algorithm, create_output
):
    out = tmp_path / "failed"

    def fail(config, target):
        if create_output:
            target.mkdir()
        raise RuntimeError("expected training failure")

    config = install_algorithm(fail)
    with pytest.raises(RuntimeError, match="expected training failure"):
        run_training(config, out)
    failed = list_runs(tmp_path, status="failed")
    assert len(failed) == 1
    assert failed[0]["error"] == {"type": "RuntimeError", "message": "expected training failure"}
    assert failed[0]["finished_at"] is not None
    if create_output:
        assert read_json(out / "training.json")["status"] == "failed"
        assert read_json(out / "run.json")["status"] == "failed"
    else:
        assert not out.exists()
    with pytest.raises(FileExistsError):
        run_training(config, out)


@pytest.mark.parametrize("outside", [False, True])
def test_runner_rejects_missing_or_external_checkpoint(tmp_path, install_algorithm, outside):
    out = tmp_path / "trial"

    def invalid_checkpoint(config, target):
        target.mkdir()
        checkpoint = tmp_path / "external.pt" if outside else target / "missing.pt"
        if outside:
            checkpoint.write_bytes(b"outside")
        return checkpoint

    config = install_algorithm(invalid_checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        run_training(config, out)
    assert list_runs(tmp_path)[0]["status"] == "failed"


def test_reservation_stays_exclusive_after_status_json_replacement(tmp_path):
    config = {"algorithm": "ippo", "env": "spread", "seed": 0, "total_steps": 8}
    out = tmp_path / "trial"
    record = reserve_run(config, out)
    record.update(status="failed")
    with pytest.raises(FileExistsError, match="reserved"):
        reserve_run(config, out)
    out.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        reserve_run(config, out)


@pytest.mark.parametrize("field,value", [("algorithm", "../other"), ("env", "a/b")])
def test_automatic_directory_components_cannot_escape_root(tmp_path, field, value):
    config = {"algorithm": "ippo", "env": "spread", "seed": 0, "total_steps": 8, field: value}
    with pytest.raises(ValueError, match="Unsafe"):
        allocate_run_dir(config, tmp_path)
    assert not list(tmp_path.iterdir())


def test_listing_ignores_unmanaged_setup_and_smoke_artifacts(tmp_path):
    for directory in ("setup-rl", "smoke-old", "train/old"):
        target = tmp_path / directory
        target.mkdir(parents=True)
        atomic_write_json(target / "training.json", {"algorithm": "ippo"})
    assert list_runs(tmp_path) == []
    assert list_runs(tmp_path / "not-created") == []
    with pytest.raises(ValueError, match="Unknown run status"):
        list_runs(tmp_path, status="not-a-status")


def test_listing_ignores_incomplete_or_invalid_index_records(tmp_path):
    index = tmp_path / ".run-index"
    index.mkdir()
    (index / "partial.json").write_text('{"schema_version":', encoding="utf-8")
    atomic_write_json(index / "list.json", [])
    atomic_write_json(index / "incomplete.json", {"schema_version": 1, "kind": "training"})
    assert list_runs(tmp_path) == []


def test_atomic_json_failure_preserves_existing_record(tmp_path):
    path = tmp_path / "record.json"
    atomic_write_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"loss": float("nan")})
    assert read_json(path) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]
