import json

import pytest

from teamwork.api import AlgorithmAPIError, PolicySession
from teamwork.api.storage import RunStore, write_json


def test_store_ignores_unrelated_folders_and_retains_old_snapshot(tmp_path):
    store = RunStore(tmp_path)
    (tmp_path / "run_manual_notes").mkdir()
    run_id = store.create({"total_steps": 1}, {"agents": ["a"]}, "test")
    before = store.read(run_id)
    after = store.update(run_id, status="failed", error={"code": "test"})
    assert before["status"] == "created"
    assert after["status"] == "failed"
    assert [item["run_id"] for item in store.list()] == [run_id]
    assert json.loads((store.path(run_id) / "manifest.json").read_text()) == after


def test_unreadable_manifest_has_structured_error(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create({"total_steps": 1}, {}, "test")
    (store.path(run_id) / "manifest.json").write_text("{broken json")
    with pytest.raises(AlgorithmAPIError) as caught:
        store.read(run_id)
    assert caught.value.code == "INVALID_MANIFEST"


def test_artifact_cannot_escape_workspace_through_symlink(tmp_path):
    store = RunStore(tmp_path / "workspace")
    run_id = store.create({"total_steps": 1}, {}, "test")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.pt").write_text("outside")
    try:
        (store.path(run_id) / "training").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(AlgorithmAPIError) as caught:
        store.artifact(run_id, "checkpoint")
    assert caught.value.code == "ARTIFACT_NOT_FOUND"


def test_failed_json_write_preserves_original_document(tmp_path):
    path = tmp_path / "data.json"
    write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(tmp_path.glob(".writing-*")) == []


def test_corrupt_checkpoint_is_reported_before_inference(tmp_path):
    path = tmp_path / "model.pt"
    path.write_text("invalid checkpoint")
    with pytest.raises(AlgorithmAPIError) as caught:
        PolicySession(path)
    assert caught.value.code == "INVALID_CHECKPOINT"
