"""公共入口只调度接口；自动产物路径必须避免覆盖已有结果。"""

import json

import pytest
import yaml

from teamwork import cli


@pytest.mark.parametrize("explicit_out", [False, True])
def test_train_dispatches_optional_output_without_algorithm_details(
    tmp_path, monkeypatch, capsys, explicit_out
):
    config_path = tmp_path / "method.yaml"
    config_path.write_text(
        yaml.safe_dump({"algorithm": "mappo_ff", "env": "spread", "seed": 0, "total_steps": 8}),
        encoding="utf-8",
    )
    received = {}

    def train(config, out, *, runs_root):
        received.update(config=config, out=out, root=runs_root)
        return tmp_path / "allocated" / "model.pt"

    monkeypatch.setattr(cli, "run_training", train)
    arguments = ["train", "--config", str(config_path), "--runs-root", str(tmp_path / "runs")]
    if explicit_out:
        arguments += ["--out", str(tmp_path / "chosen")]
    cli.main(arguments)
    assert received["config"]["algorithm"] == "mappo_ff"
    assert received["out"] == (str(tmp_path / "chosen") if explicit_out else None)
    assert received["root"] == str(tmp_path / "runs")
    assert "model.pt" in capsys.readouterr().out


def test_runs_command_preserves_filters_and_json_records(monkeypatch, capsys):
    record = {"status": "completed", "algorithm": "mappo_ff", "env": "spread", "path": "run"}
    received = {}

    def list_runs(root, **filters):
        received.update(root=root, **filters)
        return [record]

    monkeypatch.setattr(cli, "list_runs", list_runs)
    cli.main(["runs", "--algorithm", "mappo_ff", "--status", "completed", "--json"])
    assert received == {
        "root": "runs", "status": "completed", "algorithm": "mappo_ff", "env": None
    }
    assert json.loads(capsys.readouterr().out) == [record]


def test_repeated_default_evaluations_preserve_previous_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = tmp_path / "runs" / "evaluation.json"
    previous.parent.mkdir()
    previous.write_text('{"previous": true}', encoding="utf-8")
    result = {"env": "spread", "episodes": []}
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: result)
    cli.main(["evaluate", "--episodes", "1"])
    cli.main(["evaluate", "--episodes", "1"])
    outputs = list((tmp_path / "runs" / "evaluations").glob("*.json"))
    assert len(outputs) == 2
    assert all(json.loads(path.read_text(encoding="utf-8")) == result for path in outputs)
    assert json.loads(previous.read_text(encoding="utf-8")) == {"previous": True}
