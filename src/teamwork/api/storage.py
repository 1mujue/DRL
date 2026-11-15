"""按运行 ID 存储交付产物与任务清单，避免调用方拼接模型路径。"""

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .errors import AlgorithmAPIError, invalid


def timestamp():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".writing-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunStore:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, run_id):
        if not isinstance(run_id, str) or re.fullmatch(r"run_[0-9a-f]{32}", run_id) is None:
            raise invalid("run_id", "Invalid run ID")
        path = self.root / run_id
        if path.is_symlink() or not path.is_dir():
            raise AlgorithmAPIError("RUN_NOT_FOUND", "Run not found", run_id=run_id)
        return path

    def create(self, config, spec, owner):
        run_id = "run_" + uuid4().hex
        path = self.root / run_id
        path.mkdir()
        write_json(
            path / "manifest.json",
            {
                "run_id": run_id,
                "api_version": "1.0",
                "status": "created",
                "owner": owner,
                "created_at": timestamp(),
                "updated_at": timestamp(),
                "config": config,
                "environment_spec": spec,
                "evaluations": [],
                "error": None,
            },
        )
        return run_id

    def read(self, run_id):
        path = self.path(run_id) / "manifest.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("run_id") != run_id:
                raise ValueError("Manifest run ID mismatch")
            return record
        except (OSError, ValueError) as exc:
            raise AlgorithmAPIError(
                "INVALID_MANIFEST", "Run manifest is unreadable", run_id=run_id
            ) from exc

    def update(self, run_id, **changes):
        record = self.read(run_id)
        record.update(changes, updated_at=timestamp())
        write_json(self.path(run_id) / "manifest.json", record)
        return record

    def list(self):
        records = [
            self.read(p.name)
            for p in self.root.glob("run_*")
            if re.fullmatch(r"run_[0-9a-f]{32}", p.name) and p.is_dir() and not p.is_symlink()
        ]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def artifact(self, run_id, name):
        paths = {
            "checkpoint": "training/model.pt",
            "config": "training/config.json",
            "episodes": "training/episodes.csv",
            "updates": "training/updates.csv",
            "training": "training/training.json",
            "provenance": "training/provenance.json",
            "log": "train.log",
            "request": "request.json",
        }
        if name not in paths:
            raise invalid("artifact", f"Unknown artifact: {name}")
        root = self.path(run_id)
        path = (root / paths[name]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise AlgorithmAPIError(
                "ARTIFACT_NOT_FOUND", "Artifact not found", run_id=run_id, artifact=name
            )
        return path

    def evaluation_path(self, run_id, evaluation_id):
        if (
            not isinstance(evaluation_id, str)
            or re.fullmatch(r"eval_[0-9a-f]{32}", evaluation_id) is None
        ):
            raise invalid("evaluation_id", "Invalid evaluation ID")
        root = self.path(run_id)
        path = (root / "evaluations" / f"{evaluation_id}.json").resolve()
        if not path.is_relative_to(root):
            raise invalid("evaluation_id", "Evaluation path is outside the run directory")
        return path
