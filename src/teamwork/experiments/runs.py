"""实验目录预约、状态记录与查询；不依赖任何算法内部实现。"""

import hashlib
import json
import os
import re
import socket
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

RUN_SCHEMA_VERSION = 1
RUN_STATUSES = ("running", "completed", "failed")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def utc_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(path, value):
    """同目录临时文件 + replace，读者不会看到半份 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _component(value, field):
    value = str(value)
    if not _COMPONENT.fullmatch(value):
        raise ValueError(f"Unsafe {field} for a run directory: {value!r}")
    return value


def allocate_run_dir(config, runs_root="runs"):
    """只计算目录名；实际 run 目录仍由 Algorithm.train 创建。"""
    root = Path(runs_root).resolve()
    algorithm = _component(config["algorithm"], "algorithm")
    environment = _component(config["env"], "env")
    seed = _component(config["seed"], "seed")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / algorithm / "train" / environment / f"{stamp}-s{seed}-{uuid4().hex[:8]}"
    # 也防止已有的符号链接把自动生成的路径导向 runs_root 之外。
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Generated run directory escapes runs_root")
    return resolved


@dataclass
class RunRecord:
    path: Path
    index_path: Path
    data: dict

    def update(self, **fields):
        self.data.update(fields)
        atomic_write_json(self.index_path, self.data)
        if self.path.is_dir() and self.path.resolve() == self.path:
            atomic_write_json(self.path / "run.json", self.data)


def reserve_run(config, out=None, runs_root="runs"):
    """预约路径并留下 running 记录，不提前创建算法的输出目录。"""
    path = Path(out).resolve() if out is not None else allocate_run_dir(config, runs_root)
    if os.path.lexists(path):
        raise FileExistsError(f"Run directory already exists: {path}")
    canonical = os.path.normcase(str(path)).encode("utf-8")
    identity = hashlib.sha256(canonical).hexdigest()
    index_dir = path.parent / ".run-index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{identity}.json"
    lock_path = index_dir / f"{identity}.lock"
    if index_path.exists():
        raise FileExistsError(f"Run path was already reserved: {path}")
    # lock 永久保留：原子替换状态 JSON 不能释放同一路径的预约。
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"Run path was already reserved: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    record = RunRecord(
        path,
        index_path,
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "kind": "training",
            "run_id": path.name,
            "path": str(path),
            "algorithm": config["algorithm"],
            "env": config["env"],
            "training_seed": config["seed"],
            "requested_joint_steps": config["total_steps"],
            "device": config.get("device", "cpu"),
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "wall_seconds": None,
            "checkpoint": None,
            "error": None,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        },
    )
    try:
        # 再检查一次，覆盖首次检查与预约之间其他程序创建 out 的情况。
        if os.path.lexists(path):
            raise FileExistsError(f"Run directory appeared during reservation: {path}")
        atomic_write_json(index_path, record.data)
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return record


def list_runs(root="runs", *, status=None, algorithm=None, env=None):
    """只读取受管实验索引，不把安装验收、图表和缓存目录当作实验。"""
    if status is not None and status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status {status!r}; choose from {RUN_STATUSES}")
    root = Path(root)
    if not root.exists():
        return []
    records = []
    for index_path in root.rglob(".run-index/*.json"):
        try:
            record = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            continue
        if record.get("schema_version") != RUN_SCHEMA_VERSION or record.get("kind") != "training":
            continue
        if status is not None and record.get("status") != status:
            continue
        if algorithm is not None and record.get("algorithm") != algorithm:
            continue
        if env is not None and record.get("env") != env:
            continue
        records.append(record)
    return sorted(records, key=lambda item: (item.get("started_at", ""), item["path"]), reverse=True)
