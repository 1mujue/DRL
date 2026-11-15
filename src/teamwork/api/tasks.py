"""进程隔离的本地训练任务；调用进程不受训练随机种子及 Torch 全局状态影响。"""

import os
import re
import subprocess
import sys
import threading
import time
from uuid import uuid4

from .errors import AlgorithmAPIError, invalid
from .storage import timestamp, write_json


class TrainingTasks:
    def __init__(self, store, max_concurrent=1):
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise invalid("max_concurrent", "max_concurrent must be a positive integer")
        self.store = store
        self.max_concurrent = max_concurrent
        self.owner = uuid4().hex
        self.processes = {}
        self.lock = threading.RLock()
        self.closed = False

    def submit(self, request):
        with self.lock:
            if self.closed:
                raise AlgorithmAPIError("API_CLOSED", "Training task manager is closed")
            for run_id in list(self.processes):
                self.status(run_id)
            if len(self.processes) >= self.max_concurrent:
                raise AlgorithmAPIError("BUSY", "Training capacity reached; wait or cancel a task")
            config, spec = request.validate()
            run_id = self.store.create(config, spec, self.owner)
            path = self.store.path(run_id)
            write_json(path / "request.json", request.to_dict())
            write_json(path / "resolved.json", config)
            command = [
                sys.executable,
                "-m",
                "teamwork.cli",
                "train",
                "--config",
                str(path / "resolved.json"),
                "--out",
                str(path / "training"),
            ]
            env = dict(
                os.environ,
                SDL_VIDEODRIVER="dummy",
                SDL_AUDIODRIVER="dummy",
                PYGAME_HIDE_SUPPORT_PROMPT="1",
                PYTHONUNBUFFERED="1",
            )
            try:
                with (path / "train.log").open("wb") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=env
                    )
            except OSError as exc:
                self.store.update(
                    run_id,
                    status="failed",
                    finished_at=timestamp(),
                    error={"code": "START_FAILED", "message": str(exc)},
                )
                raise AlgorithmAPIError("START_FAILED", str(exc), run_id=run_id) from exc
            self.processes[run_id] = process
            self.store.update(run_id, status="running", pid=process.pid, started_at=timestamp())
            return self.status(run_id)

    def status(self, run_id):
        with self.lock:
            record = self.store.read(run_id)
            process = self.processes.get(run_id)
            if process is not None:
                return_code = process.poll()
                if return_code is not None:
                    success = (
                        return_code == 0
                        and (self.store.path(run_id) / "training/model.pt").is_file()
                    )
                    error = (
                        None
                        if success
                        else {
                            "code": "TRAINING_FAILED",
                            "message": "Training subprocess failed",
                            "log_tail": self.logs(run_id),
                        }
                    )
                    record = self.store.update(
                        run_id,
                        status="succeeded" if success else "failed",
                        return_code=return_code,
                        finished_at=timestamp(),
                        error=error,
                    )
                    del self.processes[run_id]
            matches = re.findall(r"(?:^|\n)steps=(\d+)", self.logs(run_id))
            steps = int(matches[-1]) if matches else 0
            if record["status"] == "succeeded":
                steps = record["config"]["total_steps"]
            record["progress"] = {
                "logged_steps": steps,
                "requested_steps": record["config"]["total_steps"],
                "fraction": min(steps / record["config"]["total_steps"], 1.0),
            }
            record["managed"] = run_id in self.processes
            return record

    def logs(self, run_id, limit=8192):
        if type(limit) is not int or not 1 <= limit <= 1048576:
            raise invalid("limit", "Log limit must be in [1, 1048576] bytes")
        path = self.store.path(run_id) / "train.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")

    @staticmethod
    def validate_timeout(timeout):
        if timeout is not None and (
            type(timeout) not in (int, float) or not 0 <= timeout < float("inf")
        ):
            raise invalid("timeout", "timeout must be finite and nonnegative")

    def wait(self, run_id, timeout=None):
        self.validate_timeout(timeout)
        started = time.monotonic()
        while True:
            record = self.status(run_id)
            if record["status"] in ("succeeded", "failed", "cancelled"):
                return record
            if not record["managed"]:
                raise AlgorithmAPIError(
                    "UNMANAGED_TASK", "Task belongs to another API instance", run_id=run_id
                )
            if timeout is not None and time.monotonic() - started >= timeout:
                raise AlgorithmAPIError("TIMEOUT", "Task is still running", run_id=run_id)
            time.sleep(0.05)

    def cancel(self, run_id):
        with self.lock:
            record = self.status(run_id)
            if record["status"] in ("succeeded", "failed", "cancelled"):
                return record
            process = self.processes.get(run_id)
            if process is None:
                raise AlgorithmAPIError(
                    "UNMANAGED_TASK",
                    "Cannot cancel a task owned by another instance",
                    run_id=run_id,
                )
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            del self.processes[run_id]
            self.store.update(
                run_id,
                status="cancelled",
                return_code=process.returncode,
                finished_at=timestamp(),
                error=None,
            )
            return self.status(run_id)

    def close(self):
        with self.lock:
            for run_id in list(self.processes):
                self.cancel(run_id)
            self.closed = True
