"""供项目后端调用的 IPPO/IQL Python SDK，无需修改训练器或评估器。"""

import json
import os
import subprocess
import sys
import threading
from uuid import uuid4

from ..evaluation.compare import compare_results
from .catalog import catalog
from .contracts import EvaluationRequest, TrainingRequest
from .errors import AlgorithmAPIError, invalid
from .policy import PolicySession
from .storage import RunStore, write_json
from .tasks import TrainingTasks


class AlgorithmAPI:
    def __init__(self, workspace, max_concurrent=1):
        self.store = RunStore(workspace)
        self.tasks = TrainingTasks(self.store, max_concurrent)
        self.lock = threading.RLock()
        self.closed = False

    def _open(self):
        if self.closed:
            raise AlgorithmAPIError("API_CLOSED", "Algorithm API is closed")

    @staticmethod
    def catalog():
        return catalog()

    def validate_training(self, request):
        self._open()
        if isinstance(request, dict):
            request = TrainingRequest.from_dict(request)
        if not isinstance(request, TrainingRequest):
            raise invalid("request", "Use TrainingRequest or a request dictionary")
        config, spec = request.validate()
        return {"config": config, "environment_spec": spec}

    def submit_training(self, request):
        self._open()
        if isinstance(request, dict):
            request = TrainingRequest.from_dict(request)
        if not isinstance(request, TrainingRequest):
            raise invalid("request", "Use TrainingRequest or a request dictionary")
        return self.tasks.submit(request)

    def train(self, request, timeout=None):
        self.tasks.validate_timeout(timeout)
        record = self.submit_training(request)
        record = self.wait(record["run_id"], timeout)
        if record["status"] != "succeeded":
            raise AlgorithmAPIError(
                "TRAINING_FAILED",
                "Training did not succeed",
                run_id=record["run_id"],
                status=record["status"],
                error=record["error"],
            )
        return record

    def status(self, run_id):
        self._open()
        return self.tasks.status(run_id)

    def wait(self, run_id, timeout=None):
        self._open()
        return self.tasks.wait(run_id, timeout)

    def cancel(self, run_id):
        self._open()
        return self.tasks.cancel(run_id)

    def list_runs(self):
        self._open()
        return [self.status(record["run_id"]) for record in self.store.list()]

    def logs(self, run_id, limit=8192):
        self._open()
        return self.tasks.logs(run_id, limit)

    def artifacts(self, run_id):
        self._open()
        self.store.path(run_id)
        result = {}
        for name in (
            "checkpoint",
            "config",
            "episodes",
            "updates",
            "training",
            "provenance",
            "log",
            "request",
        ):
            try:
                path = self.store.artifact(run_id, name)
                result[name] = {"path": str(path), "bytes": path.stat().st_size}
            except AlgorithmAPIError as exc:
                if exc.code != "ARTIFACT_NOT_FOUND":
                    raise
        return result

    def _completed(self, run_id):
        record = self.status(run_id)
        if record["status"] != "succeeded":
            raise AlgorithmAPIError(
                "RUN_NOT_READY",
                "A successful training run is required",
                run_id=run_id,
                status=record["status"],
            )
        return record

    def open_policy(self, run_id, seed=0):
        self._completed(run_id)
        return PolicySession(self.store.artifact(run_id, "checkpoint"), seed)

    def predict(self, run_id, observations, deterministic=True):
        with self.open_policy(run_id) as policy:
            return policy.predict(observations, deterministic)

    def evaluate(self, run_id, request=None, timeout=None):
        with self.lock:
            record = self._completed(run_id)
            if request is None:
                request = EvaluationRequest()
            if isinstance(request, dict):
                request = EvaluationRequest.from_dict(request)
            if not isinstance(request, EvaluationRequest):
                raise invalid("request", "Use EvaluationRequest or a request dictionary")
            request.validate()
            if timeout is not None and (
                type(timeout) not in (int, float) or not 0 < timeout < float("inf")
            ):
                raise invalid("timeout", "Evaluation timeout must be finite and positive")
            evaluation_id = "eval_" + uuid4().hex
            path = self.store.evaluation_path(run_id, evaluation_id)
            path.parent.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "teamwork.cli",
                "evaluate",
                "--checkpoint",
                str(self.store.artifact(run_id, "checkpoint")),
                "--episodes",
                str(request.episodes),
                "--seed",
                str(request.seed),
                "--out",
                str(path),
            ]
            config = record["config"]
            if config["env"] == "tag":
                command += ["--team", config["train_team"], "--opponent", config["train_opponent"]]
            log_path = path.with_suffix(".log")
            env = dict(
                os.environ,
                SDL_VIDEODRIVER="dummy",
                SDL_AUDIODRIVER="dummy",
                PYGAME_HIDE_SUPPORT_PROMPT="1",
            )
            try:
                with log_path.open("wb") as log:
                    result = subprocess.run(
                        command,
                        check=False,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=timeout,
                    )
                if result.returncode != 0:
                    raise AlgorithmAPIError(
                        "EVALUATION_FAILED", "Evaluation subprocess failed", log=str(log_path)
                    )
            except subprocess.TimeoutExpired as exc:
                raise AlgorithmAPIError(
                    "TIMEOUT", "Evaluation subprocess was stopped", log=str(log_path)
                ) from exc
            except OSError as exc:
                raise AlgorithmAPIError("EVALUATION_FAILED", str(exc), log=str(log_path)) from exc
            data = json.loads(path.read_text(encoding="utf-8"))
            reference = {
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "episodes": request.episodes,
                "seed": request.seed,
            }
            latest = self.store.read(run_id)
            self.store.update(run_id, evaluations=[*latest["evaluations"], reference])
            return {**reference, "result": data}

    def evaluation(self, run_id, evaluation_id):
        self._open()
        path = self.store.evaluation_path(run_id, evaluation_id)
        if not path.is_file():
            raise AlgorithmAPIError("EVALUATION_NOT_FOUND", "Evaluation result not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def compare(self, references, metric):
        self._open()
        if not isinstance(references, list) or not references:
            raise invalid("references", "Provide a nonempty list of evaluation references")
        paths = []
        for reference in references:
            if (
                not isinstance(reference, dict)
                or not {"run_id", "evaluation_id"} <= reference.keys()
            ):
                raise invalid("references", "Each reference needs run_id and evaluation_id")
            self.evaluation(reference["run_id"], reference["evaluation_id"])
            paths.append(
                self.store.evaluation_path(reference["run_id"], reference["evaluation_id"])
            )
        try:
            return compare_results(paths, metric)
        except (ValueError, KeyError) as exc:
            raise AlgorithmAPIError("INCOMPARABLE_RESULTS", str(exc)) from exc

    def export_manifest(self, run_id, path):
        record = self.status(run_id)
        write_json(path, {"run": record, "artifacts": self.artifacts(run_id)})

    def close(self):
        with self.lock:
            if not self.closed:
                self.tasks.close()
                self.closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()
