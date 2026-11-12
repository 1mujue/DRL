"""可复用的 CPU 推理会话，校验观测与动作，隔离随机推理的 RNG。"""

import threading

import numpy as np
import torch

from ..algorithms.registry import make_algorithm
from ..algorithms.training import training_agents
from ..environments.registry import make_env
from .contracts import boolean, seed_value
from .errors import AlgorithmAPIError, invalid

_RANDOM_LOCK = threading.RLock()


class PolicySession:
    def __init__(self, checkpoint, seed=0):
        seed_value(seed)
        self.lock = threading.RLock()
        self.closed = False
        env = None
        try:
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.config = data["config"]
            env = make_env(self.config["env"], **self.config.get("env_kwargs", {}))
            observations, _ = env.reset(seed=seed)
            self.agents = tuple(training_agents(env, self.config))
            self.sizes = {a: len(observations[a]) for a in env.possible_agents}
            self.actions = {a: env.action_space(a).n for a in self.agents}
            with _RANDOM_LOCK, torch.random.fork_rng(devices=[]):
                self.policy = make_algorithm(self.config["algorithm"]).load_policy(
                    checkpoint, env, observations
                )
            self.rng_state = torch.Generator().manual_seed(seed).get_state()
        except Exception as exc:
            raise AlgorithmAPIError("INVALID_CHECKPOINT", f"Cannot load checkpoint: {exc}") from exc
        finally:
            if env is not None:
                env.close()

    def _observations(self, observations):
        if not isinstance(observations, dict) or any(not isinstance(a, str) for a in observations):
            raise invalid("observations", "Observations must map agent IDs to vectors")
        missing = set(self.agents) - observations.keys()
        extra = observations.keys() - self.sizes.keys()
        if missing or extra:
            raise invalid(
                "observations",
                f"Missing controlled agents: {sorted(missing)}; unknown agents: {sorted(extra)}",
            )
        result = {}
        for agent, value in observations.items():
            try:
                array = np.asarray(value, dtype=np.float32)
            except (ValueError, TypeError, OverflowError) as exc:
                raise invalid("observations", f"Invalid numeric vector for {agent}") from exc
            if array.shape != (self.sizes[agent],) or not np.isfinite(array).all():
                raise invalid("observations", f"{agent} requires {self.sizes[agent]} finite values")
            result[agent] = array
        return result

    def _check_open(self):
        if self.closed:
            raise AlgorithmAPIError("SESSION_CLOSED", "Policy session is closed")

    def _act(self, observations, deterministic):
        if deterministic:
            actions = self.policy.act(observations, deterministic=True)
        else:
            with _RANDOM_LOCK, torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.rng_state)
                actions = self.policy.act(observations, deterministic=False)
                self.rng_state = torch.get_rng_state()
        if set(actions) != set(self.agents) or any(
            type(value) is not int or not 0 <= value < self.actions[a]
            for a, value in actions.items()
        ):
            raise AlgorithmAPIError("INVALID_ACTION", "Policy returned invalid actions")
        return actions

    def predict(self, observations, deterministic=True):
        with self.lock:
            self._check_open()
            boolean(deterministic, "deterministic")
            return self._act(self._observations(observations), deterministic)

    def predict_batch(self, observations, deterministic=True):
        with self.lock:
            self._check_open()
            boolean(deterministic, "deterministic")
            if not isinstance(observations, list) or not observations:
                raise invalid("observations", "Batch must be a nonempty list of observation maps")
            prepared = [self._observations(item) for item in observations]
            return [self._act(item, deterministic) for item in prepared]

    def reset(self, seed=None):
        with self.lock:
            self._check_open()
            if seed is not None:
                seed_value(seed)
                self.rng_state = torch.Generator().manual_seed(seed).get_state()
            self.policy.reset()

    def close(self):
        with self.lock:
            self.closed = True
            self.policy = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()
