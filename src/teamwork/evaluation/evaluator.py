"""公共评估器：执行策略只提供动作，指标由场景协议唯一决定。"""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from ..algorithms.baselines import baseline_actions
from ..algorithms.registry import make_algorithm
from ..environments.registry import make_env
from ..experiments.runner import seed_all
from .metrics import make_metrics, protocol_description, resolved_env_kwargs, summarize


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _code_hash():
    # 同一版本号下源码意外改变，也不得悄悄与旧口径结果合并。
    root = Path(__file__).resolve().parents[1]
    files = (
        "evaluation/metrics.py",
        "evaluation/evaluator.py",
        "environments/mpe.py",
        "environments/gym_tuple.py",
        "environments/registry.py",
        "algorithms/baselines.py",
    )
    content = b"".join((root / file).read_bytes() for file in files)
    return hashlib.sha256(content).hexdigest()


def evaluate(
    name,
    policy="random",
    episodes=10,
    seed=10000,
    checkpoint=None,
    env_kwargs=None,
    gif=None,
    opponent=None,
    team="all",
    deterministic=True,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if opponent and (name != "tag" or team not in {"predators", "prey"}):
        raise ValueError("opponent requires Tag and --team predators/prey")
    if team != "all" and not opponent:
        raise ValueError("Selecting a Tag team requires a fixed opponent")
    data = None
    if checkpoint:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["config"]["env"] != name:
            raise ValueError("Checkpoint environment mismatch")
        trained_team = data["config"].get("train_team", "all")
        if trained_team != "all" and (
            team != trained_team or opponent != data["config"].get("train_opponent")
        ):
            raise ValueError("Evaluation team and opponent must match fixed-opponent training")
        trained = resolved_env_kwargs(name, data["config"].get("env_kwargs"))
        if env_kwargs is not None and resolved_env_kwargs(name, env_kwargs) != trained:
            raise ValueError("Evaluation settings do not match checkpoint environment")
        env_kwargs = trained
    settings = resolved_env_kwargs(name, env_kwargs)
    seed_all(seed)
    env = make_env(name, render_mode="rgb_array" if gif else None, **settings)
    rows, frames = [], []
    execution_policy = None
    benchmark = None
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            if ep == 0:
                packages = {}
                environment_package = {
                    "spread": "mpe2",
                    "tag": "mpe2",
                    "lbf": "lbforaging",
                    "rware": "rware",
                }.get(name)
                for package in ("pettingzoo", "gymnasium", "numpy", environment_package):
                    if package is None:
                        continue
                    try:
                        packages[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError:
                        packages[package] = None
                benchmark = {
                    "protocol": protocol_description(name),
                    "env_kwargs": settings,
                    "spaces": asdict(env.describe()),
                    "environment_packages": packages,
                    "evaluation_seeds": list(range(seed, seed + episodes)),
                    "deterministic": deterministic,
                    "opponent": opponent,
                    "team": team,
                    "protocol_code_sha256": _code_hash(),
                }
                if data:
                    execution_policy = make_algorithm(data["config"]["algorithm"]).load_policy(
                        checkpoint, env, obs
                    )
            for i, agent in enumerate(env.agents):
                env.action_space(agent).seed(seed + ep * 100 + i)
            if execution_policy:
                execution_policy.reset()
            suite = make_metrics(name)
            suite.reset(env.task_snapshot())
            totals = dict.fromkeys(env.possible_agents, 0.0)
            steps = 0
            while env.agents:
                actions = (
                    execution_policy.act(obs, deterministic=deterministic)
                    if execution_policy
                    else baseline_actions(env, obs, policy)
                )
                if opponent:
                    other = baseline_actions(env, obs, opponent)
                    actions = {
                        agent: actions[agent]
                        if agent.startswith("adversary_") == (team == "predators")
                        else other[agent]
                        for agent in env.agents
                    }
                if gif and ep == 0:
                    frames.append(env.render())
                obs, rewards, _, _, _ = env.step(actions)
                steps += 1
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                suite.observe(env.task_snapshot())
            metrics = suite.finish(totals, env.possible_agents)
            expected = {d["name"] for d in benchmark["protocol"]["metrics"]}
            if set(metrics) != expected:
                raise ValueError("Metric suite output does not match its registered definitions")
            rows.append({"seed": seed + ep, "steps": steps, "returns": totals, "metrics": metrics})
        if gif:
            gif = Path(gif)
            gif.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(gif, frames, duration=100, loop=0)
    finally:
        env.close()
    benchmark_id = hashlib.sha256(
        json.dumps(benchmark, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "env": name,
        "algorithm": data["config"]["algorithm"] if data else policy,
        "training_seed": data["config"]["seed"] if data else None,
        "training_steps": data.get("steps") if data else None,
        "training_setup": {
            "team": data["config"].get("train_team", "all"),
            "opponent": data["config"].get("train_opponent"),
        } if data else None,
        "model_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if data else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "benchmark_id": benchmark_id,
        "benchmark": benchmark,
        # 双方同时更换策略，会同时改变对手强度，不允许直接作为算法排行榜。
        "comparison_allowed": name != "tag" or bool(opponent),
        "episodes": rows,
        "summary": summarize(rows),
        "mean_returns": {a: float(np.mean([r["returns"][a] for r in rows])) for a in totals},
    }
