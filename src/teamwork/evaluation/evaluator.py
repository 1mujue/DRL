"""公共评估器：执行策略只提供动作，指标由场景协议唯一决定。"""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from ..algorithms.baselines import SCRIPTED_HANDOVER_POLICY, baseline_actions, make_baseline_policy
from ..algorithms.registry import make_algorithm
from ..environments.registry import environment_registration, make_env
from ..experiments.runner import seed_all
from .lift_timing import GRASP_CONFIRMATION_STEPS, LiftTimingTrial
from .metrics import make_metrics, protocol_description, resolved_env_kwargs, summarize


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _code_hash(extra_sources=()):
    # 同一版本号下源码意外改变，也不得悄悄与旧口径结果合并。
    root = Path(__file__).resolve().parents[1]
    files = (
        "evaluation/metrics.py",
        "evaluation/evaluator.py",
        "environments/mpe.py",
        "environments/gym_tuple.py",
        "environments/registry.py",
        "algorithms/baselines.py",
        *extra_sources,
    )
    content = b"".join((root / file).read_bytes() for file in files)
    return hashlib.sha256(content).hexdigest()


def _training_initialization(data):
    if data is None:
        return None
    config = data["config"]
    cloning = config.get("behavior_cloning")
    if cloning is not None:
        if not isinstance(cloning, dict):
            raise ValueError("Invalid demonstration provenance")
        digest = cloning.get("demonstration_sha256")
        steps = cloning.get("demonstration_joint_steps")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or type(steps) is not int
            or steps < 1
        ):
            raise ValueError("Checkpoint is missing valid demonstration provenance")
        if cloning.get("method") != "behavior_cloning":
            raise ValueError("Unknown demonstration initialization method")
        method = "behavior_cloning_then_online" if data.get("steps", 0) else "behavior_cloning"
        return {
            "method": method,
            "demonstration_sha256": digest,
            "demonstration_joint_steps": steps,
        }
    if config.get("warm_start_checkpoint") or config.get("warm_start_sha256"):
        return {
            "method": "checkpoint_warm_start",
            "checkpoint_sha256": config.get("warm_start_sha256"),
        }
    return {"method": "from_scratch"}


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
    *,
    pot_mass_scale=None,
    record_trace=False,
    lift_timing=None,
    gripper_delay_steps=0,
    delayed_arm="arm_1",
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
    if type(gripper_delay_steps) is not int or gripper_delay_steps < 0:
        raise ValueError("gripper_delay_steps must be a nonnegative integer")
    if lift_timing is not None and (
        name != "two_arm_lift"
        or not data
        or data["config"].get("actor_type") != "lift_pose"
        or not env_kwargs.get("pose_feedback")
    ):
        raise ValueError("Lift timing requires a TwoArmLift lift_pose checkpoint")
    if gripper_delay_steps and lift_timing is None:
        raise ValueError("Select a lift timing mode to apply the gripper delay")
    initialization = _training_initialization(data)
    settings = resolved_env_kwargs(name, env_kwargs)
    if pot_mass_scale is not None:
        if name != "two_arm_lift":
            raise ValueError("Pot mass evaluation is only supported for two_arm_lift")
        settings.pop("pot_mass_range", None)
        settings["pot_mass_scale"] = pot_mass_scale
    if record_trace and name not in {"two_arm_lift", "two_arm_handover"}:
        raise ValueError("Task traces are currently supported for the two robotic environments")
    if not data and policy == SCRIPTED_HANDOVER_POLICY and not settings.get("pose_feedback"):
        raise ValueError("scripted_handover requires pose_feedback in the environment config")
    seed_all(seed)
    baseline = None if data else make_baseline_policy(name, policy)
    env = make_env(name, render_mode="rgb_array" if gif else None, **settings)
    rows, frames = [], []
    execution_policy = baseline
    benchmark = None
    timing_trial = None
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            if ep == 0:
                packages = {}
                registration = environment_registration(name)
                for package in ("pettingzoo", "gymnasium", "numpy", *registration.packages):
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
                    "protocol_code_sha256": _code_hash(registration.benchmark_sources),
                }
                if data:
                    execution_policy = make_algorithm(data["config"]["algorithm"]).load_policy(
                        checkpoint, env, obs
                    )
                if lift_timing is not None:
                    timing_trial = LiftTimingTrial(
                        execution_policy, lift_timing, env.horizon, delayed_arm, gripper_delay_steps
                    )
                    benchmark["gripper_delay"] = {"arm": delayed_arm, "steps": gripper_delay_steps}
            for i, agent in enumerate(env.agents):
                env.action_space(agent).seed(seed + ep * 100 + i)
            if timing_trial:
                timing_trial.reset()
            elif execution_policy:
                execution_policy.reset()
            suite = make_metrics(name)
            initial = env.task_snapshot()
            suite.reset(initial)
            trace = [initial] if record_trace else None
            totals = dict.fromkeys(env.possible_agents, 0.0)
            steps = 0
            while env.agents:
                if timing_trial:
                    actions = timing_trial.act(
                        obs, env.local_grasp_contacts(), deterministic=deterministic
                    )
                elif execution_policy:
                    actions = execution_policy.act(obs, deterministic=deterministic)
                else:
                    actions = baseline_actions(env, obs, policy)
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
                snapshot = env.task_snapshot()
                suite.observe(snapshot)
                if trace is not None:
                    if timing_trial:
                        snapshot = {**snapshot, "control": timing_trial.diagnostic}
                    trace.append(snapshot)
            if gif and ep == 0:
                frames.append(env.render())
            metrics = suite.finish(totals, env.possible_agents)
            expected = {d["name"] for d in benchmark["protocol"]["metrics"]}
            if set(metrics) != expected:
                raise ValueError("Metric suite output does not match its registered definitions")
            row = {"seed": seed + ep, "steps": steps, "returns": totals, "metrics": metrics}
            if trace is not None:
                row["trace"] = {"control_hz": env.metadata["render_fps"], "samples": trace}
            rows.append(row)
        if gif:
            gif = Path(gif)
            gif.parent.mkdir(parents=True, exist_ok=True)
            fps = env.metadata.get("render_fps", 10)
            imageio.mimsave(gif, frames, duration=round(1000 / fps), loop=0)
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
        }
        if data
        else None,
        "training_initialization": initialization,
        **(
            {"lift_timing": {"mode": lift_timing, "confirmation_steps": GRASP_CONFIRMATION_STEPS}}
            if timing_trial
            else {}
        ),
        "training_env_kwargs": trained if data else None,
        "warm_start_sha256": data["config"].get("warm_start_sha256") if data else None,
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
