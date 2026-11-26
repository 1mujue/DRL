"""Collect successful TwoArmHandover demonstrations and fit a continuous actor."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from train_stable_lift import train_clone

from teamwork.algorithms.baselines.two_arm_handover.policy import ScriptedHandoverPolicy
from teamwork.algorithms.names import CENTRALIZED_PPO_CONTINUOUS, canonical_algorithm_name
from teamwork.environments.registry import make_env
from teamwork.evaluation.evaluator import save_result
from teamwork.evaluation.metrics import make_metrics, resolved_env_kwargs
from teamwork.experiments.runner import seed_all


def collect_episode(env, teacher, seed):
    agents = env.possible_agents
    local, _ = env.reset(seed=seed)
    teacher.reset()
    suite = make_metrics("two_arm_handover")
    initial = env.task_snapshot()
    suite.reset(initial)
    frames, targets, executed = [], [], []
    trace = [initial]
    totals = dict.fromkeys(agents, 0.0)

    while env.agents:
        commands = teacher.commands(local)
        actions = teacher.actions_from_commands(commands)
        frames.append(np.stack([local[agent] for agent in agents]))
        targets.append(np.stack([commands[agent] for agent in agents]))
        executed.append(np.stack([actions[agent] for agent in agents]))
        local, rewards, *_ = env.step(actions)
        snapshot = env.task_snapshot()
        suite.observe(snapshot)
        trace.append(snapshot)
        for agent, reward in rewards.items():
            totals[agent] += float(reward)

    return (frames, targets, executed), {
        "seed": seed,
        "steps": len(frames),
        "metrics": suite.finish(totals, agents),
        "trace": {"control_hz": env.metadata["render_fps"], "samples": trace},
    }


def record_successes(config, start_seed, episodes, max_attempts, out):
    seed_all(start_seed)
    env = make_env(config["env"], **config["env_kwargs"])
    teacher = ScriptedHandoverPolicy()
    attempts, demonstrations, successes = [], [], []
    try:
        for seed in range(start_seed, start_seed + max_attempts):
            data, result = collect_episode(env, teacher, seed)
            metrics = result["metrics"]
            attempts.append({key: result[key] for key in ("seed", "steps", "metrics")})
            print(
                f"seed={seed} success={int(metrics['handover_success'])} "
                f"first_step={metrics['first_handover_step']}",
                flush=True,
            )
            if metrics["handover_success"]:
                demonstrations.append(data)
                successes.append(result)
                if len(demonstrations) == episodes:
                    break
        agents = env.possible_agents
    finally:
        env.close()

    save_result(
        {"env_kwargs": config["env_kwargs"], "requested_successes": episodes, "episodes": attempts},
        out / "attempts.json",
    )
    save_result({"episodes": successes}, out / "successful_traces.json")
    if len(demonstrations) < episodes:
        raise ValueError(
            f"Collected {len(demonstrations)} of {episodes} successful episodes. "
            f"See {out / 'attempts.json'}."
        )
    observations, targets, actions = zip(*demonstrations, strict=True)
    seeds = np.asarray([result["seed"] for result in successes], dtype=np.int64)
    destination = out / "demonstrations.npz"
    np.savez_compressed(
        destination,
        agents=np.asarray(agents),
        observations=np.asarray(observations, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        seeds=seeds,
        successful=seeds,
    )
    return destination


def train_handover_clone(config, out, *, start_seed, episodes, max_attempts):
    if start_seed < 0 or not 0 < episodes <= max_attempts:
        raise ValueError("Use a nonnegative seed and 0 < demonstration episodes <= max attempts")
    if (
        canonical_algorithm_name(config.get("algorithm")) != CENTRALIZED_PPO_CONTINUOUS
        or config.get("env") != "two_arm_handover"
        or config.get("actor_type") != "handover_pose"
    ):
        raise ValueError("Use centralized_ppo_continuous with TwoArmHandover and handover_pose")
    settings = resolved_env_kwargs(config["env"], config.get("env_kwargs"))
    if not settings.get("pose_feedback"):
        raise ValueError("The scripted handover needs pose_feedback")
    config = {**config, "algorithm": CENTRALIZED_PPO_CONTINUOUS, "env_kwargs": settings}
    out.mkdir(parents=True, exist_ok=False)
    demonstrations = record_successes(config, start_seed, episodes, max_attempts, out)
    return train_clone(config, out / "behavior_clone", demonstrations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/algorithms/robotics/mappo_handover_clone.yaml")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demo-seed", type=int, default=67100)
    parser.add_argument("--demo-episodes", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=48)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        train_handover_clone(
            config,
            args.out,
            start_seed=args.demo_seed,
            episodes=args.demo_episodes,
            max_attempts=args.max_attempts,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
