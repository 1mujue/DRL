"""Train a stable TwoArmLift policy from shared demonstrations and continuous RL updates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from teamwork.algorithms.baselines.two_arm_lift.policy import ScriptedLiftPolicy
from teamwork.algorithms.continuous_off_policy import VARIANTS, build_actors
from teamwork.algorithms.names import canonical_algorithm_name
from teamwork.algorithms.ppo.trainer import CONTINUOUS, create_agents
from teamwork.environments.registry import make_env
from teamwork.evaluation.metrics import make_metrics, resolved_env_kwargs
from teamwork.experiments.runner import run_training, seed_all, write_provenance


def record_demonstrations(env, seeds, destination):
    teacher = ScriptedLiftPolicy()
    observations, commands, successful = [], [], []
    for seed in seeds:
        frames, targets = [], []
        teacher.reset()
        local, _ = env.reset(seed=seed)
        suite = make_metrics("two_arm_lift")
        suite.reset(env.task_snapshot())
        totals = dict.fromkeys(env.possible_agents, 0.0)
        while env.agents:
            command = teacher.commands(local)
            frames.append(np.stack([local[agent] for agent in env.possible_agents]))
            targets.append(np.stack([command[agent] for agent in env.possible_agents]))
            actions = {
                agent: value.clip(-1, 1).astype(np.float32) for agent, value in command.items()
            }
            local, rewards, *_ = env.step(actions)
            suite.observe(env.task_snapshot())
            for agent, reward in rewards.items():
                totals[agent] += reward
        observations.append(frames)
        commands.append(targets)
        if suite.finish(totals, env.possible_agents)["stable_dual_lift"]:
            successful.append(seed)
        print(f"demonstration_seed={seed} stable={seed in successful}", flush=True)
    np.savez_compressed(
        destination,
        observations=np.asarray(observations, dtype=np.float32),
        targets=np.asarray(commands, dtype=np.float32),
        seeds=np.asarray(list(seeds), dtype=np.int64),
        successful=np.asarray(successful, dtype=np.int64),
    )


def train_clone(config, out, demonstration_path):
    config = {**config, "algorithm": canonical_algorithm_name(config["algorithm"])}
    seed_all(config["seed"])
    env = make_env(config["env"], **config["env_kwargs"])
    try:
        local, _ = env.reset(seed=config["seed"])
        if config["algorithm"] in VARIANTS:
            agents = build_actors(env, local, config, "cpu")
        else:
            agents = create_agents(
                env, local, config["algorithm"], "cpu", actor_type=config["actor_type"]
            )
        with np.load(demonstration_path, allow_pickle=False) as archive:
            observations = torch.from_numpy(archive["observations"])
            targets = torch.from_numpy(archive["targets"])
            seeds = archive["seeds"].tolist()
            successful = archive["successful"].tolist()
        expected_prefix = (len(seeds), env.horizon, len(agents))
        if (
            observations.shape[:3] != expected_prefix
            or targets.shape != (*expected_prefix, 7)
            or observations.shape[-1] != len(next(iter(local.values())))
        ):
            raise ValueError("Demonstrations do not match the configured environment")
        if not successful:
            raise ValueError("No demonstration completed the task")
        errors = {}
        for index, (agent, model) in enumerate(agents.items()):
            vectors = observations[:, :, index].flatten(end_dim=1)
            commands = targets[:, :, index].flatten(end_dim=1)
            errors[agent] = model.actor.fit_demonstrations(vectors, commands)
            if hasattr(model, "log_std"):
                with torch.no_grad():
                    model.log_std.fill_(-3.8)
    finally:
        env.close()

    metadata = {
        "method": "behavior_cloning",
        "demonstration_seeds": seeds,
        "successful_demonstration_seeds": successful,
        "demonstration_joint_steps": int(observations.shape[0] * observations.shape[1]),
        "demonstration_sha256": hashlib.sha256(demonstration_path.read_bytes()).hexdigest(),
        "command_rmse": errors,
        "online_joint_steps": 0,
    }
    out.mkdir()
    (out / "training.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_provenance(out, "cpu")
    checkpoint = out / "model.pt"
    torch.save(
        {
            "config": {**config, "behavior_cloning": metadata},
            "steps": 0,
            "agents": {agent: model.state_dict() for agent, model in agents.items()},
        },
        checkpoint,
    )
    print(f"clone={checkpoint} command_rmse={errors}", flush=True)
    return checkpoint, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/algorithms/robotics/mappo_stable.yaml")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demonstrations", type=Path)
    parser.add_argument("--demo-seed", type=int, default=31000)
    parser.add_argument("--demo-episodes", type=int, default=16)
    parser.add_argument("--online-steps", "--ppo-steps", dest="online_steps", type=int)
    parser.add_argument(
        "--algorithm", type=canonical_algorithm_name, choices=sorted(CONTINUOUS | VARIANTS.keys())
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if (
        args.demo_episodes < 1
        or args.demo_seed < 0
        or (args.seed is not None and args.seed < 0)
        or (args.online_steps is not None and args.online_steps < 0)
    ):
        parser.error("demo episodes must be positive and seeds and online steps nonnegative")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["algorithm"] = args.algorithm or canonical_algorithm_name(config["algorithm"])
    if args.seed is not None:
        config["seed"] = args.seed
    if config["algorithm"] not in CONTINUOUS | VARIANTS.keys():
        parser.error("use a supported continuous algorithm")
    if config["env"] != "two_arm_lift" or config.get("actor_type") != "lift_pose":
        parser.error("use a TwoArmLift configuration with actor_type: lift_pose")
    config["env_kwargs"] = resolved_env_kwargs(config["env"], config.get("env_kwargs"))
    steps = config["total_steps"] if args.online_steps is None else args.online_steps
    args.out.mkdir(parents=True, exist_ok=False)
    demonstration_path = args.demonstrations
    if demonstration_path is None:
        demonstration_path = args.out / "demonstrations.npz"
        env = make_env(config["env"], **config["env_kwargs"])
        try:
            seeds = range(args.demo_seed, args.demo_seed + args.demo_episodes)
            record_demonstrations(env, seeds, demonstration_path)
        finally:
            env.close()
    checkpoint, metadata = train_clone(config, args.out / "behavior_clone", demonstration_path)
    if steps:
        config.update(
            warm_start_checkpoint=str(checkpoint),
            behavior_cloning=metadata,
            total_steps=steps,
        )
        stage = "ppo" if config["algorithm"] in CONTINUOUS else "online"
        print(run_training(config, args.out / stage), flush=True)


if __name__ == "__main__":
    main()
