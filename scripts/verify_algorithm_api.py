"""生成算法接入验收报告：两个算法、四环境、进程任务、模型调用与统一评估。"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from teamwork.algorithms.baselines import baseline_actions
from teamwork.api import AlgorithmAPI, AlgorithmAPIError, TrainingRequest
from teamwork.api.storage import write_json
from teamwork.environments.registry import make_env

METRICS = {
    "spread": "final_coverage",
    "tag": "capture_success",
    "lbf": "success",
    "rware": "deliveries_per_100_steps",
}


def exercise_policy(api, record):
    config = record["config"]
    env = make_env(config["env"], **config["env_kwargs"])
    calls = 0
    started = time.perf_counter()
    try:
        obs, _ = env.reset(seed=10000)
        with api.open_policy(record["run_id"]) as policy:
            batch = policy.predict_batch([obs, obs])
            if batch[0] != batch[1]:
                raise RuntimeError("Deterministic batch predictions differ")
            while env.agents and calls < 100:
                actions = policy.predict(obs)
                if any(not env.action_space(a).contains(v) for a, v in actions.items()):
                    raise RuntimeError("Invalid policy action")
                if config["env"] == "tag":
                    actions = {**baseline_actions(env, obs, config["train_opponent"]), **actions}
                obs, _, _, _, _ = env.step(actions)
                calls += 1
    finally:
        env.close()
    return {"valid_action_steps": calls, "load_and_rollout_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", nargs="+", choices=tuple(METRICS), default=list(METRICS))
    args = parser.parse_args()
    if args.steps < 1000 or args.episodes < 1 or args.seed < 0:
        parser.error("Use at least 1000 training steps, positive episodes and nonnegative seed")
    if len(set(args.envs)) != len(args.envs):
        parser.error("Duplicate environments")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "steps_per_model": args.steps,
        "training_seed": args.seed,
        "evaluation_episodes": args.episodes,
        "evaluation_seed": 10000,
        "purpose": "integration_acceptance",
        "runs": [],
        "comparisons": {},
    }
    started = time.perf_counter()
    write_json(args.out / "acceptance.json", report)
    with AlgorithmAPI(args.out / "workspace") as api:
        write_json(args.out / "catalog.json", api.catalog())
        for environment in args.envs:
            references = []
            for algorithm in ("ippo", "iql"):
                tag = {"team": "predators", "opponent": "rule"} if environment == "tag" else {}
                request = TrainingRequest(
                    algorithm, environment, steps=args.steps, seed=args.seed, **tag
                )
                print(f"Training {algorithm} / {environment} / {args.steps} steps", flush=True)
                submitted = api.submit_training(request)
                record = api.wait(submitted["run_id"], timeout=1800)
                if record["status"] != "succeeded":
                    raise RuntimeError(json.dumps(record["error"], ensure_ascii=False))
                run_id = record["run_id"]
                artifacts = api.artifacts(run_id)
                data = torch.load(
                    artifacts["checkpoint"]["path"], map_location="cpu", weights_only=True
                )
                if data["steps"] != args.steps:
                    raise RuntimeError("Checkpoint step budget differs from request")
                if algorithm == "iql" and not all(data["updates"].values()):
                    raise RuntimeError("IQL has no gradient updates")
                inference = exercise_policy(api, record)
                evaluated = api.evaluate(run_id, {"episodes": args.episodes}, timeout=300)
                references.append(evaluated)
                training = json.loads(Path(artifacts["training"]["path"]).read_text())
                entry = {
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "env": environment,
                    "actual_steps": data["steps"],
                    "updates": data.get("updates"),
                    "training_seconds": training["wall_seconds"],
                    **inference,
                    "evaluation_id": evaluated["evaluation_id"],
                    "evaluation_summary": evaluated["result"]["summary"],
                    "checkpoint_sha256": hashlib.sha256(
                        Path(artifacts["checkpoint"]["path"]).read_bytes()
                    ).hexdigest(),
                    "artifacts": artifacts,
                }
                report["runs"].append(entry)
                api.export_manifest(run_id, args.out / f"{algorithm}_{environment}_manifest.json")
                write_json(args.out / "acceptance.json", report)
                print(f"Accepted {algorithm} / {environment} / run_id={run_id}", flush=True)
            comparison = api.compare(references, METRICS[environment])
            report["comparisons"][environment] = comparison
            write_json(args.out / f"{environment}_comparison.json", comparison)
    report["wall_seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    write_json(args.out / "acceptance.json", report)
    print(f"Acceptance passed. Report: {args.out / 'acceptance.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except AlgorithmAPIError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False), flush=True)
        raise SystemExit(1) from error
