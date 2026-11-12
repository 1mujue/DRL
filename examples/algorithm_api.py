"""组长可直接运行的最小接入示例：训练 → 推理 → 评估 → 比较。"""

import argparse
import json

from teamwork.api import AlgorithmAPI, AlgorithmAPIError, TrainingRequest
from teamwork.environments.registry import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, default=4096)
    args = parser.parse_args()
    with AlgorithmAPI(args.workspace) as api:
        references = []
        for algorithm in ("ippo", "iql"):
            record = api.train(TrainingRequest(algorithm, steps=args.steps), timeout=300)
            run_id = record["run_id"]
            env = make_env("spread", **record["config"]["env_kwargs"])
            try:
                obs, _ = env.reset(seed=10000)
                with api.open_policy(run_id) as policy:
                    print(json.dumps({"algorithm": algorithm, "actions": policy.predict(obs)}))
            finally:
                env.close()
            evaluation = api.evaluate(run_id, {"episodes": 3}, timeout=60)
            references.append(evaluation)
            print(json.dumps({"run_id": run_id, "artifacts": api.artifacts(run_id)}, indent=2))
        print(json.dumps(api.compare(references, "final_coverage"), indent=2))


if __name__ == "__main__":
    try:
        main()
    except AlgorithmAPIError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False))
        raise SystemExit(1) from error
