"""校验多个 eval.json 的 benchmark_id、training_steps、training_seed 是否一致。"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals", nargs="+", required=True)
    args = p.parse_args()

    rows = []
    for f in args.evals:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rows.append({
            "run": Path(f).parent.name,
            "algorithm": d.get("algorithm"),
            "benchmark_id": d.get("benchmark_id"),
            "training_steps": d.get("training_steps"),
            "training_seed": d.get("training_seed"),
        })

    for r in rows:
        print(r)
    ids = {r["benchmark_id"] for r in rows}
    steps = {r["training_steps"] for r in rows}
    print(f"\nunique benchmark_id: {len(ids)}")
    print(f"unique training_steps: {sorted(steps)}")
    if len(ids) != 1:
        print("WARNING: benchmark_id 不一致，不能进同一张对比表")
    if len(steps) != 1:
        print("WARNING: training_steps 不一致，compare_results 会拒绝")


if __name__ == "__main__":
    main()