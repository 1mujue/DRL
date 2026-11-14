"""把多个 run 的 episodes.csv 聚合成均值±标准差曲线。"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_curve(csv_path, window=20):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r["steps"]) for r in rows])
    agent_cols = [c for c in rows[0] if c not in ("episode", "steps")]
    ret = np.array([sum(float(r[c]) for c in agent_cols) for r in rows])
    if len(ret) >= window:
        kernel = np.ones(window) / window
        ret = np.convolve(ret, kernel, mode="valid")
        steps = steps[window - 1:]
    return steps, ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="runs/compare.png")
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()

    groups = defaultdict(list)
    for run in args.runs:
        algo = Path(run).name.split("_")[0]
        groups[algo].append(load_curve(Path(run) / "episodes.csv", args.window))

    plt.figure(figsize=(8, 5))
    for algo, curves in groups.items():
        n = min(len(s) for s, _ in curves)
        s = curves[0][0][:n]
        vals = np.stack([v[:n] for _, v in curves])
        mean, std = vals.mean(0), vals.std(0)
        plt.plot(s, mean, label=algo)
        plt.fill_between(s, mean - std, mean + std, alpha=0.2)
    plt.xlabel("environment steps")
    plt.ylabel(f"team return (moving avg {args.window})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()