import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
import yaml

from .algorithms.registry import available_algorithms
from .environments.registry import available_environments
from .evaluation.compare import compare_results
from .evaluation.evaluator import evaluate, save_result
from .evaluation.metrics import protocol_description, resolved_env_kwargs
from .evaluation.plots import plot_comparison, plot_training_curve
from .experiments.runner import run_training
from .experiments.runs import list_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="MARL coursework command line")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    met = sub.add_parser("metrics", help="查看一个场景的固定指标定义")
    met.add_argument("--env", choices=available_environments(), required=True)
    comp = sub.add_parser("compare", help="检查可比性并按独立训练种子汇总结果")
    comp.add_argument("results", nargs="+")
    comp.add_argument("--metric", required=True)
    comp.add_argument("--out", required=True)
    comp.add_argument("--plot")
    tr = sub.add_parser("train")
    tr.add_argument("--config", required=True)
    tr.add_argument("--out", help="指定新实验目录；省略时按算法/场景/种子自动分配")
    tr.add_argument("--runs-root", default="runs", help="自动分配训练目录的根目录")
    tr.add_argument("--steps", type=int)
    tr.add_argument("--seed", type=int)
    tr.add_argument("--device", choices=["cpu", "cuda"])
    tr.add_argument("--algorithm", choices=available_algorithms())
    tr.add_argument("--env", choices=available_environments())
    runs = sub.add_parser("runs", help="查看受管训练实验及其状态")
    runs.add_argument("--root", default="runs")
    runs.add_argument("--status", choices=["running", "completed", "failed"])
    runs.add_argument("--algorithm", choices=available_algorithms())
    runs.add_argument("--env", choices=available_environments())
    runs.add_argument("--json", action="store_true", help="输出完整 JSON 记录")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--env", choices=available_environments(), default="spread")
    ev.add_argument("--policy", choices=["random", "rule"], default="random")
    ev.add_argument("--checkpoint")
    ev.add_argument("--config", help="Use the same environment settings as a training YAML")
    ev.add_argument("--episodes", type=int, default=10)
    ev.add_argument("--seed", type=int, default=10000)
    ev.add_argument("--out", help="省略时保存到模型所属实验的 evaluations/ 子目录")
    ev.add_argument("--gif")
    ev.add_argument("--team", choices=["all", "predators", "prey"], default="all")
    ev.add_argument("--opponent", choices=["random", "rule"])
    plot = sub.add_parser("plot")
    plot.add_argument("--csv", required=True)
    plot.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "runs":
        records = list_runs(
            args.root, status=args.status, algorithm=args.algorithm, env=args.env
        )
        if args.json:
            print(json.dumps(records, ensure_ascii=False, indent=2))
        elif not records:
            print("No managed training runs found.")
        else:
            print(f"{'STATUS':<10} {'ALGORITHM':<10} {'ENV':<9} {'SEED':<6} PATH")
            for record in records:
                print(
                    f"{record['status']:<10} {record['algorithm']:<10} "
                    f"{record['env']:<9} {record['training_seed']!s:<6} {record['path']}"
                )
    elif args.command == "metrics":
        print(json.dumps(protocol_description(args.env), ensure_ascii=False, indent=2))
    elif args.command == "compare":
        result = compare_results(args.results, args.metric)
        save_result(result, args.out)
        if args.plot:
            plot_comparison(result, args.plot)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "doctor":
        versions = {}
        for package in ("mpe2", "pettingzoo", "gymnasium", "torch", "numpy", "lbforaging", "rware"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        gpu = torch.cuda.is_available()
        report = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "cuda_available": gpu,
            "torch_cuda": torch.version.cuda,
        }
        if gpu:
            report["gpu"] = torch.cuda.get_device_name(0)
            # A successful availability check alone does not prove kernel compatibility.
            report["cuda_matmul"] = float(
                (torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).sum().item()
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "train":
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if args.env and args.env != config["env"]:
            # 换场景时不能把原场景的 N、障碍物等参数一起带过去。
            config["env"] = args.env
            config["env_kwargs"] = resolved_env_kwargs(args.env)
        if args.algorithm:
            config["algorithm"] = args.algorithm
        for key, value in (
            ("total_steps", args.steps),
            ("seed", args.seed),
            ("device", args.device),
        ):
            if value is not None:
                config[key] = value
        if config["total_steps"] < 1:
            parser.error("total_steps must be positive")
        print(run_training(config, args.out, runs_root=args.runs_root))
    elif args.command == "evaluate":
        env_name, env_kwargs = args.env, {}
        if args.config:
            config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            env_name, env_kwargs = config["env"], config.get("env_kwargs", {})
        if args.checkpoint:
            config = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]
            if args.config and (
                env_name != config["env"] or env_kwargs != config.get("env_kwargs", {})
            ):
                parser.error("Evaluation config does not match checkpoint environment")
            env_name, env_kwargs = config["env"], config.get("env_kwargs", {})
        result = evaluate(
            env_name,
            args.policy,
            args.episodes,
            args.seed,
            args.checkpoint,
            env_kwargs,
            args.gif,
            args.opponent,
            args.team,
        )
        output = args.out or _evaluation_output(args.checkpoint)
        save_result(result, output)
        print(f"Evaluation saved to {output}")
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    elif args.command == "plot":
        plot_training_curve(args.csv, args.out)


def _evaluation_output(checkpoint):
    """每次评估使用独立文件，避免默认路径覆盖之前的结果。"""
    from datetime import UTC, datetime
    from uuid import uuid4

    root = Path(checkpoint).resolve().parent if checkpoint else Path("runs")
    # 周期模型放在 checkpoints/；评估结果仍归到该实验根目录。
    if checkpoint and root.name == "checkpoints":
        root = root.parent
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / "evaluations" / f"eval-{stamp}-{uuid4().hex[:8]}.json"


if __name__ == "__main__":
    main()
