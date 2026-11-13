"""调用 evaluation.compare 的统一聚合器，输出最终对比表。"""
import argparse
import json
from pathlib import Path

from teamwork.evaluation.compare import compare_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals", nargs="+", required=True, help="eval.json 路径列表")
    p.add_argument("--metric", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    result = compare_results(args.evals, args.metric)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"saved {args.out}")
    print(text)


if __name__ == "__main__":
    main()