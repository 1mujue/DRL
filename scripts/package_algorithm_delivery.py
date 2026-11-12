"""打包算法 SDK、四场景验收产物与接口说明，生成可核验的交付 ZIP。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = args.acceptance.resolve()
    report = json.loads((source / "acceptance.json").read_text(encoding="utf-8"))
    expected = {(a, e) for a in ("ippo", "iql") for e in ("spread", "tag", "lbf", "rware")}
    if (
        report.get("status") != "passed"
        or {(run["algorithm"], run["env"]) for run in report["runs"]} != expected
    ):
        parser.error(
            "Require a passed acceptance report covering both algorithms in all four environments"
        )
    if not args.wheel.is_file() or args.wheel.suffix != ".whl":
        parser.error("Provide an existing wheel file")
    for run in report["runs"]:
        checkpoint = Path(run["artifacts"]["checkpoint"]["path"]).resolve()
        if not checkpoint.is_relative_to(source) or digest(checkpoint) != run["checkpoint_sha256"]:
            parser.error(f"Checkpoint is missing or changed: {run['run_id']}")
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.wheel, args.out / args.wheel.name)
    shutil.copytree(source, args.out / "acceptance")
    for name in (
        "docs/API_INTEGRATION.md",
        "docs/API_ACCEPTANCE.md",
        "requirements.lock.txt",
        "examples/algorithm_api.py",
        "scripts/verify_algorithm_api.py",
    ):
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    index = {
        "source_commit": revision,
        "api_version": "1.0",
        "wheel": args.wheel.name,
        "acceptance_report": "acceptance/acceptance.json",
        "workspace": "acceptance/workspace",
        "runs": [],
    }
    for run in report["runs"]:
        index["runs"].append(
            {
                "run_id": run["run_id"],
                "algorithm": run["algorithm"],
                "env": run["env"],
                "checkpoint": str(
                    Path("acceptance/workspace") / run["run_id"] / "training/model.pt"
                ),
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
        )
    dump(args.out / "delivery.json", index)
    readme = f"""# IPPO / IQL 算法接入交付

源码提交：`{revision}`；Python 3.12；API 版本 1.0。

安装 SDK（依赖需按部署平台安装，具体说明见 docs/API_INTEGRATION.md）：

```bash
python -m pip install './{args.wheel.name}[extra-envs]' -c requirements.lock.txt
python examples/algorithm_api.py --workspace runs/api_example
```

## 已训练模型

`delivery.json` 列出可迁移的相对模型路径。用下面方式直接加载本包内的成功运行记录：

```python
from teamwork.api import AlgorithmAPI

with AlgorithmAPI("acceptance/workspace") as api:
    records = api.list_runs()
    print([(r["run_id"], r["config"]["algorithm"], r["config"]["env"]) for r in records])
```

详细输入输出见 docs/API_INTEGRATION.md；测试与训练结果见 docs/API_ACCEPTANCE.md。
原始验收记录保留产出机器上的路径信息，迁移后以 delivery.json 和 SDK 查询的路径为准。
SHA256SUMS.json 记录包内文件校验值。训练预算和任务效果以验收记录为准。
"""
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    checksums = {
        str(p.relative_to(args.out)): digest(p) for p in sorted(args.out.rglob("*")) if p.is_file()
    }
    dump(args.out / "SHA256SUMS.json", checksums)
    archive = Path(
        shutil.make_archive(str(args.out), "zip", root_dir=args.out.parent, base_dir=args.out.name)
    )
    with zipfile.ZipFile(archive) as bundle:
        broken = bundle.testzip()
        if broken:
            raise RuntimeError(f"Archive checksum failed: {broken}")
    print(
        json.dumps(
            {
                "archive": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": digest(archive),
                "files": len(checksums),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
