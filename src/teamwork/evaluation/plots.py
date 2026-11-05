"""公共可视化写在这里：训练回报曲线、同场景算法对比图。"""

import csv
from pathlib import Path


def plot_comparison(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["algorithms"])
    values = [result["algorithms"][a]["mean"] for a in names]
    errors = [result["algorithms"][a]["std_across_training_seeds"] or 0 for a in names]
    labels = [f"{a}\n(n={result['algorithms'][a]['n_models']} models)" for a in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.set_ylabel(result["metric"]["name"])
    ax.set_title(f"{result['env']} | std across training seeds; n=1 has no error estimate")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curve(csv_path, out):

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No complete episode in CSV")
    for agent in rows[0]:
        if agent not in {"episode", "steps"}:
            plt.plot(
                [int(r["steps"]) for r in rows],
                [float(r[agent]) for r in rows],
                label=agent,
                alpha=0.7,
            )
    plt.xlabel("Joint environment steps")
    plt.ylabel("Episode return")
    plt.legend()
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
