"""Load a single recorded robotics evaluation episode."""

import json
from pathlib import Path


def read_trace(evaluation_path, environment, seed):
    result = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    if result["env"] != environment:
        raise ValueError(f"Trajectory plot requires a {environment} evaluation")
    episode = next((row for row in result["episodes"] if seed is None or row["seed"] == seed), None)
    if episode is None:
        raise ValueError(f"Evaluation does not contain episode seed {seed}")
    if "trace" not in episode:
        raise ValueError("Evaluate with --record-trace to plot the trajectory")
    return result, episode
