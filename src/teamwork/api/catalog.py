"""接入清单及算法自身的参数，不依赖仓库工作目录。"""

from copy import deepcopy
from importlib.util import find_spec

from ..algorithms.iql.trainer import DEFAULTS as IQL_DEFAULTS
from ..algorithms.ppo.trainer import DEFAULTS as IPPO_DEFAULTS
from ..environments.registry import available_environments
from ..evaluation.metrics import protocol_description, resolved_env_kwargs

API_VERSION = "1.0"
ALGORITHMS = {
    "ippo": {"name": "Independent PPO", "family": "policy_gradient", "defaults": IPPO_DEFAULTS},
    "iql": {
        "name": "Independent Q-Learning (DQN)",
        "family": "value_based",
        "defaults": IQL_DEFAULTS,
    },
}
INTEGER_PARAMETERS = {
    "ippo": {"rollout_steps", "batch_size", "epochs"},
    "iql": {
        "hidden_size",
        "buffer_size",
        "batch_size",
        "learning_starts",
        "train_interval",
        "target_update_interval",
        "epsilon_decay_steps",
    },
}


def catalog():
    algorithms = []
    for key, descriptor in ALGORITHMS.items():
        algorithms.append(
            {
                "id": key,
                **deepcopy(descriptor),
                "actions": "discrete",
                "observations": "local_vector",
                "parameter_sharing": False,
                "checkpoint_resume": False,
            }
        )
    environments = []
    for name in available_environments():
        package = {"lbf": "lbforaging", "rware": "rware"}.get(name, "mpe2")
        environments.append(
            {
                "id": name,
                "installed": find_spec(package) is not None,
                "defaults": resolved_env_kwargs(name),
                "metrics": protocol_description(name),
            }
        )
    return {"api_version": API_VERSION, "algorithms": algorithms, "environments": environments}
