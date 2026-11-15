"""接入清单及算法自身的参数，不依赖仓库工作目录。"""

from copy import deepcopy
from importlib.util import find_spec

from ..algorithms.happo.trainer import DEFAULTS as HAPPO_DEFAULTS
from ..algorithms.hasac.trainer import DEFAULTS as HASAC_DEFAULTS
from ..algorithms.iql.trainer import DEFAULTS as IQL_DEFAULTS
from ..algorithms.ppo.trainer import DEFAULTS as IPPO_DEFAULTS
from ..environments.registry import available_environments
from ..evaluation.metrics import protocol_description, resolved_env_kwargs

API_VERSION = "1.0"


def numeric_parameters(defaults, **extra):
    """SDK 请求合同只接受数值参数。

    HAPPO/HASAC 的布尔开关（share_policy、learn_alpha 等）与空值选项（actor_order）
    是固定策略，不在 API 上开放覆盖，由训练器自身的默认配置兜底。
    """
    values = {
        key: value
        for key, value in defaults.items()
        if type(value) in (int, float) and key not in {"algorithm", "seed", "device", "env"}
    }
    values.update(extra)
    return values


ALGORITHMS = {
    "ippo": {"name": "Independent PPO", "family": "policy_gradient", "defaults": IPPO_DEFAULTS},
    "iql": {
        "name": "Independent Q-Learning (DQN)",
        "family": "value_based",
        "defaults": IQL_DEFAULTS,
    },
    "happo": {
        "name": "Heterogeneous-Agent PPO",
        "family": "policy_gradient",
        "defaults": numeric_parameters(
            HAPPO_DEFAULTS,
            # 与 HAPPO resolve_config 的 setdefault 同一语义，便于通过 SDK 分别覆盖。
            actor_lr=HAPPO_DEFAULTS["lr"],
            critic_lr=HAPPO_DEFAULTS["lr"],
        ),
    },
    "hasac_discrete": {
        "name": "Heterogeneous-Agent SAC (discrete)",
        "family": "actor_critic",
        "defaults": numeric_parameters(HASAC_DEFAULTS),
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
    "happo": {"rollout_steps", "batch_size", "epochs", "hidden_size", "hidden_layers"},
    "hasac_discrete": {
        "hidden_size",
        "hidden_layers",
        "replay_capacity",
        "batch_size",
        "learning_starts_episodes",
        "updates_per_episode",
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
