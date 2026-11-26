"""RWARE configuration, episode boundaries, and delivery snapshots."""

from ..adapters.gym_tuple import TupleParallel


class RWAREEnvironment(TupleParallel):
    package = "rware"
    family = "rware"
    default_id = "rware-tiny-2ag-v2"

    @staticmethod
    def gym_options(kwargs):
        if isinstance(kwargs.get("reward_type"), str):
            from rware.warehouse import RewardType

            kwargs["reward_type"] = RewardType[kwargs["reward_type"].upper()]
        return kwargs

    def episode_end(self, terminated, truncated):
        base = self.base.unwrapped
        if terminated and base.max_steps and base._cur_steps >= base.max_steps:
            return False, True
        return terminated, truncated

    def task_snapshot(self):
        reward_type = self.base.unwrapped.reward_type.name
        if reward_type == "TWO_STAGE":
            raise ValueError("RWARE metrics v1 require INDIVIDUAL or GLOBAL rewards")
        # RWARE 2.0 counts deliveries by summing individual or averaging global rewards.
        deliveries = (
            self._last_rewards.mean() if reward_type == "GLOBAL" else self._last_rewards.sum()
        )
        return {"deliveries_this_step": float(deliveries), "reward_type": reward_type}
