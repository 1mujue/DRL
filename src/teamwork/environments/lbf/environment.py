"""LBF episode boundaries and food-count snapshots."""

import numpy as np

from ..adapters.gym_tuple import TupleParallel


class LBFEnvironment(TupleParallel):
    package = "lbforaging"
    family = "lbf"
    default_id = "Foraging-8x8-2p-1f-coop-v3"

    def episode_end(self, terminated, truncated):
        if terminated:
            # LBF also reports its internal time limit as termination.
            task_complete = self.base.unwrapped.field.sum() == 0
            return bool(task_complete), not task_complete
        return terminated, truncated

    def task_snapshot(self):
        return {"food_remaining": int(np.count_nonzero(self.base.unwrapped.field))}
