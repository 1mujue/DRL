"""robosuite 双 Panda 机械臂交接场景。"""

from typing import ClassVar

import numpy as np

from ..base import TwoPandaEnvironment


class TwoArmHandover(TwoPandaEnvironment):
    task = "TwoArmHandover"
    camera = "agentview"
    metadata: ClassVar = {
        "name": "two_arm_handover",
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        render_mode=None,
        *,
        horizon=200,
        reward_shaping=True,
        scene_seed=0,
        pose_feedback=False,
    ):
        if type(scene_seed) is not int or scene_seed < 0:
            raise ValueError("TwoArmHandover scene_seed must be a nonnegative integer")
        if type(pose_feedback) is not bool:
            raise ValueError("TwoArmHandover pose_feedback must be boolean")
        self.pose_feedback = pose_feedback
        self._reference = None
        if pose_feedback:
            from .reference import HANDOVER_PHASES

            self.pose_phases = HANDOVER_PHASES
        # Hammer geometry is sampled when robosuite builds the scene. Fix that seed separately
        # from the per-episode seed, which controls the initial placement and orientation.
        super().__init__(
            render_mode,
            horizon=horizon,
            reward_shaping=reward_shaping,
            prehensile=True,
            seed=scene_seed,
        )

    def object_observations(self, raw, index):
        return raw["hammer_pos"], raw["handle_xpos"], raw["hammer_quat"]

    def reset(self, seed=None, options=None):
        self._reference = None
        return super().reset(seed, options)

    def local_grasp_contacts(self):
        env = self._env
        return {
            "arm_0": bool(env._check_grasp(env.robots[0].gripper["right"], env.hammer)),
            "arm_1": bool(
                env._check_grasp(env.robots[1].gripper["right"], env.hammer.handle_geoms)
            ),
        }

    def local_observations(self, raw):
        observations = super().local_observations(raw)
        if not self.pose_feedback:
            return observations
        if self._reference is None:
            from .reference import HandoverReference

            self._reference = HandoverReference(observations, self._env.hammer.handle_length)
        return self._reference.features(observations, self._step, self.local_grasp_contacts())

    def task_snapshot(self):
        env = self._env
        contacts = self.local_grasp_contacts()
        # sin(pi/2 - arccos(z)) == z. Evaluate directly to keep a vertical hammer
        # finite even when the rotation matrix rounds slightly outside [-1, 1].
        rotation = env.sim.data.body_xmat[env.hammer_body_id].reshape(3, 3)
        axis_z = np.clip(rotation[2, 2], -1.0, 1.0)
        angle_offset = (env.hammer.handle_length / 2 + 2 * env.hammer.head_halfsize) * axis_z
        handle_z = env.sim.data.geom_xpos[env.hammer_handle_geom_id][2]
        table_z = env.sim.data.site_xpos[env.table_top_id][2]
        return {
            "object_height_m": float(handle_z - env.hammer.top_offset[2] - angle_offset - table_z),
            "giver_grasp": contacts["arm_0"],
            "receiver_grasp": contacts["arm_1"],
        }
