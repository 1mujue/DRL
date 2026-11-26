"""robosuite TwoArmLift 包装为一臂一智能体的并行环境。"""

import math
from typing import ClassVar

import numpy as np

from ..base import TwoPandaEnvironment

LIFT_PHASE_START = 105


def _positive_scale(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class TwoArmLift(TwoPandaEnvironment):
    task = "TwoArmLift"
    metadata: ClassVar = {"name": "two_arm_lift", "render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        *,
        horizon=200,
        reward_shaping=True,
        pose_feedback=False,
        pot_mass_scale=1.0,
        pot_mass_range=None,
    ):
        if type(pose_feedback) is not bool:
            raise ValueError("TwoArmLift pose_feedback must be boolean")
        if not _positive_scale(pot_mass_scale):
            raise ValueError("pot_mass_scale must be finite and positive")
        if pot_mass_range is not None and (
            not isinstance(pot_mass_range, (list, tuple))
            or len(pot_mass_range) != 2
            or not all(map(_positive_scale, pot_mass_range))
            or pot_mass_range[0] >= pot_mass_range[1]
            or pot_mass_scale != 1.0
        ):
            raise ValueError(
                "pot_mass_range must have increasing positive bounds without a fixed mass"
            )
        self.pot_mass_range = tuple(pot_mass_range) if pot_mass_range is not None else None
        self.current_pot_mass_scale = 1.0
        self._mass_rng = np.random.default_rng()
        self.pose_feedback = pose_feedback
        self._grasp_references = {}
        super().__init__(render_mode, horizon=horizon, reward_shaping=reward_shaping)
        model = self._env.sim.model._model
        root = self._env.pot_body_id
        self._pot_bodies = [root]
        for index in range(root + 1, model.nbody):
            if model.body_parentid[index] in self._pot_bodies:
                self._pot_bodies.append(index)
        self._nominal_mass = model.body_mass[self._pot_bodies].copy()
        self._nominal_inertia = model.body_inertia[self._pot_bodies].copy()
        if pot_mass_scale != 1.0:
            self._apply_pot_mass_scale(pot_mass_scale)

    def _apply_pot_mass_scale(self, scale):
        """Apply density scaling to the original inertial parameters, never the previous draw."""
        from mujoco import mj_setConst

        sim = self._env.sim
        model = sim.model._model
        model.body_mass[self._pot_bodies] = self._nominal_mass * scale
        model.body_inertia[self._pot_bodies] = self._nominal_inertia * scale
        mj_setConst(model, sim.data._data)
        self.current_pot_mass_scale = scale

    def object_observations(self, raw, index):
        return raw["pot_pos"], raw[f"handle{index}_xpos"], raw["pot_quat"]

    def reset(self, seed=None, options=None):
        self._grasp_references.clear()
        if self.pot_mass_range is not None:
            # A separate stream keeps the same seeded placements in both experiments.
            if seed is not None:
                self._mass_rng = np.random.default_rng(
                    np.random.SeedSequence(int(seed), spawn_key=(1,))
                )
            self._apply_pot_mass_scale(float(self._mass_rng.uniform(*self.pot_mass_range)))
        observations, infos = super().reset(seed, options)
        for info in infos.values():
            info["pot_mass_scale"] = self.current_pot_mass_scale
        return observations, infos

    def local_observations(self, raw):
        observations = super().local_observations(raw)
        if self.pose_feedback:
            from ..pose import pose_error
            from .reference import grasp_reference

            for index, (agent, observation) in enumerate(observations.items()):
                if agent not in self._grasp_references:
                    self._grasp_references[agent] = grasp_reference(observation)
                error = pose_error(observation, self._grasp_references[agent], index)
                observations[agent] = np.concatenate(
                    (observation, error, [self._step / self.horizon]), dtype=np.float32
                )
        return observations

    def local_grasp_contacts(self):
        """Idealized local tactile flags, one per gripper and assigned handle."""
        env = self._env
        pot = env.pot
        return {
            agent: bool(env._check_grasp(robot.gripper["right"], geoms))
            for agent, robot, geoms in zip(
                self.possible_agents,
                env.robots,
                (pot.handle0_geoms, pot.handle1_geoms),
                strict=True,
            )
        }

    def task_snapshot(self):
        env = self._env
        data = env.sim.data
        # Match the official height reference. Grasp requires both fingers on the assigned handle.
        pot = env.pot
        height = data.site_xpos[env.pot_center_id, 2] - pot.top_offset[2]
        height -= data.site_xpos[env.table_top_id, 2]
        upright_cosine = data.body_xmat[env.pot_body_id].reshape(3, 3)[2, 2]
        return {
            "lift_height_m": float(height),
            "tilt_degrees": float(np.degrees(np.arccos(np.clip(upright_cosine, -1, 1)))),
            "grasping": tuple(self.local_grasp_contacts().values()),
        }
