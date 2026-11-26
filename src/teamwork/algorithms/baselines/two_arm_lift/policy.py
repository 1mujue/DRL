"""Scripted positive control for the opposed Panda setup, using only local observations."""

import numpy as np

from ....environments.robotics.pose import POSITION_STEP, pose_error
from ....environments.robotics.two_arm_lift.environment import LIFT_PHASE_START
from ....environments.robotics.two_arm_lift.reference import grasp_reference
from ...base import Policy

APPROACH_STEPS = 40
CLOSE_STEP = 75


class ScriptedLiftPolicy(Policy):
    def __init__(self, *, single_arm=False):
        self.single_arm = single_arm
        self.reset()

    def reset(self):
        self.step = 0
        self.targets = {}

    def commands(self, observations):
        """Pre-clipping controller commands, useful as behavior-cloning targets."""
        actions = {}
        for index, agent in enumerate(("arm_0", "arm_1")):
            if self.single_arm and index == 1:
                actions[agent] = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                continue
            observation = observations[agent]
            if agent not in self.targets:
                self.targets[agent] = grasp_reference(observation)
            error = pose_error(observation, self.targets[agent], index)
            if self.step < APPROACH_STEPS:
                lift = 0.06
            elif self.step < LIFT_PHASE_START:
                lift = 0
            else:
                lift = min(0.19, (self.step - LIFT_PHASE_START) * 0.004)
            # Both opposed bases share the world's vertical axis.
            error[2] += lift / POSITION_STEP
            actions[agent] = np.concatenate(
                (0.9 * error[:3], 0.8 * error[3:], [1 if self.step >= CLOSE_STEP else -1])
            )
        self.step += 1
        return actions

    def act(self, observations, deterministic=True):
        return {
            agent: command.clip(-1, 1).astype(np.float32)
            for agent, command in self.commands(observations).items()
        }
