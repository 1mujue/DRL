"""Scripted training demonstrations using guided handover features."""

import numpy as np

from ....environments.robotics.two_arm_handover.reference import ARMS, HANDOVER_PHASES
from ...base import Policy


class ScriptedHandoverPolicy(Policy):
    def commands(self, observations):
        commands = {}
        for index, arm in enumerate(ARMS):
            state = observations[arm]
            phase = round(float(state[-1]) * HANDOVER_PHASES)
            close = phase in (1, 2) if index == 0 else phase in (2, 3)
            error = state[-7:-1]
            commands[arm] = np.concatenate((0.9 * error[:3], 0.8 * error[3:], [1 if close else -1]))
        return commands

    @staticmethod
    def actions_from_commands(commands):
        return {arm: np.tanh(command).astype(np.float32) for arm, command in commands.items()}

    def act(self, observations, deterministic=True):
        return self.actions_from_commands(self.commands(observations))
