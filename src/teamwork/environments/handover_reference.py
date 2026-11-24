"""Local OSC reference features for a guided tabletop handover."""

import numpy as np
from scipy.spatial.transform import Rotation

from .lift_coordinates import pose_error

ARMS = ("arm_0", "arm_1")
HANDOVER_PHASES = 4
GIVER_CLOSE = 86
RECEIVER_CLOSE = 295
SHARED_CONFIRMATION = 5


def handle_axis(observation):
    return Rotation.from_quat(observation[13:17]).as_matrix()[:, 2]


def grip_orientation(axis, quaternion):
    horizontal = axis.copy()
    horizontal[2] = 0
    norm = np.linalg.norm(horizontal)
    horizontal = horizontal / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    crossbar = np.cross(horizontal, down)
    if np.dot(crossbar, Rotation.from_quat(quaternion).as_matrix()[:, 0]) < 0:
        crossbar *= -1
    return np.column_stack((crossbar, np.cross(down, crossbar), down))


class HandoverReference:
    """Track shared stage and map each arm's own pose to its local reference error."""

    def __init__(self, observations, handle_length):
        giver, receiver = (observations[arm] for arm in ARMS)
        axis = handle_axis(giver)
        self.offset = handle_length / 2 - 0.025
        self.pick = giver[10:13] + self.offset * axis
        self.pick_orientation = grip_orientation(axis, giver[3:7])
        self.above = self.pick + [0, 0, 0.28]
        self.transfer = np.array([self.pick[0] * 0.5, -0.045, self.above[2]])
        yaw = -np.pi / 2 - np.arctan2(axis[1], axis[0])
        self.yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        self.receiver_home = receiver[:3].copy()
        self.receiver_home_orientation = Rotation.from_quat(receiver[3:7]).as_matrix()
        self.receiver_target = None
        self.shared_steps = 0
        self.release_step = None

    def _observe_contacts(self, step, contacts):
        # The release gate exchanges one contact bit per arm. Both must stay in contact first.
        if self.release_step is None and step > RECEIVER_CLOSE:
            self.shared_steps = self.shared_steps + 1 if all(contacts.values()) else 0
            if self.shared_steps >= SHARED_CONFIRMATION:
                self.release_step = step

    def _giver_reference(self, step):
        if step < 45:
            position = self.pick + [0, 0, 0.07]
        elif step < 120:
            position = self.pick
        elif step < 170:
            position = self.pick + [0, 0, min(0.28, (step - 120) * 0.007)]
        else:
            progress = np.clip((step - 170) / 80, 0, 1)
            position = self.above + progress * (self.transfer - self.above)
        if self.release_step is not None:
            progress = np.clip((step - self.release_step - 6) / 35, 0, 1)
            position = self.transfer + progress * np.array([0, -0.11, 0.02])
        angle = np.clip((step - 170) / 80, 0, 1) * self.yaw
        rotation = Rotation.from_rotvec([0, 0, angle]).as_matrix() @ self.pick_orientation
        return position, rotation

    def _receiver_reference(self, observation, step):
        if step < 199:
            return self.receiver_home, self.receiver_home_orientation
        axis = handle_axis(observation)
        orientation = grip_orientation(axis, observation[3:7])
        grip = observation[10:13] - self.offset * axis
        if step < 255:
            return np.array(
                [self.transfer[0], max(0.058, grip[1]), self.transfer[2] + 0.065]
            ), orientation
        if step < 300:
            return grip, orientation
        if self.receiver_target is None:
            self.receiver_target = grip.copy()
        if self.release_step is None:
            return self.receiver_target, orientation
        progress = np.clip((step - self.release_step - 6) / 35, 0, 1)
        return self.receiver_target + [0, 0, 0.023 * progress], orientation

    def features(self, observations, step, contacts):
        self._observe_contacts(step, contacts)
        if self.release_step is not None:
            phase = 3
        elif step >= RECEIVER_CLOSE:
            phase = 2
        else:
            phase = int(step >= GIVER_CLOSE)

        references = (
            self._giver_reference(step),
            self._receiver_reference(observations["arm_1"], step),
        )
        return {
            arm: np.concatenate(
                (
                    observations[arm],
                    pose_error(observations[arm], references[index], index),
                    [phase / HANDOVER_PHASES],
                ),
                dtype=np.float32,
            )
            for index, arm in enumerate(ARMS)
        }
