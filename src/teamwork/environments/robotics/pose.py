"""Local grasp-relative coordinates for the opposed Panda controllers."""

import numpy as np
from scipy.spatial.transform import Rotation

WORLD_TO_BASE = (
    Rotation.from_euler("z", -90, degrees=True).as_matrix(),
    Rotation.from_euler("z", 90, degrees=True).as_matrix(),
)
POSITION_STEP = 0.05
ROTATION_STEP = 0.5
DOWN = np.array([0, 0, -1])


def pose_error(observation, reference, index):
    """Translation and rotation relative to the grasp reference, in OSC input units."""
    position, quaternion = observation[:3], observation[3:7]
    target_position, target_orientation = reference
    orientation = Rotation.from_quat(quaternion).as_matrix()
    base = WORLD_TO_BASE[index]
    translation = base @ (target_position - position) / POSITION_STEP
    rotation = (
        Rotation.from_matrix(base @ target_orientation @ orientation.T @ base.T).as_rotvec()
        / ROTATION_STEP
    )
    return np.concatenate((translation, rotation))
