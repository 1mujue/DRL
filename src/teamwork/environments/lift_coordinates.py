"""Local grasp-relative coordinates for the opposed Panda controllers."""

import numpy as np
from scipy.spatial.transform import Rotation

from .two_arm_lift import unpack_lift_observation

WORLD_TO_BASE = (
    Rotation.from_euler("z", -90, degrees=True).as_matrix(),
    Rotation.from_euler("z", 90, degrees=True).as_matrix(),
)
POSITION_STEP = 0.05
ROTATION_STEP = 0.5
DOWN = np.array([0, 0, -1])


def grasp_reference(observation):
    """Locate the assigned handle's outward bar from its initial local observation."""
    _, quaternion, pot, handle = unpack_lift_observation(observation)
    radial = handle - pot
    radial[2] = 0
    radial /= np.linalg.norm(radial)
    initial_x = Rotation.from_quat(quaternion).as_matrix()[:, 0]
    x_axis = radial if np.dot(radial, initial_x) >= 0 else -radial
    orientation = np.column_stack((x_axis, np.cross(DOWN, x_axis), DOWN))
    return handle + 0.005 * radial, orientation


def pose_error(observation, reference, index):
    """Translation and rotation relative to the grasp reference, in OSC input units."""
    position, quaternion, _, _ = unpack_lift_observation(observation)
    target_position, target_orientation = reference
    orientation = Rotation.from_quat(quaternion).as_matrix()
    base = WORLD_TO_BASE[index]
    translation = base @ (target_position - position) / POSITION_STEP
    rotation = (
        Rotation.from_matrix(base @ target_orientation @ orientation.T @ base.T).as_rotvec()
        / ROTATION_STEP
    )
    return np.concatenate((translation, rotation))
