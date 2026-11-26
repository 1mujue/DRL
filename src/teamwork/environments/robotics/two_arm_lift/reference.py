"""Grasp reference derived from the lift observation."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..pose import DOWN


def unpack_lift_observation(observation):
    """Read own grip position/orientation, shared pot position, and own handle."""
    position, orientation, pot, handle, _ = np.split(observation, (3, 7, 10, 13))
    return position, orientation, pot, handle


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
