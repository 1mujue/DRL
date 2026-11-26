"""Cardinal action encoding shared by the MPE rule policies."""

import numpy as np


def cardinal_action(delta):
    axis = int(np.argmax(np.abs(delta)))
    return (2 if delta[0] > 0 else 1) if axis == 0 else (4 if delta[1] > 0 else 3)
