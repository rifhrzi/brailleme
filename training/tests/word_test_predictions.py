"""Convert synthetic loss targets into ideal decoder outputs for tests only."""
import numpy as np


def center_predictions(target):
    # Gaussian noncenter values weight focal negatives; they are not predicted
    # objects and have no supervised geometry/dots. Leave source targets intact.
    output = target.copy()
    output[..., 0] = np.where(target[..., 0] >= .999, target[..., 0], 0)
    return output
