"""Independent, deterministic view geometry draws for an opt-in ablation."""
from __future__ import annotations

import numpy as np

VERTICAL_STRETCH_RANGE = (0.9, 1.1)
VERTICAL_STREAM_ID = 0x56535452


def vertical_stretch_factors(seed, dataset_epoch, batch_index, count):
    """Leave the existing image/lighting generator entirely untouched.

    The caller uses one factor per record, and applies it only when the existing
    affine augmentation is selected. This utility never mutates pixels, boxes,
    labels, point visibility, global NumPy state, or a caller-owned generator.
    """
    values = (seed, dataset_epoch, batch_index, count)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0 for value in values):
        raise ValueError('View-geometry seed, epoch, batch index and count must be nonnegative integers')
    sequence = np.random.SeedSequence([int(seed), int(dataset_epoch), int(batch_index), VERTICAL_STREAM_ID])
    generator = np.random.default_rng(sequence)
    return generator.uniform(*VERTICAL_STRETCH_RANGE, size=int(count))
