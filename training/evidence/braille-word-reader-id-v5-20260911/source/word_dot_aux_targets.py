"""Pure NumPy queries for explicit, complete DSBI TRAIN cells only."""
import numpy as np

FEATURE_ORIGIN = 3.0
FEATURE_STRIDE = 4.0
FEATURE_WIDTH = 120
FEATURE_HEIGHT = 48


def complete_dot_queries(point_state, masks, complete):
    """Return fine-grid XY queries and explicit occupancy labels, never guess.

    The caller obtains point_state only from load_point_states' hash-bound DSBI
    TRAIN metadata. Completeness is the existing irreversible per-raster mask.
    No point state means no auxiliary targets, including all Angelina records.
    """
    if point_state is None:
        return np.empty((0, 2), np.float32), np.empty((0, 1), np.float32)
    points = np.asarray(point_state.points, np.float64)
    survivors = np.asarray(point_state.survivors)
    masks = np.asarray(masks)
    complete = np.asarray(complete)
    count = len(masks)
    if (points.shape != (count, 6, 2) or not np.isfinite(points).all()
            or survivors.shape != (count, 6) or survivors.dtype != np.bool_
            or complete.shape != (count,) or complete.dtype != np.bool_
            or not np.issubdtype(masks.dtype, np.integer)
            or np.any(masks < -1) or np.any(masks > 63)):
        raise ValueError('Aligned finite points, explicit mask labels and boolean visibility are required')
    bits = ((masks[:, None] >> np.arange(6)) & 1).astype(np.float32)
    queries = (points - FEATURE_ORIGIN) / FEATURE_STRIDE
    inside = ((queries[..., 0] >= 0) & (queries[..., 0] <= FEATURE_WIDTH - 1)
              & (queries[..., 1] >= 0) & (queries[..., 1] <= FEATURE_HEIGHT - 1)).all(axis=1)
    # False survivor means either absent or lost; only the mask defines absence.
    occupied_survive = ((bits == 0) | survivors).all(axis=1)
    keep = complete & (masks >= 1) & (masks <= 63) & inside & occupied_survive
    return (queries[keep].reshape(-1, 2).astype(np.float32),
            bits[keep].reshape(-1, 1).copy())


def pad_dot_queries(rows):
    """Pad to this actual batch's maximum query count; no truncation or cap."""
    if not rows:
        raise ValueError('A batch must contain at least one record')
    maximum = max(1, max(len(queries) for queries, labels in rows))
    queries = np.zeros((len(rows), maximum, 2), np.float32)
    labels = np.full((len(rows), maximum, 1), -1, np.float32)
    for index, (positions, values) in enumerate(rows):
        if positions.shape != (len(values), 2) or values.shape != (len(values), 1):
            raise ValueError('One explicit label per query is required')
        queries[index, :len(positions)] = positions
        labels[index, :len(values)] = values
    return queries, labels
