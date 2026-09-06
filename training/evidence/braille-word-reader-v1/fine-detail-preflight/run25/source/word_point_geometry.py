"""Track known occupied dot centers through Pillow raster operations.

Points use pixel-index coordinates: integer (0, 0) is the first sample center.
Pillow affine matrices instead use pixel-corner coordinates, hence +.5/- .5.
Visibility is the conservative sample-center hull [0, width-1] x [0, height-1]
after EVERY raster stage. Once lost, a point cannot regain visibility.

This module neither loads data nor changes training targets. The future caller
must resolve points only from hash-bound DSBI TRAIN metadata using the exact
(source, row, column, annotation_line, mask) identity. Angelina rectangles do
not provide these points; unknown grid slots and explicit mask zero must not
be passed as positive cells. Six source points retain their actual middle row.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _size(width, height):
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError('Raster width and height must be positive integers')


def safe_center_mask(points, width, height):
    """Conservative hull, not the wider [-.5, width-.5) pixel footprint."""
    _size(width, height)
    points = np.asarray(points, np.float64)
    if points.ndim < 2 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError('Point coordinates must be finite pairs')
    return ((points[..., 0] >= 0) & (points[..., 0] <= width-1)
            & (points[..., 1] >= 0) & (points[..., 1] <= height-1))


@dataclass(frozen=True)
class OccupiedPointState:
    """One six-point row per known cell; survivors already mask unoccupied bits.

    Arrays are copied and read-only so cached metadata or another batch cannot
    be modified through the state. Each transformation returns a new state.
    """
    points: np.ndarray
    survivors: np.ndarray
    width: int = 480
    height: int = 192

    def __post_init__(self):
        _size(self.width, self.height)
        points = np.array(self.points, dtype=np.float64, copy=True)
        survivors = np.array(self.survivors, copy=True)
        if points.ndim != 3 or points.shape[1:] != (6, 2) or not np.isfinite(points).all():
            raise ValueError('Each known cell needs six finite source points')
        if survivors.dtype != np.bool_ or survivors.shape != points.shape[:2]:
            raise ValueError('Point survivors must be an aligned boolean array')
        survivors &= safe_center_mask(points, self.width, self.height)
        points.setflags(write=False)
        survivors.setflags(write=False)
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, 'survivors', survivors)

    @classmethod
    def from_source_crop(cls, source_points, masks, crop, *, width=480, height=192):
        """Apply the existing integer crop then Pillow resize to exact centers.

        Both crop and resize perform irreversible visibility checks. Masks must
        be the aligned, explicitly known nonzero six-dot labels (all 1..63).
        """
        _size(width, height)
        if (len(crop) != 4 or any(type(v) is not int for v in crop)
                or crop[0] < 0 or crop[1] < 0 or crop[2] <= crop[0] or crop[3] <= crop[1]):
            raise ValueError('Source crop must be a positive integer rectangle')
        points = np.asarray(source_points, np.float64)
        labels = np.asarray(masks)
        if points.ndim != 3 or points.shape[1:] != (6, 2) or not np.isfinite(points).all():
            raise ValueError('Each known cell needs six finite source points')
        if labels.shape != (len(points),):
            raise ValueError('Masks must align with known source cells')
        if len(labels) and (not np.issubdtype(labels.dtype, np.integer)
                            or np.any(labels < 1) or np.any(labels > 63)):
            raise ValueError('Only explicitly known masks 1..63 have point states')
        occupied = ((labels.astype(np.int64)[:, None] >> np.arange(6)) & 1).astype(bool)
        local = points-np.asarray(crop[:2])
        crop_width, crop_height = crop[2]-crop[0], crop[3]-crop[1]
        survivors = occupied & safe_center_mask(local, crop_width, crop_height)
        scale = np.asarray([width/crop_width, height/crop_height])
        resized = (local+.5)*scale-.5
        return cls(resized, survivors, width, height)

    @property
    def has_occupied_center(self):
        return self.survivors.any(axis=1)

    def _updated(self, points, permutation=None):
        survivors = self.survivors & safe_center_mask(points, self.width, self.height)
        if permutation is not None:
            points, survivors = points[:, permutation], survivors[:, permutation]
        return OccupiedPointState(points, survivors, self.width, self.height)

    def affine(self, scale_x, scale_y, translate_x=0., translate_y=0.):
        """Use the same sampled parameters as word_data.affine_patch; no RNG."""
        if (not all(math.isfinite(v) for v in (scale_x, scale_y, translate_x, translate_y))
                or scale_x <= 0 or scale_y <= 0):
            raise ValueError('Affine scales must be positive and parameters finite')
        scale = np.asarray([scale_x, scale_y])
        offset = np.asarray([self.width/2*(1-scale_x)+translate_x,
                             self.height/2*(1-scale_y)+translate_y])
        return self._updated((self.points+.5)*scale+offset-.5)

    def rotate(self, angle_degrees):
        """Pillow positive-angle rotation, expand=False, default center."""
        if not math.isfinite(angle_degrees):
            raise ValueError('Rotation angle must be finite')
        angle = angle_degrees % 360.
        # Pillow has exact cardinal transforms. Trig epsilon at 180/360 could
        # otherwise turn a retained corner into an out-of-hull point. Generic
        # rotate preserves slot identity; half_turn explicitly remaps its bits.
        if angle == 0:
            return self
        if angle == 180:
            return self._updated(np.asarray([self.width-1, self.height-1])-self.points)
        if angle in (90, 270):
            rotation = np.asarray([[0., 1.], [-1., 0.]]) * (1 if angle == 90 else -1)
        else:
            theta = -np.deg2rad(angle)
            rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        center = np.asarray([self.width/2, self.height/2])
        return self._updated((self.points+.5-center)@rotation.T+center-.5)

    def mirror_horizontal(self):
        points = self.points.copy()
        points[..., 0] = self.width-1-points[..., 0]
        # After reflecting coordinates, old right slots are the new left slots.
        return self._updated(points, [3, 4, 5, 0, 1, 2])

    def half_turn(self):
        return self._updated(np.asarray([self.width-1, self.height-1])-self.points,
                             [5, 4, 3, 2, 1, 0])


def visible_partial_candidates(boxes, state, *, min_width=4., min_height=12.):
    """Return (indices, visible boxes) for geometry-only partial supervision.

    The caller must opt in for verified DSBI cells only and set ALL six dot
    labels to unknown. Complete boxes, earlier cuts now wholly inside, missing
    occupied centers, and whole-canvas sizes get no candidate. Source boxes and
    full-cell training targets are never modified. Unknown slots have no state.
    """
    boxes = np.asarray(boxes, np.float64)
    if (boxes.shape != (len(state.points), 4) or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('Valid known-cell boxes must align with point states')
    if not all(math.isfinite(v) and v > 0 for v in (min_width, min_height)):
        raise ValueError('Visible minimum dimensions must be positive and finite')
    visible = np.clip(boxes, [0, 0, 0, 0], [state.width, state.height]*2)
    sizes = visible[:, 2:]-visible[:, :2]
    crosses = ((boxes[:, 0] < 0) | (boxes[:, 1] < 0)
               | (boxes[:, 2] > state.width) | (boxes[:, 3] > state.height))
    eligible = (crosses & state.has_occupied_center
                & (sizes[:, 0] >= min_width) & (sizes[:, 1] >= min_height)
                & (sizes[:, 0] < state.width) & (sizes[:, 1] < state.height))
    indices = np.flatnonzero(eligible)
    return indices, visible[indices]
