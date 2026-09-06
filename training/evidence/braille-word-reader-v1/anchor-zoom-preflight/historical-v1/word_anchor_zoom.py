"""PRIVATE: retain one original full cell in the existing isotropic affine."""
import numpy as np


def anchor_preserving_zoom(boxes, eligible, zoom, translate_x, translate_y):
    """Cap magnification only; no RNG, raster, label, or visibility mutation.

    The nearest-canvas-center originally full cell is the anchor (index ties).
    A cap infeasible at the existing minimum .8 leaves the original draw intact.
    This promises affine containment only; later rotation can still cut cells.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or eligible.shape != boxes.shape[:1]:
        raise ValueError('Eligibility must identify each known original full box')
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite([zoom, translate_x, translate_y])):
        raise ValueError('Geometry must be finite')
    if not .8 <= zoom <= 1.7 or abs(translate_x) > .04*480 or abs(translate_y) > .04*192:
        raise ValueError('Anchor cap uses the existing isotropic illumination affine only')
    inside = (boxes[:,0]>=0)&(boxes[:,1]>=0)&(boxes[:,2]<=480)&(boxes[:,3]<=192)
    positive = (boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    ids = np.flatnonzero(eligible & inside & positive)
    if not len(ids):
        return float(zoom)
    centers = (boxes[ids,:2]+boxes[ids,2:])/2
    anchor = int(ids[np.argmin(np.sum((centers-[240.,96.])**2, axis=1))])
    bound = float('inf')
    for lo, hi, center, shift, size in ((0,2,240.,translate_x,480.), (1,3,96.,translate_y,192.)):
        for edge in (lo, hi):
            delta = float(boxes[anchor,edge]-center)
            if delta < 0:
                bound = min(bound, (center+shift)/(-delta))
            elif delta > 0:
                bound = min(bound, (size-center-shift)/delta)
    if bound >= zoom or bound <= .8:
        return float(zoom)
    # Numerical containment only: move at most eight float64 ULPs below the
    # analytic boundary, evaluating the SAME arithmetic as affine_patch.
    capped = float(bound)
    for _ in range(8):
        capped = float(np.nextafter(capped, 0.))
        if capped < .8:
            return float(zoom)
        transformed = boxes[anchor]*capped + np.array([240*(1-capped)+translate_x,96*(1-capped)+translate_y]*2)
        if transformed[0]>=0 and transformed[1]>=0 and transformed[2]<=480 and transformed[3]<=192:
            return capped
    raise ValueError('Could not establish numerical affine containment')
