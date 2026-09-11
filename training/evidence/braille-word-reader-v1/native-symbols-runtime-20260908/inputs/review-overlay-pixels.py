"""Offline screenshot/feed alignment and green rectangle measurements; no acceptance decision."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, label, map_coordinates
from scipy.optimize import least_squares, linear_sum_assignment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_box(box, matrix, translation):
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ matrix.T + translation
    return [float(corners[:, 0].min()), float(corners[:, 1].min()),
            float(corners[:, 0].max()), float(corners[:, 1].max())]


def green_rectangles(rgb, preview):
    """Keep components and side occupancy visible; touching boxes may remain ambiguous."""
    x, y, w, h = preview
    color = np.array([8, 116, 67])  # Current success token; source bound in report.
    signed = rgb.astype(np.int16)
    mask = (np.linalg.norm(signed - color, axis=2) <= 32)
    mask &= (signed[:, :, 1] > signed[:, :, 0] + 25) & (signed[:, :, 1] > signed[:, :, 2] + 12)
    within = np.zeros(mask.shape, dtype=bool)
    within[y:y+h, x:x+w] = True
    mask &= within
    labels, count = label(mask, np.ones((3, 3)))
    rows = []
    for number, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        ys, xs = bounds
        height, width = ys.stop - ys.start, xs.stop - xs.start
        pixels = labels[bounds] == number
        if width < 8 or height < 8 or int(pixels.sum()) < 20:
            continue
        thickness = min(8, max(2, min(width, height)//5))
        coverage = [pixels[:thickness].any(axis=0).mean(), pixels[-thickness:].any(axis=0).mean(),
                    pixels[:, :thickness].any(axis=1).mean(), pixels[:, -thickness:].any(axis=1).mean()]
        rows.append({'box_xyxy': [xs.start, ys.start, xs.stop, ys.stop],
                     'green_pixels': int(pixels.sum()), 'side_coverage_top_bottom_left_right': [float(v) for v in coverage],
                     'rectangle_candidate': bool(min(coverage) >= .65),
                     'interpretation': 'Connected green component; touching/overlapping rectangles can merge.'})
    return mask, rows


def fit_pixels(feed_gray, screen_gray, exclude, bounds, initial, stride=3):
    left, top, right, bottom = [int(v) for v in bounds]
    yy, xx = np.mgrid[top:bottom:stride, left:right:stride]
    keep = ~exclude[yy, xx]
    xx, yy = xx[keep], yy[keep]
    observed = gaussian_filter(screen_gray, .75)[yy, xx]
    if len(observed) < 300 or float(observed.std()) < .003:
        return {'status': 'insufficient_texture', 'samples': len(observed)}
    points = np.stack([xx, yy])
    smooth = gaussian_filter(feed_gray, .45)
    matrix, translation = initial
    a, b, c, d = matrix.ravel()
    start = np.array([a, b, c, d, *translation, 1., 0.])
    margin = max(abs(a), abs(d)) * .04
    delta = np.array([margin, margin, margin, margin, 30., 30., .5, .2])

    def residual(parameters):
        mat = parameters[:4].reshape(2, 2)
        coordinates = np.linalg.solve(mat, points - parameters[4:6, None])
        expected = map_coordinates(smooth, [coordinates[1], coordinates[0]], order=1, mode='nearest')
        return expected * parameters[6] + parameters[7] - observed

    fit = least_squares(residual, start, bounds=(start-delta, start+delta),
                        loss='soft_l1', f_scale=.008, x_scale='jac', max_nfev=50)
    error = residual(fit.x)
    prediction = error + observed
    corr = float(np.corrcoef(prediction, observed)[0, 1])
    rmse = float(np.sqrt(np.mean(error**2))*255)
    identifiable = bool(fit.success and np.isfinite(corr) and corr >= .90 and rmse <= 12)
    return {'status': 'usable_pixel_alignment' if identifiable else 'unreliable_pixel_alignment',
            'converged': bool(fit.success), 'samples': len(observed), 'parameters': fit.x.tolist(),
            'matrix': fit.x[:4].reshape(2, 2).tolist(), 'translation': fit.x[4:6].tolist(),
            'correlation': corr if np.isfinite(corr) else None, 'rmse_8bit': rmse,
            'p95_abs_residual_8bit': float(np.quantile(np.abs(error), .95)*255),
            'sample_bounds_xyxy': [left, top, right, bottom], 'sample_stride': stride}


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap/union if union > 0 else 0.


def compare_boxes(rectangles, expected):
    eligible = [(i, row) for i, row in enumerate(rectangles) if row['rectangle_candidate']]
    if not eligible or not expected:
        return []
    scores = np.array([[iou(row['box_xyxy'], box) for box in expected] for _, row in eligible])
    row_indices, col_indices = linear_sum_assignment(-scores)
    comparisons = []
    for ri, ci in zip(row_indices, col_indices):
        if scores[ri, ci] <= 0:
            continue
        actual = np.array(eligible[ri][1]['box_xyxy'])
        exp = np.array(expected[ci])
        comparisons.append({'overlay_component': eligible[ri][0], 'annotation_cell': int(ci),
                            'iou': float(scores[ri, ci]), 'edge_delta_left_top_right_bottom_px': (actual-exp).tolist(),
                            'center_distance_px': float(np.linalg.norm((actual[:2]+actual[2:]-exp[:2]-exp[2:])/2))})
    return comparisons


def review(screenshot, manifest_path, entry_name, output, preview=None):
    if output.exists():
        raise ValueError('Output already exists; preserve prior evidence.')
    manifest = json.loads(manifest_path.read_text())
    candidates = [row for row in manifest['rows'] if row['output'] == entry_name]
    if len(candidates) != 1:
        raise ValueError('Entry must identify exactly one frozen manifest row.')
    row = candidates[0]
    mapping_path = Path(manifest['mapping_report'])
    if sha(mapping_path) != manifest['mapping_report_sha256']:
        raise ValueError('Frozen calibration hash mismatch.')
    mapping = json.loads(mapping_path.read_text())
    feed_path = manifest_path.parent / row['output']
    source_path = Path(row['source'])
    if sha(feed_path) != row['output_sha256'] or sha(source_path) != row['source_sha256']:
        raise ValueError('Frozen feed/source pixels changed.')
    feed = np.asarray(Image.open(feed_path).convert('RGB'))
    screen = np.asarray(Image.open(screenshot).convert('RGB'))
    old_preview = np.array(mapping['preview_rectangle_xywh'])
    preview = list(map(int, preview if preview is not None else old_preview))
    x, y, w, h = preview
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > screen.shape[1] or y+h > screen.shape[0]:
        raise ValueError('Preview must be inside the actual screenshot.')
    scale = np.diag(np.array([w, h])/old_preview[2:])
    initial_matrix = scale @ np.array(mapping['feed_to_preview']['matrix'])
    initial_translation = scale @ np.array(mapping['feed_to_preview']['translation']) + [x, y]
    projected = project_box(row['placement'], initial_matrix, initial_translation)
    bounds = [max(x, int(projected[0])+4), max(y, int(projected[1])+4),
              min(x+w, int(projected[2])-4), min(y+h, int(projected[3])-4)]
    mask, rectangles = green_rectangles(screen, preview)
    # White outer frame is also UI, and otherwise dominates squared residuals.
    # Use a fixed color rule, independent of alignment quality or model output.
    white = (screen.min(axis=2) >= 235) & ((screen.max(axis=2).astype(np.int16)-screen.min(axis=2)) <= 16)
    within = np.zeros(mask.shape, dtype=bool)
    within[y:y+h, x:x+w] = True
    white &= within
    excluded = binary_dilation(mask | white, iterations=8)
    weights = np.array([.2126, .7152, .0722])/255
    fit = fit_pixels(feed @ weights, screen @ weights, excluded, bounds, (initial_matrix, initial_translation))
    variants, expected, comparisons, roi = [], [], [], None
    if fit['status'] == 'usable_pixel_alignment':
        matrix, translation = np.array(fit['matrix']), np.array(fit['translation'])
        left, top, right, bottom = bounds
        middle = (left+right)//2
        for piece in [[left, top, middle, bottom], [middle, top, right, bottom]]:
            variants.append(fit_pixels(feed @ weights, screen @ weights, excluded, piece, (matrix, translation)))
        sx, sy = row['realized_scale_xy']
        px, py = row['placement'][:2]
        for box in row.get('original_annotation', {}).get('boxes', []):
            feed_box = [px+box[0]*sx, py+box[1]*sy, px+box[2]*sx, py+box[3]*sy]
            expected.append(project_box(feed_box, matrix, translation))
        if row.get('roi_source_normalized'):
            b = row['roi_source_normalized']; sw, sh = row['source_size']
            roi = project_box([px+b['x']*sw*sx, py+b['y']*sh*sy,
                               px+(b['x']+b['width'])*sw*sx, py+(b['y']+b['height'])*sh*sy], matrix, translation)
        comparisons = compare_boxes(rectangles, expected)
    report = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Offline pixel geometry review, not model/physical-camera/readiness acceptance.',
        'script_sha256': sha(Path(__file__)),
        'inputs': {name: {'path': str(path.resolve()), 'sha256': sha(path)} for name, path in
                   [('screenshot', screenshot), ('manifest', manifest_path), ('calibration', mapping_path), ('feed', feed_path), ('source', source_path)]},
        'entry': entry_name, 'expected_text': row.get('expected_text'), 'classification': row['classification'],
        'preview_xywh': preview, 'screen_size': [screen.shape[1], screen.shape[0]],
        'green_detection': {'rgb': [8, 116, 67], 'euclidean_tolerance': 32, 'mask_pixels': int(mask.sum()),
                            'masked_from_alignment_with_dilation_px': 8, 'components': rectangles},
        'white_overlay_mask': {'minimum_channel': 235, 'maximum_channel_spread': 16,
                               'mask_pixels': int(white.sum()), 'dilation_px': 8,
                               'limit': 'Any equally bright source pixels are excluded too; no source texture is invented.'},
        'fit': fit, 'spatial_sensitivity_fits': variants,
        'projected_annotation_cells_xyxy': expected, 'projected_startup_roi_xyxy': roi,
        'overlay_annotation_comparisons': comparisons,
        'matched_annotation_count': len(comparisons), 'unmatched_annotation_count': len(expected)-len(comparisons),
        'comparison_status': 'measured_no_acceptance_threshold' if expected else 'not_assessed_no_usable_alignment_or_cell_annotations',
        'limits': [
            'Alignment uses only grayscale source/screenshot pixels with green and near-white overlays masked; model predictions never determine fitting or input choice.',
            'Fit is local to frozen calibration (matrix ±4%, translation ±30px); poor correlation/RMSE or little texture keeps annotation comparison unassessed.',
            'Source annotations bound original cell patches, not the physical relief-dot ink extents. Green border thickness changes bounding boxes by several screen pixels.',
            'Closed green components are candidates; touching/overlapping rectangles may merge and cannot be separated confidently by this helper.',
            'Affine sensitivity across left/right image subsets is empirical variation, not a confidence interval; retain at least one screenshot pixel edge uncertainty.',
            'A single screenshot cannot validate temporal tracking, native sensor transforms, photo mapping, text accuracy or physical paper.',
            'Startup context has no complete cell annotation: ROI and overlay positions alone are not individual-cell correctness.',
        ],
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshot', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--entry', required=True, help='Output PNG filename from calibrated manifest')
    parser.add_argument('--preview', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; existing files are rejected')
    args = parser.parse_args()
    result = review(args.screenshot, args.manifest, args.entry, args.output, args.preview)
    print(json.dumps({'output': str(args.output), 'fit': result['fit']['status'],
                      'green_components': len(result['green_detection']['components']),
                      'annotation_comparisons': len(result['overlay_annotation_comparisons'])}))


if __name__ == '__main__':
    main()
