"""Measure six sampling locations from exact TRAIN points; no model backend."""
import hashlib
import json
from pathlib import Path
import numpy as np

work = Path('.braille-training-work/word-scan-v1-20260905')
manifest_path = work / 'dsbi-training-contexts-v2/train.json'
point_path = work / 'dsbi-point-metadata-v1/points.json'
records = json.loads(manifest_path.read_bytes())
metadata = json.loads(point_path.read_bytes())['records']
unique, complete = {}, []
max_reconstruction_error = 0.
max_fraction_repeat_error = 0.
for row in records:
    entry = metadata[row['image']]
    boxes = np.asarray(row['boxes'], np.float64).reshape(-1, 4)
    source_points = np.asarray(entry['source_points'], np.float64).reshape(-1, 6, 2)
    crop = np.asarray(row['crop'], np.float64)
    scale = np.asarray([480, 192]) / (crop[2:]-crop[:2])
    points = (source_points-crop[:2]+.5)*scale-.5
    sizes = boxes[:, 2:]-boxes[:, :2]
    fractions = (points+.5-boxes[:, None, :2])/sizes[:, None, :]
    reconstructed = boxes[:, None, :2]+fractions*sizes[:, None, :]-.5
    if len(points):
        max_reconstruction_error = max(max_reconstruction_error, float(np.max(np.abs(points-reconstructed))))
    for index, identity in enumerate(row['source_cells']):
        key = (row['source'], identity['row'], identity['col'], identity['annotation_line'], row['masks'][index])
        current = fractions[index]
        if key in unique:
            max_fraction_repeat_error = max(max_fraction_repeat_error, float(np.max(np.abs(current-unique[key]['fractions']))))
        else:
            unique[key] = {'fractions': current, 'complete_seen': False}
        if row['originally_complete'][index]:
            unique[key]['complete_seen'] = True
            complete.append((boxes[index], points[index], current))
assert len(unique) == 9983 and all(row['complete_seen'] for row in unique.values())
assert max_fraction_repeat_error < 1e-12 and max_reconstruction_error < 1e-10
values = np.stack([row['fractions'] for row in unique.values()])
priors = np.median(values, axis=0)
errors = []
for box, points, _ in complete:
    approximated = box[None, :2]+priors*(box[None, 2:]-box[None, :2])-.5
    errors.append(approximated-points)
errors = np.stack(errors)
source_stats = {}
for source in sorted(set(key[0] for key in unique)):
    subset = np.stack([row['fractions'] for key, row in unique.items() if key[0] == source])
    source_stats[source] = {'unique_known_cells': len(subset), 'median_corner_fractions': np.median(subset, axis=0).tolist()}
percentiles = [0, 5, 25, 50, 75, 95, 100]
report = {
    'status': 'passed',
    'scope': 'Exact known DSBI TRAIN source-grid coordinates, including unoccupied slots in known nonzero cells. Unique cells each contribute once to priors. No Angelina/validation/external pixels or model outputs used.',
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'point_metadata_sha256': hashlib.sha256(point_path.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'coordinate_convention': 'q=(source_point-crop_origin+.5)*scale-.5 is pixel-index; f=(q+.5-box_left_top)/box_size; reconstructed q=box_left_top+f*box_size-.5; stride4 feature=(q-3)/4.',
    'slot_order': ['left_top', 'left_middle', 'left_bottom', 'right_top', 'right_middle', 'right_bottom'],
    'axis_order': ['x', 'y'],
    'unique_cells': len(unique),
    'complete_cached_occurrences': len(complete),
    'max_exact_cached_reconstruction_error_pixels': max_reconstruction_error,
    'max_same_source_fraction_difference_across_crops': max_fraction_repeat_error,
    'median_corner_fractions': priors.tolist(),
    'fraction_percentiles': {str(percentile): np.percentile(values, percentile, axis=0).tolist() for percentile in percentiles},
    'fixed_prior_input_pixel_absolute_error_percentiles': {str(percentile): np.percentile(np.abs(errors), percentile, axis=(0, 1)).tolist() for percentile in percentiles},
    'per_source': source_stats,
    'limitations': [
        'Global median fractions are sampling priors, not exact point predictions for every cell.',
        'Angelina rectangles do not provide exact point locations; DSBI priors cannot be claimed verified for all Angelina images.',
        'Rotation turns a cell into an axis-aligned enclosing box; these axis-aligned priors do not encode rotation.',
        'Convolution features integrate neighborhoods; a tap at a nominal point does not isolate that dot.',
        'This audit does not establish recognition improvement or justify model activation.',
    ],
}
output = work / 'slot-prior-audit.json'
output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ('unique_cells', 'complete_cached_occurrences', 'median_corner_fractions', 'fixed_prior_input_pixel_absolute_error_percentiles', 'max_exact_cached_reconstruction_error_pixels', 'max_same_source_fraction_difference_across_crops')}, indent=2))
print(hashlib.sha256(output.read_bytes()).hexdigest())
