"""Measure TRAIN-only annotated physical grid ratios; never edit source targets."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent
metadata_path = root/'dsbi-point-metadata-v1/points.json'
expected_metadata = 'ea9592a727f7f3abcc6b017f8f22e143e90ae1541f623c43adf30104021937e8'
assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == expected_metadata
metadata = json.loads(metadata_path.read_text())
unique = {}
for row in metadata['records'].values():
    for cell, mask, points in zip(row['source_cells'], row['masks'], row['source_points']):
        key = row['source'], cell['row'], cell['col'], cell['annotation_line']
        value = {'mask': mask, 'points': points}
        assert key not in unique or unique[key] == value
        unique[key] = value
assert len(unique) == 9983
sources = defaultdict(list)
for key, row in sorted(unique.items()):
    points = np.asarray(row['points'], np.float64)
    x_pitch = points[3, 0]-points[0, 0]
    y_pitch = (points[2, 1]-points[0, 1])/2
    assert x_pitch > 0 and y_pitch > 0
    sources[key[0]].append({'ratio': float(y_pitch/x_pitch),
                            'x_pitch': float(x_pitch), 'mean_y_pitch': float(y_pitch),
                            'upper_y_pitch': float(points[1, 1]-points[0, 1]),
                            'lower_y_pitch': float(points[2, 1]-points[1, 1])})


def summary(values):
    a = np.asarray(values, np.float64)
    return dict(zip(('minimum', 'p05', 'median', 'p95', 'maximum'), map(float, np.quantile(a, [0, .05, .5, .95, 1]))))


source_report = {name: {'unique_cells': len(rows), 'y_over_x_pitch': summary([row['ratio'] for row in rows]),
                         'x_pitch_pixels': summary([row['x_pitch'] for row in rows]),
                         'mean_y_pitch_pixels': summary([row['mean_y_pitch'] for row in rows])}
                 for name, rows in sorted(sources.items())}
ratios = [row['ratio'] for rows in sources.values() for row in rows]
report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'scope': 'Exact DSBI TRAIN grid annotation ratios only; no model, validation, external photos or target edits.',
          'point_metadata_sha256': expected_metadata, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'unique_positive_cells': len(unique), 'nonempty_recto_sources': len(sources),
          'ratio_definition': '((bottom_y-top_y)/2)/(right_x-left_x), using supplied deskewed source pixel coordinates.',
          'unique_cell_weighted_y_over_x_pitch': summary(ratios),
          'equal_source_weighted_median_ratios': summary([row['y_over_x_pitch']['median'] for row in source_report.values()]),
          'by_source': source_report,
          'limitations': ['Grid lines are source annotations, not new optical measurements of dot centers.',
                         'DSBI source images are already deskewed scanner images; these ratios do not establish Angelina camera perspective statistics.',
                         'No annotation bounding-box aspect or external box bias is treated as an optical pitch measurement.',
                         'Paper IV.B supports a separate plus/minus10percent vertical stretch augmentation, but no accuracy benefit is assumed.']}
output = root/'train-pitch-ratio-audit.json'
assert not output.exists(), 'Do not overwrite retained ratio evidence'
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'by_source'}, indent=2))
