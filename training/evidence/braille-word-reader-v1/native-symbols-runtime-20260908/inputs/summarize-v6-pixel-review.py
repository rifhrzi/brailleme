"""Summarize actual v6 screenshots/log metadata without ADB or model inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
STAGE = Path('C:/bm-symbols-apk-20260908/evidence')
FILES = {'tracking_review': OUT/'overlay-review-v6-tracking-masked-v2.json',
         'photo_review': OUT/'overlay-review-v6-first-photo-v1.json',
         'tracking_screenshot': STAGE/'v6-tracker-live.png', 'photo_screenshot': STAGE/'v6-first-photo.png',
         'photo_log': STAGE/'v6-first-photo.log'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(lines, name):
    matches = [re.search(r"'"+name+r"', '(\{.*\})'", line) for line in lines]
    return [json.loads(match.group(1)) for match in matches if match]


def union(boxes):
    array = np.array(boxes)
    return [float(array[:, 0].min()), float(array[:, 1].min()), float(array[:, 2].max()), float(array[:, 3].max())]


def area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def intersection(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


tracking = json.loads(FILES['tracking_review'].read_text())
photo = json.loads(FILES['photo_review'].read_text())
assert tracking['fit']['status'] == photo['fit']['status'] == 'usable_pixel_alignment'
matrix = np.array(tracking['fit']['matrix'])
translation = np.array(tracking['fit']['translation'])
inverse = np.linalg.inv(matrix)
placement = np.array([62., 383.])


def source_box(box):
    corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    projected = (corners-translation) @ inverse.T - placement
    return [float(projected[:, 0].min()), float(projected[:, 1].min()),
            float(projected[:, 0].max()), float(projected[:, 1].max())]


green = sorted([row['box_xyxy'] for row in tracking['green_detection']['components'] if row['rectangle_candidate']], key=lambda b: b[0])
assert len(green) == 11
rgb = np.asarray(Image.open(FILES['tracking_screenshot']).convert('RGB'))
white = (rgb.min(axis=2) >= 235) & (rgb.max(axis=2).astype(np.int16)-rgb.min(axis=2) <= 16)
inside = np.zeros(white.shape, bool)
inside[817:1254, 158:921] = True
ys, xs = np.nonzero(white & inside)
white_box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
lines = FILES['photo_log'].read_text().splitlines()
capture = event(lines, 'BRAILLE_WORD_CAPTURE_GEOMETRY')[-1]
timing = event(lines, 'BRAILLE_WORD_PHOTO_METRICS')[-1]
metrics = event(lines, 'BRAILLE_WORD_TRACKING_METRICS')[-1]
roi = capture['roi']; iw, ih = capture['upright']['width'], capture['upright']['height']
preview = tracking['preview_xywh']; px, py, pw, ph = preview
scale = max(pw/iw, ph/ih)
offset = np.array([px-(iw*scale-pw)/2, py-(ih*scale-ph)/2])
photo_roi = [roi['x']*iw, roi['y']*ih, (roi['x']+roi['width'])*iw, (roi['y']+roi['height'])*ih]
capture_screen = [photo_roi[0]*scale+offset[0], photo_roi[1]*scale+offset[1],
                  photo_roi[2]*scale+offset[0], photo_roi[3]*scale+offset[1]]
known_roi = tracking['projected_startup_roi_xyxy']
known_source = source_box(known_roi)
green_union = union(green)
pixel_corners = np.array([[158, 817], [920, 817], [920, 1253], [158, 1253]])
feed_corners = (pixel_corners-translation) @ inverse.T
photo_matrix, photo_translation = np.array(photo['fit']['matrix']), np.array(photo['fit']['translation'])
photo_corners = feed_corners @ photo_matrix.T + photo_translation
delta = photo_corners-pixel_corners
overlap = area(intersection(capture_screen, known_roi))
report = {
    'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Actual v6 emulator screenshot pixels and recorded capture metadata; no ADB, inference, source changes or acceptance claim.',
    'inputs': {name: {'path': str(path), 'sha256': sha(path)} for name, path in FILES.items()},
    'script_sha256': sha(Path(__file__)),
    'preview_xywh': preview,
    'pixel_alignment': {'tracking_correlation': tracking['fit']['correlation'], 'tracking_rmse_8bit': tracking['fit']['rmse_8bit'],
                        'photo_correlation': photo['fit']['correlation'], 'photo_rmse_8bit': photo['fit']['rmse_8bit'],
                        'photo_minus_tracking_corners_px': delta.tolist(),
                        'maximum_corner_distance_px': float(np.linalg.norm(delta, axis=1).max()),
                        'uncertainty': 'Approximately one screenshot pixel including edge interpretation; left/right fit variations retained in input reports, not a confidence interval.'},
    'known_fixture_roi': {'expected_text': 'rawenstwa', 'source_xyxy': known_source, 'screen_xyxy': known_roi,
                         'whole_scene_label': False, 'individual_cells_annotated': False},
    'detected_overlay': {'green_rectangle_count': len(green), 'green_screen_xyxy': green,
                         'green_source_xyxy': [source_box(box) for box in green],
                         'green_union_screen_xyxy': green_union, 'green_union_source_xyxy': source_box(green_union),
                         'white_outer_screen_xyxy': white_box, 'white_outer_source_xyxy': source_box(white_box)},
    'capture_metadata': capture,
    'capture_projection': {'upright_photo_roi_xyxy': photo_roi, 'screen_xyxy': capture_screen,
                           'source_xyxy': source_box(capture_screen),
                           'screen_minus_white_outer_edge_px': (np.array(capture_screen)-white_box).tolist(),
                           'known_roi_area_covered_fraction': overlap/area(known_roi),
                           'known_roi_height_covered_fraction': max(0, min(capture_screen[3], known_roi[3])-max(capture_screen[1], known_roi[1]))/(known_roi[3]-known_roi[1]),
                           'projection_basis': 'Recorded upright photo 960x1280 and square preview975px with cover display; independent screenshot fit confirms global content alignment. Metadata projection is not a saved raw-photo pixel crop.'},
    'runtime_observed': {'photo_timing': timing, 'tracking_metrics': metrics,
                         'admitted_per_second_over_metric_elapsed': metrics['admitted']/(metrics['elapsedMs']/1000)},
    'findings': [
        'Live-preview and displayed-photo source pixels globally align to within approximately one screenshot pixel, after brightness correction.',
        'The 11 green boxes and frozen capture ROI cover the upper region of the middle source row, substantially above the known rawenstwa ROI; they do not describe that complete annotated word.',
        'Captured metadata projected through the observed square cover display is close to the visible outer target (differences retained per edge); this supports consistency of displayed target and frozen photo target.',
        'This does not isolate model localization from frame-crop-to-preview coordinate handling. Both can yield a misplaced target while global preview/photo alignment remains correct. Raw frame input plus proposal coordinates would be needed to isolate that path.',
        'The first actual photo is rejected after1907ms. One rejected capture is not a latency percentile or scan-accuracy success.',
    ],
    'limits': ['No individual full-scene cell labels are asserted; a cell-count accuracy or model-confusion claim cannot be calculated.',
               'Screenshot and shutter were not simultaneous; EMA/target motion, border thickness and snapshot timing affect small edge differences.',
               'Known ROI is a prior native startup fixture annotation, not a label of every word/row in the full source.',
               'No crop tightness or detection threshold was adjusted based on these results; original calibrated inputs remain frozen.',
               'All results are emulator-only; no physical-paper or production-readiness claim.'],
}
target = OUT/'v6-pixel-geometry-review.json'
with target.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps({'report': str(target), 'sha256': sha(target), 'pixel_alignment': report['pixel_alignment'],
                  'green_union_source_xyxy': report['detected_overlay']['green_union_source_xyxy'],
                  'white_outer_screen_xyxy': white_box, 'capture_projection': report['capture_projection']}, indent=2))
