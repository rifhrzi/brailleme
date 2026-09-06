"""Ideal rotated straight-row contract probe; no model or image input."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path.cwd()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
OUT = WORK / 'tilted-row-contract-diagnostic-v1'
assert not OUT.exists()
OUT.mkdir()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
source_names = ['training/scripts/word_decode.py', 'training/scripts/word_data.py',
    'training/scripts/benchmark_angelina.py', 'training/scripts/word_anchor_zoom.py',
    'src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/types/word-scan.ts',
    'scripts/word-native-parity/geometry-reference.cjs',
    '.braille-training-work/word-scan-v1-20260905/tilted_row_contract_diagnostic.py',
    '.braille-training-work/word-scan-v1-20260905/tilted_row_contract_diagnostic.cjs']
protocol = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Synthetic ideal outputs only. No ML, pixels, model outputs, validation405, external607, selection, calibration, or decoder changes.',
    'lengths': [1, 8, 9, 12], 'angles_degrees': [0, -3, 3, -5, 5],
    'unrotated_cell_width_pixels': 18, 'unrotated_cell_height_pixels': 32,
    'center_pitch_pixels': 28, 'rotation_origin_pixels': [240, 96],
    'canvas_pixels': [480, 192], 'guide_xyxy': [0, 0, 1, 1],
    'word': 'MAMAMAMAMAMA', 'score_and_selected_dot_probability': 1,
    'geometry': 'Rotate every center and all cell corners by the same angle. Encode the exact resulting axis-aligned bounding rectangle at the stride4 center site, casting tensor fields to float32.',
    'source_sha256': {name: sha(ROOT / name) for name in source_names},
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

import numpy as np
sys.path.insert(0, str(ROOT / 'training/scripts'))
from word_decode import decode, DEFAULT_THRESHOLDS, DECODER_VERSION
from benchmark_angelina import LETTER_BY_MASK
masks = {letter.upper(): mask for mask, letter in LETTER_BY_MASK.items()}
rows = []
with (OUT / 'outputs.f32le').open('xb') as raw:
    for length in protocol['lengths']:
        text = protocol['word'][:length]
        for angle in protocol['angles_degrees']:
            theta = math.radians(angle)
            cosine, sine = math.cos(theta), math.sin(theta)
            width, height = 18 * abs(cosine) + 32 * abs(sine), 18 * abs(sine) + 32 * abs(cosine)
            output = np.zeros((48, 120, 11), dtype='<f4')
            centers, sites, pixel_boxes = [], set(), []
            for index, letter in enumerate(text):
                dx = (index - (length - 1) / 2) * 28
                cx, cy = 240 + dx * cosine, 96 + dx * sine
                gx, gy = math.floor(cx / 4), math.floor(cy / 4)
                assert (gy, gx) not in sites
                sites.add((gy, gx))
                output[gy, gx] = [1, cx / 4 - gx, cy / 4 - gy, width / 480, height / 192,
                    *[(masks[letter] >> bit) & 1 for bit in range(6)]]
                centers.append([cx, cy])
                pixel_boxes.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
            output.tofile(raw)
            result = decode(output, DEFAULT_THRESHOLDS, [0., 0., 1., 1.])
            assert len(result['cells']) == length
            assert all(box[0] >= DEFAULT_THRESHOLDS['edgeMargin'] * 480 and box[1] >= DEFAULT_THRESHOLDS['edgeMargin'] * 192
                and box[2] <= (1 - DEFAULT_THRESHOLDS['edgeMargin']) * 480 and box[3] <= (1 - DEFAULT_THRESHOLDS['edgeMargin']) * 192 for box in pixel_boxes)
            assert width >= DEFAULT_THRESHOLDS['minCellWidth'] * 480 and height >= DEFAULT_THRESHOLDS['minCellHeight'] * 192
            gap = 0 if length == 1 else 28 * cosine / height
            assert gap <= DEFAULT_THRESHOLDS['maxGapRatio'] and 28 * cosine >= width
            rows.append({'index': len(rows), 'length': length, 'angle_degrees': angle,
                'expected_text': text, 'actual_python_text': result['text'], 'actual_python_reason': result['reason'],
                'actual_python_cells': result['cells'], 'pixel_centers': centers, 'pixel_boxes': pixel_boxes,
                'aabb_width_pixels': width, 'aabb_height_pixels': height,
                'center_y_span_pixels': (length - 1) * 28 * abs(sine),
                'row_span_limit_pixels': DEFAULT_THRESHOLDS['maxRowDeviation'] * height,
                'neighbor_gap_ratio': gap, 'pairwise_overlap': 0,
                'all_size_gap_nms_edge_conditions_pass': True})
for name, digest in protocol['source_sha256'].items():
    assert sha(ROOT / name) == digest
assert not any(name in sys.modules for name in ['torch', 'tensorflow', 'keras'])
report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': protocol['scope'],
    'decoder_version': DECODER_VERSION, 'protocol_sha256': sha(OUT / 'protocol.json'),
    'source_sha256': protocol['source_sha256'], 'thresholds': DEFAULT_THRESHOLDS,
    'raw_outputs_sha256': sha(OUT / 'outputs.f32le'), 'raw_outputs_size_bytes': (OUT / 'outputs.f32le').stat().st_size,
    'no_model_backend_imported': True, 'source_hashes_verified_before_after': True, 'rows': rows}
(OUT / 'python-reference.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'cases': len(rows), 'rejected_cases': [{key: row[key] for key in ['length', 'angle_degrees', 'actual_python_reason']} for row in rows if row['actual_python_reason']]}, indent=2))
