"""Decompose fixed held-word errors without modifying labels or thresholds."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from word_data import HEIGHT, WIDTH, save_json
from word_decode import decode, prepare_calibration_outputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-cache', action='store_true', help='Use the saved epoch callback cache, explicitly labelled as TF32 diagnostic only.')
    args = parser.parse_args()
    manifest = args.data/'validation-words.json'
    records = json.loads(manifest.read_text())['records']
    if args.training_cache:
        selected = json.loads((args.run/'selected-validation.json').read_text())
        prediction_file = args.run/'best-word-validation.npz'
        predictions = np.load(prediction_file)['outputs']
        precision = 'Training callback cache; cuDNN TF32 may be enabled. Diagnostic only, not a full-precision selection oracle.'
    else:
        selected = json.loads((args.run/'selected-validation-fullprecision.json').read_text())
        prediction_file = args.run/'backend-fullprecision.npz'
        cached = np.load(prediction_file)
        oracle_manifest = args.data/selected['oracle']['fixture_file']
        assert digest(oracle_manifest) == str(cached['fixture_sha256'])
        oracle_records = json.loads(oracle_manifest.read_text())['records']
        indices = {row['image']: i for i, row in enumerate(oracle_records)}
        predictions = cached['outputs'][[indices[row['image']] for row in records]]
        precision = 'Frozen full-float32 validation oracle.'
    assert len(predictions) == len(records) == 135
    thresholds = selected['thresholds']
    local_outputs = prepare_calibration_outputs(predictions, minimum_proposal=0.)
    totals, outcomes, unmatched_masks = Counter(), Counter(), Counter()
    details = []
    for record, output, peaks in zip(records, predictions, local_outputs):
        result = decode(output, thresholds, record['selection'])
        outcomes['correct' if result['text'] == record['expected'] else ('wrong_accepted' if result['text'] else result['reason'])] += 1
        ys, xs = np.where(peaks[..., 0] > 0.05)
        rows = peaks[ys, xs]
        centers = np.column_stack(((xs+rows[:, 1])*4, (ys+rows[:, 2])*4))
        targets = np.asarray(record['boxes'], np.float64)
        target_centers = (targets[:, :2]+targets[:, 2:])/2
        heights = targets[:, 3]-targets[:, 1]
        used = set()
        cells = []
        for index, (center, height, mask) in enumerate(zip(target_centers, heights, record['masks'])):
            totals['target_cells'] += 1
            candidates = [(float(np.linalg.norm(pc-center)/height), pi) for pi, pc in enumerate(centers) if pi not in used]
            distance, chosen = min(candidates) if candidates else (float('inf'), -1)
            detail = {'index': index, 'mask': mask}
            if distance > .45:
                totals['no_local_peak_over_005'] += 1
                unmatched_masks[str(mask)] += 1
                detail['status'] = 'no_local_peak_over_005'
            else:
                used.add(chosen)
                row = rows[chosen].astype(np.float64)
                score = float(row[0])
                predicted_mask = sum(int(value >= .5) << bit for bit, value in enumerate(row[5:]))
                true_confidence = min(value if (mask >> bit) & 1 else 1-value for bit, value in enumerate(row[5:]))
                width, predicted_height = row[3]*WIDTH, row[4]*HEIGHT
                score_category = 'below_proposal' if score < thresholds['proposalScore'] else ('below_accept' if score < thresholds['acceptScore'] else 'accepted_score')
                totals[score_category] += 1
                totals['matched_wrong_mask' if predicted_mask != mask else 'matched_correct_mask'] += 1
                totals['matched_true_bits_confident' if true_confidence >= thresholds['dotConfidence'] else 'matched_true_bits_not_confident'] += 1
                detail.update(status=score_category, score=score, predicted_mask=predicted_mask, true_bit_confidence=float(true_confidence), center_distance_per_height=distance,
                              signed_center_x_per_height=float((centers[chosen, 0]-center[0])/height), signed_center_y_per_height=float((centers[chosen, 1]-center[1])/height),
                              width_ratio=float(width/(targets[index, 2]-targets[index, 0])), height_ratio=float(predicted_height/height))
            cells.append(detail)
        if result['text'] != record['expected']:
            details.append({'image': record['image'], 'source': record['source'], 'expected': record['expected'], 'predicted': result['text'], 'reason': result['reason'], 'cells': cells})
    report = {'scope': 'Natural held validation only. No external predictions, gradients, threshold selection, relabeling, or fixture filtering.',
              'precision': precision, 'prediction_sha256': digest(prediction_file), 'validation_fixture_sha256': digest(manifest), 'thresholds': thresholds,
              'matching': 'Greedy nearest unused local peak with score >.05, center distance <=.45 target height; approximate diagnostic, not detector AP.',
              'word_outcomes': dict(outcomes), 'cell_counts': dict(totals), 'unmatched_masks': dict(unmatched_masks), 'failed_words': details}
    save_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'failed_words'}, indent=2))


if __name__ == '__main__':
    main()
