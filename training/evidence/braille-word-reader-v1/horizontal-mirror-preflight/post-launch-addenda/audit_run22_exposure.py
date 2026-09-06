"""Reproduce the frozen TRAIN-only expected mirror label frequency report."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent
report_path = root/'run22/training-mirror-exposure.json'
report = json.loads(report_path.read_text())
config_path = root/'run20/training-config.json'
config = json.loads(config_path.read_text())
manifest_path = root/'data'/config['train_manifest']
assert hashlib.sha256(config_path.read_bytes()).hexdigest() == report['sampling_config_sha256']
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == report['manifest_sha256']
families = defaultdict(list)
for row in json.loads(manifest_path.read_text()):
    if row['boxes']:
        families[row.get('source_family', '/'.join(row['source'].split('/')[:2]))].append(row)
weighted, total = np.zeros(6, np.float64), 0.
details = {}
for family, rows in sorted(families.items()):
    quota = config['sampling']['families'][family]['records_per_epoch']
    counts, complete = np.zeros(6, np.float64), 0
    for row in rows:
        for box, mask, was_complete in zip(row['boxes'], row['masks'], row.get('originally_complete', [True]*len(row['boxes']))):
            if was_complete and box[0] >= 0 and box[1] >= 0 and box[2] <= 480 and box[3] <= 192:
                complete += 1
                counts += np.array([(int(mask) >> i) & 1 for i in range(6)])
    weight = quota/len(rows)
    weighted += counts*weight
    total += complete*weight
    details[family] = {'positive_records': len(rows), 'expected_record_draws_per_epoch': quota,
                       'complete_occurrences_before_augmentation': complete,
                       'occupied_dot_counts': counts.astype(int).tolist()}
before = weighted/total
after = (before+before[[3, 4, 5, 0, 1, 2]])/2
assert details == report['families']
assert total == report['expected_complete_cell_occurrences_per_epoch']
np.testing.assert_array_equal(before, report['before_mirror_dot_frequencies'])
np.testing.assert_array_equal(after, report['after_probability_half_mirror_expected_frequencies'])
print(json.dumps({'status': 'passed', 'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'scope': 'Exact source-label arithmetic reproduction; no images, inference or accuracy estimate.'}, indent=2))
