"""Reconstruct every DSBI context directly from raw grid text and original RGB.

Independent of word_dsbi.py, its preparer, and their coordinate helpers.
"""
from pathlib import Path
from collections import Counter
import json
import hashlib
import time
import argparse
import numpy as np
from PIL import Image

started = time.perf_counter()
root = Path.cwd()
work = root/'.braille-training-work/word-scan-v1-20260905'
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--folder', default='dsbi-training-contexts-v1')
args = parser.parse_args()
folder = work/args.folder
manifest_path = folder/'train.json'
index_path = folder/'source-index.json'
records = json.loads(manifest_path.read_bytes())
index = json.loads(index_path.read_bytes())
dataset = Path(index['root'])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_hash, index_hash = sha(manifest_path), sha(index_path)
assert manifest_hash == index['generatedManifest']['sha256']
expected_unique, seen_unique = set(), set()
occurrences = ignored = pixel_bytes = 0
max_error = 0.
conservative_flags = []
kinds, families = Counter(), Counter()

for source in index['sources']:
    image_path = dataset/source['original_image']
    annotation_path = dataset/source['original_annotation']
    assert sha(image_path) == source['image_sha256']
    assert sha(annotation_path) == source['annotation_sha256']
    lines = annotation_path.read_text(encoding='utf-8-sig').splitlines()
    cells, slots, unknown = [], set(), []
    if lines:
        xs = list(map(float, lines[1].split()))
        ys = list(map(float, lines[2].split()))
        def full_box(row, col):
            x0, x1 = xs[2*(col-1):2*col]
            y0, _, y1 = ys[3*(row-1):3*row]
            pad = .3*(x1-x0)
            return [x0-pad, y0-pad, x1+pad, y1+pad]
        for line_number, line in enumerate(lines[3:], 4):
            if not line.strip():
                continue
            values = list(map(int, line.split()))
            row, col = values[:2]
            mask = sum(bit*(2**number) for number, bit in enumerate(values[2:]))
            assert len(values) == 8 and all(bit in (0, 1) for bit in values[2:])
            assert (row, col) not in slots
            slots.add((row, col))
            if mask:
                identity = {'row': row, 'col': col, 'annotation_line': line_number}
                cells.append((full_box(row, col), mask, identity))
                expected_unique.add((source['source'], row, col))
        for row in range(1, len(ys)//3+1):
            for col in range(1, len(xs)//2+1):
                if (row, col) not in slots:
                    unknown.append(full_box(row, col))
    with Image.open(image_path) as original:
        rgb = original.convert('RGB')
    for record in (record for record in records if record['source'] == source['source']):
        crop = record['crop']
        x0, y0, x1, y1 = crop
        assert all(type(value) is int for value in crop)
        assert 0 <= x0 < x1 <= rgb.width and 0 <= y0 < y1 <= rgb.height
        intersects = lambda b: b[2]>x0 and b[3]>y0 and b[0]<x1 and b[1]<y1
        transform = lambda b: [(b[0]-x0)*480/(x1-x0), (b[1]-y0)*192/(y1-y0),
                               (b[2]-x0)*480/(x1-x0), (b[3]-y0)*192/(y1-y0)]
        selected = [cell for cell in cells if intersects(cell[0])]
        boxes = np.asarray([transform(cell[0]) for cell in selected], np.float64).reshape(-1, 4)
        assert [cell[1] for cell in selected] == record['masks']
        assert [cell[2] for cell in selected] == record['source_cells']
        actual_boxes = np.asarray(record['boxes']).reshape(-1, 4)
        assert boxes.shape == actual_boxes.shape
        if len(boxes):
            error = float(np.max(np.abs(boxes-actual_boxes)))
            max_error = max(max_error, error)
            assert error < 1e-10
        complete = [b[0]>=x0 and b[1]>=y0 and b[2]<=x1 and b[3]<=y1 for b, _, _ in selected]
        for i, recorded in enumerate(record['originally_complete']):
            if complete[i] != recorded:
                assert complete[i] and not recorded, 'Unsafe recovered completeness'
                assert max(abs(actual_boxes[i]-boxes[i])) < 1e-10
                assert boxes[i,0] == 0 or boxes[i,1] == 0 or boxes[i,2] == 480 or boxes[i,3] == 192
                conservative_flags.append({'image':record['image'], 'source_cell':selected[i][2],
                                           'source_box':selected[i][0], 'crop':crop,
                                           'recorded_box':actual_boxes[i].tolist()})
        for (_, _, cell), valid in zip(selected, complete):
            if valid:
                seen_unique.add((source['source'], cell['row'], cell['col']))
        expected_ignored = np.asarray([transform(b) for b in unknown if intersects(b)]).reshape(-1, 4)
        actual_ignored = np.asarray(record['ignore_boxes']).reshape(-1, 4)
        assert expected_ignored.shape == actual_ignored.shape
        if len(expected_ignored):
            # Order is part of the frozen generator contract, too.
            error = float(np.max(np.abs(expected_ignored-actual_ignored)))
            max_error = max(max_error, error)
            assert error < 1e-10
        cached = work/'data'/record['image']
        assert sha(cached) == record['image_sha256']
        expected_rgb = rgb.crop(crop).resize((480, 192), Image.Resampling.BILINEAR).tobytes()
        with Image.open(cached) as image:
            assert image.size == (480, 192) and image.mode == 'RGB'
            assert image.tobytes() == expected_rgb
        pixel_bytes += len(expected_rgb)
        occurrences += len(selected)
        ignored += len(expected_ignored)
        kinds[record['kind']] += 1
        families[record['source_family']] += 1
    print(json.dumps({'source': source['source'], 'audited_records': sum(families.values())}), flush=True)

assert len(expected_unique) == 9983 and seen_unique == expected_unique
assert sum(families.values()) == len(records) == 2338
assert sha(manifest_path) == manifest_hash and sha(index_path) == index_hash
report = {'status': 'passed' if not conservative_flags else 'conservative_boundary_flags_found',
          'conservative_boundary_flags': conservative_flags,
          'scope': __doc__, 'script_sha256': sha(Path(__file__)),
          'manifest_sha256': manifest_hash, 'source_index_sha256': index_hash,
          'original_sources': len(index['sources']), 'contexts': len(records),
          'unique_positive_source_cells': len(expected_unique), 'complete_unique_source_cells_seen': len(seen_unique),
          'all_intersecting_positive_occurrences_checked': occurrences, 'all_unknown_ignore_boxes_checked': ignored,
          'pixel_bytes_compared': pixel_bytes, 'pixel_mismatches': 0,
          'max_coordinate_error': max_error, 'by_kind': dict(kinds), 'by_family': dict(families),
          'elapsed_seconds': time.perf_counter()-started,
          'limitations': ['Uses source annotations as truth; no independent physical or optical-label certification.',
                          'Verifies pre-augmentation cached crops; training augmentation exposure is audited separately.']}
(folder/'independent-context-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
