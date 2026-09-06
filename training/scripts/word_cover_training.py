"""Prepare additional real TRAIN contexts for audited uncovered source cells.

This creates a separate proposed manifest. It never rewrites existing training,
validation or external fixtures, and it does not load a recognition model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from word_data import HEIGHT, WIDTH, load_page, save_json


def context_for_cell(box, image_size, width_per_height):
    """Use a 2.5:1 original-pixel context, moving it inside the source page."""
    image_w, image_h = image_size
    cell_h = float(box[3]-box[1])
    crop_h = max(1, int(np.ceil(cell_h*width_per_height*HEIGHT/WIDTH)))
    crop_w = max(1, int(np.ceil(crop_h*WIDTH/HEIGHT)))
    if crop_w > image_w or crop_h > image_h:
        return None
    cx, cy = (box[:2]+box[2:])/2
    left = min(max(0, int(np.floor(cx-crop_w/2))), image_w-crop_w)
    top = min(max(0, int(np.floor(cy-crop_h/2))), image_h-crop_h)
    crop = [left, top, left+crop_w, top+crop_h]
    if box[0] < crop[0] or box[1] < crop[1] or box[2] > crop[2] or box[3] > crop[3]:
        return None
    return crop


def prepare(args):
    output = args.output.resolve()
    if (output/'combined-training-manifest.json').exists():
        raise SystemExit('Use a new output directory; prepared data is immutable')
    output.mkdir(parents=True, exist_ok=True)
    source_split_path = args.data/'source-split.json'
    source_split = json.loads(source_split_path.read_text())
    allowed_sources = set(source_split['images']['train'])
    parent_path = args.data/args.parent_manifest
    parent_records = json.loads(parent_path.read_text())
    missing_rows = json.loads(args.uncovered.read_text())
    if not isinstance(missing_rows, list):
        raise ValueError('Uncovered input must be an array of source/annotation_row records')
    if any(row['source'] not in allowed_sources for row in missing_rows):
        raise ValueError('Supplement contains a source outside the frozen training partition')
    by_source = {}
    for row in missing_rows:
        by_source.setdefault(row['source'], set()).add(int(row['annotation_row'])-1)
    additions, events, failed, source_hashes = [], [], [], {}
    factors = (3.125, 5., 7.5)
    for source, pending_rows in sorted(by_source.items()):
        path = args.dataset/source
        image, boxes, masks = load_page(args.dataset, path)
        source_hashes[source] = {
            'image': hashlib.sha256(path.read_bytes()).hexdigest(),
            'csv': hashlib.sha256(path.with_suffix('.csv').read_bytes()).hexdigest(),
        }
        if any(index < 0 or index >= len(boxes) for index in pending_rows):
            raise ValueError(f'Invalid source annotation row: {source}')
        for row in (row for row in missing_rows if row['source']==source):
            index = int(row['annotation_row'])-1
            if int(row['mask']) != int(masks[index]) or np.max(np.abs(np.asarray(row['box'])-boxes[index])) > .001:
                raise ValueError(f'Uncovered audit no longer matches source pixels/CSV: {source}')
        pending = set(pending_rows)
        unable_to_fit = set()
        for anchor in sorted(pending_rows):
            if anchor not in pending:
                continue
            factor = factors[anchor % len(factors)]
            crop = context_for_cell(boxes[anchor], image.size, factor)
            if crop is None:
                unable_to_fit.add(anchor)
                continue
            intersects = ((boxes[:, 2]>crop[0]) & (boxes[:, 3]>crop[1]) &
                          (boxes[:, 0]<crop[2]) & (boxes[:, 1]<crop[3]))
            indices = np.flatnonzero(intersects)
            transformed = ((boxes[indices]-np.asarray(crop[:2]*2))*
                           np.asarray([WIDTH/(crop[2]-crop[0]), HEIGHT/(crop[3]-crop[1])]*2))
            complete = ((boxes[indices, 0]>=crop[0]) & (boxes[indices, 1]>=crop[1]) &
                        (boxes[indices, 2]<=crop[2]) & (boxes[indices, 3]<=crop[3]))
            newly_covered = sorted(set(indices[complete].tolist()) & pending)
            if anchor not in newly_covered:
                raise ValueError('Anchor is not a complete cell in its prepared context')
            filename = f'{len(additions):06d}.png'
            image.crop(crop).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR).save(output/filename, compress_level=1)
            image_path = Path(os.path.relpath(output/filename, args.data.resolve())).as_posix()
            additions.append({'image': image_path, 'source': source, 'crop': crop,
                              'boxes': transformed.tolist(), 'masks': masks[indices].tolist(),
                              'originally_complete': complete.tolist(),
                              'annotation_rows': (indices+1).tolist(),
                              'supplement_id': 'unique-training-cell-coverage-v3'})
            events.append({'image': image_path, 'source': source,
                           'anchor_annotation_row': anchor+1, 'context_width_per_cell_height': factor,
                           'newly_covered_annotation_rows': [index+1 for index in newly_covered]})
            pending.difference_update(newly_covered)
        # Report original-image limitations explicitly; no uncertain relabeling.
        if pending-unable_to_fit:
            raise ValueError('Some uncovered source cells were silently omitted')
        failed.extend({'source': source, 'annotation_row': index+1,
                       'reason': 'complete_cell_context_does_not_fit_original_image'} for index in sorted(pending))
    combined = parent_records+additions
    combined_path = output/'combined-training-manifest.json'
    save_json(combined_path, combined)
    save_json(output/'supplement-records.json', additions)
    save_json(output/'coverage-events.json', events)
    save_json(output/'preparation-provenance.json', {
        'scope': 'Proposed TRAIN-only original-pixel supplemental contexts; no model outputs used.',
        'parent_manifest': str(parent_path),
        'parent_manifest_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        'source_split_sha256': hashlib.sha256(source_split_path.read_bytes()).hexdigest(),
        'uncovered_audit_sha256': hashlib.sha256(args.uncovered.read_bytes()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_records_preserved': len(parent_records), 'added_contexts': len(additions),
        'requested_uncovered_cells': sum(len(rows) for rows in by_source.values()),
        'new_unique_complete_cells': sum(len(row['newly_covered_annotation_rows']) for row in events),
        'uncovered_source_limits': failed, 'source_sha256': source_hashes,
        'combined_manifest_sha256': hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        'context_protocol': 'Row-index modulo3 chooses width3.125/5/7.5 cell heights, aspect2.5, center on audited missing cell and shift inside original page. Greedily cover other missing cells in that context. Keep every intersecting original annotation with its complete flag.',
        'validation_or_external_data_modified': False, 'active_training_manifest_modified': False,
    })
    print(json.dumps({'added_contexts': len(additions), 'combined_records': len(combined),
                      'new_unique_cells': sum(len(row['newly_covered_annotation_rows']) for row in events),
                      'uncovered_source_limits': len(failed)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('C:/bm-datasets/AngelinaDataset'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parent-manifest', default='train-coverage-v2.json')
    parser.add_argument('--uncovered', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())
