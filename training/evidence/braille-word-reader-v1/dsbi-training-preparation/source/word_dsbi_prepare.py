"""Prepare append-only DSBI TRAIN contexts and provenance, without training.

Uses the supplied deskewed recto JPEGs, full grid boxes and original six bits.
Unlisted slots are separate ignore_boxes; explicit mask zero is background.
No existing Angelina manifest, source split, validation or test fixture is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from word_dsbi import DSBI_COORDINATE_SPACE, DSBI_REVISION, read_dsbi_annotation, validate_dsbi_image_bounds


WIDTH, HEIGHT = 480, 192
COLUMN_SPANS = (1, 2, 4, 6, 8, 12)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def read_members(path):
    members = []
    for token in Path(path).read_text(encoding='utf-8-sig').splitlines():
        token = token.strip()
        if not token:
            continue
        relative = PurePosixPath(token.replace('\\', '/'))
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.jpg':
            raise ValueError('DSBI split entries must be relative JPEG paths')
        members.append((token, relative))
    if len({path for _, path in members}) != len(members):
        raise ValueError('Duplicate DSBI source in split metadata')
    return members


def unknown_slot_boxes(annotation):
    known = {(cell.row, cell.col) for cell in annotation.cells}
    boxes = []
    for row in range(len(annotation.horizontal_lines)//3):
        top, _, bottom = annotation.horizontal_lines[row*3:row*3+3]
        for col in range(len(annotation.vertical_lines)//2):
            if (row+1, col+1) in known:
                continue
            left, right = annotation.vertical_lines[col*2:col*2+2]
            extension = .3*(right-left)
            boxes.append((left-extension, top-extension, right+extension, bottom+extension))
    return np.asarray(boxes, np.float64).reshape(-1, 4)


def bounded_context(boxes, image_width, image_height):
    left, top = boxes[:, :2].min(axis=0)
    right, bottom = boxes[:, 2:].max(axis=0)
    pitch = float(np.median((boxes[:, 2]-boxes[:, 0])/1.6))
    cell_height = float(np.max(boxes[:, 3]-boxes[:, 1]))
    width = max(right-left+2*pitch, 3.2*cell_height)
    height = width/2.5
    if width > image_width or height > image_height:
        raise ValueError('Source image cannot contain this full training context')
    x = max(0., min(image_width-width, (left+right-width)/2))
    y = max(0., min(image_height-height, (top+bottom-height)/2))
    return [math.floor(x), math.floor(y), min(image_width, math.ceil(x+width)), min(image_height, math.ceil(y+height))]


def plan_contexts(annotation, image_width, image_height, source_number):
    """Every listed slot anchors one deterministic row band, without word claims."""
    if annotation.is_empty:
        width = min(320., image_width, image_height*2.5)
        height = width/2.5
        for fy in (.2, .5, .8):
            for fx in (.2, .5, .8):
                x = max(0., min(image_width-width, fx*image_width-width/2))
                y = max(0., min(image_height-height, fy*image_height-height/2))
                yield [math.floor(x), math.floor(y), math.ceil(x+width), math.ceil(y+height)], (), 'documented_empty_recto'
        return
    by_row = defaultdict(list)
    for cell in annotation.cells:
        by_row[cell.row].append(cell)
    for row, members in sorted(by_row.items()):
        members.sort(key=lambda cell: cell.col)
        first, band = 0, 0
        while first < len(members):
            span = COLUMN_SPANS[(source_number+row+band) % len(COLUMN_SPANS)]
            last_col = members[first].col+span
            last = first+1
            while last < len(members) and members[last].col < last_col:
                last += 1
            anchors = tuple(members[first:last])
            box = np.asarray([cell.box() for cell in anchors])
            yield bounded_context(box, image_width, image_height), anchors, 'annotated_row_band'
            first, band = last, band+1


def transform_boxes(boxes, crop):
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    transformed = (boxes-np.asarray(crop[:2]*2))*np.asarray([WIDTH/(crop[2]-crop[0]), HEIGHT/(crop[3]-crop[1])]*2)
    visible = (transformed[:, 2]>0)&(transformed[:, 3]>0)&(transformed[:, 0]<WIDTH)&(transformed[:, 1]<HEIGHT)
    return transformed[visible], visible


def context_labels(annotation, crop, unknown_boxes=None):
    positive = [cell for cell in annotation.cells if not cell.is_background]
    source_boxes = np.asarray([cell.box() for cell in positive], np.float64).reshape(-1, 4)
    boxes, visible = transform_boxes(source_boxes, crop)
    cells = [cell for cell, keep in zip(positive, visible) if keep]
    # The bounded crop is inside the original raster. Compare the original
    # full source box before resize: an exact right/bottom boundary can become
    # 480/192 + rounding epsilon after multiplication. No tolerance may restore
    # a truly cropped source cell.
    full_source_boxes = source_boxes[visible]
    complete = (full_source_boxes[:, 0]>=crop[0])&(full_source_boxes[:, 1]>=crop[1])&(full_source_boxes[:, 2]<=crop[2])&(full_source_boxes[:, 3]<=crop[3])
    if unknown_boxes is None:
        unknown_boxes = unknown_slot_boxes(annotation)
    ignored, _ = transform_boxes(unknown_boxes, crop)
    return {'boxes': boxes.tolist(), 'masks': [cell.mask for cell in cells], 'originally_complete': complete.tolist(), 'ignore_boxes': ignored.tolist(), 'source_cells': [{'row': cell.row, 'col': cell.col, 'annotation_line': cell.annotation_line} for cell in cells]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--download-provenance', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Existing cache root, used only to express relative image paths.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output directory; append-only preparation never overwrites an existing dataset')
    provenance = json.loads(args.download_provenance.read_text())
    if provenance['revision'] != DSBI_REVISION:
        raise ValueError('DSBI download revision differs from the pinned parser revision')
    for entry in provenance['sourceFiles']:
        path = args.root/entry['file']
        if path.stat().st_size != entry['sizeBytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Downloaded source differs from provenance: {path}')
    train, test = read_members(args.root/'train.txt'), read_members(args.root/'test.txt')
    if len(train) != 26 or len(test) != 88 or {path for _, path in train}&{path for _, path in test}:
        raise ValueError('Pinned DSBI train/test membership differs')
    protected_names = ('source-split.json', 'train.json', 'validation.json', 'validation-words.json', 'validation-stress-selection.json', 'external-words.json')
    protected = {name: sha256(args.data/name) for name in protected_names}
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    protocol = {'schemaVersion': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Append-only DSBI TRAIN context preparation; no training, inference, validation selection or test pixels.', 'dataset_revision': DSBI_REVISION, 'source_coordinate_space': DSBI_COORDINATE_SPACE, 'image_transform_before_crop': 'Identity on provided already deskewed +recto.jpg; JPEG decode to RGB; never reapply annotation angle.', 'cell_box': 'Full two-column/three-row grid, .3 horizontal column pitch extension on every edge.', 'positive_anchor_policy': {'row_bands': 'Listed slots in row/column order; span determined only by source ordinal, one-based row and band ordinal.', 'column_spans_cycle': COLUMN_SPANS, 'context_width': 'max(anchor union width + two horizontal pitches, 3.2 * maximum full cell height)', 'context_aspect': 2.5, 'pixel_crop_rounding': 'floor left/top; ceil right/bottom after fitting the complete rectangle inside the source image'}, 'unlisted_slots': 'Separate transformed full ignore_boxes; never positive objectness/geometry/dots.', 'explicit_zero_masks': 'Retained in source ledger as background, excluded from positive boxes/masks.', 'empty_recto_side': 'Nine fixed contexts at 20/50/80 percent x/y; 320x128 source pixels, bounded if needed; justified only by documented empty side annotation.', 'resize': {'width': WIDTH, 'height': HEIGHT, 'resampling': 'Pillow Image.Resampling.BILINEAR', 'pillow_version': PILLOW_VERSION, 'png_compress_level': 1}, 'download_provenance_sha256': sha256(args.download_provenance), 'generator_sha256': sha256(__file__), 'parser_sha256': sha256(Path(__file__).with_name('word_dsbi.py')), 'protected_original_hashes': protected}
    protocol['originally_complete'] = 'Compare the full original source box to the bounded integer source crop before resize, with exact inequalities and no restoration tolerance. Crops remain inside the original deskewed raster.'
    save_json(args.output/'preparation-protocol.json', protocol)
    records, sources, coverage = [], [], []
    for source_number, (train_member, relative) in enumerate(train):
        original_image = PurePosixPath('data')/relative.parent/(relative.stem+'+recto.jpg')
        original_annotation = original_image.with_suffix('.txt')
        image_path, annotation_path = args.root/original_image, args.root/original_annotation
        annotation = read_dsbi_annotation(annotation_path)
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        validate_dsbi_image_bounds(annotation, image.width, image.height)
        source = (PurePosixPath('dsbi')/relative.parent/(relative.stem+'+recto.jpg')).as_posix()
        family = PurePosixPath(source).parent.as_posix()
        unknown = unknown_slot_boxes(annotation)
        occurrences, complete_occurrences = Counter(), Counter()
        first_record = len(records)
        for crop, anchors, kind in plan_contexts(annotation, image.width, image.height, source_number):
            labels = context_labels(annotation, crop, unknown)
            patch = image.crop(crop).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
            image_file = args.output/'images'/f'{len(records):06d}.png'
            patch.save(image_file, compress_level=1)
            record = {'image': os.path.relpath(image_file.resolve(), args.data.resolve()).replace(os.sep, '/'), 'image_sha256': sha256(image_file), 'source': source, 'source_family': family, 'source_coordinate_space': DSBI_COORDINATE_SPACE, 'crop': crop, 'kind': kind, 'anchor_slots': [{'row': cell.row, 'col': cell.col, 'mask': cell.mask, 'annotation_line': cell.annotation_line} for cell in anchors], **labels}
            for cell, complete in zip(record['source_cells'], record['originally_complete']):
                key = (cell['row'], cell['col'])
                occurrences[key] += 1
                if complete:
                    complete_occurrences[key] += 1
            records.append(record)
        positive = [cell for cell in annotation.cells if not cell.is_background]
        for cell in positive:
            key = (cell.row, cell.col)
            if complete_occurrences[key] == 0:
                raise ValueError(f'Positive source cell has no complete context: {source} {key}')
            coverage.append({'source': source, 'row': cell.row, 'col': cell.col, 'mask': cell.mask, 'annotation_line': cell.annotation_line, 'context_occurrences': occurrences[key], 'complete_context_occurrences': complete_occurrences[key]})
        sources.append({'source': source, 'source_family': family, 'split': 'train', 'train_member': train_member, 'original_image': original_image.as_posix(), 'original_annotation': original_annotation.as_posix(), 'image_sha256': sha256(image_path), 'annotation_sha256': sha256(annotation_path), 'image_size': [image.width, image.height], 'source_coordinate_space': DSBI_COORDINATE_SPACE, 'angle_degrees_not_applied': annotation.angle_degrees, 'positive_cell_count': len(positive), 'explicit_zero_mask_count': sum(cell.is_background for cell in annotation.cells), 'explicit_zero_cells': [{'row': cell.row, 'col': cell.col, 'annotation_line': cell.annotation_line} for cell in annotation.cells if cell.is_background], 'unlisted_grid_slot_count': len(unknown), 'empty_recto_annotation': annotation.is_empty, 'generated_record_count': len(records)-first_record})
        print(f'DSBI TRAIN {source_number+1}/26: {source}, {len(records)-first_record} contexts', flush=True)
    if len(coverage) != 9983 or len({(row['source'], row['row'], row['col']) for row in coverage}) != 9983:
        raise ValueError('Pinned DSBI positive coverage differs from the independent source census')
    if any(mask == 0 for record in records for mask in record['masks']):
        raise ValueError('Explicit background must not become an object')
    manifest = args.output/'train.json'
    save_json(manifest, records)
    save_json(args.output/'source-cell-coverage.json', {'schemaVersion': 1, 'scope': 'Every original nonzero DSBI TRAIN cell has at least one complete cached context; no claim about subsequent stochastic training exposure.', 'positive_source_cells': len(coverage), 'rows': coverage})
    index = {'schemaVersion': 1, 'dataset': 'DSBI', 'revision': DSBI_REVISION, 'root': str(args.root.resolve()), 'trainList': 'train.txt', 'trainListSha256': sha256(args.root/'train.txt'), 'testList': 'test.txt', 'testListSha256': sha256(args.root/'test.txt'), 'testImagePixelsAccessed': False, 'source_coordinate_space': DSBI_COORDINATE_SPACE, 'sources': sources, 'generatedManifest': {'path': 'train.json', 'sha256': sha256(manifest), 'recordCount': len(records)}, 'coverage': {'path': 'source-cell-coverage.json', 'sha256': sha256(args.output/'source-cell-coverage.json'), 'positiveSourceCells': len(coverage)}, 'preparationProtocol': {'path': 'preparation-protocol.json', 'sha256': sha256(args.output/'preparation-protocol.json')}, 'ignore_box_semantics': 'Unlisted full source grid slots; objectness loss omitted, no geometry or dot target even if they cross the canvas.', 'explicit_zero_mask_semantics': 'Known background; never a positive cell or letter.', 'positive_families': sorted({row['source_family'] for row in records if row['boxes']})}
    save_json(args.output/'source-index.json', index)
    for name, expected in protected.items():
        if sha256(args.data/name) != expected:
            raise ValueError(f'Protected original artifact changed: {name}')
    counts = {'records': len(records), 'positive_records': sum(bool(row['boxes']) for row in records), 'negative_records': sum(not row['boxes'] for row in records), 'documented_empty_recto_records': sum(row['kind']=='documented_empty_recto' for row in records), 'source_positive_cells': len(coverage), 'complete_source_coverage': all(row['complete_context_occurrences'] for row in coverage), 'source_families': dict(Counter(row['source_family'] for row in records)), 'ignored_box_occurrences': sum(len(row['ignore_boxes']) for row in records), 'manifest_sha256': sha256(manifest), 'source_index_sha256': sha256(args.output/'source-index.json'), 'protected_original_artifacts_unchanged': True}
    save_json(args.output/'preparation-summary.json', counts)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
