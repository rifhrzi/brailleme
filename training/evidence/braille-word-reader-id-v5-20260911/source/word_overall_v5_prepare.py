"""Render the reserved 2026-09-11 final holdout, without model inference.

The primary 500+40 cases use the original stress composer unchanged. Native
contiguous photo windows form a separate diagnostic with an explicit original
mask transcription; they never replace or dilute the primary denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageStat

from word_dsbi import DSBI_REVISION, read_dsbi_annotation
from word_overall_v5_protocol import GROUPS, VERSION, read, sha, verify_hashes, write_new
from word_symbol_v4_prepare import DonorPicker, make_case, sample_text, transform_pool
from word_symbols_id_v4 import encode_text, parse_word_masks


def git_blob(value):
    return hashlib.sha1(f'blob {len(value)}\0'.encode()+value).hexdigest()


def download_reserved(experiment, dataset):
    reservation = read(experiment/'test-source-reservation.json')
    protocol = read(experiment/'protocol.json')
    verify_hashes(reservation['input_sha256'])
    verify_hashes(protocol['source_sha256'])
    verify_hashes({experiment/'protocol.json': reservation['protocol_sha256']})
    if reservation['revision'] != DSBI_REVISION or reservation['reserved_count'] != 24:
        raise ValueError('Unexpected reserved dataset revision or page count')
    selected = reservation['pages']
    if set(selected) & set(reservation['excluded_publisher_train']+reservation['excluded_development_observed_test']):
        raise ValueError('Reserved source overlaps development data')
    request = Request(f'https://api.github.com/repos/yeluo1994/DSBI/git/trees/{DSBI_REVISION}?recursive=1',
                      headers={'User-Agent': 'BrailleMe-source-reserved-evaluation'})
    with urlopen(request, timeout=60) as response:
        tree = json.load(response)
    if tree.get('truncated'):
        raise ValueError('Git tree must be complete for file verification')
    blobs = {row['path']: row['sha'] for row in tree['tree'] if row['type'] == 'blob'}
    relative_files = [f'data/{PurePosixPath(member).with_suffix("").as_posix()}+recto{suffix}'
                      for member in selected for suffix in ('.jpg', '.txt')]
    base = f'https://raw.githubusercontent.com/yeluo1994/DSBI/{DSBI_REVISION}/'
    def fetch(relative):
        target = dataset/relative
        if relative not in blobs:
            raise ValueError('Reserved file is absent from pinned tree: '+relative)
        if target.exists():
            value = target.read_bytes()
            if git_blob(value) != blobs[relative]:
                raise ValueError('Existing dataset bytes differ; never overwrite: '+str(target))
        else:
            with urlopen(base+quote(relative), timeout=90) as response:
                value = response.read()
            if git_blob(value) != blobs[relative]:
                raise ValueError('Downloaded file differs from pinned Git blob')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(value)
        return dict(path=str(target), file=relative, git_blob_sha1=blobs[relative],
                    sha256=hashlib.sha256(value).hexdigest(), bytes=len(value))
    with ThreadPoolExecutor(max_workers=6) as executor:
        files = list(executor.map(fetch, relative_files))
    prior_hashes = set(reservation['trained_image_sha256']+reservation['old_observed_image_sha256'])
    fresh_hashes = [row['sha256'] for row in files if row['file'].endswith('.jpg')]
    if len(set(fresh_hashes)) != len(fresh_hashes) or set(fresh_hashes) & prior_hashes:
        raise ValueError('Fresh source JPEG bytes duplicate prior/training or reserved pages')
    report = dict(version=VERSION, revision=DSBI_REVISION, files=files,
        source_reservation_sha256=sha(experiment/'test-source-reservation.json'),
        downloaded_utc=datetime.now(timezone.utc).isoformat(), old_train_or_observed_image_overlap=[],
        source_pages=selected, annotation_inspection_after_reservation=True,
        model_inference_performed=False, physical_camera_validated=False)
    write_new(experiment/'test-download-manifest.json', report)
    return report


def donors_and_pages(download):
    file_rows = {row['file']: row for row in download['files']}
    donors, pages, page_audit = [], [], []
    for member in download['source_pages']:
        relative = f'data/{PurePosixPath(member).with_suffix("").as_posix()}+recto.jpg'
        image_path = Path(file_rows[relative]['path'])
        annotation_path = image_path.with_suffix('.txt')
        annotation = read_dsbi_annotation(annotation_path)
        with Image.open(image_path) as image:
            width, height = image.size
        source = 'dsbi-overall80-test/'+relative[5:]
        complete = []
        for cell in annotation.cells:
            box = cell.box()
            if cell.mask and box[0] >= 1 and box[1] >= 1 and box[2] <= width-1 and box[3] <= height-1:
                complete.append(cell)
                donors.append(dict(source=source, image=str(image_path), box=box, original_mask=cell.mask,
                    annotation_line=cell.annotation_line, annotation_path=str(annotation_path), source_grid=[cell.row, cell.col]))
        pages.append(dict(member=member, source=source, image=str(image_path), annotation=str(annotation_path),
                          cells=complete, width=width, height=height))
        page_audit.append(dict(member=member, source=source, image_sha256=sha(image_path),
            annotation_sha256=sha(annotation_path), complete_positive_cells=len(complete),
            excluded_zero_or_incomplete=len(annotation.cells)-len(complete),
            out_of_bounds_unused_grid_lines=sum(value >= width for value in annotation.vertical_lines)+sum(value >= height for value in annotation.horizontal_lines)))
    return donors, pages, page_audit


def required_masks():
    required = set()
    for text in ('abcdefghijkl', 'mnopqrstuvwx', 'yz', 'ABC', '0123456789'):
        required.update(encode_text(text))
    for mark in '.,?!:;-()':
        required.update(encode_text('(a)' if mark in '()' else 'a-a' if mark == '-' else 'a'+mark))
    return required


def render_primary(experiment, donors, seed):
    pool = transform_pool(donors, required_masks())
    picker, rng = DonorPicker(pool), random.Random(seed)
    rows = []
    for group in GROUPS:
        for index in range(100):
            value = sample_text(group, index, rng)
            masks = encode_text(value)
            if parse_word_masks(masks).get('text') != value:
                raise ValueError('Encoder/parser disagreement')
            path = experiment/'data/test'/f'{len(rows):06d}.png'
            boxes, provenance = make_case(picker, masks, rng, path)
            rows.append(dict(image=str(path), source=f'overall-v5/test/{group}/{len(rows):06d}',
                source_family=f'overall-v5/{group}', boxes=boxes, masks=masks, expected=value, symbol_group=group,
                origin='original_v4_stress_composer_with_fresh_source_photographs', source_cells=provenance))
    invalid = [[32], [60], [32,32], [1,60,1], [60,1,32,3], [60,5], [1,32,32,3], [32,60,1]]
    for index in range(40):
        masks = invalid[index % len(invalid)]
        if parse_word_masks(masks)['status'] != 'rejected':
            raise ValueError('Mandatory invalid sequence became valid')
        path = experiment/'data/test'/f'{len(rows):06d}.png'
        boxes, provenance = make_case(picker, masks, rng, path)
        rows.append(dict(image=str(path), source=f'overall-v5/test/rejection/{len(rows):06d}',
            source_family='overall-v5/rejection', boxes=boxes, masks=masks, expected='', symbol_group='rejection',
            origin='original_v4_stress_composer_with_fresh_source_photographs', source_cells=provenance))
    manifest = dict(version=VERSION, scope='primary_balanced_stress_recomposed_public_scans_not_physical_camera',
        split='test', seed=seed, source_reservation_sha256=sha(experiment/'test-source-reservation.json'),
        protocol_sha256=sha(experiment/'protocol.json'), records=rows)
    write_new(experiment/'test-symbols.json', manifest)
    write_new(experiment/'test-image-hashes.json', {row['image']: sha(row['image']) for row in rows})
    return rows


def native_windows(cells):
    """Identity contiguous cells, original annotation bits, no invented spaces."""
    by_row = defaultdict(dict)
    for cell in cells:
        by_row[cell.row][cell.col] = cell
    candidates = []
    for row in sorted(by_row):
        values = by_row[row]
        for start in sorted(values):
            window = []
            for col in range(start, start+12):
                if col not in values or not values[col].mask:
                    break
                window.append(values[col])
                if len(window) < 2:
                    continue
                parsed = parse_word_masks([cell.mask for cell in window])
                if parsed['status'] == 'success' and 2 <= len(parsed['text']) <= 12:
                    candidates.append(dict(cells=list(window), expected=parsed['text'], row=row,
                                           start_col=start, end_col=col))
    return candidates


def native_group(text):
    if any(mark in text for mark in '.,?!:;-()'):
        return 'punctuation'
    if text.isdigit():
        return 'numbers'
    if text.isupper():
        return 'capital_word'
    if any(char.isupper() for char in text):
        return 'capital_single'
    return 'lowercase'


def render_native_window(page, candidate, path, rng):
    boxes = np.asarray([cell.box() for cell in candidate['cells']], np.float64)
    # One native rectangle retains real inter-cell pixels and illumination.
    # Small outer margin remains below the distance to the next cell's dots.
    h = float(np.median(boxes[:, 3]-boxes[:, 1]))
    rectangle = [max(0, math.floor(boxes[:, 0].min()-.05*h)),
                 max(0, math.floor(boxes[:, 1].min()-.05*h)),
                 min(page['width'], math.ceil(boxes[:, 2].max()+.05*h)),
                 min(page['height'], math.ceil(boxes[:, 3].max()+.05*h))]
    with Image.open(page['image']) as source:
        crop = source.convert('RGB').crop(rectangle)
    ratio = min(rng.randint(64, 106)/crop.height, 424/crop.width)
    size = max(1, round(crop.width*ratio)), max(1, round(crop.height*ratio))
    resized = crop.resize(size, Image.Resampling.BILINEAR)
    background = tuple(round(value) for value in ImageStat.Stat(crop).median)
    canvas = Image.new('RGB', (480, 192), background)
    left, top = (480-size[0])//2+rng.randint(-7, 7), (192-size[1])//2+rng.randint(-10, 10)
    canvas.paste(resized, (left, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, compress_level=1)
    sx, sy = size[0]/crop.width, size[1]/crop.height
    transformed = (boxes-np.asarray(rectangle[:2]*2))*[sx, sy, sx, sy]+[left, top, left, top]
    return transformed.tolist(), rectangle, [sx, sy]


def render_native(experiment, pages, seed, maximum=200):
    rng, options, available = random.Random(seed), {}, {}
    for page in pages:
        windows = native_windows(page['cells'])
        rng.shuffle(windows)
        options[page['source']] = windows
        available[page['source']] = dict(candidates=len(windows),
            groups=dict(Counter(native_group(item['expected']) for item in windows)))
    used, chosen = defaultdict(set), []
    while len(chosen) < maximum:
        before = len(chosen)
        for page in pages:
            windows = options[page['source']]
            while windows:
                candidate = windows.pop()
                identities = {(cell.row, cell.col) for cell in candidate['cells']}
                if not identities & used[page['source']]:
                    chosen.append((page, candidate))
                    used[page['source']].update(identities)
                    break
            if len(chosen) >= maximum:
                break
        if len(chosen) == before:
            break
    rows = []
    for page, candidate in chosen:
        path = experiment/'data/native-context'/f'{len(rows):06d}.png'
        boxes, rectangle, scale = render_native_window(page, candidate, path, rng)
        cells = candidate['cells']
        rows.append(dict(image=str(path), source=f'overall-v5/native/{len(rows):06d}',
            source_family=page['source'], source_page=page['source'], original_image=page['image'],
            annotation_path=page['annotation'], original_rectangle=rectangle, resize_scale=scale,
            source_grid=[[cell.row, cell.col] for cell in cells], original_boxes=[cell.box() for cell in cells],
            masks=[cell.mask for cell in cells], boxes=boxes, expected=candidate['expected'],
            symbol_group=native_group(candidate['expected']), origin='identity_native_contiguous_scan_window',
            label_interpretation='literal_original_six_dot_mask_transcription_under_v4_not_Chinese_language_translation'))
    manifest = dict(version=VERSION, split='native_context_diagnostic',
        scope='separate_native_contiguous_context_audit_never_primary_accuracy_denominator', seed=seed,
        selection='seeded page round-robin over grammar-valid 2..12-cell contiguous windows; no reused source cells within native audit',
        source_reservation_sha256=sha(experiment/'test-source-reservation.json'), records=rows)
    write_new(experiment/'native-context.json', manifest)
    write_new(experiment/'native-context-image-hashes.json', {row['image']: sha(row['image']) for row in rows})
    return rows, available


def prepare(experiment, dataset):
    experiment, dataset = Path(experiment).resolve(), Path(dataset).resolve()
    for name in ('test-download-manifest.json', 'test-symbols.json', 'native-context.json', 'corpus-audit.json',
                 'test-evaluation-reservation.json', 'data/test', 'data/native-context'):
        if (experiment/name).exists():
            raise ValueError('Preparation destination already exists: '+name)
    protocol = read(experiment/'protocol.json')
    reservation = read(experiment/'test-source-reservation.json')
    if protocol['version'] != VERSION or reservation['version'] != VERSION:
        raise ValueError('Protocol version differs')
    download = download_reserved(experiment, dataset)
    donors, pages, page_audit = donors_and_pages(download)
    print(f'FRESH_RESERVED_PAGES {len(pages)} COMPLETE_CELLS {len(donors)}', flush=True)
    seed = reservation['seed']+200006  # Same split seed offset as v4's TEST.
    primary = render_primary(experiment, donors, seed)
    print(f'PRIMARY_RENDERED {len(primary)}', flush=True)
    native, native_available = render_native(experiment, pages, seed+100003)
    positive = [row for row in primary if row['expected']]
    if len(positive) != 500 or sum(len(row['expected']) >= 2 for row in positive) != 480:
        raise ValueError('Primary difficulty/denominator drift')
    actual_pages = {donor['source'] for row in primary for donor in row['source_cells']}
    expected_pages = {page['source'] for page in pages}
    if actual_pages != expected_pages:
        raise ValueError('Primary samples must cover every reserved source page')
    verify_hashes(protocol['input_sha256'])
    verify_hashes(protocol['source_sha256'])
    verify_hashes({row['path']: row['sha256'] for row in download['files']})
    source_paths = [Path(__file__), Path(__file__).with_name('word_overall_v5_protocol.py'),
                    Path(__file__).with_name('word_symbol_v4_prepare.py'), Path(__file__).with_name('word_dsbi.py'),
                    Path(__file__).with_name('word_symbols_id_v4.py')]
    artifact_paths = [experiment/name for name in ('protocol.json', 'test-source-reservation.json',
        'test-download-manifest.json', 'test-symbols.json', 'test-image-hashes.json', 'native-context.json', 'native-context-image-hashes.json')]
    audit = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        split_before_annotation_inspection=True, source_overlap=[], duplicate_source_image_bytes=[],
        reserved_pages=len(pages), source_pages=page_audit,
        primary=dict(total=len(primary), positives=len(positive), invalid=40,
            group_counts=dict(Counter(row['symbol_group'] for row in primary)),
            positive_lengths=dict(Counter(len(row['expected']) for row in positive)),
            source_pages_used=len(actual_pages), multi_character=480,
            transforms=dict(Counter(donor['synthetic_transform'] for row in primary for donor in row['source_cells'])),
            cross_page_positive_compositions=sum(len({donor['source'] for donor in row['source_cells']}) > 1 for row in positive)),
        native_context=dict(total=len(native), source_pages_used=len({row['source_page'] for row in native}),
            groups=dict(Counter(row['symbol_group'] for row in native)),
            lengths=dict(Counter(len(row['expected']) for row in native)), availability=native_available,
            interpretation='Original bit sequences transcribed under v4; not natural Indonesian text or Chinese translation; no copied/transformed individual cells'),
        development_files=protocol['development_files'], training_images_copied=0,
        old_test_used_for_gradients=False, model_inference_performed=False,
        physical_camera_validated=False, source_sha256={str(path): sha(path) for path in source_paths},
        artifact_sha256={str(path): sha(path) for path in artifact_paths})
    write_new(experiment/'corpus-audit.json', audit)
    print('FROZEN_NEW_HOLDOUT '+json.dumps(dict(primary=audit['primary'], native_total=len(native),
        native_groups=audit['native_context']['groups'], audit=str(experiment/'corpus-audit.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('C:/bm-symbols-20260908/overall80-20260911'))
    parser.add_argument('--dataset', type=Path, default=Path('C:/bm-datasets/DSBI-test-6937eb8'))
    args = parser.parse_args()
    prepare(args.experiment, args.dataset)


if __name__ == '__main__':
    main()
