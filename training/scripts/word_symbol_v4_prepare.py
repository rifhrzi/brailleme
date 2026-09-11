"""Prepare source-disjoint photo-cell compositions; never train or infer TEST.

Reuses existing TRAIN/validation PNGs. Fresh DSBI TEST pages are selected from
the pinned publisher split before any model inference, then used only for TEST
compositions. These are recomposed public photographs, not physical camera data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, ImageStat

from word_dsbi import DSBI_REVISION, read_dsbi_annotation, validate_dsbi_image_bounds

WIDTH, HEIGHT = 480, 192
GROUPS = ('lowercase', 'capital_single', 'capital_word', 'numbers', 'punctuation')
PUNCTUATION = '.,?!:;-()'
TRANSFORMS = {'identity': (0, 1, 2, 3, 4, 5),
              'horizontal': (3, 4, 5, 0, 1, 2),
              'vertical': (2, 1, 0, 5, 4, 3),
              'half_turn': (5, 4, 3, 2, 1, 0)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def transform_mask(mask, transform):
    return sum(((int(mask) >> old) & 1) << new
               for old, new in enumerate(TRANSFORMS[transform]))


def transform_pool(donors, required_masks):
    """Add all exact geometric variants, including already present rare masks.

    Return new dictionaries; neither source annotations nor donor objects mutate.
    """
    result = defaultdict(list)
    for donor in donors:
        for transform in TRANSFORMS:
            mask = transform_mask(donor['original_mask'], transform)
            if mask in required_masks:
                result[mask].append(dict(donor, mask=mask, synthetic_transform=transform))
    missing = set(required_masks) - set(result)
    if missing:
        raise ValueError(f'Photograph pool lacks masks even after exact transforms: {sorted(missing)}')
    return dict(result)


def old_donors(data, records):
    donors, seen = [], set()
    for row in records:
        for box, mask in zip(row['boxes'], row['masks']):
            x0, y0, x1, y1 = box
            if not (1 <= x0 < x1 <= WIDTH-1 and 1 <= y0 < y1 <= HEIGHT-1):
                continue
            if x1-x0 < 8 or y1-y0 < 16:
                continue
            key = (row['source'], tuple(row.get('crop') or []), tuple(round(v, 2) for v in box))
            if key in seen:
                continue
            seen.add(key)
            donors.append(dict(source=row['source'], image=str((data/row['image']).resolve()),
                               box=box, original_mask=int(mask), source_crop=row.get('crop')))
    return donors


def split_members(path):
    result = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        value = PurePosixPath(line.strip().replace('\\', '/'))
        if not line.strip():
            continue
        if value.is_absolute() or '..' in value.parts or value.suffix != '.jpg':
            raise ValueError('Unsafe DSBI member path')
        result.append(value.as_posix())
    if len(result) != len(set(result)):
        raise ValueError('Duplicate DSBI members')
    return result


def select_test_pages(members, count, seed):
    """Round-robin families with seeded order, independent of labels/predictions."""
    families = defaultdict(list)
    for member in sorted(members):
        families[str(PurePosixPath(member).parent)].append(member)
    rng = random.Random(seed)
    for values in families.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(count, len(members)):
        for family in sorted(families):
            if families[family] and len(selected) < count:
                selected.append(families[family].pop())
    return selected


def fresh_test_donors(repo, output, dataset, seed, page_count):
    restored = Path('C:/bm-datasets/DSBI-train-6937eb8')
    manifest = json.loads((repo/'training/evidence/braille-word-reader-v1/dsbi-training-preparation/download-manifest.json').read_text())
    known_hashes = {row['file']: row['sha256'] for row in manifest['sourceFiles']}
    for name in ('train.txt', 'test.txt'):
        if sha(restored/name) != known_hashes[name]:
            raise ValueError('Pinned DSBI split metadata hash mismatch')
    train, test = split_members(restored/'train.txt'), split_members(restored/'test.txt')
    if len(train) != 26 or len(test) != 88 or set(train) & set(test):
        raise ValueError('Publisher DSBI split differs')
    selected = select_test_pages(test, page_count, seed)
    save(output/'test-source-selection.json', dict(revision=DSBI_REVISION, seed=seed,
         policy='Seeded family round-robin from publisher TEST before label inspection or model inference',
         pages=selected, train_overlap=[], test_list_sha256=sha(restored/'test.txt')))
    base = f'https://raw.githubusercontent.com/yeluo1994/DSBI/{DSBI_REVISION}/'
    # Git blobs bind every downloaded byte to the immutable source revision.
    request = Request(f'https://api.github.com/repos/yeluo1994/DSBI/git/trees/{DSBI_REVISION}?recursive=1',
                      headers={'User-Agent': 'BrailleMe-auditable-corpus'})
    with urlopen(request, timeout=60) as response:
        tree = json.load(response)
    blobs = {row['path']: row['sha'] for row in tree['tree'] if row['type'] == 'blob'}
    paths = [f'data/{PurePosixPath(member).with_suffix("").as_posix()}+recto{suffix}'
             for member in selected for suffix in ('.jpg', '.txt')]
    def fetch(relative):
        destination = dataset/relative
        content = destination.read_bytes() if destination.exists() else None
        def blob_hash(value):
            return hashlib.sha1(f'blob {len(value)}\0'.encode()+value).hexdigest()
        if content is None or blob_hash(content) != blobs[relative]:
            with urlopen(base+quote(relative), timeout=90) as response:
                content = response.read()
            if blob_hash(content) != blobs[relative]:
                raise ValueError('Pinned DSBI TEST git blob mismatch: '+relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return dict(path=str(destination), file=relative, git_blob_sha1=blobs[relative],
                    sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    with ThreadPoolExecutor(max_workers=6) as executor:
        files = list(executor.map(fetch, paths))
    old_image_hashes={row['sha256'] for row in manifest['sourceFiles'] if row['file'].endswith('.jpg')}
    if any(row['sha256'] in old_image_hashes for row in files if row['file'].endswith('.jpg')):
        raise ValueError('TEST page pixels duplicate an already trained DSBI image')
    donors, pages = [], []
    for member in selected:
        relative = f'data/{PurePosixPath(member).with_suffix("").as_posix()}+recto.jpg'
        image_path = dataset/relative
        annotation_path = image_path.with_suffix('.txt')
        annotation = read_dsbi_annotation(annotation_path)
        with Image.open(image_path) as image:
            width, height = image.size
        # Some publisher pages have an unused terminal grid line exactly at
        # image width (OPD+6). It does not describe a listed cell. Full source
        # rectangles below must fit strictly inside; never clamp their truth.
        out_of_bounds_grid_lines = sum(v>=width for v in annotation.vertical_lines)+sum(v>=height for v in annotation.horizontal_lines)
        source = 'dsbi-test/'+relative[5:]
        count = 0
        for cell in annotation.cells:
            box = cell.box()
            if cell.mask and box[0]>=1 and box[1]>=1 and box[2]<=width-1 and box[3]<=height-1:
                donors.append(dict(source=source, image=str(image_path), box=box,
                                   original_mask=cell.mask, annotation_line=cell.annotation_line,
                                   annotation_path=str(annotation_path), source_grid=[cell.row,cell.col]))
                count += 1
        pages.append(dict(source=source, complete_positive_cells=count, source_member=member,
                          out_of_bounds_grid_lines=out_of_bounds_grid_lines,
                          excluded_zero_or_incomplete_cells=len(annotation.cells)-count))
    save(output/'test-download-manifest.json', dict(revision=DSBI_REVISION, files=files,
         pages=pages, applied_rotation=0, boxes='Full annotated 2x3 positions plus .3 column pitch margin',
         test_inference_performed=False, physical_camera_validated=False))
    return donors


@lru_cache(maxsize=24)
def source_pixels(image_path):
    with Image.open(image_path) as image:
        return image.convert('RGB')


@lru_cache(maxsize=4096)
def crop_pixels(image_path, box, transform):
    crop = source_pixels(image_path).crop(tuple(round(v) for v in box))
    if transform == 'horizontal':
        crop = ImageOps.mirror(crop)
    elif transform == 'vertical':
        crop = ImageOps.flip(crop)
    elif transform == 'half_turn':
        crop = crop.transpose(Image.Transpose.ROTATE_180)
    return crop


class DonorPicker:
    def __init__(self, pool):
        self.pool = pool
        self.groups = {}
        for mask, donors in pool.items():
            transforms = defaultdict(lambda: defaultdict(list))
            for donor in donors:
                transforms[donor['synthetic_transform']][donor['source']].append(donor)
            self.groups[mask] = dict(transforms)

    def choose(self, mask, rng):
        # Equal transform/source selection prevents plentiful mask-A donors from
        # eliminating the rare original dot-6 photographs from training.
        transforms = self.groups[mask]
        sources = transforms[rng.choice(sorted(transforms))]
        return rng.choice(sources[rng.choice(sorted(sources))])


def make_case(picker, masks, rng, path):
    selected = [picker.choose(mask, rng) for mask in masks]
    cell_h = rng.randint(64, 106)
    cell_w = min(round(cell_h*rng.uniform(.48, .59)), int((WIDTH-56)/(len(masks)*1.28)))
    cell_h = min(cell_h, int(cell_w/.43))
    gap = max(4, round(cell_w*rng.uniform(.23, .34)))
    total = len(masks)*cell_w+(len(masks)-1)*gap
    left = (WIDTH-total)//2+rng.randint(-7, 7)
    top = (HEIGHT-cell_h)//2+rng.randint(-10, 10)
    crops = [crop_pixels(d['image'], tuple(d['box']), d['synthetic_transform']).resize(
             (cell_w, cell_h), Image.Resampling.BILINEAR) for d in selected]
    background = tuple(round(v) for v in ImageStat.Stat(crops[0]).median)
    image = Image.new('RGB', (WIDTH, HEIGHT), background)
    boxes = []
    for index, crop in enumerate(crops):
        x = left+index*(cell_w+gap)
        image.paste(crop, (x, top))
        boxes.append([x, top, x+cell_w, top+cell_h])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=1)
    return boxes, selected


def sample_text(group, index, rng):
    length = 1 if index % 20 == 18 else 12 if index % 20 == 19 else 2+index % 7
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    word = ''.join(rng.choice(alphabet) for _ in range(length))
    if group == 'lowercase':
        return word
    if group == 'capital_single':
        return word[0].upper()+word[1:]
    if group == 'capital_word':
        return word.upper()
    if group == 'numbers':
        return ''.join(rng.choice('0123456789') for _ in range(length))
    mark = PUNCTUATION[index % len(PUNCTUATION)]
    if mark in '()':
        return '('+word[:max(1, min(length-2, 8))]+')'
    if mark == '-':
        return word[:max(1,min(length-2, 8))]+'-'+rng.choice(alphabet)
    return word[:max(1, length-1)]+mark


def assert_source_disjoint(splits):
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index+1:]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f'Source leakage {left}/{right}: {sorted(overlap)[:4]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('C:/bm-symbols-20260908/accuracy80-20260910'))
    parser.add_argument('--test-dataset', type=Path, default=Path('C:/bm-datasets/DSBI-test-6937eb8'))
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--train-per-group', type=int, default=1200)
    parser.add_argument('--eval-per-group', type=int, default=100)
    parser.add_argument('--test-pages', type=int, default=18)
    args = parser.parse_args()
    from word_symbols_id_v4 import encode_text, parse_word_masks
    repo, output = args.repo.resolve(), args.output.resolve()
    if (output/'corpus-audit.json').exists() or any((output/f'{name}-symbols.json').exists() for name in ('train','validation','test')):
        raise SystemExit('Frozen corpus exists; never overwrite a completed manifest')
    output.mkdir(parents=True, exist_ok=True)
    data = repo/'.braille-training-work/word-scan-v1-20260905/data'
    train_file = data/'../combined-training-v5/combined-training-manifest.json'
    validation_file = data/'validation.json'
    train = json.loads(train_file.read_text())
    validation = json.loads(validation_file.read_text())
    records = {'train': train, 'validation': validation}
    source_sets = {name: {row['source'] for row in values} for name, values in records.items()}
    donors = {name: old_donors(data, values) for name, values in records.items()}
    donors['test'] = fresh_test_donors(repo, output, args.test_dataset.resolve(), args.seed, args.test_pages)
    source_sets['test'] = {row['source'] for row in donors['test']}
    assert_source_disjoint(source_sets)
    # DSBI test identities use a distinct prefix, so verify actual source image
    # basenames too: a naming convention alone never establishes isolation.
    natural_train_dsbi = {Path(row['source']).name for row in train if row['source'].startswith('dsbi/')}
    if natural_train_dsbi & {Path(row['image']).name for row in donors['test']}:
        raise ValueError('DSBI original page overlap despite differing source prefixes')
    required = set()
    for text in ('abcdefghijkl', 'mnopqrstuvwx', 'yz', 'ABC', '0123456789'):
        required.update(encode_text(text))
    for mark in PUNCTUATION:
        required.update(encode_text('(a)' if mark in '()' else 'a-a' if mark=='-' else 'a'+mark))
    pools = {name: transform_pool(values, required) for name, values in donors.items()}
    manifests, transforms = {}, {}
    all_sources = {}
    for split_index, name in enumerate(('train', 'validation', 'test')):
        rng = random.Random(args.seed+100003*split_index)
        picker = DonorPicker(pools[name])
        rows = []
        count = args.train_per_group if name == 'train' else args.eval_per_group
        texts = set()
        for group in GROUPS:
            for index in range(count):
                value = sample_text(group, index, rng)
                masks = encode_text(value)
                parsed = parse_word_masks(masks)
                if parsed.get('text') != value:
                    raise ValueError(f'Encoder/parser disagreement: {value}')
                path = output/'data'/name/f'{len(rows):06d}.png'
                boxes, provenance = make_case(picker, masks, rng, path)
                rows.append(dict(image=str(path), source=f'synthetic-id-v4/{name}/{group}/{len(rows):06d}',
                     source_family=f'synthetic-id-v4/{group}', boxes=boxes, masks=masks, expected=value,
                     symbol_group=group, origin='synthetic_recomposed_public_cell_photographs', source_cells=provenance))
                texts.add(value)
        # Teach indicator shapes even in sequences the grammar must reject.
        invalid = [[32], [60], [32,32], [1,60,1], [60,1,32,3], [60,5], [1,32,32,3], [32,60,1]]
        for index in range(96 if name == 'train' else 40):
            masks = invalid[index % len(invalid)]
            if parse_word_masks(masks)['status'] != 'rejected':
                raise ValueError('Mandatory invalid fixture became valid')
            path = output/'data'/name/f'{len(rows):06d}.png'
            boxes, provenance = make_case(picker, masks, rng, path)
            rows.append(dict(image=str(path), source=f'synthetic-id-v4/{name}/rejection/{len(rows):06d}',
                 source_family='synthetic-id-v4/rejection', boxes=boxes, masks=masks, expected='',
                 symbol_group='rejection', origin='synthetic_recomposed_public_cell_photographs', source_cells=provenance))
        for row in rows:
            for donor in row['source_cells']:
                all_sources[donor['image']] = None
        manifest = dict(scope='synthetic_recomposed_public_cell_photographs_not_physical_camera_accuracy',
                        split=name, seed=args.seed+100003*split_index, records=rows)
        save(output/f'{name}-symbols.json', manifest)
        manifests[name] = dict(records=len(rows), groups=dict(Counter(row['symbol_group'] for row in rows)),
             positive_length_counts=dict(Counter(len(row['expected']) for row in rows if row['expected'])),
             unique_texts=len(texts), sha256=sha(output/f'{name}-symbols.json'))
        transforms[name] = dict(Counter(d['synthetic_transform'] for row in rows for d in row['source_cells']))
        save(output/f'{name}-image-hashes.json', {row['image']: sha(row['image']) for row in rows})
        print('PREPARED '+name+' '+json.dumps(manifests[name]), flush=True)
    save(output/'source-image-hashes.json', {path: sha(path) for path in sorted(all_sources)})
    audit = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
         split_before_composition=True, source_sets={name: sorted(values) for name, values in source_sets.items()},
         source_overlap=[], manifests=manifests, transforms=transforms,
         source_manifest_sha256={'old_train':sha(train_file),'old_validation':sha(validation_file)},
         generator_sha256=sha(__file__), symbol_parser_sha256=sha(Path(__file__).with_name('word_symbols_id_v4.py')),
         test_policy='Fresh publisher DSBI TEST pages, never gradients/selection/threshold calibration; evaluate only frozen selected checkpoint once',
         validation_policy='Previously observed validation pages; fresh composed strings, used for checkpoint selection',
         physical_camera_validated=False, source_kind='synthetic_recomposed_public_cell_photographs',
         required_masks=sorted(required), pool_counts={name:{str(mask):len(values) for mask,values in pool.items()} for name,pool in pools.items()},
         capitalization_fix='All four exact transforms in TRAIN and evaluation, balanced by transform then source; original labels immutable')
    save(output/'corpus-audit.json', audit)
    # A cheap, auditable TRAIN-only pilot exercises every group and transform.
    training_rows=json.loads((output/'train-symbols.json').read_text())['records']
    pilot=[]
    for group in (*GROUPS,'rejection'):
        pilot.extend([row for row in training_rows if row['symbol_group']==group][:96])
    save(output/'pilot-train-symbols.json', dict(scope='TRAIN only balanced smoke/pilot slice', records=pilot))
    print('FROZEN '+str(output/'corpus-audit.json'), flush=True)


if __name__ == '__main__':
    main()
