"""Audit append-only supplemental TRAIN sources without changing frozen splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_path(value):
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in value:
        raise ValueError(f'Source path must remain within its dataset: {value}')
    return path


def expand_negative_records(records, base_repeat, supplemental_sources=(), supplemental_repeat=None):
    """Preserve original negative exposure when adding separately labeled data."""
    supplemental_sources = set(supplemental_sources)
    if base_repeat < 1 or (supplemental_sources and (supplemental_repeat is None or supplemental_repeat < 1)):
        raise ValueError('Negative repeat counts must be explicit positive integers')
    base = [row for row in records if not row['boxes'] and row['source'] not in supplemental_sources]
    extra = [row for row in records if not row['boxes'] and row['source'] in supplemental_sources]
    expanded = records + base*(base_repeat-1) + extra*((supplemental_repeat or base_repeat)-1)
    description = {'base_unique_negative_records': len(base), 'base_repeat': base_repeat,
                   'supplemental_unique_negative_records': len(extra), 'supplemental_repeat': supplemental_repeat,
                   'negative_draws': len(base)*base_repeat + len(extra)*(supplemental_repeat or base_repeat),
                   'total_epoch_records': len(expanded)}
    return expanded, description


def audit_supplemental_sources(records, data, base_manifest, index_paths):
    """Require an exact base-plus-supplements list and verify all source pixels."""
    if not index_paths:
        if base_manifest is not None:
            raise ValueError('A base manifest is only used with supplemental source indexes')
        return None
    if base_manifest is None:
        raise ValueError('Supplemental sources require an explicit preserved base manifest')
    base_path = data/base_manifest
    expected_records = json.loads(base_path.read_text())
    split = json.loads((data/'source-split.json').read_text())['images']
    excluded = set(split['validation']) | set(split['external'])
    original = json.loads((data/'train.json').read_text())
    negative_sources = {row['source'] for row in original if not row['boxes']}
    for row in expected_records:
        if row['source'] in excluded or row['source'] not in set(split['train']) | negative_sources:
            raise ValueError('The preserved base contains a non-training source')
    all_sources = set(split['train']) | excluded | negative_sources
    supplemental_sources = set()
    indexes = []
    for index_path in index_paths:
        index_path = data/index_path
        index = json.loads(index_path.read_text())
        if index['schemaVersion'] != 1 or index['dataset'] != 'DSBI':
            raise ValueError('Unsupported supplemental source index')
        if index['revision'] != '6937eb8973efe5ff337e5c018485cb626303f5c2':
            raise ValueError('Supplemental DSBI revision changed')
        root = Path(index['root'])
        if not root.is_absolute():
            root = index_path.parent/root
        members = {}
        for name in ('train', 'test'):
            file = root/relative_path(index[name+'List'])
            if sha256(file) != index[name+'ListSha256']:
                raise ValueError('Supplemental split metadata changed')
            values = [relative_path(line.strip()).as_posix() for line in file.read_text().splitlines() if line.strip()]
            if len(values) != len(set(values)):
                raise ValueError('Duplicate supplemental split member')
            members[name] = set(values)
        if members['train'] & members['test']:
            raise ValueError('Supplemental TRAIN/TEST membership overlaps')
        source_map = {}
        for source in index['sources']:
            identity = source['source']
            member = relative_path(source['train_member']).as_posix()
            family = str(PurePosixPath(identity).parent)
            if source['split'] != 'train' or member not in members['train'] or member in members['test']:
                raise ValueError('A supplemental source is outside declared TRAIN membership')
            if identity in all_sources or identity in source_map or source['source_family'] != family:
                raise ValueError('Duplicate supplemental identity or incorrect source family')
            image = relative_path(source['original_image'])
            annotation = relative_path(source['original_annotation'])
            original_member = PurePosixPath(member)
            expected_image = PurePosixPath('data')/original_member.parent/(original_member.stem+'+recto.jpg')
            if identity != str(PurePosixPath('dsbi')/original_member.parent/expected_image.name):
                raise ValueError('Supplemental source identity does not preserve its metadata family')
            if image != expected_image or annotation != expected_image.with_suffix('.txt'):
                raise ValueError('Supplemental image does not correspond to its TRAIN member')
            if sha256(root/image) != source['image_sha256'] or sha256(root/annotation) != source['annotation_sha256']:
                raise ValueError('Supplemental original image or annotation changed')
            source_map[identity] = source
        generated = index['generatedManifest']
        manifest = Path(generated['path'])
        if not manifest.is_absolute():
            manifest = index_path.parent/manifest
        if sha256(manifest) != generated['sha256']:
            raise ValueError('Supplemental generated manifest changed')
        extra = json.loads(manifest.read_text())
        if len(extra) != generated['recordCount']:
            raise ValueError('Supplemental record count changed')
        for row in extra:
            if row['source'] not in source_map or row['source_family'] != source_map[row['source']]['source_family']:
                raise ValueError('Supplemental record has an unregistered source')
            if len(row['boxes']) != len(row['masks']) or any(type(mask) is not int or not 1 <= mask <= 63 for mask in row['masks']):
                raise ValueError('Supplemental positives must be complete known six-dot masks')
            if sha256(data/row['image']) != row['image_sha256']:
                raise ValueError('Supplemental cached TRAIN image changed')
        if {row['source'] for row in extra} != set(source_map):
            raise ValueError('Some supplemental sources have no generated training record')
        expected_records.extend(extra)
        all_sources.update(source_map)
        supplemental_sources.update(source_map)
        indexes.append({'path': str(index_path.resolve()), 'sha256': sha256(index_path),
                        'dataset': index['dataset'], 'revision': index['revision'],
                        'generated_manifest_sha256': sha256(manifest), 'sources': len(source_map),
                        'families': sorted({row['source_family'] for row in source_map.values()}),
                        'records': len(extra)})
    if records != expected_records:
        raise ValueError('Combined training records must exactly preserve base plus all supplemental records')
    images = [row['image'] for row in records]
    if len(images) != len(set(images)):
        raise ValueError('Duplicate cached image in combined training manifest')
    return {'status': 'passed', 'preserved_base_manifest': str(base_manifest),
            'preserved_base_manifest_sha256': sha256(base_path),
            'original_source_split_sha256': sha256(data/'source-split.json'),
            'supplemental_sources': sorted(supplemental_sources),
            'indexes': indexes, 'combined_records': len(records),
            'scope': 'Exact append-only records, split membership and original/cached pixel hashes; no label or validation changes.'}
