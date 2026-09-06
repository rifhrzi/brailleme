"""Build and verify separate exact DSBI TRAIN point metadata; no image edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from word_dsbi import DSBI_REVISION, read_dsbi_annotation
from word_point_geometry import OccupiedPointState
from word_sources import relative_path, sha256


def derive_metadata(data, index_paths):
    """Reproduce points from hash-bound raw annotations and aligned identities."""
    data = Path(data)
    result = {'schemaVersion': 1, 'coordinate_space': 'source_pixel_index_centers',
              'indexes': [], 'records': {}}
    for name in index_paths:
        index_path = data/name
        index = json.loads(index_path.read_text())
        if index['dataset'] != 'DSBI' or index['revision'] != DSBI_REVISION:
            raise ValueError('Exact point metadata requires the pinned DSBI source')
        root = Path(index['root'])
        if not root.is_absolute():
            root = index_path.parent/root
        source_cells = {}
        for source in index['sources']:
            annotation_path = root/relative_path(source['original_annotation'])
            if source['split'] != 'train' or sha256(annotation_path) != source['annotation_sha256']:
                raise ValueError('Point annotation is not unchanged DSBI TRAIN')
            annotation = read_dsbi_annotation(annotation_path)
            source_cells[source['source']] = {
                (cell.row, cell.col, cell.annotation_line): cell
                for cell in annotation.cells if not cell.is_background}
        manifest = index_path.parent/index['generatedManifest']['path']
        if sha256(manifest) != index['generatedManifest']['sha256']:
            raise ValueError('Point source manifest changed')
        result['indexes'].append({'path': str(index_path.resolve()),
                                  'sha256': sha256(index_path),
                                  'manifest_sha256': sha256(manifest)})
        for row in json.loads(manifest.read_text()):
            if len(row['source_cells']) != len(row['masks']):
                raise ValueError('Each known cell needs an aligned source identity')
            points = []
            for identity, mask in zip(row['source_cells'], row['masks']):
                key = tuple(identity[field] for field in ('row', 'col', 'annotation_line'))
                cell = source_cells[row['source']].get(key)
                if cell is None or cell.mask != mask:
                    raise ValueError('Known point identity or occupied mask changed')
                points.append([list(pair) for pair in cell.dot_centers])
            if row['image'] in result['records']:
                raise ValueError('Duplicate point metadata image')
            result['records'][row['image']] = {
                'source': row['source'], 'image_sha256': row['image_sha256'],
                'crop': row['crop'], 'masks': row['masks'],
                'source_cells': row['source_cells'], 'source_points': points}
    return result


def load_point_states(path, data, index_paths, records):
    """Verify the separate file against raw source again, then cache states.

    Caller first runs audit_supplemental_sources for original pixels and exact
    TRAIN/TEST membership. No validation or Angelina record receives a state.
    """
    path = Path(data)/path
    actual = json.loads(path.read_text())
    expected = derive_metadata(data, index_paths)
    if actual != expected:
        raise ValueError('Exact point metadata differs from its raw TRAIN sources')
    rows = {row['image']: row for row in records}
    states = {}
    for image, metadata in actual['records'].items():
        row = rows.get(image)
        if row is None or any(row[field] != metadata[field] for field in
                              ('source', 'image_sha256', 'crop', 'masks', 'source_cells')):
            raise ValueError('Point metadata does not match this training record')
        points = np.asarray(metadata['source_points'], np.float64).reshape(-1, 6, 2)
        states[image] = OccupiedPointState.from_source_crop(points, row['masks'], row['crop'])
    return states, {'path': str(path.resolve()), 'sha256': sha256(path),
                    'records': len(states), 'known_cell_occurrences':
                    sum(len(state.points) for state in states.values()),
                    'source_verification': 'Exact equality with freshly parsed hash-bound DSBI TRAIN annotations; no inferred dot positions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source-index', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new immutable point metadata path')
    metadata = derive_metadata(args.data, args.source_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'path': str(args.output), 'sha256': sha256(args.output),
                      'records': len(metadata['records'])}))


if __name__ == '__main__':
    main()
