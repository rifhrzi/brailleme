"""Separate point metadata must exactly retain raw source identities and bits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_point_metadata import derive_metadata, load_point_states
from word_sources import sha256
from word_dsbi import DSBI_REVISION


class PointMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation = self.root/'page.txt'
        self.annotation.write_text('0\n10 30\n20 37 60\n1 1 1 1 1 1 1 1\n')
        self.row = {'image': 'image.png', 'image_sha256': 'fixture', 'source': 'dsbi/train/page.jpg',
                    'source_cells': [{'row': 1, 'col': 1, 'annotation_line': 4}],
                    'crop': [0, 0, 480, 192], 'masks': [63]}
        (self.root/'train.json').write_text(json.dumps([self.row]))
        self.index = {'dataset': 'DSBI', 'revision': DSBI_REVISION, 'root': str(self.root),
                      'sources': [{'source': self.row['source'], 'original_annotation': 'page.txt',
                                   'split': 'train', 'annotation_sha256': sha256(self.annotation)}],
                      'generatedManifest': {'path': 'train.json', 'sha256': sha256(self.root/'train.json')}}
        (self.root/'index.json').write_text(json.dumps(self.index))
        self.metadata = derive_metadata(self.root, ['index.json'])
        self.path = self.root/'points.json'
        self.path.write_text(json.dumps(self.metadata))

    def test_actual_middle_grid_coordinate_is_preserved(self):
        states, audit = load_point_states('points.json', self.root, ['index.json'], [self.row])
        self.assertEqual(states['image.png'].points[0, 1, 1], 37)
        self.assertEqual(audit['known_cell_occurrences'], 1)

    def test_fabricated_point_is_rejected_against_raw_source(self):
        self.metadata['records']['image.png']['source_points'][0][1][1] = 40
        self.path.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, 'differs from its raw TRAIN'):
            load_point_states('points.json', self.root, ['index.json'], [self.row])

    def test_changed_raw_annotation_fails_hash(self):
        self.annotation.write_text(self.annotation.read_text().replace('37', '38'))
        with self.assertRaisesRegex(ValueError, 'unchanged DSBI TRAIN'):
            load_point_states('points.json', self.root, ['index.json'], [self.row])

    def test_source_cell_order_or_mask_cannot_change(self):
        changed = copy.deepcopy(self.row)
        changed['masks'] = [1]
        with self.assertRaisesRegex(ValueError, 'does not match this training record'):
            load_point_states('points.json', self.root, ['index.json'], [changed])


if __name__ == '__main__':
    unittest.main()
