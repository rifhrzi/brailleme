"""Supplemental public data must remain append-only and TRAIN-only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_sources import audit_supplemental_sources, expand_negative_records, sha256


class SupplementalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'data'
        self.dataset = self.root/'dsbi'
        self.data.mkdir()
        (self.dataset/'data/Massage').mkdir(parents=True)
        (self.dataset/'train.txt').write_text('Massage/M+1.jpg\n')
        (self.dataset/'test.txt').write_text('Massage/M+2.jpg\n')
        (self.dataset/'data/Massage/M+1+recto.jpg').write_bytes(b'public image fixture')
        (self.dataset/'data/Massage/M+1+recto.txt').write_bytes(b'public annotation fixture')
        (self.data/'extra.png').write_bytes(b'cached TRAIN image fixture')
        self.base = {'image': 'base.png', 'source': 'books/train/page.jpg', 'boxes': [[1, 2, 3, 4]], 'masks': [1]}
        self.extra = {'image': 'extra.png', 'image_sha256': sha256(self.data/'extra.png'),
                      'source': 'dsbi/Massage/M+1+recto.jpg', 'source_family': 'dsbi/Massage',
                      'boxes': [[10, 20, 30, 40]], 'masks': [63], 'originally_complete': [True]}
        self.write('base.json', [self.base])
        self.write('train.json', [self.base])
        self.write('source-split.json', {'images': {'train': [self.base['source']], 'validation': ['books/held/page.jpg'], 'external': []}})
        self.write('extra.json', [self.extra])
        self.index = {'schemaVersion': 1, 'dataset': 'DSBI', 'revision': '6937eb8973efe5ff337e5c018485cb626303f5c2',
                      'root': str(self.dataset), 'trainList': 'train.txt', 'testList': 'test.txt',
                      'trainListSha256': sha256(self.dataset/'train.txt'), 'testListSha256': sha256(self.dataset/'test.txt'),
                      'sources': [{'source': self.extra['source'], 'source_family': self.extra['source_family'], 'split': 'train',
                                   'train_member': 'Massage/M+1.jpg', 'original_image': 'data/Massage/M+1+recto.jpg',
                                   'original_annotation': 'data/Massage/M+1+recto.txt',
                                   'image_sha256': sha256(self.dataset/'data/Massage/M+1+recto.jpg'),
                                   'annotation_sha256': sha256(self.dataset/'data/Massage/M+1+recto.txt')}],
                      'generatedManifest': {'path': 'extra.json', 'sha256': sha256(self.data/'extra.json'), 'recordCount': 1}}
        self.write('index.json', self.index)

    def write(self, path, value):
        (self.data/path).write_text(json.dumps(value))

    def audit(self, records=None):
        return audit_supplemental_sources(records if records is not None else [self.base, self.extra],
                                          self.data, 'base.json', ['index.json'])

    def test_exact_append_and_hashes_pass(self):
        self.assertEqual(self.audit()['status'], 'passed')

    def test_cannot_drop_original_or_supplemental_records(self):
        for rows in ([self.base], [self.extra]):
            with self.assertRaisesRegex(ValueError, 'exactly preserve'):
                self.audit(rows)

    def test_cannot_change_original_masks(self):
        base = copy.deepcopy(self.base)
        base['masks'] = [2]
        with self.assertRaisesRegex(ValueError, 'exactly preserve'):
            self.audit([base, self.extra])

    def test_test_member_cannot_masquerade_as_train(self):
        self.index['sources'][0]['train_member'] = 'Massage/M+2.jpg'
        self.write('index.json', self.index)
        with self.assertRaisesRegex(ValueError, 'outside declared TRAIN'):
            self.audit()

    def test_cached_pixels_and_original_annotation_are_hash_bound(self):
        for path in (self.data/'extra.png', self.dataset/'data/Massage/M+1+recto.txt'):
            before = path.read_bytes()
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.audit()
            path.write_bytes(before)

    def test_base_cannot_include_held_source(self):
        base = dict(self.base, source='books/held/page.jpg')
        self.write('base.json', [base])
        with self.assertRaisesRegex(ValueError, 'non-training source'):
            self.audit([base, self.extra])

    def test_adding_negatives_preserves_all_original_exposure(self):
        old = {'image': 'old-negative.png', 'source': 'not_braille/old.jpg', 'boxes': [], 'masks': []}
        extra = {'image': 'extra-negative.png', 'source': self.extra['source'], 'boxes': [], 'masks': []}
        rows, description = expand_negative_records([self.base, old, extra], 27, [extra['source']], 2)
        self.assertEqual(sum(row['image'] == old['image'] for row in rows), 27)
        self.assertEqual(sum(row['image'] == extra['image'] for row in rows), 2)
        self.assertEqual(sum(row['image'] == self.base['image'] for row in rows), 1)
        self.assertEqual(description['negative_draws'], 29)

    def test_supplemental_negative_repeat_cannot_be_implicit(self):
        with self.assertRaisesRegex(ValueError, 'explicit positive'):
            expand_negative_records([self.base], 27, [self.extra['source']])


if __name__ == '__main__':
    unittest.main()
