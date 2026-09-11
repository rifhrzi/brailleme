"""TRAIN-only bounded sampling weights, independent of parser acceptance."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_v4_mine import difficulty, load_sampling_weights, sha, train_rows


class DifficultyTests(unittest.TestCase):
    def output(self, bits):
        value = np.zeros((48,120,11), np.float32)
        value[4,3,5:] = bits
        return value

    def test_correct_confident_weak_and_incorrect_have_weights_one_two_three(self):
        row = dict(boxes=[[8,12,16,20]], masks=[1])
        for bits,expected in [([.9,.1,.1,.1,.1,.1],1), ([.65,.1,.1,.1,.1,.1],2),
                              ([.9,.9,.1,.1,.1,.1],3)]:
            with self.subTest(expected=expected):
                result = difficulty(self.output(bits), row)
                self.assertEqual(result['weight'], expected)
                self.assertEqual(result['complete_cells'], 1)

    def test_invalid_grammar_indicator_still_receives_dot_supervision(self):
        row = dict(boxes=[[8,12,16,20]], masks=[32], expected='')
        result = difficulty(self.output([.1,.1,.1,.1,.1,.9]), row)
        self.assertEqual(result['weight'], 1)
        result = difficulty(self.output([.9,.1,.1,.1,.1,.1]), row)
        self.assertEqual(result['weight'], 3)

    def test_partial_or_originally_incomplete_boxes_are_not_classification_targets(self):
        for row in [dict(boxes=[[-1,12,16,20]], masks=[1]),
                    dict(boxes=[[8,12,16,20]], masks=[1], originally_complete=[False])]:
            result = difficulty(self.output([.9,.9,.9,.9,.9,.9]), row)
            self.assertEqual(result['weight'], 1)
            self.assertEqual(result['complete_cells'], 0)
            self.assertEqual(result['skipped_cells'], 1)

    def test_annotation_and_output_errors_fail(self):
        row = dict(boxes=[[8,12,16,20]], masks=[1])
        output = self.output([.9,.1,.1,.1,.1,.1])
        for invalid in [output[:1], output+2, output*np.nan]:
            with self.assertRaises(ValueError):
                difficulty(invalid, row)
        with self.assertRaises(ValueError):
            difficulty(output, dict(row,masks=[True]))
        with self.assertRaises(ValueError):
            difficulty(output, dict(row,masks=[]))


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root/'train-symbols.json'
        self.index = self.root/'train-image-hashes.json'
        self.checkpoint = self.root/'model.keras'
        self.checkpoint.write_bytes(b'checkpoint fixture')
        self.image = str(self.root/'data/train/000.png')
        self.row = dict(source='synthetic-id-v4/train/capital_single/000',image=self.image,
                        boxes=[[8,12,16,20]],masks=[1],symbol_group='capital_single')
        self.write(self.manifest,dict(split='train',records=[self.row]))
        self.write(self.index,{self.image:'a'*64})
        self.weights = self.root/'weights.json'
        self.report = dict(split='train',scope='TRAIN_only_dot_mask_sampling',
            manifest_path=str(self.manifest),manifest_sha256=sha(self.manifest),
            image_index_path=str(self.index),image_index_sha256=sha(self.index),
            checkpoint_path=str(self.checkpoint),checkpoint_sha256=sha(self.checkpoint),
            weights_by_source={self.row['source']:2})
        self.write(self.weights,self.report)

    def write(self,path,value):
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_valid_complete_mapping_loads(self):
        self.assertEqual(load_sampling_weights(self.weights,self.manifest,self.index),self.report['weights_by_source'])

    def test_validation_split_and_validation_pixels_are_rejected(self):
        self.write(self.manifest,dict(split='validation',records=[self.row]))
        with self.assertRaisesRegex(ValueError,'TRAIN-only'):
            train_rows(self.manifest,self.index)
        validation_image = str(self.root/'data/validation/000.png')
        self.write(self.manifest,dict(split='train',records=[dict(self.row,image=validation_image)]))
        self.write(self.index,{validation_image:'a'*64})
        with self.assertRaisesRegex(ValueError,'Only TRAIN image paths'):
            train_rows(self.manifest,self.index)

    def test_source_coverage_and_integer_bounds_required(self):
        for weights in [{}, {'unexpected':2}, {self.row['source']:0}, {self.row['source']:4},
                        {self.row['source']:True}, {self.row['source']:2.5}]:
            self.write(self.weights,dict(self.report,weights_by_source=weights))
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                load_sampling_weights(self.weights,self.manifest,self.index)

    def test_manifest_image_index_and_checkpoint_hashes_are_bound(self):
        for field in ['manifest_sha256','image_index_sha256','checkpoint_sha256']:
            self.write(self.weights,dict(self.report,**{field:'0'*64}))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'binding changed'):
                load_sampling_weights(self.weights,self.manifest,self.index)
        self.write(self.weights,dict(self.report,split='test'))
        with self.assertRaisesRegex(ValueError,'only from TRAIN'):
            load_sampling_weights(self.weights,self.manifest,self.index)


if __name__ == '__main__':
    unittest.main()
