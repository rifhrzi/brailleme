"""Validation threshold searches must retain whole-word and per-group gates."""
import os
from pathlib import Path
import sys
import unittest

os.environ['KERAS_BACKEND'] = 'torch'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

import numpy as np
from word_data import make_target
from word_select import grouped_calibration
from word_test_predictions import center_predictions


def cell_output(mask, score=1.):
    output = center_predictions(make_target(np.asarray([[210.,70.,240.,120.]]), np.asarray([mask])))
    output[...,0] *= score
    return output


class WordSelectionTests(unittest.TestCase):
    def test_extended_grid_can_accept_supported_weak_peaks(self):
        outputs = np.stack([cell_output(1,.25), cell_output(1,.25)])
        rows = [{'expected':'A','validation_group':group} for group in ('natural','stress')]
        _, old, _, old_rank = grouped_calibration(outputs,rows)
        thresholds, new, _, new_rank = grouped_calibration(outputs,rows,'weak-peaks-v2')
        self.assertEqual(old['correct'],0)
        self.assertFalse(old_rank[0])
        self.assertEqual(new['correct'],2)
        self.assertTrue(new_rank[0])
        self.assertLessEqual(thresholds['proposalScore'],thresholds['acceptScore'])
        self.assertLessEqual(thresholds['acceptScore'],.25)

    def test_pooled_precision_cannot_hide_failed_group(self):
        outputs = np.stack([cell_output(1)]*21)
        rows = [{'expected':'A','validation_group':'natural'}]*20+[{'expected':'B','validation_group':'stress'}]
        _, metrics, groups, rank = grouped_calibration(outputs,rows,'weak-peaks-v2')
        self.assertGreaterEqual(metrics['accepted_accuracy'],.95)
        self.assertEqual(groups['stress']['accepted_accuracy'],0.)
        self.assertFalse(rank[0])

    def test_quality_rejected_zeros_count_in_whole_word_denominator(self):
        good=cell_output(1)
        outputs=np.stack([good,np.zeros_like(good),good])
        rows=[{'expected':'A','validation_group':'natural'},{'expected':'A','validation_group':'natural'},{'expected':'A','validation_group':'stress'}]
        _,metrics,groups,rank=grouped_calibration(outputs,rows,'weak-peaks-v2')
        self.assertEqual(metrics['total'],3)
        self.assertEqual(metrics['correct'],2)
        self.assertEqual(groups['natural']['exact_word_accuracy'],.5)
        self.assertEqual(rank[2],.75)
        self.assertTrue(rank[0])


if __name__=='__main__':
    unittest.main()
