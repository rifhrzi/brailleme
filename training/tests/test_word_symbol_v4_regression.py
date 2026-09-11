"""Paired regression metrics cannot hide rejection or change ground truth."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_v4_regression import compare_decisions, load_bound_oracle, lowercase_rows, sha, test_exposure_state


class RegressionTests(unittest.TestCase):
    def test_previous_test_exposure_is_not_claimed_as_unopened_by_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            self.assertEqual(test_exposure_state(experiment), dict(opened_by_this_evaluation=False,
                reserved_before_this_evaluation=False, completed_before_this_evaluation=False))
            (experiment/'test-evaluation-reservation.json').write_text('{}')
            state = test_exposure_state(experiment)
            self.assertFalse(state['opened_by_this_evaluation'])
            self.assertTrue(state['reserved_before_this_evaluation'])
            self.assertFalse(state['completed_before_this_evaluation'])
            (experiment/'final-test-evaluation.json').write_text('{}')
            state = test_exposure_state(experiment)
            self.assertFalse(state['opened_by_this_evaluation'])
            self.assertTrue(state['reserved_before_this_evaluation'])
            self.assertTrue(state['completed_before_this_evaluation'])

    def test_wrong_and_rejected_words_remain_in_exact_denominator(self):
        rows = [dict(expected=value) for value in ['ab', 'bc', 'cd']]
        baseline = [dict(text=value) for value in ['ab', 'bc', 'wrong']]
        candidate = [dict(text=value) for value in ['ab', '', 'wrong']]
        summary = compare_decisions(rows, baseline, candidate)
        self.assertEqual(summary['baseline']['correct'], 2)
        self.assertEqual(summary['candidate']['exact_accuracy'], 1/3)
        self.assertEqual(summary['candidate']['accepted_precision'], .5)
        self.assertEqual(summary['lost'], 1)
        self.assertEqual(summary['gained'], 0)
        self.assertTrue(summary['regression_veto'])

    def test_precision_regression_veto_even_with_equal_exact_accuracy(self):
        rows = [dict(expected='a'), dict(expected='b')]
        summary = compare_decisions(rows, [dict(text='a'), dict(text='')],
                                    [dict(text='a'), dict(text='wrong')])
        self.assertEqual(summary['baseline']['correct'], summary['candidate']['correct'])
        self.assertIn('accepted_precision_regressed', summary['reason_codes'])

    def test_negative_rejections_are_not_counted_as_correct_words(self):
        rows = [dict(expected=''), dict(expected='')]
        summary = compare_decisions(rows, [dict(text=''), dict(text='')],
                                    [dict(text=''), dict(text='noise')])
        self.assertEqual(summary['candidate']['correct'], 0)
        self.assertIsNone(summary['candidate']['exact_accuracy'])
        self.assertEqual(summary['candidate']['false_accepts'], 1)
        self.assertTrue(summary['regression_veto'])

    def test_rejecting_incorrect_acceptance_is_not_a_regression(self):
        rows = [dict(expected='a'), dict(expected='b')]
        summary = compare_decisions(rows, [dict(text='a'), dict(text='wrong')],
                                    [dict(text='a'), dict(text='')])
        self.assertFalse(summary['regression_veto'])

    def test_prediction_count_mismatch_fails(self):
        with self.assertRaises(ValueError):
            compare_decisions([dict(expected='ab')], [dict(text='ab')], [])

    def test_lowercase_is_a_derived_copy_and_geometry_stays_identical(self):
        original = dict(image='image.png', source='page', expected='ABC', masks=[1,3,9], selection=[0.,0.,1.,1.])
        copied = lowercase_rows([original], Path.cwd())[0]
        self.assertEqual(original['expected'], 'ABC')
        self.assertEqual(copied['expected'], 'abc')
        self.assertEqual(copied['masks'], original['masks'])
        self.assertEqual(copied['selection'], original['selection'])
        with self.assertRaises(ValueError):
            lowercase_rows([dict(original, expected='ABC!')], Path.cwd())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            lowercase_rows([original, original], Path.cwd())

    def test_cache_must_bind_real_checkpoint_fixture_and_tflite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, checkpoint, model, cache = [root/name for name in ['fixture.json','model.keras','model.tflite','oracle.npz']]
            for file in (fixture, checkpoint, model):
                file.write_bytes(file.name.encode())
            values = dict(outputs=np.zeros((2,48,120,11), np.float32), checkpoint_sha256=sha(checkpoint),
                          fixture_sha256=sha(fixture), model_sha256=sha(model), oracle_kind='tflite')
            np.savez(cache, **values)
            self.assertEqual(load_bound_oracle(cache, fixture, checkpoint, 2, model).shape[0], 2)
            for field in ['checkpoint_sha256','fixture_sha256','model_sha256','oracle_kind']:
                with self.subTest(field=field):
                    np.savez(cache, **dict(values, **{field:'incorrect'}))
                    with self.assertRaises(ValueError):
                        load_bound_oracle(cache, fixture, checkpoint, 2, model)
            np.savez(cache, **values)
            with self.assertRaises(ValueError):
                load_bound_oracle(cache, fixture, checkpoint, 3, model)


if __name__ == '__main__':
    unittest.main()
