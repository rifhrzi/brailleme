from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_decode_id_v4 import decode
from word_overall_v5_tta import (POLICIES, first_accepted, infer_decisions,
    invert_outputs, match_cells, matched_average, runtime_decisions,
    selected_cells, transformed_images, assert_validation_manifest, method_specification,
    assert_development_open)

THRESHOLDS = dict(proposalScore=.25, acceptScore=.25, dotConfidence=.7,
    nmsIou=.25, minCellWidth=.008, minCellHeight=.06, edgeMargin=.008,
    maxRowDeviation=.55, maxGapRatio=1.2, maxOverlapRatio=.5)


def cell_output(first_dot=.99):
    value = np.zeros((48, 120, 11), np.float32)
    # Offset exactly1 crosses a grid boundary: floor(center) would attach wrong.
    value[24, 40] = [.9, 1., .75, .04, .2, first_dot, .01, .01, .01, .01, .01]
    return value


class RuntimeTests(unittest.TestCase):
    def test_method_specification_roundtrips_and_cannot_mutate_runtime_order(self):
        value = method_specification()
        self.assertEqual(json.loads(json.dumps(value)), value)
        value['policies']['cascade'].clear()
        self.assertEqual(method_specification()['policies']['cascade'][0], 'identity')

    def test_fixed_cascade_and_label_blind_first_accept(self):
        self.assertEqual(POLICIES['cascade'], ('identity', 'matched_logit_four',
            'matched_prob_four', 'matched_prob_vertical', 'matched_prob_half_turn', 'matched_prob_horizontal'))
        class Poison(dict):
            def __getitem__(self, key):
                if key == 'expected':
                    raise AssertionError('Runtime must not inspect answer labels')
                return super().__getitem__(key)
        identity, other = Poison(text='wrong', expected='correct'), Poison(text='correct')
        self.assertIs(first_accepted([identity, other]), identity)
        identity['text'] = ''
        self.assertIs(first_accepted([identity, other]), other)
        self.assertIs(first_accepted([identity, dict(text='')]), identity)

    def test_inverse_preserves_geometry_and_physical_dot_positions(self):
        values = cell_output()[None]
        values[0, 24, 40, 5:] = [1, 0, 0, 0, 0, 0]
        for name, gy, gx, bit in [('horizontal', 24, 79, 3), ('vertical', 23, 40, 2), ('half_turn', 23, 79, 5)]:
            inverted = invert_outputs(values, name)
            self.assertEqual(np.argmax(inverted[0, gy, gx, 5:]), bit)
            np.testing.assert_allclose(invert_outputs(inverted, name), values, rtol=0, atol=1e-7)
        images = np.arange(2*3*4*3).reshape(2, 3, 4, 3)
        for name in ('horizontal', 'vertical', 'half_turn'):
            np.testing.assert_array_equal(transformed_images(transformed_images(images, name), name), images)

    def test_attachment_uses_actual_nms_grid_and_only_replaces_dots(self):
        base, other = cell_output(.6), cell_output(.99)
        base_cells = selected_cells(base, THRESHOLDS, None, decode)
        self.assertEqual(base_cells[0][0], (24, 40))
        other_cells = selected_cells(other, THRESHOLDS, None, decode)
        combined, count = matched_average(base, ('horizontal',), base_cells, {'horizontal': other_cells})
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(combined[..., :5], base[..., :5])
        np.testing.assert_array_equal(combined[24, 41], base[24, 41])
        self.assertGreater(combined[24, 40, 5], .7)
        duplicate = other_cells + other_cells
        self.assertEqual(match_cells(base_cells, duplicate), {0: 0})
        distant = [((24, 1), dict(box=[.8, .8, .9, .9], dots=[.99]*6))]
        self.assertEqual(match_cells(base_cells, distant), {})

    def test_quality_and_original_clipping_cannot_be_rescued_by_tta(self):
        good = cell_output()
        quality = runtime_decisions({'identity': good}, THRESHOLDS, quality_rejected=True)
        self.assertTrue(all(decision['reason'] == 'no_grid' for decision in quality['decisions'].values()))
        clipped = good.copy()
        clipped[24, 40, 3] = .9
        result = runtime_decisions(dict(identity=clipped, horizontal=good, vertical=good, half_turn=good),
                                   THRESHOLDS, views_are_inverted=True)
        self.assertTrue(all(decision['reason'] == 'clipped' for decision in result['decisions'].values()))
        self.assertTrue(all(decision['cells'][0]['box'] == decode(clipped, THRESHOLDS)['cells'][0]['box']
                            for decision in result['decisions'].values()))

    def test_lazy_engine_api_only_augments_uncertain_quality_passing_cells(self):
        pixels = np.random.default_rng(5).uniform(.1, .9, (4, 192, 480, 3)).astype(np.float32)
        pixels[2] = 0
        initial = np.stack([cell_output(), cell_output(.6), cell_output(), np.zeros((48, 120, 11), np.float32)])
        supplied = [initial] + [invert_outputs(cell_output()[None], name) for name in ('horizontal', 'vertical', 'half_turn')]
        calls = []
        def predict(images):
            calls.append(images.copy())
            return supplied[len(calls)-1]
        result = infer_decisions(predict, pixels, THRESHOLDS, policy='cascade', batch_size=8)
        self.assertEqual([len(batch) for batch in calls], [4, 1, 1, 1])
        self.assertEqual([decision['text'] for decision in result['decisions']], ['a', 'a', '', ''])
        self.assertEqual(result['routes'], ['identity', 'matched_logit_four', 'identity', 'identity'])
        for i, name in enumerate(('horizontal', 'vertical', 'half_turn'), 1):
            np.testing.assert_array_equal(calls[i], transformed_images(pixels[1:2], name))
        self.assertEqual(result['latency']['mean_forward_count'], 1.75)
        calls.clear()
        identity = infer_decisions(predict, pixels, THRESHOLDS, policy='identity', batch_size=8)
        self.assertEqual(len(calls), 1)
        self.assertEqual([decision['text'] for decision in identity['decisions']], ['a', '', '', ''])

    def test_validation_cli_split_guard_rejects_train_and_test(self):
        for split in ('train', 'test', None):
            with self.assertRaises(ValueError):
                assert_validation_manifest(dict(split=split, records=[]))
        self.assertEqual(assert_validation_manifest(dict(split='validation', records=[])), [])

    def test_method_freeze_stops_development_before_test_exposure(self):
        for name in ('final-method-selection.json', 'test-evaluation-reservation.json', 'final-test-evaluation.json'):
            with tempfile.TemporaryDirectory() as directory:
                assert_development_open(directory)
                (Path(directory)/name).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'frozen or TEST opened'):
                    assert_development_open(directory)


if __name__ == '__main__':
    unittest.main()
