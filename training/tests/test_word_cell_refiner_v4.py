"""CPU checks for refiner provenance, real-proposal attachment and crop behavior."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_cell_refiner_v4 import (CellRefiner, crop_cell, selected_grids, replace_dots,
    save_refiner, load_refiner, balanced_indices, augment_crops, assert_unopened,
    load_prepared, NativeAdam)
from word_decode_id_v4 import DEFAULT_THRESHOLDS, decode


class CellRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_crop_constant_border_and_fractional_ramp(self):
        constant = np.full((192, 480, 3), .37, np.float32)
        crop = crop_cell(constant, [-.1, -.1, .1, .1])
        self.assertEqual(crop.shape, (64, 40, 1))
        np.testing.assert_allclose(crop, .37, atol=1e-7)
        ramp = np.broadcast_to((np.arange(480, dtype=np.float32)/479)[None, :, None], (192, 480, 3))
        value = crop_cell(ramp, [.3, .3, .7, .7])
        # Expansion: x starts .24*480, spans .52*480; sample center minus .5.
        expected = (.24*480+(np.arange(40)+.5)*(.52*480)/40-.5)/479
        np.testing.assert_allclose(value[32, :, 0], expected, atol=1e-7)
        for box in ([0, 0, 0, 1], [0, 0, 1, np.nan]):
            with self.assertRaises(ValueError):
                crop_cell(constant, box)

    def test_attachment_does_not_round_predicted_center_and_changes_only_dots(self):
        output = np.zeros((48, 120, 11), np.float32)
        output[20, 10, :5] = [.8, 0., .5, .1425454169511795, .2]
        output[20, 10, 5:] = .5
        thresholds = dict(DEFAULT_THRESHOLDS, acceptScore=.25, dotConfidence=.7)
        attachments = selected_grids(output, thresholds)
        self.assertEqual(attachments[0][0], (20, 10))
        box = attachments[0][1]
        self.assertEqual(int(((box[0]+box[2])/2)*120), 9)  # Actual rounding counterexample.
        updated = replace_dots(output[None], [(0, 20, 10)], np.asarray([[1, 0, 0, 0, 0, 0]], np.float32), 1)
        np.testing.assert_array_equal(updated[..., :5], output[None, ..., :5])
        untouched = output.copy()
        untouched[20, 10, 5:] = [1, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(updated[0], untouched)
        self.assertEqual(decode(updated[0], thresholds)['text'], 'a')
        unchanged = replace_dots(output[None], [(0, 20, 10)], np.ones((1, 6)), 0)
        np.testing.assert_array_equal(unchanged, output[None])

    def test_low_dot_confidence_is_refinable_but_geometry_rejection_remains(self):
        output = np.zeros((48, 120, 11), np.float32)
        output[20, 1, :5] = [.8, .5, .5, .1, .2]
        output[20, 1, 5:] = .5
        thresholds = dict(DEFAULT_THRESHOLDS, acceptScore=.25, dotConfidence=.7)
        self.assertEqual(decode(output, thresholds)['reason'], 'clipped')
        grids = [(0, *grid) for grid, _ in selected_grids(output, thresholds)]
        refined = replace_dots(output[None], grids, np.asarray([[1, 0, 0, 0, 0, 0]]), 1)
        self.assertEqual(decode(refined[0], thresholds)['reason'], 'clipped')

    def test_suppressed_proposals_are_never_changed(self):
        output = np.zeros((48, 120, 11), np.float32)
        output[20, 50, :5] = [.9, 1., .5, .05, .2]
        output[20, 51, :5] = [.8, 0., .5, .05, .2]
        output[..., 5:] = .5
        thresholds = dict(DEFAULT_THRESHOLDS, acceptScore=.25, dotConfidence=.7)
        attached = selected_grids(output, thresholds)
        self.assertEqual([grid for grid, _ in attached], [(20, 50)])
        refined = replace_dots(output[None], [(0, 20, 50)], np.zeros((1, 6)), .5)
        np.testing.assert_array_equal(refined[0, 20, 51], output[20, 51])
        np.testing.assert_array_equal(refined[0, 20, 50, 5:], .25)
        for wrong in [[(0, -1, 50)], [(0, 20, 50), (0, 20, 50)], [(0, 20., 50)], [(1, 20, 50)]]:
            with self.assertRaisesRegex(ValueError, 'Attachments'):
                replace_dots(output[None], wrong, np.zeros((len(wrong), 6)), 1)

    def test_positional_model_budget_roundtrip_and_finite_native_backward(self):
        torch.manual_seed(21)
        model = CellRefiner()
        self.assertLess(sum(p.numel() for p in model.parameters()), 250000)
        inputs = torch.rand(8, 64, 40, 1)
        before = model(inputs).detach()
        self.assertEqual(before.shape, (8, 6))
        self.assertTrue(torch.isfinite(model(torch.zeros_like(inputs))).all())
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'model.npz'
            save_refiner(model, checkpoint)
            torch.testing.assert_close(load_refiner(checkpoint)(inputs), before, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_refiner(model, checkpoint)
        optimizer = NativeAdam(model.parameters(), .001)
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(augment_crops(inputs)), torch.ones(8, 6))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model(inputs), before))

    def test_mask_sampling_balances_indicators_without_label_changes(self):
        masks = np.asarray([1]*100+[32]*2+[60]*3+[54]*4, np.uint8)
        choices = balanced_indices(masks, 128, np.random.default_rng(3))
        self.assertEqual(dict(zip(*np.unique(masks[choices], return_counts=True))), {1:32, 32:32, 54:32, 60:32})
        np.testing.assert_array_equal(choices, balanced_indices(masks, 128, np.random.default_rng(3)))

    def test_final_evaluation_reservation_and_nontrain_preparation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assert_unopened(root)
            (root/'test-evaluation-reservation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Final evaluation'):
                assert_unopened(root)
            path = root/'data.npz'
            path.with_suffix('.json').write_text(json.dumps(dict(split='validation')))
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                load_prepared(path, root)


if __name__ == '__main__':
    unittest.main()
