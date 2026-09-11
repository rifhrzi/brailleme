"""CPU checks for selective weighting; no model/GPU inference or TEST access."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from word_symbol_v4_loss import MINORITY_MASKS, minority_centers, validate_minority_weight, validate_dot_loss_scale, word_relative_loss
from word_torch_port import word_relative_loss as original_loss
from word_symbol_v4_train import StepTiming, requested_stop


def fixture():
    truth = torch.zeros((1, 2, len(MINORITY_MASKS) + 3, 11), dtype=torch.float64)
    for index, mask in enumerate((*MINORITY_MASKS, 1)):
        truth[0, 0, index] = truth.new_tensor([1., .2, .4, .12, .18] + [int(bool(mask & (1 << bit))) for bit in range(6)])
    # A partial cell resembles capital32 but contains one unknown dot.
    truth[0, 0, -2] = truth.new_tensor([1., .2, .4, .12, .18, -1, 0, 0, 0, 0, 1])
    # Ignored objectness/geometry and a noncenter negative with a minority mask.
    truth[0, 0, -1, 0] = -1
    truth[0, 1, 0, -1] = 1
    prediction = torch.linspace(.2, .8, truth.numel(), dtype=torch.float64).reshape(truth.shape)
    return truth, prediction


class MinorityLossTests(unittest.TestCase):
    def test_dot_scale_changes_only_known_dot_gradients_including_minority_extra(self):
        truth, prediction = fixture()
        for minority_weight in [1., 1.5, 3.]:
            before = prediction.clone().requires_grad_()
            after = prediction.clone().requires_grad_()
            word_relative_loss(truth, before, minority_weight).backward()
            word_relative_loss(truth, after, minority_weight, dot_loss_scale=3.).backward()
            with self.subTest(minority_weight=minority_weight):
                self.assertTrue(torch.equal(after.grad[..., :5], before.grad[..., :5]))
                torch.testing.assert_close(after.grad[..., 5:], before.grad[..., 5:] * 3., rtol=1e-12, atol=1e-12)
                self.assertEqual(float(after.grad[0, 0, -2, 5]), 0.)
                self.assertEqual(float(after.grad[0, 0, -1].abs().sum()), 0.)
                self.assertTrue(torch.equal(after.grad[0, 1], before.grad[0, 1]))

    def test_dot_scale_one_preserves_previous_minority_weighted_loss_exactly(self):
        truth, prediction = fixture()
        # Independent reconstruction of the previous loss from minority-only
        # components and the unchanged original baseline.
        weight = 1.5
        selected = minority_centers(truth)
        centers = (truth[..., :1] >= .999).to(torch.float32)
        normalizer = centers.sum().clamp_min(1.)
        probability = prediction[..., :1].clamp(1e-5, 1 - 1e-5)
        localization = (-selected * (1 - probability).square() * probability.log()).sum() / normalizer
        delta = torch.cat((truth[..., 1:3] - prediction[..., 1:3],
                           truth[..., 3:5].clamp_min(1e-3).log() - prediction[..., 3:5].clamp_min(1e-3).log()), dim=-1)
        absolute = delta.abs()
        geometry = (torch.where(absolute < 1., .5 * delta.square(), absolute - .5) * selected).sum() / normalizer
        bits = prediction[..., 5:].clamp(1e-5, 1 - 1e-5)
        known = (truth[..., 5:] >= 0).to(torch.float32)
        labels = truth[..., 5:].clamp_min(0.)
        dot = (-(labels * bits.log() + (1 - labels) * (1 - bits).log()) * selected * known).sum() / (centers * known).sum().clamp_min(1.)
        previous = original_loss(truth, prediction) + (weight - 1.) * (localization + 5 * geometry + 3 * dot)
        self.assertTrue(torch.equal(word_relative_loss(truth, prediction, weight), previous))

    def test_default_one_is_exact_original_loss_and_gradient(self):
        truth, prediction = fixture()
        before = prediction.clone().requires_grad_()
        after = prediction.clone().requires_grad_()
        expected = original_loss(truth, before)
        actual = word_relative_loss(truth, after)
        self.assertTrue(torch.equal(expected, actual))
        expected.backward()
        actual.backward()
        self.assertTrue(torch.equal(before.grad, after.grad))

    def test_every_minority_positive_gradient_is_exact_weight_and_all_others_unchanged(self):
        truth, prediction = fixture()
        for weight in [2., 3.5, 5.]:
            before, after = prediction.clone().requires_grad_(), prediction.clone().requires_grad_()
            original_loss(truth, before).backward()
            word_relative_loss(truth, after, weight).backward()
            target = before.grad.clone()
            target[0, 0, :len(MINORITY_MASKS)] *= weight
            with self.subTest(weight=weight):
                torch.testing.assert_close(after.grad, target, rtol=1e-12, atol=1e-12)
                self.assertTrue(torch.equal(after.grad[0, 0, len(MINORITY_MASKS):], before.grad[0, 0, len(MINORITY_MASKS):]))
                self.assertTrue(torch.equal(after.grad[0, 1], before.grad[0, 1]))
                self.assertEqual(float(after.grad[0, 0, -2, 5]), 0.)
                self.assertEqual(float(after.grad[0, 0, -1].abs().sum()), 0.)

    def test_only_complete_binary_positive_labels_are_selected(self):
        truth, _ = fixture()
        self.assertEqual(int(minority_centers(truth).sum()), len(MINORITY_MASKS))
        truth[0, 0, 0, 5] = .2
        self.assertEqual(int(minority_centers(truth).sum()), len(MINORITY_MASKS) - 1)

    def test_invalid_weights_fail_before_training(self):
        truth, prediction = fixture()
        for value in [0, .99, -1, True, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                word_relative_loss(truth, prediction, value)
        self.assertEqual(validate_minority_weight('2.5'), 2.5)
        for value in [0, .99, -1, True, float('nan'), float('inf'), None]:
            with self.subTest(dot_scale=value), self.assertRaisesRegex(ValueError, 'dot_loss_scale'):
                word_relative_loss(truth, prediction, dot_loss_scale=value)
        self.assertEqual(validate_dot_loss_scale('3'), 3.)


class StepTimingTests(unittest.TestCase):
    def test_stop_request_is_explicit_and_does_not_consume_sentinel(self):
        with tempfile.NamedTemporaryFile(suffix='.braille-stop', delete=True) as handle:
            path = Path(handle.name)
            self.assertEqual(requested_stop(False, path), 'stop_file_after_completed_epoch')
            self.assertTrue(path.exists())
            self.assertEqual(requested_stop(True, path), 'sigint_after_completed_epoch')
            self.assertIsNone(requested_stop(False, path.with_suffix('.absent')))

    def test_disabled_timing_does_not_synchronize_cuda(self):
        timer = StepTiming(False)
        with patch('word_symbol_v4_train.torch.cuda.synchronize') as synchronize:
            with timer.measure('forward'):
                pass
            synchronize.assert_not_called()
        self.assertEqual(timer.summary(), {})

    def test_enabled_stages_aggregate_and_data_wait_is_host_only(self):
        timer = StepTiming(True)
        with patch('word_symbol_v4_train.torch.cuda.synchronize') as synchronize, \
             patch('word_symbol_v4_train.time.perf_counter', side_effect=[1., 1.25, 2., 2.5, 3., 3.1]):
            with timer.measure('forward'):
                pass
            with timer.measure('forward'):
                pass
            with timer.measure('data_wait', synchronize=False):
                pass
            self.assertEqual(synchronize.call_count, 4)
        self.assertEqual(timer.summary()['forward'], dict(total_seconds=.75, steps=2, mean_milliseconds=375.))
        self.assertAlmostEqual(timer.summary()['data_wait']['mean_milliseconds'], 100.)


if __name__ == '__main__':
    unittest.main()
