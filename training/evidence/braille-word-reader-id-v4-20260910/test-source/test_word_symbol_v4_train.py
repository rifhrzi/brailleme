"""CPU tests for selection integrity, group sampling and the actual Adam update."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_v4_train import (
    NativeAdam, ReplayPatches, REQUIRED_GROUPS, audit_rows, rank, reserve_test, score, verify_hashes,
    bind_sampling_checkpoint, sha,
)


class AccuracyTests(unittest.TestCase):
    def test_missing_predictions_and_nan_cannot_shrink_denominator(self):
        rows = [dict(source='one', expected='ABC', symbol_group='capital_word')]
        for outputs in [np.zeros((0, 1)), np.asarray([[float('nan')]])]:
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                score(outputs, rows, {})

    def test_rejected_words_count_as_wrong_and_negative_accepts_fail_gate(self):
        rows = [dict(source=str(i), expected=text, symbol_group=group) for i, (text, group) in enumerate([
            ('AB', 'capital_word'), ('BC', 'capital_word'), ('', 'rejection')])]
        answers = [dict(text='AB', reason='accepted', cells=[]),
                   dict(text='', reason='low_confidence', cells=[]),
                   dict(text='A', reason='accepted', cells=[])]
        with patch('word_symbol_v4_train.decode', side_effect=answers):
            groups, _ = score(np.zeros((3, 1)), rows, {})
        self.assertEqual(groups['capital_word']['exact_accuracy'], .5)
        self.assertEqual(groups['capital_word']['accepted_precision'], 1.)
        self.assertEqual(groups['rejection']['false_accepts'], 1)
        self.assertFalse(rank(groups)[0])

    def test_single_character_success_cannot_hide_failed_words(self):
        group = dict(positive_total=100, accepted_precision=1., exact_accuracy=.9, false_accepts=0,
                     multi_character=dict(positive_total=20, accepted_precision=1., exact_accuracy=.5))
        self.assertEqual(rank({'capital_word': group})[1], .5)
        with self.assertRaises(ValueError):
            rank({})

    def test_each_requested_punctuation_has_a_separate_denominator(self):
        rows = [dict(source=str(i), expected=text, symbol_group='punctuation')
                for i, text in enumerate(['abc!', 'abc?', '(abc)'])]
        with patch('word_symbol_v4_train.decode', side_effect=[
                dict(text='abc!', reason='accepted', cells=[]),
                dict(text='', reason='rejected', cells=[]),
                dict(text='(abc)', reason='accepted', cells=[])]):
            groups, _ = score(np.zeros((3, 1)), rows, {})
        details = groups['punctuation']['by_punctuation']
        self.assertEqual(details['!']['exact_accuracy'], 1.)
        self.assertEqual(details['?']['exact_accuracy'], 0.)
        self.assertEqual(details['(']['positive_total'], 1)
        self.assertIsNone(details['.']['exact_accuracy'])

    def test_all_groups_and_unique_identities_required(self):
        rows = [dict(source=group, expected='' if group == 'rejection' else 'ab',
                     boxes=[], masks=[], symbol_group=group) for group in sorted(REQUIRED_GROUPS)]
        audit_rows(rows)
        with self.assertRaisesRegex(ValueError, 'missing semantic'):
            audit_rows(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit_rows(rows+[rows[0]])


class SelectionTests(unittest.TestCase):
    def test_hard_sampling_checkpoint_must_equal_resume_and_be_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'checkpoint.keras'
            checkpoint.write_bytes(b'resumed checkpoint')
            weights = root/'weights.json'
            metadata = dict(split='train', scope='TRAIN_only_dot_mask_sampling', checkpoint_sha256=sha(checkpoint))
            weights.write_text(json.dumps(metadata))
            self.assertEqual(bind_sampling_checkpoint(weights, checkpoint), metadata)
            for changed in [dict(metadata, split='validation'), dict(metadata, scope='test'),
                            dict(metadata, checkpoint_sha256='0'*64)]:
                weights.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    bind_sampling_checkpoint(weights, checkpoint)

    def test_second_run_cannot_reopen_the_same_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            runs = [experiment/'first', experiment/'second']
            for run in runs:
                run.mkdir()
                (run/'selection.json').write_text('{}')
            reserve_test(experiment, runs[0], {'sha256': 'a'*64})
            with self.assertRaises(FileExistsError):
                reserve_test(experiment, runs[1], {'sha256': 'b'*64})
            reservation = json.loads((experiment/'test-evaluation-reservation.json').read_text())
            self.assertEqual(reservation['checkpoint_sha256'], 'a'*64)

    def test_changed_or_missing_frozen_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'input.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                verify_hashes({path: 'a'*64})
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                verify_hashes({path.with_name('missing'): 'a'*64})


class SamplerTests(unittest.TestCase):
    def test_hard_weights_are_group_normalized_and_preserve_half_replay(self):
        symbols = [dict(source=f'{group}-{weight}', symbol_group=group)
                   for group in ('capital_word', 'punctuation') for weight in (1, 3)]
        weights = {row['source']: int(row['source'][-1]) for row in symbols}
        args = (Path('.'), [{}]*20, symbols, 8, 1000, 123)
        first, second = ReplayPatches(*args, source_weights=weights), ReplayPatches(*args, source_weights=weights)
        np.testing.assert_array_equal(first.order, second.order)
        self.assertEqual(np.count_nonzero(first.order < 20), 4000)
        self.assertEqual(first.sampling_log[0]['symbol_draws'], dict(capital_word=2000, punctuation=2000))
        for group, indices in first.group_indices.items():
            np.testing.assert_array_equal(first.group_probabilities[group], [.25, .75])
            draws = sum(first.order == indices[1])
            self.assertAlmostEqual(draws/2000, .75, delta=.04)
        default, explicit_none = ReplayPatches(*args), ReplayPatches(*args, source_weights=None)
        np.testing.assert_array_equal(default.order, explicit_none.order)
        for invalid in [{}, dict(weights, extra=1), {key: 0 for key in weights}, {key: True for key in weights}]:
            with self.assertRaises(ValueError):
                ReplayPatches(*args, source_weights=invalid)

    def test_equal_semantic_sampling_half_replay_reproducible_and_changes_epoch(self):
        symbols = [dict(symbol_group=g) for g in ['capital_word']*50+['numbers']*2+['punctuation']]
        args = (Path('.'), [{}]*20, symbols, 12, 10, 123)
        first, second = ReplayPatches(*args), ReplayPatches(*args)
        np.testing.assert_array_equal(first.order, second.order)
        self.assertEqual(np.count_nonzero(first.order < 20), 60)
        self.assertEqual(first.sampling_log[0]['symbol_draws'],
                         dict(capital_word=20, numbers=20, punctuation=20))
        previous = first.order.copy()
        first.on_epoch_end()
        self.assertFalse(np.array_equal(first.order, previous))


class OptimizerTests(unittest.TestCase):
    def test_two_adam_updates_match_independent_numpy_equations(self):
        parameter = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        frozen = torch.nn.Parameter(torch.tensor([7.]), requires_grad=False)
        optimizer = NativeAdam([parameter, frozen], .01)
        expected = np.asarray([1., -2.])
        first, second = np.zeros(2), np.zeros(2)
        for step, gradient in enumerate([[1., -2.], [.5, -1.]], 1):
            gradient = np.asarray(gradient)
            parameter.grad = torch.tensor(gradient)
            first = .9*first+.1*gradient
            second = .999*second+.001*gradient**2
            expected -= .01*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-7)
            optimizer.step()
            np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(frozen.item(), 7.)
        optimizer.zero_grad()
        self.assertIsNone(parameter.grad)

    def test_global_gradient_clip_and_nonfinite_rejection(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = NativeAdam([parameter], .01)
        parameter.grad = torch.tensor([100.])
        optimizer.step()
        self.assertAlmostEqual(optimizer.moments[0][0].item(), 1., places=6)
        parameter.grad = torch.tensor([float('inf')])
        before = parameter.detach().clone()
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            optimizer.step()
        torch.testing.assert_close(parameter.detach(), before)
        self.assertEqual(optimizer.steps, 1)


if __name__ == '__main__':
    unittest.main()
