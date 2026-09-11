"""Promotion comparison tests; never create or rewrite model evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from word_symbol_deployment_evidence import compare_words, symbol_results


def report(predictions):
    words = [dict(image=f'{index}.png', source=f'page/{index}', expected=value,
                  selection=[0, 0, 1, 1], predicted=prediction, rejection='' if prediction else 'low_confidence')
             for index, (value, prediction) in enumerate(zip(['a', 'b', 'c'], predictions, strict=True))]
    return dict(decoder_version='six-dot-id-v3', thresholds={'acceptScore': .25}, fixture_manifest_sha256='f' * 64,
                words=words, whole_words=dict(total=3, correct=sum(row['expected'] == row['predicted'] for row in words)))


class DeploymentTests(unittest.TestCase):
    def test_counts_transitions_from_aligned_predictions_and_keeps_reasons(self):
        before, after = report(['a', 'b', '']), report(['a', '', ''])
        counts, rows = compare_words(before, after, 3)
        self.assertEqual(counts, dict(total=3, baseline_correct=2, candidate_correct=1,
                                     lost=1, gained=0, both_correct=1, neither_correct=1))
        self.assertEqual(rows[1]['baseline']['text'], 'b')
        self.assertEqual(rows[1]['candidate']['reason'], 'low_confidence')

    def test_refuses_identity_threshold_decoder_or_fixture_drift(self):
        before, after = report(['a', 'b', '']), report(['a', '', ''])
        for field, value in [('thresholds', {'acceptScore': .2}), ('decoder_version', 'old'), ('fixture_manifest_sha256', 'different')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_words(before, {**after, field: value}, 3)
        changed = copy.deepcopy(after)
        changed['words'][0]['selection'] = [.1, 0, 1, 1]
        with self.assertRaises(ValueError):
            compare_words(before, changed, 3)

    def test_refuses_no_regression_and_false_aggregate(self):
        with self.assertRaisesRegex(ValueError, 'No external'):
            compare_words(report(['a', '', '']), report(['a', 'b', '']), 3)
        before = report(['a', 'b', ''])
        before['whole_words']['correct'] = 3
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            compare_words(before, report(['a', '', '']), 3)

    def test_symbol_gates_count_invalid_acceptance_independently(self):
        fixtures, replay = [], []
        for group, count in [('lowercase', 140), ('capital_single', 140), ('capital_word', 140), ('numbers', 64), ('rejection', 32)]:
            for _ in range(count):
                index = len(fixtures)
                expected = '' if group == 'rejection' else 'a'
                fixtures.append(dict(image=f'{index}.png', expected=expected, origin='synthetic', symbol_group=group))
                replay.append(dict(index=index, image=f'{index}.png', expected=expected, python_text=expected, python_reason='' if expected else 'invalid_sequence'))
        _, _, _, gates = symbol_results(fixtures, {'rows': replay})
        self.assertTrue(gates)
        replay[-1].update(python_text='a', python_reason='')
        rows, groups, mandatory, gates = symbol_results(fixtures, {'rows': replay})
        self.assertFalse(gates)
        self.assertEqual(groups['rejection']['false_acceptances'], 1)
        self.assertEqual((len(rows), len(mandatory)), (516, 32))


if __name__ == '__main__':
    unittest.main()
