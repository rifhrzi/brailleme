"""Primary overall target, source isolation, and deterministic reservation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_overall_v5_protocol import GROUPS, metrics, reserve_pages, validation_rank


class OverallProtocolTests(unittest.TestCase):
    def rows(self):
        rows = []
        for group in GROUPS:
            for index in range(100):
                expected = 'a' if group != 'punctuation' and index >= 95 else 'ab'
                rows.append(dict(source=f'{group}/{index}', group=group, expected=expected,
                    predicted=expected if index < 80 else 'wrong'))
        return rows+[dict(source=f'invalid/{index}', group='rejection', expected='', predicted='') for index in range(40)]

    def test_eighty_overall_is_independent_of_precision_and_worst_group(self):
        rows = self.rows()
        # Move20 correct cases from first group to the second:60% and100%.
        for row in rows[60:80]: row['predicted'] = 'wrong'
        for row in rows[180:200]: row['predicted'] = row['expected']
        value = metrics(rows, require_primary=True)
        self.assertEqual(value['overall_exact_accuracy'], .8)
        self.assertTrue(value['target80_met'])
        self.assertFalse(value['precision95_met'])
        self.assertEqual(value['groups']['lowercase']['correct'], 60)

    def test_rejections_count_wrong_and_invalid_rows_never_pad_accuracy(self):
        rows = self.rows()
        for row in rows:
            if row['predicted'] == 'wrong': row['predicted'] = ''
        value = metrics(rows, require_primary=True)
        self.assertEqual(value['positive_total'], 500)
        self.assertEqual(value['overall_exact_accuracy'], .8)
        self.assertEqual(value['accepted_precision'], 1.)
        rows[-1]['predicted'] = 'a'
        changed = metrics(rows, require_primary=True)
        self.assertTrue(changed['target80_met'])
        self.assertFalse(changed['invalid_rejection_met'])

    def test_missing_groups_duplicate_examples_and_shortened_scope_rejected(self):
        for rows in (self.rows()[:-1], self.rows()+self.rows()[:1]):
            with self.assertRaises(ValueError): metrics(rows, require_primary=True)
        rows = self.rows()
        rows[0]['expected'] = 'a'
        with self.assertRaises(ValueError): metrics(rows, require_primary=True)

    def test_rank_follows_exact_not_a_precision_gate(self):
        lower, higher = self.rows(), self.rows()
        for row in lower:
            if row['predicted'] == 'wrong': row['predicted'] = ''
        higher[80]['predicted'] = higher[80]['expected']
        self.assertGreater(validation_rank(higher), validation_rank(lower))

    def test_reservation_is_disjoint_seeded_family_balanced_without_labels(self):
        train = ['train/a.jpg']
        test = [f'{family}/{index}.jpg' for family in ('a', 'b', 'c') for index in range(10)]
        observed = ['a/0.jpg', 'b/0.jpg', 'c/0.jpg']
        selected, remaining = reserve_pages(train, test, observed, 12, 11)
        self.assertEqual((selected, remaining), reserve_pages(train, list(reversed(test)), observed, 12, 11))
        self.assertEqual(len(selected), 12)
        self.assertFalse(set(selected) & set(train+observed+remaining))
        self.assertEqual({family:sum(item.startswith(family+'/') for item in selected) for family in ('a','b','c')}, {'a':4,'b':4,'c':4})
        for badtrain, badobserved in [(test[:1], observed), (train, ['unknown.jpg'])]:
            with self.assertRaises(ValueError): reserve_pages(badtrain, test, badobserved, 12)


if __name__ == '__main__':
    unittest.main()
