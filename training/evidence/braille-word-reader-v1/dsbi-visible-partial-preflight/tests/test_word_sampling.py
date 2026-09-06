"""Balanced fine-tuning must not silently discard source records or negatives."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_sampling import BalancedFamilyOrder


class WordSamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = ([{'source':f'books/large/{i}.jpg','boxes':[[1,2,3,4]]} for i in range(8)]
                     +[{'source':f'handwritten/small/{i}.jpg','boxes':[[1,2,3,4]]} for i in range(2)]
                     +[{'source':'not_braille/paper.jpg','boxes':[]} for _ in range(4)])

    def test_balanced_epoch_preserves_negative_indices_and_total_size(self):
        sampler=BalancedFamilyOrder(self.rows,42)
        for epoch in range(5):
            order=sampler.order(epoch)
            self.assertEqual(len(order),14)
            self.assertEqual(int((order<8).sum()),5)
            self.assertEqual(int(((order>=8)&(order<10)).sum()),5)
            self.assertEqual(sorted(order[order>=10].tolist()),[10,11,12,13])

    def test_cycles_cover_every_original_record_and_are_reproducible(self):
        first=BalancedFamilyOrder(self.rows,42)
        second=BalancedFamilyOrder(self.rows,42)
        seen=set()
        for epoch in range(first.describe()['complete_cycle_epochs']):
            np.testing.assert_array_equal(first.order(epoch),second.order(epoch))
            seen.update(map(int,first.order(epoch)))
        self.assertEqual(seen,set(range(len(self.rows))))
        self.assertFalse(np.array_equal(first.order(0),first.order(1)))

    def test_actual_order_audit_tracks_cumulative_source_coverage(self):
        sampler=BalancedFamilyOrder(self.rows,42)
        seen=set()
        first=sampler.audit(sampler.order(0),seen)
        final=sampler.audit(sampler.order(1),seen)
        self.assertEqual(first['families']['books/large']['cumulative_unique_records'],5)
        self.assertEqual(final['families']['books/large']['cumulative_unique_records'],8)
        self.assertEqual(final['cumulative_unique_record_indices'],14)


if __name__=='__main__':
    unittest.main()
