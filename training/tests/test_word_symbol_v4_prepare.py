"""Regression checks for corpus isolation and rare-indicator augmentation."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_v4_prepare import (DonorPicker, GROUPS, assert_source_disjoint,
    old_donors, sample_text, select_test_pages, transform_mask, transform_pool)


class CorpusV4Tests(unittest.TestCase):
    def test_existing_capital_mask_also_gets_transformed_donors(self):
        original=[dict(source='page/a', image='a.png', box=[1,1,9,17], original_mask=1),
                  dict(source='page/b', image='b.png', box=[1,1,9,17], original_mask=32)]
        pool=transform_pool(original, {1,32})
        self.assertEqual({(d['original_mask'],d['synthetic_transform']) for d in pool[32]},
                         {(1,'half_turn'),(32,'identity')})
        self.assertNotIn('mask', original[0])
        self.assertNotIn('synthetic_transform', original[1])

    def test_exact_transforms_are_involutions_on_all_masks(self):
        for transform in ('identity','horizontal','vertical','half_turn'):
            for mask in range(64):
                self.assertEqual(transform_mask(transform_mask(mask,transform),transform),mask)

    def test_missing_masks_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'lacks masks'):
            transform_pool([dict(source='a',original_mask=1)], {60})

    def test_selection_is_deterministic_and_spans_families(self):
        members=[f'{family}/{index}.jpg' for family in ('a','b','c') for index in range(8)]
        first=select_test_pages(members,12,10)
        self.assertEqual(first,select_test_pages(list(reversed(members)),12,10))
        self.assertEqual(len(set(first)),12)
        self.assertEqual({v.split('/')[0] for v in first},{'a','b','c'})

    def test_source_overlap_aborts(self):
        with self.assertRaisesRegex(ValueError,'Source leakage'):
            assert_source_disjoint({'train':['a'],'test':['a']})

    def test_partial_old_donors_do_not_become_complete_targets(self):
        rows=[dict(source='p',image='a.png',boxes=[[-1,2,20,30],[1,2,20,30]],masks=[32,1])]
        self.assertEqual([r['original_mask'] for r in old_donors(Path('.'),rows)],[1])

    def test_generated_text_is_valid_and_mostly_multicharacter(self):
        from word_symbols_id_v4 import encode_text, parse_word_masks
        rng=random.Random(10)
        for group in GROUPS:
            values=[sample_text(group,i,rng) for i in range(120)]
            self.assertGreaterEqual(sum(len(v)>=2 for v in values)/len(values),.9)
            for value in values:
                self.assertEqual(parse_word_masks(encode_text(value)).get('text'),value)


if __name__=='__main__':
    unittest.main()
