"""No model needed to verify native source continuity and stress scope."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_dsbi import DSBICell
from word_overall_v5_prepare import native_windows, native_group, required_masks, git_blob
from word_symbol_v4_prepare import GROUPS, sample_text
from word_symbols_id_v4 import encode_text


def cell(row, col, mask):
    return DSBICell(row, col, tuple((mask >> bit) & 1 for bit in range(6)),
        tuple((col*10+x, row*20+y) for x in (0, 4) for y in (0, 4, 8)), 4+col)


class OverallPreparationTests(unittest.TestCase):
    def test_native_windows_keep_original_masks_and_stop_at_missing_or_blank_cells(self):
        originals = [cell(1,1,1), cell(1,2,3), cell(1,4,9), cell(1,5,25),
                     cell(1,6,0), cell(1,7,17), cell(2,1,32), cell(2,2,1)]
        windows = native_windows(originals)
        self.assertEqual({row['expected'] for row in windows}, {'ab', 'cd'})
        for window in windows:
            self.assertEqual([c.col for c in window['cells']], list(range(window['start_col'], window['end_col']+1)))
            self.assertTrue(all(any(c is original for original in originals) for c in window['cells']))

    def test_context_grouping_does_not_invent_missing_groups(self):
        self.assertEqual(native_group('abc'), 'lowercase')
        self.assertEqual(native_group('Abc'), 'capital_single')
        self.assertEqual(native_group('ABC'), 'capital_word')
        self.assertEqual(native_group('123'), 'numbers')
        self.assertEqual(native_group('(ab)'), 'punctuation')

    def test_primary_retains_original_96_percent_multicharacter_and_known_masks(self):
        rng = random.Random(20260911+200006)
        texts = [sample_text(group, index, rng) for group in GROUPS for index in range(100)]
        self.assertEqual(len(texts), 500)
        self.assertEqual(sum(len(value) >= 2 for value in texts), 480)
        self.assertTrue(all(set(encode_text(value)) <= required_masks() for value in texts))

    def test_git_blob_includes_object_header(self):
        self.assertEqual(git_blob(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')


if __name__ == '__main__':
    unittest.main()
