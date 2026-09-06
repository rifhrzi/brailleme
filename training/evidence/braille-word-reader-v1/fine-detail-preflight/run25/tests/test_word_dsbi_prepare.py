"""DSBI context preparation must preserve full cells and separate unknown slots."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_dsbi import parse_dsbi_annotation
from word_dsbi_prepare import context_labels, plan_contexts, unknown_slot_boxes


class DsbiContextTests(unittest.TestCase):
    def test_blank_and_unknown_slots_never_become_positive_cells(self):
        page = parse_dsbi_annotation('0\n20 40 70 90 120 140\n30 50 70\n1 1 1 0 0 0 0 0\n1 2 0 0 0 0 0 0')
        ignored = unknown_slot_boxes(page)
        np.testing.assert_array_equal(ignored, [[114, 24, 146, 76]])
        label = context_labels(page, [0, 0, 200, 100])
        self.assertEqual(label['masks'], [1])
        self.assertEqual(label['originally_complete'], [True])
        self.assertEqual(label['source_cells'], [{'row': 1, 'col': 1, 'annotation_line': 4}])
        np.testing.assert_allclose(label['boxes'], [[33.6, 46.08, 110.4, 145.92]])
        np.testing.assert_allclose(label['ignore_boxes'], [[273.6, 46.08, 350.4, 145.92]])

    def test_cut_source_box_remains_full_and_unknown_box_can_cross_canvas(self):
        page = parse_dsbi_annotation('0\n20 40 70 90\n30 50 70\n1 1 1 0 0 0 0 0')
        label = context_labels(page, [20, 25, 85, 80])
        self.assertEqual(label['originally_complete'], [False])
        self.assertLess(label['boxes'][0][0], 0)
        self.assertLess(label['boxes'][0][1], 0)
        self.assertGreater(label['ignore_boxes'][0][2], 480)
        self.assertEqual(label['masks'], [1])

    def test_exact_source_boundary_stays_complete_despite_resize_rounding(self):
        # Actual TRAIN source geometry from Shaver Yang Fengting: the full
        # right edge equals the crop, but factor-first resize rounds above 480.
        page = parse_dsbi_annotation('0\n189 209\n570 591 612\n1 1 1 0 0 0 0 0')
        exact = context_labels(page, [41, 556, 215, 626])
        self.assertGreater(exact['boxes'][0][2], 480)
        self.assertEqual(exact['originally_complete'], [True])
        # A real source-raster cut remains incomplete; no epsilon can restore it.
        cut = context_labels(page, [41, 556, 214, 626])
        self.assertEqual(cut['originally_complete'], [False])
        self.assertGreater(cut['boxes'][0][2], 480)

    def test_every_listed_positive_anchors_once_and_is_complete_somewhere(self):
        vertical = ' '.join(str(20+50*column+offset) for column in range(16) for offset in (0, 20))
        lines = ['0', vertical, '30 50 70 120 140 160']
        slots = [(1, column) for column in (1, 2, 4, 7, 8, 12, 16)]+[(2, 1), (2, 16)]
        lines += [f'{row} {col} 1 0 0 0 0 0' for row, col in slots]
        page = parse_dsbi_annotation('\n'.join(lines))
        anchored, complete = [], set()
        for crop, anchors, _ in plan_contexts(page, 850, 240, 0):
            self.assertTrue(0 <= crop[0] < crop[2] <= 850)
            self.assertTrue(0 <= crop[1] < crop[3] <= 240)
            anchored.extend((cell.row, cell.col) for cell in anchors)
            label = context_labels(page, crop)
            complete.update((cell['row'], cell['col']) for cell, full in zip(label['source_cells'], label['originally_complete']) if full)
        self.assertCountEqual(anchored, slots)
        self.assertEqual(complete, set(slots))

    def test_only_a_documented_empty_annotation_produces_empty_side_tiles(self):
        page = parse_dsbi_annotation('')
        planned = list(plan_contexts(page, 1700, 2300, 0))
        self.assertEqual(len(planned), 9)
        for crop, anchors, kind in planned:
            self.assertEqual(kind, 'documented_empty_recto')
            self.assertEqual(anchors, ())
            label = context_labels(page, crop)
            self.assertEqual(label['boxes'], [])
            self.assertEqual(label['ignore_boxes'], [])


if __name__ == '__main__':
    unittest.main()
