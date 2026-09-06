"""DSBI parser semantics: side identity, full-cell grid and hostile/malformed input."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_dsbi import parse_dsbi_annotation, validate_dsbi_image_bounds


HEADER = '-0.4\n10 30 60 82\n15 36 60 90 110 130\n'


class DsbiParserTests(unittest.TestCase):
    def test_preserves_column_major_bits_and_actual_uneven_middle_line(self):
        parsed = parse_dsbi_annotation(HEADER+'2 2 1 0 1 0 1 1\n1 1 0 1 0 1 0 0')
        self.assertEqual(parsed.angle_degrees, -.4)
        self.assertEqual(parsed.cells[0].mask, 53)
        self.assertEqual(parsed.cells[1].mask, 10)
        self.assertEqual(parsed.cells[1].dot_centers, ((10., 15.), (10., 36.), (10., 60.), (30., 15.), (30., 36.), (30., 60.)))
        self.assertEqual(parsed.cells[1].annotation_line, 5)
        self.assertEqual(parsed.cells[1].box(), (4., 9., 36., 66.))

    def test_all_masks_use_full_grid_and_zero_is_explicit_background(self):
        expected_box = (4., 9., 36., 66.)
        for mask in range(64):
            with self.subTest(mask=mask):
                bits = ' '.join(str((mask >> bit) & 1) for bit in range(6))
                parsed = parse_dsbi_annotation(HEADER+'1 1 '+bits)
                self.assertEqual(len(parsed.cells), 1)  # Do not synthesize missing slots.
                self.assertEqual(parsed.cells[0].mask, mask)
                self.assertEqual(parsed.cells[0].is_background, mask == 0)
                self.assertEqual(parsed.cells[0].box(), expected_box)

    def test_empty_side_is_allowed_but_partial_header_is_not(self):
        for text in ('', ' \r\n\t', '\ufeff\n'):
            self.assertTrue(parse_dsbi_annotation(text).is_empty)
        for text in ('0', '0\n10 30'):
            with self.assertRaises(ValueError):
                parse_dsbi_annotation(text)

    def test_plain_numeric_tokens_only_with_finite_scalar_angle(self):
        for angle in ('nan', 'inf', '1e999', '0 1', '1/2', '__import__("os").system("echo unsafe")'):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                parse_dsbi_annotation(angle+'\n10 30\n15 35 55\n')
        parsed = parse_dsbi_annotation('\ufeff+2e-1\r\n10\t30\r\n15 35 55\r\n1 1 1 0 0 0 0 0\r\n')
        self.assertEqual(parsed.angle_degrees, .2)

    def test_rejects_invalid_or_nonmonotonic_grid_without_evaluating_it(self):
        for vertical, horizontal in [('10 30 60', '15 35 55'), ('10 30', '15 35'), ('10 10', '15 35 55'), ('30 10', '15 35 55'), ('-1 30', '15 35 55'), ('10 30', '15 55 35'), ('10 NaN', '15 35 55'), ('10 30', '15 35 1e999'), ('10 30', '15 35 (1+54)')]:
            with self.subTest(vertical=vertical, horizontal=horizontal), self.assertRaises(ValueError):
                parse_dsbi_annotation('0\n'+vertical+'\n'+horizontal)

    def test_rejects_zero_fractional_out_of_grid_indices_and_nonbinary_bits(self):
        for row in ('0 1 1 0 0 0 0 0', '1.0 1 1 0 0 0 0 0', '3 1 1 0 0 0 0 0', '1 3 1 0 0 0 0 0', '1 1 1 0 0 0 0 2', '1 1 1 0 0 0 0 0.0', '1 1 1 0 0 0 0', '1 1 1 0 0 0 0 0 0'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_dsbi_annotation(HEADER+row)

    def test_rejects_duplicate_cells_even_when_both_are_blank(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            parse_dsbi_annotation(HEADER+'1 1 0 0 0 0 0 0\n1 1 0 0 0 0 0 0')

    def test_image_bounds_check_does_not_clip_extended_boxes_or_reapply_angle(self):
        parsed = parse_dsbi_annotation('12.5\n1 21\n1 21 41\n1 1 1 0 0 0 0 0')
        validate_dsbi_image_bounds(parsed, 22, 42)
        self.assertEqual(parsed.cells[0].box(), (-5., -5., 27., 47.))
        for dimensions in ((21, 42), (22, 41), (0, 42), (22., 42), (True, 42)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                validate_dsbi_image_bounds(parsed, *dimensions)
        for margin in (-1., float('inf'), float('nan'), 1e308):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                parsed.cells[0].box(margin)


if __name__ == '__main__':
    unittest.main()
