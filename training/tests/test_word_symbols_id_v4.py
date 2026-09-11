"""Versioned grammar tests; these are not image-recognition accuracy evidence."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from word_symbols_id_v4 import (
    DECODER_VERSION, PUNCTUATION_MASKS, SOURCE_URL, encode_text, parse_word_masks,
)


class IndonesianPunctuationTests(unittest.TestCase):
    def test_primary_table_dot_patterns(self):
        dots = {'.': [2, 5, 6], ',': [2], '?': [2, 3, 6], '!': [2, 3, 5],
                ':': [2, 5], ';': [2, 3], '-': [3, 6], '(': [2, 3, 5, 6], ')': [2, 3, 5, 6]}
        self.assertEqual(PUNCTUATION_MASKS, {char: sum(1 << (dot - 1) for dot in positions)
                                             for char, positions in dots.items()})
        self.assertIn('experimental', DECODER_VERSION)
        self.assertIn('perkins.org', SOURCE_URL)

    def test_literal_sequences_include_indicator_cells(self):
        fixtures = [
            ('Baca?', [32, 3, 1, 9, 1, 38]),
            ('ABC!', [32, 32, 1, 3, 9, 22]),
            ('(001)', [54, 60, 26, 26, 1, 54]),
            ('(ABC).', [54, 32, 32, 1, 3, 9, 54, 50]),
            ('(Baca?)', [54, 32, 3, 1, 9, 1, 38, 54]),
            ('a-b', [1, 36, 3]),
        ]
        for text, masks in fixtures:
            with self.subTest(text=text):
                self.assertEqual(encode_text(text), masks)
                result = parse_word_masks(masks)
                self.assertEqual(result['text'], text)
                self.assertEqual([symbol['mask'] for symbol in result['symbols']], masks)
                self.assertEqual(''.join(symbol['output'] for symbol in result['symbols']), text)

    def test_complete_supported_vocabulary(self):
        for text in ['baca', 'Baca', 'BACA', 'iPhone', '001', '1234567890', 'baca-baca',
                     'ABC-DEF', '(1)', '12.', 'A:', 'A;', 'A,', 'A!', 'A?', '.', ',', '!', ':', ';', '-']:
            with self.subTest(text=text):
                self.assertEqual(parse_word_masks(encode_text(text))['text'], text)

    def test_quotes_and_isolated_parentheses_are_ambiguous(self):
        for masks in [[38], [54], [38, 1], [38, 1, 52], [54, 1], [1, 54],
                      [54, 54], [54, 54, 1, 54, 54]]:
            with self.subTest(masks=masks):
                self.assertEqual(parse_word_masks(masks)['status'], 'rejected')
        for text in ['?', '(', ')', '(a', 'a)', ')a(', '((a))', '?a', '"a"', '(?)']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                encode_text(text)

    def test_unsupported_semantics_and_incomplete_tokens_fail_closed(self):
        for text in ['', ' ', 'dua kata', 'a1', '1a', '-1', '12,5', '12.5', '12:30',
                     '1-2', '-a', 'a-', 'a--b', 'a.b', 'a!?', '(a!).', 'é', '١', 'A\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                encode_text(text)
        for masks in [[32], [60], [32, 50], [60, 50], [32, 32, 32, 1], [60, 1, 32, 3]]:
            self.assertEqual(parse_word_masks(masks)['status'], 'rejected')

    def test_printed_and_physical_limits(self):
        self.assertEqual(parse_word_masks(encode_text('ABCDEFGHIJK!'))['text'], 'ABCDEFGHIJK!')
        self.assertEqual(parse_word_masks([32, 1] * 12)['text'], 'A' * 12)
        self.assertEqual(parse_word_masks([1] * 12 + [50])['reason'], 'too_many_characters')
        self.assertEqual(parse_word_masks([32, 1] * 12 + [50])['reason'], 'too_many_cells')
        with self.assertRaises(ValueError):
            encode_text('ABCDEFGHIJKL!')

    def test_invalid_masks(self):
        for mask in [-1, 64, True, None, '1', 1.5, float('nan'), float('inf')]:
            self.assertEqual(parse_word_masks([mask])['reason'], 'unsupported_pattern')
        self.assertEqual(parse_word_masks([1.0])['text'], 'a')


if __name__ == '__main__':
    unittest.main()
