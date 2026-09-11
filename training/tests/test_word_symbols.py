"""Semantic fixtures are shared with the app; no image accuracy is implied."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from word_symbols import DECODER_VERSION, SYMBOL_PROTOCOL, parse_word_masks

FIXTURES = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/word-symbols-id-v1.json').read_text())


class WordSymbolTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(FIXTURES['decoderVersion'], DECODER_VERSION)
        self.assertEqual(FIXTURES['protocol'], SYMBOL_PROTOCOL)

    def test_shared_sequence_fixtures(self):
        for fixture in FIXTURES['cases']:
            with self.subTest(name=fixture['name']):
                result = parse_word_masks(fixture['masks'])
                if 'reason' in fixture:
                    self.assertEqual(result, {'status': 'rejected', 'reason': fixture['reason']})
                else:
                    self.assertEqual(result['status'], 'success')
                    self.assertEqual(result['text'], fixture['text'])
                    self.assertEqual([s['mask'] for s in result['symbols']], fixture['masks'])
                    self.assertEqual([s['output'] for s in result['symbols']], fixture['outputs'])
                    self.assertEqual([s['role'] for s in result['symbols']], fixture['roles'])

    def test_all_letters_with_and_without_capitals(self):
        for letter, mask in FIXTURES['alphabet']:
            with self.subTest(letter=letter):
                self.assertEqual(parse_word_masks([mask])['text'], letter)
                self.assertEqual(parse_word_masks([32, mask])['text'], letter.upper())

    def test_separate_printed_and_physical_length_limits(self):
        for masks, expected in [([32, 32] + [1] * 12, 'A' * 12), ([32, 1] * 12, 'A' * 12), ([60] + [26] * 12, '0' * 12)]:
            self.assertEqual(parse_word_masks(masks)['text'], expected)
        self.assertEqual(parse_word_masks([1] * 13), {'status': 'rejected', 'reason': 'too_many_characters'})
        self.assertEqual(parse_word_masks([1] * 25), {'status': 'rejected', 'reason': 'too_many_cells'})

    def test_invalid_masks(self):
        for value in [-1, 64, 1.2, float('nan'), float('inf'), True, '1', None]:
            self.assertEqual(parse_word_masks([value]), {'status': 'rejected', 'reason': 'unsupported_pattern'})
        # JavaScript has a single number type; integral floats have the same meaning.
        self.assertEqual(parse_word_masks([1.0])['text'], 'a')


if __name__ == '__main__':
    unittest.main()
