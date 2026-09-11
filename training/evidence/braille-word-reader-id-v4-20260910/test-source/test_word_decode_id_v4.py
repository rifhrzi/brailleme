"""Model-output fixtures verify geometry plus grammar, not learned accuracy."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from word_decode import decode as legacy_decode, DEFAULT_THRESHOLDS
from word_decode_id_v4 import decode
from word_symbols_id_v4 import encode_text

THRESHOLDS = dict(DEFAULT_THRESHOLDS, proposalScore=.25, acceptScore=.25, dotConfidence=.7)


def tensor(masks):
    output = np.zeros((48, 120, 11), np.float32)
    for index, mask in enumerate(masks):
        cx = 110 + index * 18
        gx = int(cx // 4)
        output[24, gx] = [.99, cx / 4 - gx, 0, 12 / 480, 32 / 192] + [float(bool(mask & (1 << bit))) for bit in range(6)]
    return output


class ExperimentalDecoderTests(unittest.TestCase):
    def test_symbols_survive_localization_with_exact_physical_roles(self):
        for text in ['Baca?', '(ABC)', '(001)', 'baca-baca', 'ABC!']:
            masks = encode_text(text)
            result = decode(tensor(masks), THRESHOLDS)
            with self.subTest(text=text):
                self.assertEqual(result['text'], text)
                self.assertEqual([cell['mask'] for cell in result['cells']], masks)
                self.assertEqual(''.join(cell['output'] for cell in result['cells']), text)
                self.assertEqual(len(result['cells']), len(masks))

    def test_legacy_geometry_and_basic_semantics_remain_identical(self):
        for text in ['abc', 'Baca', 'ABC', '001']:
            output = tensor(encode_text(text))
            self.assertEqual(decode(output, THRESHOLDS), legacy_decode(output, THRESHOLDS))
            for selection in [[.23, .45, .5, .55], [0, 0, 1, 1]]:
                self.assertEqual(decode(output, THRESHOLDS, selection), legacy_decode(output, THRESHOLDS, selection))

    def test_weak_punctuation_and_ambiguous_cells_reject_complete_result(self):
        output = tensor(encode_text('a?'))
        output[24, 32, 5] = .5
        self.assertEqual(decode(output, THRESHOLDS)['reason'], 'low_confidence')
        for mask in [38, 54]:
            self.assertEqual(decode(tensor([mask]), THRESHOLDS)['reason'], 'ambiguous_symbol')
        self.assertEqual(legacy_decode(tensor(encode_text('a?')), THRESHOLDS)['reason'], 'unsupported_pattern')


if __name__ == '__main__':
    unittest.main()
