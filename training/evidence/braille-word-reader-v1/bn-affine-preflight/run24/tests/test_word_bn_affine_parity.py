"""A numerical tolerance must never conceal a changed decoder decision."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_bn_affine_parity import check_outputs, decision_rows, require_same_decisions
from word_decode import DEFAULT_THRESHOLDS


class BNAffineParityTests(unittest.TestCase):
    def test_near_threshold_change_is_not_hidden_by_float_tolerance_or_quality(self):
        high = np.zeros((1, 48, 120, 11), dtype=np.float32)
        high[0, 24, 60] = [.25001, .5, .5, .05, .3, .99, .01, .01, .01, .01, .01]
        low = high.copy(); low[0, 24, 60, 0] = .24999
        np.testing.assert_allclose(high, low, atol=1e-4, rtol=1e-4)
        images = np.ones((1, 192, 480, 3), dtype=np.float32)
        thresholds = dict(DEFAULT_THRESHOLDS, proposalScore=.25, acceptScore=.25, dotConfidence=.7)
        before, after = decision_rows(high, images, thresholds), decision_rows(low, images, thresholds)
        self.assertEqual(before[0]['raw']['text'], 'A')
        self.assertEqual(after[0]['raw']['reason'], 'no_grid')
        self.assertEqual(before[0]['effective'], after[0]['effective'])
        with self.assertRaisesRegex(ValueError, 'decoder decisions differ'):
            require_same_decisions(before, after)

    def test_output_contract_rejects_wrong_dtype_shape_and_nonprobabilities(self):
        valid = np.zeros((6, 48, 120, 11), dtype=np.float32)
        check_outputs(valid)
        for bad in (valid.astype(np.float64), valid[:5], valid+np.nan, valid-1, valid+2):
            with self.assertRaises(ValueError):
                check_outputs(bad)


if __name__ == '__main__':
    unittest.main()
