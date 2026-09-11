"""Export contracts that prevent numeric parity masking runtime incompatibility."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_v4_export import (
    decision_signature, runtime_quality_rejected, validate_outputs, validate_tensor_interface,
)


class ExportContractTests(unittest.TestCase):
    def test_dark_white_and_textureless_inputs_must_be_quality_rejected(self):
        for intensity in (0., .03, .5, .99, 1.):
            with self.subTest(intensity=intensity):
                self.assertTrue(runtime_quality_rejected(np.full((1,192,480,3), intensity, np.float32)))
        image = np.tile(np.linspace(.2, .8, 480, dtype=np.float32)[None,:,None], (192,1,3))
        self.assertFalse(runtime_quality_rejected(image))

    def test_same_text_cannot_hide_different_indicator_or_cell_order(self):
        base = dict(text='A', reason='accepted', cells=[dict(mask=32, role='capital', output=''),
                                                       dict(mask=1, role='letter', output='A')])
        altered = dict(base, cells=list(reversed(base['cells'])))
        self.assertNotEqual(decision_signature(base), decision_signature(altered))
        altered = dict(base, cells=[dict(base['cells'][0], mask=60), base['cells'][1]])
        self.assertNotEqual(decision_signature(base), decision_signature(altered))
        # Floating-point confidence is evaluated by allclose, not discrete equality.
        altered = dict(base, cells=[dict(cell, score=.9000001) for cell in base['cells']])
        self.assertEqual(decision_signature(base), decision_signature(altered))

    def test_interface_requires_exact_single_float32_model(self):
        inputs = [dict(shape=np.array([1,192,480,3]), dtype=np.float32)]
        outputs = [dict(shape=np.array([1,48,120,11]), dtype=np.float32)]
        validate_tensor_interface(inputs, outputs)
        for bad_inputs, bad_outputs in [(inputs*2, outputs), (inputs, []),
                ([dict(inputs[0], dtype=np.uint8)], outputs),
                (inputs, [dict(outputs[0], shape=np.array([1,48,120,7]))])]:
            with self.subTest(inputs=bad_inputs, outputs=bad_outputs), self.assertRaises(ValueError):
                validate_tensor_interface(bad_inputs, bad_outputs)

    def test_probabilities_require_full_finite_oracle_shape(self):
        good = np.zeros((2,48,120,11), np.float32)
        validate_outputs(good, 2)
        with self.assertRaises(ValueError):
            validate_outputs(good[:1], 2)
        for value in (float('nan'), float('inf'), -1., 1.01):
            changed = good.copy()
            changed[0,0,0,0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_outputs(changed, 2)


if __name__ == '__main__':
    unittest.main()
