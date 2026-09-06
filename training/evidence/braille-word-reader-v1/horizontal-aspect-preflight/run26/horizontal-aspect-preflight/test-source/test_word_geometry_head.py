"""Analytic sampling contract before any geometry-conditioned model training."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import keras
from keras import ops
import numpy as np
from word_geometry_head import GeometrySlotSampler, bilinear_feature_samples


class GeometryHeadTests(unittest.TestCase):
    def test_bilinear_ramp_border_and_batch_identity(self):
        yy, xx = np.mgrid[:4, :5].astype(np.float32)
        features = np.stack([np.stack([2*xx+3*yy, xx], axis=-1),
                             np.stack([100+2*xx+3*yy, 100+xx], axis=-1)])
        x = np.asarray([[1.25, -2., 9.], [2.75, 0., 4.]], np.float32)
        y = np.asarray([[1.5, 2., 9.], [.25, -3., 3.]], np.float32)
        actual = ops.convert_to_numpy(bilinear_feature_samples(ops.convert_to_tensor(features),
                                                               ops.convert_to_tensor(x), ops.convert_to_tensor(y)))
        x, y = np.clip(x, 0, 4), np.clip(y, 0, 3)
        expected = np.stack([2*x+3*y, x], axis=-1)+np.asarray([0., 100.])[:, None, None]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_point_corner_offset_lattice_and_slot_order(self):
        fractions = [[.2, .15], [.2, .5], [.2, .85], [.8, .15], [.8, .5], [.8, .85]]
        layer = GeometrySlotSampler(fractions)
        yy, xx = np.mgrid[:48, :120].astype(np.float32)
        # Features encode input pixel-index coordinates of their nominal grid.
        features = np.stack([xx*4+3, yy*4+3], axis=-1)[None]
        probabilities = np.zeros((1, 48, 120, 11), np.float32)
        probabilities[..., 1:3] = .5
        probabilities[..., 3] = 40/480
        probabilities[..., 4] = 60/192
        result = ops.convert_to_numpy(layer([features, probabilities]))[0, 24, 60].reshape(6, 2)
        expected = np.asarray([[(60+.5)*4+(fx-.5)*40-.5,
                                (24+.5)*4+(fy-.5)*60-.5] for fx, fy in fractions])
        np.testing.assert_allclose(result, expected, atol=3e-5, rtol=0)
        restored = GeometrySlotSampler.from_config(layer.get_config())
        np.testing.assert_array_equal(ops.convert_to_numpy(restored([features, probabilities])),
                                      ops.convert_to_numpy(layer([features, probabilities])))

    def test_geometry_and_feature_gradients_survive_interpolation(self):
        if keras.backend.backend() != 'torch':
            self.skipTest('CPU Torch gradient check')
        import torch
        features = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5, 1).requires_grad_()
        x = torch.tensor([[1.25]], requires_grad=True)
        y = torch.tensor([[1.5]], requires_grad=True)
        bilinear_feature_samples(features, x, y).sum().backward()
        self.assertEqual(float(x.grad.item()), 1.)
        self.assertEqual(float(y.grad.item()), 5.)
        self.assertEqual(float(features.grad.sum()), 1.)

    def test_unreviewed_lattice_and_invalid_priors_fail(self):
        fractions = [[.2, .15]]*6
        with self.assertRaises(ValueError):
            GeometrySlotSampler(fractions, feature_origin=0.)
        with self.assertRaises(ValueError):
            GeometrySlotSampler([[float('nan'), .2]]*6)

    def test_slot_geometry_gradient_reaches_all_four_inherited_geometry_values(self):
        if keras.backend.backend() != 'torch':
            self.skipTest('CPU Torch gradient check')
        import torch
        fractions = [[.2, .15], [.2, .51], [.2, .89], [.83, .15], [.83, .51], [.83, .89]]
        yy, xx = np.mgrid[:48, :120].astype(np.float32)
        features = torch.tensor((2*xx+3*yy)[None, ..., None])
        probabilities = torch.full((1, 48, 120, 11), .1, requires_grad=True)
        result = GeometrySlotSampler(fractions)([features, probabilities])
        result[0, 24, 60].sum().backward()
        self.assertTrue(bool(torch.all(probabilities.grad[0, 24, 60, 1:5] != 0)))
        self.assertEqual(float(probabilities.grad[0, 24, 60, 0]), 0.)


if __name__ == '__main__':
    unittest.main()
