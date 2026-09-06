"""Verify affine gradients without mutable BN statistics or parent mutation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
import keras
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_bn_affine import affine_backbone_spec, build_bn_affine_parent


def small_parent(training=False):
    inputs = keras.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 3, padding='same')(inputs)
    x = keras.layers.BatchNormalization(trainable=False)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(4, 1)(x)
    x = keras.layers.BatchNormalization(trainable=False)(x)
    backbone = keras.Model(inputs, x, name='mobilenetv2_half')
    pixels = keras.Input((8, 8, 3))
    outputs = backbone(pixels, training=False) if training is False else backbone(pixels)
    return keras.Model(pixels, keras.layers.Conv2D(11, 1, activation='sigmoid')(outputs))


class BNAffineTests(unittest.TestCase):
    def test_initial_values_outputs_and_reload_are_identical(self):
        parent = small_parent()
        model, spec = build_bn_affine_parent(parent)
        self.assertEqual(spec['trainable_affine_layers'], 2)
        self.assertEqual(affine_backbone_spec(parent)['trainable_affine_layers'], 0)
        pixels = np.random.default_rng(53).uniform(0, 1, (3, 8, 8, 3)).astype(np.float32)
        expected = parent(pixels, training=False).detach().numpy()
        np.testing.assert_array_equal(model(pixels, training=False).detach().numpy(), expected)
        np.testing.assert_array_equal(model(pixels, training=True).detach().numpy(), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'affine.keras'
            model.save(path)
            loaded = keras.models.load_model(path, compile=False)
            self.assertEqual(affine_backbone_spec(loaded), spec)
            np.testing.assert_array_equal(loaded(pixels, training=True).detach().numpy(), expected)

    def test_affine_gradients_do_not_update_moving_moments(self):
        model, _ = build_bn_affine_parent(small_parent())
        batchnorm = [layer for layer in model.get_layer('mobilenetv2_half').layers
                     if isinstance(layer, keras.layers.BatchNormalization)]
        before = [layer.get_weights() for layer in batchnorm]
        pixels = np.random.default_rng(37).uniform(0, 1, (2, 8, 8, 3)).astype(np.float32)
        loss = torch.square(model(pixels, training=True)-.2).mean()
        loss.backward()
        for layer, weights in zip(batchnorm, before):
            for affine in (layer.gamma, layer.beta):
                self.assertIsNotNone(affine.value.grad)
                self.assertGreater(float(affine.value.grad.abs().max()), 0.)
            for moment in (layer.moving_mean, layer.moving_variance):
                self.assertFalse(moment.value.requires_grad)
                self.assertIsNone(moment.value.grad)
            for expected, actual in zip(weights, layer.get_weights()):
                np.testing.assert_array_equal(actual, expected)

    def test_implicit_training_and_already_open_affine_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'explicit training=False'):
            build_bn_affine_parent(small_parent(training=None))
        opened, _ = build_bn_affine_parent(small_parent())
        with self.assertRaisesRegex(ValueError, 'entirely frozen BN parent'):
            build_bn_affine_parent(opened)

    def test_plain_trainable_bn_is_rejected_despite_saved_false_call(self):
        parent = small_parent()
        for layer in parent.get_layer('mobilenetv2_half').layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
        with self.assertRaisesRegex(ValueError, 'inside layer.call'):
            affine_backbone_spec(parent)


if __name__ == '__main__':
    unittest.main()
