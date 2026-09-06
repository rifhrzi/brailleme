"""A frozen-moment violation must fail before publishing any epoch checkpoint."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import keras
import numpy as np
from word_bn_affine import build_bn_affine_parent
from word_train import WordCheckpoint, moving_moments_sha256, validated_bn_affine


def small_model():
    image = keras.Input((4, 4, 3))
    bn = keras.layers.BatchNormalization(trainable=False, name='test_bn')
    backbone = keras.Model(image, bn(image), name='mobilenetv2_half')
    pixels = keras.Input((4, 4, 3))
    parent = keras.Model(pixels, backbone(pixels, training=False))
    return build_bn_affine_parent(parent)[0]


class BNAffineCheckpointTest(unittest.TestCase):
    def test_only_stored_moments_participate_in_guard_hash(self):
        model = small_model()
        bn = model.get_layer('mobilenetv2_half').get_layer('test_bn')
        original = moving_moments_sha256(model)
        bn.gamma.assign(bn.gamma.numpy()+.25)
        bn.beta.assign(bn.beta.numpy()-.25)
        self.assertEqual(moving_moments_sha256(model), original)
        bn.moving_mean.assign(np.ones(3, np.float32))
        self.assertNotEqual(moving_moments_sha256(model), original)

    def test_changed_moments_abort_before_last_or_retained_save(self):
        model = small_model()
        original = moving_moments_sha256(model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'validation-words.json').write_text('{"records": []}')
            callback = WordCheckpoint(root, root, 1, retain_epochs=True, expected_bn_moments=original)
            callback.set_model(model)
            bn = model.get_layer('mobilenetv2_half').get_layer('test_bn')
            bn.moving_variance.assign(np.full(3, 1.25, np.float32))
            with patch.object(model, 'save') as save, self.assertRaisesRegex(RuntimeError, 'before writing'):
                callback.on_epoch_end(0, {})
            save.assert_not_called()
            self.assertFalse((root/'last.keras').exists())
            self.assertFalse((root/'retained-epochs').exists())

    def test_custom_class_cannot_enter_training_without_explicit_proofs(self):
        with self.assertRaisesRegex(ValueError, 'explicit audited'):
            validated_bn_affine(small_model(), None, None, None)


if __name__ == '__main__':
    unittest.main()
