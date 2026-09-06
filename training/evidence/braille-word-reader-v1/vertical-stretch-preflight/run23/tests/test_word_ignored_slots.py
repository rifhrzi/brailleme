"""Unknown source-grid slots must never acquire labels through augmentation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
from word_data import make_target
from word_train import Patches


class IgnoredSlotTests(unittest.TestCase):
    def test_unknown_full_or_partial_slots_have_no_positive_target(self):
        for partial in (False, True):
            target = make_target(np.empty((0, 4)), np.empty(0, int),
                                 partial_geometry_supervision=partial,
                                 ignore_boxes=[[100, 40, 130, 100], [200, -20, 240, 70]])
            self.assertTrue(np.any(target[..., 0] == -1))
            self.assertFalse(np.any(target[..., 0] > 0))
            self.assertEqual(float(np.abs(target[..., 1:]).sum()), 0.)

    def test_off_canvas_unknown_does_not_mask_other_pixels(self):
        target = make_target(np.empty((0, 4)), np.empty(0, int),
                             ignore_boxes=[[-30, -40, -5, -10], [490, 200, 520, 240]])
        self.assertEqual(float(np.abs(target).sum()), 0.)

    def test_empty_optional_field_preserves_existing_target(self):
        boxes, masks = np.asarray([[100, 40, 130, 100]], np.float32), np.asarray([1])
        np.testing.assert_array_equal(make_target(boxes, masks), make_target(boxes, masks, ignore_boxes=[]))

    def test_augmentation_cannot_restore_unknown_slot_supervision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (480, 192), (160, 160, 160)).save(root/'sample.png')
            record = {'image': 'sample.png', 'boxes': [], 'masks': [],
                      'ignore_boxes': [[180, -20, 225, 70], [100, 50, 135, 115]]}
            patches = Patches(root, [record], 1, True, 20260905, illumination_augmentation=True,
                              horizontal_mirror_augmentation=True, partial_geometry_supervision=True)
            _, target = patches[0]
            self.assertFalse(np.any(target[..., 0] > 0))
            self.assertTrue(np.any(target[..., 0] == -1))
            self.assertEqual(float(np.abs(target[..., 1:]).sum()), 0.)


if __name__ == '__main__':
    unittest.main()
