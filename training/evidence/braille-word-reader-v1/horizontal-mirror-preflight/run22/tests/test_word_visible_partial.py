"""Real partial geometry may teach rejection without inventing six dot labels."""
import contextlib
import io
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
from word_data import make_target, augment_patch
from word_decode import decode
from word_point_geometry import OccupiedPointState, visible_partial_candidates
from word_train import Patches, parse_args


class VisiblePartialTests(unittest.TestCase):
    def state(self):
        points = np.asarray([[[460, 65], [460, 91], [460, 120], [500, 65], [500, 91], [500, 120]]])
        return OccupiedPointState.from_source_crop(points, [11], [0, 0, 480, 192])

    def test_visible_fragment_is_unknown_bits_and_rejects_ideal_short_word(self):
        boxes = np.asarray([[100, 40, 135, 110], [446, 44, 513, 145]], np.float64)
        fragments = visible_partial_candidates(boxes[1:], self.state())[1]
        target = make_target(boxes, np.asarray([1, 11]), [True, False], visible_partial_boxes=fragments)
        proposals = target[target[..., 0] == 1]
        self.assertEqual(len(proposals), 2)
        self.assertEqual(int(np.sum(np.all(proposals[:, 5:] == -1, axis=1))), 1)
        ideal = np.clip(target, 0, 1)
        self.assertEqual(decode(ideal)['reason'], 'clipped')

    def test_unknown_slots_and_angelina_partial_stay_without_targets(self):
        target = make_target(np.asarray([[446, 44, 513, 145]]), np.asarray([11]), [False],
                             ignore_boxes=[[200, -20, 240, 50]])
        self.assertFalse(np.any(target[..., 0] > 0))
        self.assertEqual(float(np.abs(target[..., 1:]).sum()), 0.)

    def test_tracking_preserves_rng_and_pixels_for_paired_augmentation(self):
        image = Image.fromarray(np.random.default_rng(7).integers(80, 200, (192, 480, 3), dtype=np.uint8))
        boxes = np.asarray([[446, 44, 513, 145]], np.float64)
        for seed in range(24):
            first_rng, second_rng = np.random.default_rng(seed), np.random.default_rng(seed)
            old = augment_patch(image, boxes, first_rng, illumination_augmentation=True)
            tracked = augment_patch(image, boxes, second_rng, illumination_augmentation=True, point_state=self.state())
            for a, b in zip(old, tracked[:3]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(first_rng.bit_generator.state, second_rng.bit_generator.state)

    def test_patches_excludes_unknown_slots_from_point_count_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (480, 192), (140, 140, 140)).save(root/'sample.png')
            row = {'image': 'sample.png', 'boxes': [[446, 44, 513, 145]], 'masks': [11],
                   'originally_complete': [False], 'ignore_boxes': [[200, -20, 240, 50]]}
            old_pixels, old = Patches(root, [row], 1, False, 7)[0]
            pixels, target = Patches(root, [row], 1, False, 7, point_states={'sample.png': self.state()})[0]
            np.testing.assert_array_equal(old_pixels, pixels)
            self.assertFalse(np.any(old[..., 0] > 0))
            proposals = target[target[..., 0] == 1]
            self.assertEqual(len(proposals), 1)
            self.assertTrue(np.all(proposals[:, 5:] == -1))

    def test_cli_requires_explicit_metadata_and_separate_experiment(self):
        base = ['--data', 'data', '--output', 'out']
        for flags in (['--dsbi-visible-partial-supervision'], ['--dsbi-point-metadata', 'points.json'],
                      ['--dsbi-visible-partial-supervision', '--dsbi-point-metadata', 'points.json']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base+flags)
        self.assertFalse(parse_args(base).dsbi_visible_partial_supervision)


if __name__ == '__main__':
    unittest.main()
