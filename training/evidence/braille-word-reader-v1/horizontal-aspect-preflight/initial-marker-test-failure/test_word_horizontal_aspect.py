"""Check independent aspect draws and faithful single-raster point geometry."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import word_data
from word_horizontal_aspect import horizontal_aspect_factors
from word_point_geometry import OccupiedPointState


class AspectTests(unittest.TestCase):
    def test_record_stream_is_order_independent_and_leaves_other_rng_untouched(self):
        rng = np.random.default_rng(51); before = deepcopy(rng.bit_generator.state)
        state = deepcopy(np.random.get_state())
        first = horizontal_aspect_factors(20260905, 3, [12, 99, 7])
        second = horizontal_aspect_factors(20260905, 3, [7, 12, 99, 12])
        np.testing.assert_array_equal(second, first[[2, 0, 1, 0]])
        self.assertFalse(np.array_equal(first, horizontal_aspect_factors(20260905, 4, [12, 99, 7])))
        self.assertEqual(before, rng.bit_generator.state)
        for old, new in zip(state, np.random.get_state()):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(horizontal_aspect_factors(1, 1, []).shape, (0,))
        for args in ((True, 1, [2]), (1, -1, [2]), (1, 1, [2.5]), (1, 1, [-1]), (1, 1, [[1]]), (1, 1, [True])):
            with self.assertRaises(ValueError):
                horizontal_aspect_factors(*args)

    def test_one_affine_and_all_legacy_random_draws_remain_paired(self):
        yy, xx = np.mgrid[:192, :480]
        image = Image.fromarray(np.repeat(((xx+3*yy)%256).astype(np.uint8)[..., None], 3, axis=2))
        boxes = np.array([[215., 55., 265., 135.]])
        original = word_data.affine_patch
        for seed in range(18):
            captured = [[], []]
            def wrapper(index):
                def call(image, boxes, sx, sy, tx=0., ty=0.):
                    captured[index].append([sx, sy, tx, ty])
                    return original(image, boxes, sx, sy, tx, ty)
                return call
            a, b = np.random.default_rng(seed), np.random.default_rng(seed)
            with mock.patch.object(word_data, 'affine_patch', wrapper(0)):
                word_data.augment_patch(image, boxes, a, illumination_augmentation=True)
            with mock.patch.object(word_data, 'affine_patch', wrapper(1)):
                word_data.augment_patch(image, boxes, b, illumination_augmentation=True, horizontal_aspect=.55)
            self.assertEqual(a.bit_generator.state, b.bit_generator.state)
            self.assertEqual(len(captured[0]), len(captured[1])); self.assertLessEqual(len(captured[1]), 1)
            if captured[0]:
                self.assertEqual(captured[1][0][0], captured[0][0][0]*.55)
                self.assertEqual(captured[1][0][1:], captured[0][0][1:])

    def test_actual_pillow_centroid_matches_tracked_pixel_center(self):
        yy, xx = np.mgrid[:192, :480]
        for factor in (.55, 1.25):
            for px, py in ((60., 60.), (240., 96.), (410., 130.)):
                marker = np.exp(-((xx-px)**2+(yy-py)**2)/32.)*255
                image = Image.fromarray(np.repeat(marker.astype(np.uint8)[..., None], 3, axis=2))
                pts = np.tile([px, py], (1, 6, 1))
                state = OccupiedPointState.from_source_crop(pts, np.array([63]), [0, 0, 480, 192])
                sx, sy, tx, ty = factor, 1., 3.25, -2.5
                raster, _ = word_data.affine_patch(image, np.array([[px-10, py-10, px+10, py+10]]), sx, sy, tx, ty)
                moved = state.affine(sx, sy, tx, ty)
                weights = np.asarray(raster)[..., 0].astype(float)
                # Sample away from the fill borders, retaining the marker.
                weights[:8] = weights[-8:] = 0; weights[:, :8] = weights[:, -8:] = 0
                actual = np.array([(weights*xx).sum(), (weights*yy).sum()])/weights.sum()
                np.testing.assert_allclose(actual, moved.points[0, 0], atol=.04, rtol=0)

    def test_skip_and_profile_guards_prevent_extra_or_combined_rasters(self):
        image = Image.fromarray(np.full((192, 480, 3), 127, np.uint8))
        boxes = np.array([[200., 50., 260., 130.]])
        seed = next(i for i in range(50) if np.random.default_rng(i).random() >= .85)
        a = word_data.augment_patch(image, boxes, np.random.default_rng(seed), illumination_augmentation=True, horizontal_aspect=.55)
        b = word_data.augment_patch(image, boxes, np.random.default_rng(seed), illumination_augmentation=True, horizontal_aspect=1.25)
        for old, new in zip(a, b):
            np.testing.assert_array_equal(old, new)
        for profile in ({}, {'camera_augmentation': True}, {'polarity_augmentation': True},
                        {'illumination_augmentation': True, 'vertical_stretch': 1.1}):
            with self.assertRaises(ValueError):
                word_data.augment_patch(image, boxes, np.random.default_rng(1), horizontal_aspect=.55, **profile)


if __name__ == '__main__':
    unittest.main()
