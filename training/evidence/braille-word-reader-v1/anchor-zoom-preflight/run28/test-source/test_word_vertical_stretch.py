"""Check isolated view draws and actual anisotropic Pillow/point geometry."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_data import affine_patch, augment_patch
from word_point_geometry import OccupiedPointState
from word_vertical_stretch import vertical_stretch_factors


class VerticalStretchTests(unittest.TestCase):
    def test_stream_reproducibility_prefixes_and_independence(self):
        main = np.random.default_rng(20260905)
        before = copy.deepcopy(main.bit_generator.state)
        values = vertical_stretch_factors(20260905, 2, 0, 24)
        np.testing.assert_array_equal(values, vertical_stretch_factors(20260905, 2, 0, 24))
        np.testing.assert_array_equal(values[:6], vertical_stretch_factors(20260905, 2, 0, 6))
        self.assertEqual(before, main.bit_generator.state)
        self.assertFalse(np.array_equal(values, vertical_stretch_factors(20260905, 3, 0, 24)))
        self.assertFalse(np.array_equal(values, vertical_stretch_factors(20260905, 2, 1, 24)))
        self.assertTrue(np.all((values >= .9) & (values <= 1.1)))

    def test_invalid_stream_identity_is_rejected(self):
        for args in ((-1, 2, 0, 1), (1, -2, 0, 1), (1, 2, .5, 1), (1, 2, 0, -1), (True, 2, 0, 1)):
            with self.assertRaises(ValueError):
                vertical_stretch_factors(*args)
        self.assertEqual(vertical_stretch_factors(1, 2, 0, 0).shape, (0,))

    def test_height_factor_does_not_change_main_rng_or_default_behavior(self):
        source = np.random.default_rng(18).integers(0, 256, (192, 480, 3), dtype=np.uint8)
        boxes = np.array([[100., 50., 130., 110.]])
        for seed in range(12):
            generators = [np.random.default_rng(seed) for _ in range(3)]
            outputs = [augment_patch(Image.fromarray(source), boxes, generators[0], illumination_augmentation=True),
                       augment_patch(Image.fromarray(source), boxes, generators[1], illumination_augmentation=True, vertical_stretch=1.),
                       augment_patch(Image.fromarray(source), boxes, generators[2], illumination_augmentation=True, vertical_stretch=1.1)]
            for implicit, explicit in zip(outputs[0], outputs[1]):
                np.testing.assert_array_equal(implicit, explicit)
            self.assertEqual(generators[0].bit_generator.state, generators[1].bit_generator.state)
            self.assertEqual(generators[0].bit_generator.state, generators[2].bit_generator.state)
        for factor in (np.nan, np.inf, .89, 1.11):
            with self.assertRaises(ValueError):
                augment_patch(Image.fromarray(source), boxes, np.random.default_rng(1), vertical_stretch=factor)

    def test_annotated_points_match_actual_pillow_vertical_stretch(self):
        # One isolated antialiased marker at each known source slot, far from
        # the canvas edges: compare actual raster centroids to tracked points.
        points = np.array([[[100., 50.], [100., 75.], [100., 100.],
                            [125., 50.], [125., 75.], [125., 100.]]])
        initial = OccupiedPointState.from_source_crop(points, np.array([63]), [0, 0, 480, 192])
        box = np.array([[90., 40., 135., 110.]])
        for factor in (.9, 1., 1.1):
            transformed_state = initial.affine(1.05, 1.05*factor, 4., 3.)
            for slot, (x, y) in enumerate(points[0]):
                yy, xx = np.mgrid[:192, :480]
                marker = np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/8)).astype(np.uint8)
                rgb = np.repeat(marker[..., None], 3, axis=-1)
                raster, transformed_box = affine_patch(Image.fromarray(rgb), box, 1.05, 1.05*factor, 4., 3.)
                values = np.asarray(raster, dtype=np.float64)[..., 0]
                # Background fill is rounded mean0 for this bounded marker.
                centroid = np.array([(values*xx).sum(), (values*yy).sum()])/values.sum()
                np.testing.assert_allclose(centroid, transformed_state.points[0, slot], atol=.035, rtol=0)
                expected_box = (box-np.array([240., 96.]*2))*np.array([1.05, 1.05*factor]*2)+np.array([244., 99.]*2)
                np.testing.assert_allclose(transformed_box, expected_box)
                self.assertTrue(transformed_state.survivors[0, slot])


if __name__ == '__main__':
    unittest.main()
