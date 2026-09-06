"""Occupied-point survival and actual Pillow pixel-center transform oracles."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_point_geometry import OccupiedPointState, safe_center_mask, visible_partial_candidates


def state_at(points, masks=None, width=480, height=192):
    points = np.asarray(points, np.float64).reshape(-1, 6, 2)
    return OccupiedPointState.from_source_crop(points, masks if masks is not None else [63]*len(points),
                                               [0, 0, width, height], width=width, height=height)


class PointGeometryTests(unittest.TestCase):
    def test_resize_tracks_pixel_centers_and_checks_both_crop_and_resize(self):
        points = [[[10, 20], [12, 22], [19, 29], [20, 25], [9, 25], [15, 30]]]
        state = OccupiedPointState.from_source_crop(points, [63], [10, 20, 20, 30], width=5, height=5)
        np.testing.assert_array_equal(state.points[0], [[-.25, -.25], [.75, .75], [4.25, 4.25], [4.75, 2.25], [-.75, 2.25], [2.25, 4.75]])
        self.assertEqual(state.survivors.tolist(), [[False, True, False, False, False, False]])

    def test_preserves_actual_middle_line_and_all_known_masks(self):
        points = [[[10, 15], [10, 36], [10, 60], [30, 15], [30, 36], [30, 60]]]
        for mask in range(1, 64):
            with self.subTest(mask=mask):
                state = state_at(points, [mask])
                np.testing.assert_array_equal(state.points, points)
                self.assertEqual(state.survivors[0].tolist(), [bool(mask & (1 << bit)) for bit in range(6)])
        self.assertNotEqual(state.points[0, 1, 1], (state.points[0, 0, 1]+state.points[0, 2, 1])/2)

    def test_safe_hull_rejects_last_half_pixel_area_and_nonfinite_points(self):
        points = np.asarray([[0, 0], [479, 191], [479.1, 191], [480, 191], [-.1, 4], [5, 191.1]])
        self.assertEqual(safe_center_mask(points, 480, 192).tolist(), [True, True, False, False, False, False])
        for invalid in ([[np.nan, 1]], [[1, np.inf]], [1, 2, 3]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                safe_center_mask(invalid, 480, 192)

    def test_mirror_and_half_turn_remap_coordinates_and_slot_identity(self):
        points = [[[10, 20], [10, 32], [10, 60], [30, 20], [30, 32], [30, 60]]]
        original = state_at(points, [1])
        mirrored = original.mirror_horizontal()
        self.assertEqual(mirrored.survivors.tolist(), [[False, False, False, True, False, False]])
        np.testing.assert_array_equal(mirrored.points[0, 3], [469, 20])
        rotated = original.half_turn()
        self.assertEqual(rotated.survivors.tolist(), [[False, False, False, False, False, True]])
        np.testing.assert_array_equal(rotated.points[0, 5], [469, 171])
        np.testing.assert_array_equal(mirrored.mirror_horizontal().points, original.points)
        np.testing.assert_array_equal(rotated.half_turn().points, original.points)

    def test_affine_then_return_cannot_restore_an_occupied_point(self):
        state = state_at([[[479, 96], [20, 20], [50, 50], [100, 100], [120, 120], [140, 140]]], [3])
        first = state.affine(1, 1, 3, 0)
        returned = first.affine(1, 1, -3, 0)
        np.testing.assert_array_equal(returned.points, state.points)
        self.assertEqual(returned.survivors.tolist(), [[False, True, False, False, False, False]])
        self.assertEqual(state.survivors.tolist(), [[True, True, False, False, False, False]])

    def test_source_crop_loss_and_rotation_loss_never_recover(self):
        source_cut = state_at([[[-2, 20], [20, 20], [50, 50], [100, 100], [120, 120], [140, 140]]], [1])
        self.assertFalse(source_cut.affine(1, 1, 10, 0).has_occupied_center[0])
        corner = state_at([[[0, 0]]*6], [1])
        rotated = corner.rotate(4)
        self.assertFalse(rotated.has_occupied_center[0])
        self.assertFalse(rotated.rotate(-4).has_occupied_center[0])

    def test_cardinal_rotations_match_actual_pillow_corner_pixels_without_epsilon_loss(self):
        for width, height in ((480, 192), (64, 64)):
            for x, y in ((0, 0), (width-1, 0), (0, height-1), (width-1, height-1)):
                pixels = np.zeros((height, width), np.uint8)
                pixels[y, x] = 255
                image = Image.fromarray(pixels)
                state = state_at([[[x, y]]*6], [1], width=width, height=height)
                for angle in (0, 90, 180, 270, 360, 720, -180, -360):
                    with self.subTest(size=(width, height), point=(x, y), angle=angle):
                        observed = np.argwhere(np.asarray(image.rotate(angle, Image.Resampling.BILINEAR)) > 0)
                        transformed = state.rotate(angle)
                        self.assertEqual(bool(transformed.has_occupied_center[0]), bool(len(observed)))
                        if len(observed):
                            self.assertEqual(len(observed), 1)
                            np.testing.assert_array_equal(transformed.points[0, 0], observed[0, ::-1])
                            self.assertEqual(transformed.survivors.tolist(), [[True, False, False, False, False, False]])

    def test_states_copy_metadata_and_are_read_only(self):
        source = np.asarray([[[20, 20]]*6], float)
        state = state_at(source, [1])
        source[:] = -100
        self.assertTrue(state.has_occupied_center[0])
        self.assertEqual(state.points[0, 0, 0], 20)
        with self.assertRaises(ValueError):
            state.points[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            state.survivors[0, 0] = False

    def test_visible_partial_candidates_require_known_survivor_size_and_crossing(self):
        points = [[[2, 46]]*6, [[2, 46]]*6, [[12, 46]]*6, [[478, 46]]*6,
                  [[2, 46]]*6, [[-2, 46]]*6, [[2, 46]]*6]
        state = state_at(points, [1]*7)
        boxes = np.asarray([[-4, 40, 4, 52], [-4, 40, 3.99, 52], [10, 40, 14, 52],
                            [476, 40, 486, 52], [-10, 40, 490, 52], [-4, 40, 4, 52], [-4, 40, 4, 51.99]])
        before = boxes.copy()
        indices, visible = visible_partial_candidates(boxes, state)
        self.assertEqual(indices.tolist(), [0, 3])
        np.testing.assert_array_equal(visible, [[0, 40, 4, 52], [476, 40, 480, 52]])
        np.testing.assert_array_equal(boxes, before)
        # A previously cut box moved fully inside has no candidate, regardless
        # of its retained point visibility. It cannot become a full-cell target.
        self.assertNotIn(2, indices)

    def test_unknown_zero_masks_and_misaligned_or_invalid_metadata_fail(self):
        points = [[[20, 20]]*6]
        for masks in ([0], [64], [-1], [1.0], [True], [1, 2]):
            with self.subTest(masks=masks), self.assertRaises(ValueError):
                state_at(points, masks)
        for crop in ([0, 0, 0, 10], [-1, 0, 10, 10], [0., 0, 10, 10]):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                OccupiedPointState.from_source_crop(points, [1], crop)
        state = state_at(points, [1])
        with self.assertRaises(ValueError):
            visible_partial_candidates([[0, 0, 2, 2], [0, 0, 2, 2]], state)
        with self.assertRaises(ValueError):
            visible_partial_candidates([[0, 0, np.nan, 2]], state)
        for values in ((0, 1), (1, -1), (np.nan, 1), (1, 1, np.inf, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                state.affine(*values)
        with self.assertRaises(ValueError):
            state.rotate(np.inf)

    def test_empty_known_cell_list_does_not_synthesize_unknown_slots(self):
        state = state_at(np.empty((0, 6, 2)), [])
        self.assertEqual(state.has_occupied_center.tolist(), [])
        indices, boxes = visible_partial_candidates(np.empty((0, 4)), state)
        self.assertEqual(indices.tolist(), [])
        self.assertEqual(boxes.shape, (0, 4))

    def test_actual_pillow_rgb_resize_affine_rotate_and_orientation(self):
        # Three independent RGB Gaussian markers provide a measured raster
        # oracle. Centroids have .1px tolerance for interpolation/quantization.
        from word_data import affine_patch
        centers = np.asarray([[101., 77.], [327., 131.], [552., 207.]])
        yy, xx = np.indices((261, 655))
        pixels = np.stack([np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/32)) for x, y in centers], axis=-1).astype(np.uint8)
        original = Image.fromarray(pixels)
        state = OccupiedPointState.from_source_crop(np.repeat(centers[:, None, :], 6, axis=1), [1]*3, [20, 10, 620, 250])
        resized = original.crop([20, 10, 620, 250]).resize((480, 192), Image.Resampling.BILINEAR)

        def compare(image, current, slot=0):
            array = np.asarray(image, np.float64)
            y, x = np.indices(array.shape[:2])
            observed = np.asarray([[(array[..., c]*x).sum()/array[..., c].sum(),
                                    (array[..., c]*y).sum()/array[..., c].sum()] for c in range(3)])
            np.testing.assert_allclose(current.points[:, slot], observed, rtol=0, atol=.1)

        compare(resized, state)
        compare(resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT), state.mirror_horizontal(), slot=3)
        compare(resized.transpose(Image.Transpose.ROTATE_180), state.half_turn(), slot=5)
        warped, _ = affine_patch(resized, np.empty((0, 4)), .93, 1.13, 17.2, -8.3)
        affine_state = state.affine(.93, 1.13, 17.2, -8.3)
        compare(warped, affine_state)
        rotated = warped.rotate(3.7, Image.Resampling.BILINEAR, fillcolor=tuple(np.asarray(warped).mean(axis=(0, 1)).astype(int)))
        compare(rotated, affine_state.rotate(3.7))


if __name__ == '__main__':
    unittest.main()
