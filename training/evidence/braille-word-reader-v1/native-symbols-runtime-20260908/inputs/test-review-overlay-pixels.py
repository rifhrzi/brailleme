import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('overlay', Path(__file__).with_name('review-overlay-pixels.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class OverlayTests(unittest.TestCase):
    def test_rectangles_inside_preview_only(self):
        image = np.full((100, 160, 3), 180, dtype=np.uint8)
        for left, top, right, bottom in [(20, 30, 50, 80), (70, 35, 90, 75), (130, 10, 155, 50)]:
            image[top:bottom, left:left+2] = [8, 116, 67]
            image[top:bottom, right-2:right] = [8, 116, 67]
            image[top:top+2, left:right] = [8, 116, 67]
            image[bottom-2:bottom, left:right] = [8, 116, 67]
        _, rows = mod.green_rectangles(image, [0, 0, 120, 100])
        self.assertEqual([row['box_xyxy'] for row in rows], [[20, 30, 50, 80], [70, 35, 90, 75]])
        self.assertTrue(all(row['rectangle_candidate'] for row in rows))

    def test_open_line_not_rectangle(self):
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        for i in range(20, 70):
            image[i, i:i+2] = [8, 116, 67]
        _, rows = mod.green_rectangles(image, [0, 0, 100, 100])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['rectangle_candidate'])

    def test_projection_and_assignment(self):
        self.assertEqual(mod.project_box([1, 2, 4, 7], np.diag([2, 3]), np.array([10, 20])), [12, 26, 18, 41])
        rows = [{'rectangle_candidate': True, 'box_xyxy': [30, 30, 50, 70]},
                {'rectangle_candidate': True, 'box_xyxy': [1, 1, 10, 20]}]
        result = mod.compare_boxes(rows, [[1, 1, 10, 20], [30, 30, 50, 70], [70, 0, 90, 20]])
        self.assertEqual([r['annotation_cell'] for r in result], [1, 0])
        self.assertTrue(all(r['iou'] == 1 for r in result))

    def test_flat_feed_not_identifiable(self):
        gray = np.full((100, 100), .5)
        result = mod.fit_pixels(gray, gray, np.zeros((100, 100), bool), [0, 0, 100, 100], (np.eye(2), np.zeros(2)))
        self.assertEqual(result['status'], 'insufficient_texture')

    def test_known_geometry_and_brightness(self):
        yy, xx = np.mgrid[:120, :160]
        feed = .45 + .1*np.sin(xx*.11) + .06*np.cos(yy*.15) + .05*np.sin((xx+yy)*.05)
        screen = feed*.92+.015
        result = mod.fit_pixels(feed, screen, np.zeros(feed.shape, bool), [10, 10, 150, 110], (np.eye(2), np.zeros(2)), 2)
        self.assertEqual(result['status'], 'usable_pixel_alignment')
        np.testing.assert_allclose(result['matrix'], np.eye(2), atol=.002)
        np.testing.assert_allclose(result['translation'], [0, 0], atol=.05)


if __name__ == '__main__':
    unittest.main()
