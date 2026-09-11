"""Physical flip labels and loss geometry, without model inference."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_overall_v5_train import FLIPS, contained, flip_example
from word_data import make_target


class PhysicalFlipTests(unittest.TestCase):
    def test_every_mask_and_image_box_is_an_involution(self):
        pixels=np.arange(192*480*3,dtype=np.uint8).reshape(192,480,3)
        boxes=np.asarray([[0,1,20,60],[430,140,480,192],[-9,45,12,105]],float)
        masks=np.arange(64)
        for name in FLIPS:
            moved=flip_example(pixels,boxes,masks,name)
            restored=flip_example(*moved,name)
            for before,after in zip((pixels,boxes,masks),restored):
                np.testing.assert_array_equal(before,after)
            np.testing.assert_array_equal(contained(boxes),contained(moved[1]))
            self.assertTrue(moved[0].flags.c_contiguous)

    def test_capital_indicator_maps_to_actual_reflected_corner(self):
        pixels=np.zeros((192,480,3),np.uint8)
        # Dot6 is bottom-right. Horizontal=>dot3; vertical=>dot4; both=>dot1.
        for name,expected in [('identity',32),('horizontal',4),('vertical',8),('half_turn',1)]:
            _,_,masks=flip_example(pixels,np.empty((0,4)),np.asarray([32]),name)
            self.assertEqual(int(masks[0]),expected)

    def test_loss_grid_rebuilt_from_continuous_transformed_boxes(self):
        pixels=np.zeros((192,480,3),np.uint8)
        # Boundary center x=40 aligns exactly with output grid edge. Rebuilding
        # from geometry handles its mirror at x=440 without a one-grid shift.
        boxes=np.asarray([[30,40,50,80]],float)
        _,moved,masks=flip_example(pixels,boxes,np.asarray([1]),'horizontal')
        truth=make_target(moved,masks,np.asarray([True]))
        positions=np.argwhere(truth[...,0]==1)
        np.testing.assert_array_equal(positions,np.asarray([[15,110]]))
        np.testing.assert_array_equal(truth[15,110,5:],np.asarray([0,0,0,1,0,0]))
        self.assertEqual(float(truth[15,110,1]),0.)

    def test_empty_negatives_and_ignored_partial_boxes_remain_valid(self):
        pixels=np.zeros((192,480,3),np.uint8)
        for name in FLIPS:
            _,boxes,masks=flip_example(pixels,np.empty((0,4)),np.asarray([],int),name)
            self.assertEqual(boxes.shape,(0,4));self.assertEqual(masks.shape,(0,))
            self.assertFalse(make_target(boxes,masks,np.asarray([],bool))[...,0].any())
        with self.assertRaises(ValueError):
            flip_example(pixels,np.empty((0,4)),np.asarray([64]),'identity')


if __name__=='__main__':unittest.main()
