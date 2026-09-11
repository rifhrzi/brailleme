"""Independent checks of pixel geometry and detector/refiner cell attachment."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_cell_refiner_v4 import crop_cell,replace_dots,selected_grids
from word_decode_id_v4 import DEFAULT_THRESHOLDS


class GeometryReviewTests(unittest.TestCase):
    def test_linear_intensity_plane_has_exact_center_samples_and_replicated_borders(self):
        plane=np.tile(np.linspace(0,1,6,dtype=np.float32)[None,:,None],(8,1,3))
        crop=crop_cell(plane,[0,0,1,1])
        self.assertEqual(crop.shape,(64,40,1))
        self.assertAlmostEqual(float(crop[32,0,0]),0.,places=7)
        self.assertAlmostEqual(float(crop[32,-1,0]),1.,places=7)
        # The center-column half-pixel sample lands at source x=2.5975;
        # a linear plane spanning x=0..5 therefore has intensity .5195.
        self.assertAlmostEqual(float(crop[32,20,0]),.5195,places=6)
        np.testing.assert_allclose(crop[0],crop[-1],rtol=0,atol=1e-7)

    def test_rgb_channel_order_survives_the_crop(self):
        red=np.zeros((8,6,3),np.float32);red[...,0]=1
        np.testing.assert_allclose(crop_cell(red,[0,0,1,1]),.2126,rtol=0,atol=1e-7)

    def test_exact_grid_attachment_survives_center_cancellation_boundary(self):
        output=np.zeros((48,120,11),np.float32)
        output[20,10]=[.9,0.,.5,.1425454169511795,.25,.9,.1,.1,.1,.1,.1]
        thresholds=dict(DEFAULT_THRESHOLDS,dotConfidence=.7)
        attachments=selected_grids(output,thresholds)
        self.assertEqual(len(attachments),1)
        grid,box=attachments[0]
        self.assertEqual(grid,(20,10))
        self.assertEqual(int(((box[0]+box[2])/2)*120),9)

    def test_dot_replacement_preserves_geometry_objectness_and_every_other_grid(self):
        rng=np.random.default_rng(7)
        original=rng.random((2,48,120,11),dtype=np.float32)
        before=original.copy()
        attachments=[(0,20,10),(1,21,11)]
        probabilities=np.full((2,6),.75,np.float32)
        updated=replace_dots(original,attachments,probabilities,.5)
        np.testing.assert_array_equal(original,before)
        np.testing.assert_array_equal(updated[...,:5],before[...,:5])
        expected=before.copy()
        for index,gy,gx in attachments:
            expected[index,gy,gx,5:]=.5*before[index,gy,gx,5:]+.375
        np.testing.assert_array_equal(updated,expected)
        np.testing.assert_array_equal(replace_dots(original,attachments,probabilities,0.),original)


if __name__=='__main__':
    unittest.main()
