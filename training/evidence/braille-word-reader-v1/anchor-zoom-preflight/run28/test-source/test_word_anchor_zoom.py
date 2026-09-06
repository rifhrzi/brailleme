"""Protect original full-cell supervision without relabeling cut/unknown cells."""
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('KERAS_BACKEND','torch')
os.environ.setdefault('KERAS_TORCH_DEVICE','cpu')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
from word_anchor_zoom import anchor_preserving_zoom
from word_data import affine_patch,augment_patch
from word_train import Patches,parse_args


class AnchorZoomTests(unittest.TestCase):
    def test_four_edges_translations_and_containment(self):
        for box in ([20,2,80,190],[400,2,470,190],[1,50,479,130],[210,10,280,180]):
            for tx,ty in ((-19.2,-7.68),(-19.2,7.68),(19.2,-7.68),(19.2,7.68)):
                z=anchor_preserving_zoom([box],[True],1.7,tx,ty)
                self.assertGreaterEqual(z,.8);self.assertLessEqual(z,1.7)
                _,b=affine_patch(Image.new('RGB',(480,192)),np.asarray([box],float),z,z,tx,ty)
                self.assertTrue(np.all(b[:,:2]>=0) and np.all(b[:,2:]<=[480,192]))

    def test_exact_bound_uses_actual_affine_containment(self):
        boxes=np.asarray([[50,70,280,125]],np.float32);zoom=1.2847156687768433;tx=4.0959770676002165
        fixed=anchor_preserving_zoom(boxes,[True],zoom,tx,0.)
        self.assertLess(fixed,zoom)
        _,b=affine_patch(Image.new('RGB',(480,192)),boxes,fixed,fixed,tx,0.)
        self.assertGreaterEqual(b[0,0],0.)

    def test_cut_unknown_and_empty_boxes_cannot_anchor(self):
        for boxes,eligible in (([],[]),([[0,0,480,192]],[False]),([[-10,0,40,190]],[True])):
            self.assertEqual(anchor_preserving_zoom(np.asarray(boxes).reshape(-1,4),eligible,1.7,0,0),1.7)
        self.assertEqual(anchor_preserving_zoom([[210,40,250,130]],[True],.8,0,0),.8)

    def test_original_index_breaks_distance_ties_without_reading_masks(self):
        a=[[210,30,230,170],[250,70,270,130]]
        self.assertLess(anchor_preserving_zoom(a,[True,True],1.7,0,0),anchor_preserving_zoom(a[::-1],[True,True],1.7,0,0))

    def test_same_draws_one_affine_and_irreversible_eligibility(self):
        import word_data
        image=Image.new('RGB',(480,192),(127,127,127));boxes=np.asarray([[190.,15.,270.,185.],[-15.,30.,20.,150.]])
        actual=word_data.affine_patch
        for seed in range(20):
            traces=[[],[]]
            def capture(i):
                def fn(image,boxes,sx,sy,tx=0.,ty=0.):
                    traces[i].append((sx,sy,tx,ty));return actual(image,boxes,sx,sy,tx,ty)
                return fn
            a,b=np.random.default_rng(seed),np.random.default_rng(seed)
            with mock.patch.object(word_data,'affine_patch',capture(0)):
                old=augment_patch(image,boxes,a,illumination_augmentation=True,originally_complete=[True,False])
            with mock.patch.object(word_data,'affine_patch',capture(1)):
                new=augment_patch(image,boxes,b,illumination_augmentation=True,originally_complete=[True,False],anchor_zoom_cap=True)
            self.assertEqual(a.bit_generator.state,b.bit_generator.state)
            self.assertEqual(len(traces[0]),len(traces[1]));self.assertLessEqual(len(traces[1]),1)
            self.assertFalse(new[2][1])
            if traces[0]:
                self.assertEqual(traces[0][0][2:],traces[1][0][2:]);self.assertLessEqual(traces[1][0][0],traces[0][0][0])
            else:
                for x,y in zip(old,new):np.testing.assert_array_equal(x,y)

    def test_cli_defaults_off_and_rejects_mixed_ablation(self):
        base=['--data','unused','--output','unused']
        self.assertFalse(parse_args(base).anchor_zoom_cap)
        on=base+['--illumination-augmentation','--anchor-zoom-cap']
        self.assertTrue(parse_args(on).anchor_zoom_cap)
        for extra in (['--camera-augmentation'],['--half-turn-augmentation'],['--polarity-augmentation'],['--horizontal-mirror-augmentation'],['--vertical-stretch-augmentation'],['--horizontal-aspect-augmentation'],['--partial-geometry-supervision'],['--image-normalization','raw']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(on+extra)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(base+['--anchor-zoom-cap'])

    def test_direct_patches_rejects_nonisolated_profile(self):
        for kwargs in ({},{'illumination_augmentation':True,'horizontal_aspect_augmentation':True},{'illumination_augmentation':True,'horizontal_mirror_augmentation':True}):
            with self.assertRaises(ValueError):Patches(Path('.'),[],24,True,1,anchor_zoom_cap=True,**kwargs)

    def test_direct_raster_rejects_nonisotropic_or_missing_illumination(self):
        for kwargs in ({},{'illumination_augmentation':True,'horizontal_aspect':.8},{'illumination_augmentation':True,'vertical_stretch':1.1}):
            with self.assertRaises(ValueError):augment_patch(Image.new('RGB',(480,192)),np.empty((0,4)),np.random.default_rng(1),anchor_zoom_cap=True,**kwargs)


if __name__=='__main__':unittest.main()
