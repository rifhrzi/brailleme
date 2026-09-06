"""PRIVATE NumPy/Pillow proof: execute copied functions/classes, never Keras."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter

BASE=Path(__file__).resolve().parent
METRICS={'marker_centroid_errors_px':[]}
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
point=load_module('private_points',BASE/'source-original/word_point_geometry.py')
sampling=load_module('private_sampling',BASE/'source-original/word_sampling.py')
vertical=load_module('private_vertical',BASE/'source-original/word_vertical_stretch.py')
horizontal=load_module('private_horizontal',BASE/'source-private/word_horizontal_aspect.py')

def load_data(folder):
    names={'affine_patch','augment_patch','make_target','mirror_patch_horizontal','rotate_patch_180'}
    tree=ast.parse((folder/'word_data.py').read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {n.name for n in tree.body}==names
    ns=dict(np=np,Image=Image,ImageEnhance=ImageEnhance,ImageFilter=ImageFilter,WIDTH=480,HEIGHT=192,STRIDE=4,GRID_W=120,GRID_H=48)
    exec(compile(tree,str(folder/'word_data.py'),'exec'),ns)
    return ns

class DatasetStub:
    def __init__(self,**kwargs):pass

def load_patches(folder,data):
    tree=ast.parse((folder/'word_train.py').read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Patches')
    ns=dict(data,keras=types.SimpleNamespace(utils=types.SimpleNamespace(PyDataset=DatasetStub)),math=math,
        BalancedFamilyOrder=sampling.BalancedFamilyOrder,vertical_stretch_factors=vertical.vertical_stretch_factors,
        horizontal_aspect_factors=horizontal.horizontal_aspect_factors,visible_partial_candidates=point.visible_partial_candidates)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(folder/'word_train.py'),'exec'),ns)
    return ns['Patches'],ns

OLD=load_data(BASE/'source-original');NEW=load_data(BASE/'source-private')
OldPatches,OLD_PATCH=load_patches(BASE/'source-original',OLD)
NewPatches,NEW_PATCH=load_patches(BASE/'source-private',NEW)

class SpyRNG:
    def __init__(self,seed):self.generator=np.random.default_rng(seed);self.calls=[]
    def random(self,*args,**kwargs):
        value=self.generator.random(*args,**kwargs);self.calls.append(('random',args,kwargs,float(value)));return value
    def uniform(self,*args,**kwargs):
        value=self.generator.uniform(*args,**kwargs);self.calls.append(('uniform',args,kwargs,float(value)));return value

def source_image():
    yy,xx=np.mgrid[:192,:480]
    return Image.fromarray(np.stack(((xx*3+yy*5)%256,(xx*7+yy)%256,(xx+yy*3)%256),axis=-1).astype(np.uint8))

def point_state(mask=63):
    points=np.array([[[225.,65.],[225.,96.],[225.,125.],[255.,65.],[255.,96.],[255.,125.]]])
    return point.OccupiedPointState.from_source_crop(points,np.array([mask]),[0,0,480,192])

class AspectTests(unittest.TestCase):
    def test_stream_is_per_record_reproducible_order_independent_and_separate(self):
        rng=np.random.default_rng(91);before=copy.deepcopy(rng.bit_generator.state)
        global_before=copy.deepcopy(np.random.get_state())
        a=horizontal.horizontal_aspect_factors(20260905,3,[12,99,7])
        b=horizontal.horizontal_aspect_factors(20260905,3,[7,12,99,12])
        np.testing.assert_array_equal(b,a[[2,0,1,0]])
        self.assertEqual(before,rng.bit_generator.state)
        for x,y in zip(global_before,np.random.get_state()):np.testing.assert_array_equal(x,y)
        self.assertFalse(np.array_equal(a,horizontal.horizontal_aspect_factors(20260905,4,[12,99,7])))
        many=horizontal.horizontal_aspect_factors(20260905,3,np.arange(1000))
        self.assertTrue(np.all((many>=.55)&(many<1.25)))
        self.assertEqual(horizontal.horizontal_aspect_factors(1,1,[]).shape,(0,))
        for args in ((True,1,[2]),(1,-1,[2]),(1,1,[2.5]),(1,1,[-1]),(1,1,[[1]]),(1,1,[True])):
            with self.assertRaises(ValueError):horizontal.horizontal_aspect_factors(*args)

    def test_default_off_exact_pixels_boxes_eligibility_points_and_rng(self):
        box=np.array([[215.,55.,265.,135.]])
        for seed in range(18):
            a,b=SpyRNG(seed),SpyRNG(seed)
            old=OLD['augment_patch'](source_image(),box,a,illumination_augmentation=True,point_state=point_state())
            new=NEW['augment_patch'](source_image(),box,b,illumination_augmentation=True,point_state=point_state())
            for x,y in zip(old[:3],new[:3]):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(old[3].points,new[3].points)
            np.testing.assert_array_equal(old[3].survivors,new[3].survivors)
            self.assertEqual(a.calls,b.calls)

    def test_enabled_changes_only_x_of_existing_single_affine_and_preserves_all_rng(self):
        box=np.array([[215.,55.,265.,135.]])
        for seed in range(18):
            a,b=SpyRNG(seed),SpyRNG(seed)
            records=[[],[]]
            point_calls=[]
            old_affine,new_affine=OLD['affine_patch'],NEW['affine_patch']
            original_point_affine=point.OccupiedPointState.affine
            def track_points(state,sx,sy,tx=0.,ty=0.):
                point_calls.append([sx,sy,tx,ty]);return original_point_affine(state,sx,sy,tx,ty)
            def capture(index,fn):
                def wrapped(image,boxes,sx,sy,tx=0.,ty=0.):
                    records[index].append([sx,sy,tx,ty]);return fn(image,boxes,sx,sy,tx,ty)
                return wrapped
            with mock.patch.dict(OLD,affine_patch=capture(0,old_affine)),mock.patch.dict(NEW,affine_patch=capture(1,new_affine)),mock.patch.object(point.OccupiedPointState,'affine',track_points):
                OLD['augment_patch'](source_image(),box,a,illumination_augmentation=True)
                NEW['augment_patch'](source_image(),box,b,illumination_augmentation=True,horizontal_aspect=.55,point_state=point_state())
            self.assertEqual(a.calls,b.calls)
            self.assertEqual(point_calls,records[1])
            self.assertEqual(len(records[0]),len(records[1]));self.assertLessEqual(len(records[0]),1)
            if records[0]:
                self.assertEqual(records[1][0][0],records[0][0][0]*.55)
                self.assertEqual(records[1][0][1:],records[0][0][1:])

    def test_skipped_affine_ignores_factor_and_invalid_factor_rejected(self):
        seed=next(i for i in range(50) if np.random.default_rng(i).random()>=.85)
        args=(source_image(),np.array([[215.,55.,265.,135.]]))
        a=NEW['augment_patch'](*args,SpyRNG(seed),illumination_augmentation=True,horizontal_aspect=.55)
        b=NEW['augment_patch'](*args,SpyRNG(seed),illumination_augmentation=True,horizontal_aspect=1.25)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        for factor in (float('nan'),float('inf'),.54,1.26):
            with self.assertRaises(ValueError):NEW['augment_patch'](*args,SpyRNG(seed),horizontal_aspect=factor)

    def test_patches_off_exact_order_pixels_targets_all63_masks_and_unknown_slots(self):
        with tempfile.TemporaryDirectory(dir=BASE,prefix='synthetic-test-') as folder:
            root=Path(folder);records=[];states={}
            for mask in range(1,64):
                image=f'{mask}.png';source_image().save(root/image)
                records.append(dict(image=image,source=f'TRAIN/family{mask%3}/synthetic.png',boxes=[[215.,55.,265.,135.]],masks=[mask],ignore_boxes=[[40.,40.,65.,95.]]))
                states[image]=point_state(mask)
            opts=dict(root=root,records=records,batch_size=7,training=True,seed=42,illumination_augmentation=True,point_states=states)
            old=OldPatches(**opts);off=NewPatches(**opts);on=NewPatches(**opts,horizontal_aspect_augmentation=True)
            np.testing.assert_array_equal(old.order,off.order);np.testing.assert_array_equal(old.order,on.order)
            for batch in range(len(old)):
                x,y=old[batch]
                def forbid_factor_stream(*args,**kwargs):raise AssertionError('OFF must not draw width factors')
                with mock.patch.dict(NEW_PATCH,horizontal_aspect_factors=forbid_factor_stream):xo,yo=off[batch]
                xn,yn=on[batch]
                np.testing.assert_array_equal(x,xo);np.testing.assert_array_equal(y,yo)
                for slot,index in enumerate(on.order[batch*7:(batch+1)*7]):
                    positives=yn[slot][yn[slot,...,0]>=.999]
                    self.assertEqual(len(positives),1,'Unknown slots must never become positive')
                    mask=records[index]['masks'][0]
                    np.testing.assert_array_equal(positives[0,5:],[(mask>>bit)&1 for bit in range(6)])
            old.on_epoch_end();off.on_epoch_end();on.on_epoch_end()
            np.testing.assert_array_equal(old.order,off.order);np.testing.assert_array_equal(old.order,on.order)

    def test_actual_pillow_marker_centers_boxes_and_all63_occupied_masks(self):
        yy,xx=np.mgrid[:192,:480]
        state=point_state();box=np.array([[215.,55.,265.,135.]])
        for factor in (.55,1.25):
            sx,sy,tx,ty=1.05*factor,1.05,4.,3.
            transformed=state.affine(sx,sy,tx,ty)
            for slot,(x,y) in enumerate(state.points[0]):
                marker=np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/8)).astype(np.uint8)
                raster,new_box=NEW['affine_patch'](Image.fromarray(np.repeat(marker[...,None],3,axis=-1)),box,sx,sy,tx,ty)
                values=np.asarray(raster,dtype=np.float64)[...,0]
                centroid=np.array([(values*xx).sum(),(values*yy).sum()])/values.sum()
                np.testing.assert_allclose(centroid,transformed.points[0,slot],atol=.045,rtol=0)
                METRICS['marker_centroid_errors_px'].append(float(np.max(np.abs(centroid-transformed.points[0,slot]))))
                expected=box*np.array([sx,sy,sx,sy])+np.array([240*(1-sx)+tx,96*(1-sy)+ty]*2)
                np.testing.assert_allclose(new_box,expected,atol=1e-10,rtol=0)
            for mask in range(1,64):
                known=point_state(mask).affine(sx,sy,tx,ty).rotate(3.)
                np.testing.assert_array_equal(known.survivors[0],[(mask>>bit)&1 for bit in range(6)])

    def test_partial_survival_irreversible_unknown_bits_and_unoccupied_slots_excluded(self):
        pts=np.array([[[455.,75.],[455.,100.],[455.,125.],[485.,75.],[485.,100.],[485.,125.]]])
        box=np.array([[440.,65.,500.,135.]])
        known=point.OccupiedPointState.from_source_crop(pts,np.array([1]),[0,0,480,192]).affine(1.05,1.)
        _,b=NEW['affine_patch'](source_image(),box,1.05,1.)
        indices,fragments=point.visible_partial_candidates(b,known)
        np.testing.assert_array_equal(indices,[0])
        target=NEW['make_target'](b,np.array([1]),np.array([False]),visible_partial_boxes=fragments,ignore_boxes=np.array([[1.,1.,8.,30.]]))
        positives=target[target[...,0]>=.999];self.assertEqual(len(positives),1)
        np.testing.assert_array_equal(positives[0,5:],[-1]*6)
        empty=point.OccupiedPointState.from_source_crop(pts,np.array([8]),[0,0,480,192]).affine(.55,1.)
        self.assertFalse(empty.has_occupied_center[0],'A dot already outside the source crop must not return')
        lost=point.OccupiedPointState.from_source_crop(pts,np.array([1]),[0,0,480,192]).affine(1.25,1.)
        self.assertFalse(lost.has_occupied_center[0])
        recovered_positions=lost.affine(.55,1.).rotate(2.)
        self.assertTrue(point.safe_center_mask(recovered_positions.points,480,192)[0,0])
        self.assertFalse(recovered_positions.has_occupied_center[0])

if __name__=='__main__':unittest.main(verbosity=2)
