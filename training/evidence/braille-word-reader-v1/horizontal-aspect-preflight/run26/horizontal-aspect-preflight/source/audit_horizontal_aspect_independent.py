"""Bounded private-prototype review. NumPy/Pillow only; no model or real corpus."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

WORK = Path(__file__).resolve().parent
BASE = WORK / 'horizontal-aspect-prototype-v1'
OUT = WORK / 'horizontal-aspect-independent-review-v1'
OUT.mkdir(exist_ok=True)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('frozen_aspect_tests', BASE/'test_private_aspect.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
checks = {}

class IndependentTests(unittest.TestCase):
    def test_snapshot_and_unrelated_ast_identity(self):
        self.assertEqual(sha(BASE/'proof-report.json'), '7df89ed1f3be63849d893d3119a5f1105183e8141da9b4cbf0de6b3d7a41d4a6')
        self.assertEqual(sha(BASE/'index.json'), 'c53b382af5777c38ef46f7ee34ea44226cbe5e9fa8bca799fde48505b6f039b3')
        index = json.loads((BASE/'index.json').read_text())
        for name, value in index['files'].items():
            self.assertEqual(sha(BASE/name), value['sha256'], name)
        originals = sorted((BASE/'source-original').glob('*.py'))
        self.assertEqual(len(originals), 21)
        for path in originals:
            self.assertEqual(path.read_bytes(), (WORK/'run25/source'/path.name).read_bytes())
            self.assertEqual(path.read_bytes(), (WORK.parent.parent/'training/scripts'/path.name).read_bytes())
        for name, changed in [('word_data.py', 'augment_patch'), ('word_train.py', 'Patches')]:
            def remaining(folder):
                tree = ast.parse((BASE/folder/name).read_text())
                tree.body = [node for node in tree.body if getattr(node, 'name', None) != changed
                             and not (isinstance(node, ast.ImportFrom) and node.module == 'word_horizontal_aspect')]
                return ast.dump(tree, include_attributes=False)
            self.assertEqual(remaining('source-original'), remaining('source-private'))
        checks['source_identity'] = {'originals_run25_and_active_exact': 21, 'indexed_payloads_exact': len(index['files']), 'unrelated_ast_exact': True}

    def test_point_and_box_affine_rotation_independent_arithmetic_all_masks(self):
        # Deliberately include originally lost right-column dots and an ignore box.
        points = np.array([[[225.,65.],[225.,96.],[225.,125.],[255.,65.],[255.,96.],[255.,125.]],
                           [[455.,75.],[455.,100.],[455.,125.],[485.,75.],[485.,100.],[485.,125.]]])
        boxes = np.array([[215.,55.,265.,135.],[440.,65.,500.,135.],[25.,50.,65.,120.]])
        errors = []
        for mask in range(1,64):
            for factor in (.55, 1.25):
                state = module.point.OccupiedPointState.from_source_crop(points, np.array([mask,mask]), [0,0,480,192])
                expected_points = points.copy()
                expected_live = (((np.array([mask,mask])[:,None] >> np.arange(6)) & 1).astype(bool)
                                 & (points[...,0] >= 0) & (points[...,0] <= 479)
                                 & (points[...,1] >= 0) & (points[...,1] <= 191))
                expected_boxes = boxes.copy()
                expected_eligible = np.array([True, False, False])
                real_affine = module.NEW['affine_patch']
                operations = []
                def affine(image, current_boxes, sx, sy, tx=0., ty=0.):
                    operations.append(('affine', sx, sy, tx, ty))
                    return real_affine(image, current_boxes, sx, sy, tx, ty)
                actual_rotate = Image.Image.rotate
                def rotate(image, angle, *args, **kwargs):
                    operations.append(('rotate', angle))
                    return actual_rotate(image, angle, *args, **kwargs)
                rng = module.SpyRNG(mask)
                with mock.patch.dict(module.NEW, affine_patch=affine), mock.patch.object(Image.Image, 'rotate', rotate):
                    result = module.NEW['augment_patch'](module.source_image(), boxes, rng, illumination_augmentation=True,
                        originally_complete=expected_eligible.copy(), point_state=state, horizontal_aspect=factor)
                for op in operations:
                    if op[0] == 'affine':
                        _, sx, sy, tx, ty = op
                        offset = np.array([240*(1-sx)+tx,96*(1-sy)+ty])
                        expected_points = (expected_points+.5)*[sx,sy]+offset-.5
                        expected_boxes = expected_boxes*[sx,sy,sx,sy]+np.tile(offset,2)
                    else:
                        angle = math.radians(-op[1])
                        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                        expected_points = (expected_points+.5-[240,96])@rotation.T+[240,96]-.5
                        corners = np.stack([expected_boxes[:,[0,1]],expected_boxes[:,[2,1]],expected_boxes[:,[2,3]],expected_boxes[:,[0,3]]],axis=1)
                        corners = (corners-[240,96])@rotation.T+[240,96]
                        expected_boxes = np.concatenate([corners.min(axis=1),corners.max(axis=1)],axis=1)
                    expected_live &= ((expected_points[...,0]>=0)&(expected_points[...,0]<=479)
                                      &(expected_points[...,1]>=0)&(expected_points[...,1]<=191))
                    expected_eligible &= ((expected_boxes[:,0]>=0)&(expected_boxes[:,1]>=0)
                                          &(expected_boxes[:,2]<=480)&(expected_boxes[:,3]<=192))
                np.testing.assert_allclose(result[1],expected_boxes,atol=1e-10,rtol=0)
                np.testing.assert_allclose(result[3].points,expected_points,atol=1e-10,rtol=0)
                np.testing.assert_array_equal(result[3].survivors,expected_live)
                np.testing.assert_array_equal(result[2],expected_eligible)
                self.assertFalse(result[2][1], 'A source-partial cell cannot recover a full label')
                errors.append(float(np.max(np.abs(result[3].points-expected_points))))
                self.assertLessEqual(sum(op[0]=='affine' for op in operations),1)
        checks['independent_geometry'] = {'cases':len(errors),'masks':63,'endpoint_factors':[.55,1.25],
                                        'max_point_error_px':max(errors),'all_stage_survival_and':True}

    def test_balanced_patches_negative_partial_unknown_and_disabled_validation(self):
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            root = Path(folder)
            records=[]; states={}
            for index,mask in enumerate([1,8,63,21]):
                filename=f'known-{index}.png'; module.source_image().save(root/filename)
                records.append(dict(image=filename,source=f'TRAIN/family{index//3}/page.png',
                    boxes=[[215.,55.,265.,135.],[440.,65.,500.,135.]],masks=[mask,mask],
                    originally_complete=[True,False],ignore_boxes=[[25.,50.,65.,120.]]))
                pts=np.array([[[225.,65.],[225.,96.],[225.,125.],[255.,65.],[255.,96.],[255.,125.]],
                              [[455.,75.],[455.,100.],[455.,125.],[485.,75.],[485.,100.],[485.,125.]]])
                states[filename]=module.point.OccupiedPointState.from_source_crop(pts,np.array([mask,mask]),[0,0,480,192])
            Image.new('RGB',(480,192),(220,220,220)).save(root/'negative.png')
            records.extend([dict(image='negative.png',source='negative/public',boxes=[],masks=[]) for _ in range(3)])
            kwargs=dict(root=root,records=records,batch_size=3,training=True,seed=7,
                        illumination_augmentation=True,balance_source_families=True,point_states=states)
            old=module.OldPatches(**kwargs); off=module.NewPatches(**kwargs)
            on=module.NewPatches(**kwargs,horizontal_aspect_augmentation=True)
            batches=0; partial_centers=0
            for epoch in range(3):
                np.testing.assert_array_equal(old.order,off.order);np.testing.assert_array_equal(old.order,on.order)
                self.assertEqual(sorted(i for i in on.order if i>=4),[4,5,6])
                for batch in range(len(old)):
                    a=old[batch];b=off[batch]
                    for av,bv in zip(a,b):np.testing.assert_array_equal(av,bv)
                    seen=[];real_target=module.NEW_PATCH['make_target']
                    def capture(boxes,masks,eligible,*args,**kwargs):
                        result=real_target(boxes,masks,eligible,*args,**kwargs)
                        seen.append((boxes.copy(),masks.copy(),eligible.copy(),kwargs,result))
                        return result
                    with mock.patch.dict(module.NEW_PATCH,make_target=capture):on[batch]
                    for boxes,masks,eligible,extra,target in seen:
                        if not len(boxes):
                            self.assertTrue(np.all(target[...,0]==0));continue
                        self.assertEqual(len(boxes),2);self.assertEqual(len(extra['ignore_boxes']),1)
                        self.assertFalse(eligible[1])
                        occupied=np.argwhere(target[...,0]>=.999)
                        for gy,gx in occupied:
                            bits=target[gy,gx,5:]
                            if np.all(bits==-1):partial_centers+=1
                            else:
                                np.testing.assert_array_equal(bits,[(masks[0]>>bit)&1 for bit in range(6)])
                                expected=(boxes[0,:2]+boxes[0,2:])/8
                                self.assertEqual((gy,gx),(int(expected[1]),int(expected[0])))
                    batches+=1
                old.on_epoch_end();off.on_epoch_end();on.on_epoch_end()
            # Enabling the private flag on non-training data must still be inert.
            valargs=dict(kwargs,training=False)
            valold=module.OldPatches(**valargs);valnew=module.NewPatches(**valargs,horizontal_aspect_augmentation=True)
            def forbidden(*args,**kwargs):raise AssertionError('Validation must not sample aspect factors')
            with mock.patch.dict(module.NEW_PATCH,horizontal_aspect_factors=forbidden):
                for batch in range(len(valold)):
                    for av,bv in zip(valold[batch],valnew[batch]):np.testing.assert_array_equal(av,bv)
            checks['balanced_patches']={'epochs':3,'batches':batches,'original_off_pixels_targets_order_exact':True,
                'negative_indices_once_per_epoch':3,'geometry_only_centers_observed':partial_centers,
                'unknown_not_promoted':True,'validation_enabled_flag_inert':True}

started=datetime.now(timezone.utc).isoformat()
stream=io.StringIO()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(module.AspectTests),
                         unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)])
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
log=OUT/'tests.log';log.write_text(stream.getvalue(),encoding='utf-8')
report={'scope':'Independent CPU-only review of immutable private horizontal-aspect prototype; no real image corpus/model/training or protocol change',
    'status':'passed' if result.wasSuccessful() else 'failed','started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
    'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'checks':checks,
    'prototype_report':{'path':str(BASE/'proof-report.json'),'sha256':sha(BASE/'proof-report.json')},
    'prototype_index':{'path':str(BASE/'index.json'),'sha256':sha(BASE/'index.json')},
    'audit_source':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},'log_sha256':sha(log),
    'backend_modules_imported':[name for name in ('torch','tensorflow','keras') if name in sys.modules],
    'limitations':['Private profile only: illumination true; camera/polarity/half-turn/mirror/vertical false. No CLI/config adoption guard yet.',
                   'Per-record factor repeats within a dataset epoch for oversampled records; existing image RNG remains per batch.',
                   'Factor affects only existing 0.85 affine gate. Relative x/y range is 0.55..1.25; existing zoom 0.8..1.7 means effective x scale 0.44..2.125.',
                   'Unknown regions can overlap soft Gaussian shoulders of nearby known targets, per unchanged target priority; unknown slots themselves never create a target.',
                   'Synthetic CPU proof does not establish useful accuracy, real-paper dot visibility after heavy shrink/blur, or optimizer/runtime behavior.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(stream.getvalue());print(json.dumps({'status':report['status'],'report_sha256':sha(OUT/'report.json'),'checks':checks}))
sys.exit(0 if result.wasSuccessful() else 1)
