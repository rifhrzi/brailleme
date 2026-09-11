"""CPU-only curriculum invariants; no real model inference or TEST images."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_overall_v5_train import (Curriculum, annotation, assert_test_unopened,
    bind_development_inputs, contained, cosine_rate, gentle_anchor, overall_rank,
    rotate_boxes, save_new)


def row(image='image.png', group=None, *, empty=False, incomplete=False):
    value=dict(image=image,source=group or 'source',boxes=[] if empty else [[20,50,40,100]],
               masks=[] if empty else [1],originally_complete=[] if empty else [not incomplete])
    if group: value['symbol_group']=group
    return value


def grouped(correct, accepted=None, invalid=0):
    accepted=accepted or correct
    result={name:dict(positive_total=100,correct=c,accepted=a,exact_accuracy=c/100)
            for name,c,a in zip(('lowercase','capital_single','capital_word','numbers','punctuation'),correct,accepted)}
    result['rejection']=dict(false_accepts=invalid)
    return result


class CurriculumTests(unittest.TestCase):
    def test_annotation_preserves_partial_and_ignore_flags(self):
        value=row();value['boxes'] += [[-1,20,25,80],[90,30,110,70]]
        value['masks'] += [3,5];value['originally_complete']=[True,True,False]
        value['ignore_boxes']=[[200,40,220,90]]
        boxes,eligible,count=annotation(value)
        self.assertEqual(count,3);self.assertEqual(eligible.tolist(),[True,False,False,False])
        self.assertEqual(boxes.shape,(4,4))

    def test_bad_original_flags_fail(self):
        value=row();value['originally_complete']=[]
        with self.assertRaises(ValueError):annotation(value)

    def test_gentle_keeps_all_edge_targets_and_never_promotes_partial(self):
        image=Image.fromarray(np.arange(192*480*3,dtype=np.uint8).reshape(192,480,3))
        boxes=np.asarray([[0,1,12,40],[464,150,480,191],[-10,40,15,90],[150,60,170,100]],float)
        eligible=np.asarray([True,True,False,False])
        before=np.asarray(image).copy();original=boxes.copy()
        modes=set()
        for seed in range(80):
            pixels,moved,after,mode=gentle_anchor(image,boxes,eligible,np.random.default_rng(seed))
            self.assertTrue(contained(moved)[eligible].all())
            np.testing.assert_array_equal(after,eligible)
            self.assertEqual(pixels.shape,(192,480,3));self.assertEqual(pixels.dtype,np.uint8)
            modes.add(mode)
            if mode=='identity': np.testing.assert_array_equal(pixels,before)
        np.testing.assert_array_equal(boxes,original)
        np.testing.assert_array_equal(np.asarray(image),before)
        self.assertIn('identity',modes)
        self.assertTrue(modes-{'identity'})

    def test_empty_boxes_are_valid_negative_through_rotation_and_augmentation(self):
        boxes,eligible,count=annotation(row(empty=True));self.assertEqual(count,0)
        self.assertEqual(rotate_boxes(boxes,2).shape,(0,4))
        for seed in (0,1,2):
            pixels,moved,after,_=gentle_anchor(Image.new('RGB',(480,192),(130,130,130)),boxes,eligible,np.random.default_rng(seed))
            self.assertEqual(moved.shape,(0,4));self.assertEqual(after.shape,(0,));self.assertTrue(np.isfinite(pixels).all())

    def test_rotation_bounds_use_all_four_corners(self):
        boxes=np.asarray([[220,86,260,106]],float)
        np.testing.assert_allclose(rotate_boxes(boxes,90),[[230,76,250,116]],atol=1e-12)
        np.testing.assert_allclose(rotate_boxes(boxes,0),boxes)

    def test_batch_roles_determinism_and_ignore_only_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);Image.new('RGB',(480,192),(140,140,140)).save(path/'image.png')
            replay=[row(),row(incomplete=True),row(empty=True)]
            symbols=[row(group=name) for name in ('lowercase','capital_single','capital_word','numbers','punctuation','rejection')]
            data=Curriculum(path,replay,symbols,2,123)
            self.assertEqual(data.complete,[0]);self.assertEqual(data.context,[0,1]);self.assertEqual(data.negative,[2])
            first=data.batch(1,0);second=data.batch(1,0)
            self.assertEqual(first[0].shape,(24,192,480,3));self.assertEqual(first[1].shape,(24,48,120,11))
            self.assertEqual(first[2]['draws'],dict(new=11,invalid=1,complete_replay=6,context=5,negative=1))
            np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
            self.assertTrue(np.isfinite(first[1]).all());self.assertLessEqual(first[2]['eligible_after'],first[2]['eligible_before'])

    def test_overall_accuracy_precedes_precision_and_worst_group(self):
        higher=grouped([100,100,100,100,1],[100]*5,invalid=1)
        lower=grouped([80]*5)
        self.assertGreater(overall_rank(higher),overall_rank(lower))
        self.assertEqual(len(overall_rank(higher)),3)

    def test_equal_overall_precision_then_invalid_break_ties(self):
        precise=grouped([80]*5)
        imprecise=grouped([80]*5,[100]*5)
        self.assertGreater(overall_rank(precise),overall_rank(imprecise))
        self.assertGreater(overall_rank(precise),overall_rank(grouped([80]*5,invalid=1)))
        self.assertEqual(overall_rank(grouped([60,85,85,85,85])),overall_rank(precise))

    def test_learning_rate_real_recipe_and_shortest_supported_schedule(self):
        values=[cosine_rate(epoch,24) for epoch in range(1,25)]
        np.testing.assert_allclose(values[:3],[1e-5,2e-5,3e-5],rtol=1e-12)
        self.assertAlmostEqual(values[-1],5e-6)
        self.assertTrue(all(a>=b for a,b in zip(values[2:],values[3:])))
        self.assertAlmostEqual(cosine_rate(4,4),5e-6)
        for args in ((1,3),(0,24),(25,24),(1,24,1e-5,2e-5)):
            with self.assertRaises(ValueError):cosine_rate(*args)

    def test_completed_or_reserved_method_blocks_more_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);assert_test_unopened(path)
            for name in ('final-method-selection.json','test-evaluation-reservation.json','final-test-evaluation.json'):
                with self.subTest(name=name):
                    marker=path/name;marker.write_text('{}')
                    with self.assertRaises(ValueError):assert_test_unopened(path)
                    marker.unlink()

    def test_evidence_is_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'evidence.json';save_new(path,{'first':True})
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):save_new(path,{'first':False})
            self.assertEqual(path.read_bytes(),before)

    def test_protocol_rejects_relabelled_or_changed_development_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);files={name:str(path/name) for name in ('train-symbols.json','validation-symbols.json','train-image-hashes.json','validation-image-hashes.json')}
            for target in files.values():Path(target).write_text('{}')
            protocol=dict(version='overall-exact-text-v5-20260911',development_files=files,input_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files.values()})
            (path/'protocol.json').write_text(json.dumps(protocol))
            bindings=bind_development_inputs(path,path/'train-symbols.json',path/'validation-symbols.json')
            self.assertEqual(len(bindings),5)
            fake=path/'relabelled-test.json';fake.write_text('{"split":"train"}')
            with self.assertRaises(ValueError):bind_development_inputs(path,fake,path/'validation-symbols.json')
            (path/'train-symbols.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):bind_development_inputs(path,path/'train-symbols.json',path/'validation-symbols.json')


if __name__=='__main__':unittest.main()
