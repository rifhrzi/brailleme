import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import word_overall_v5_tta as flip
import word_overall_v5_zoom as zoom
from word_decode_id_v4 import decode

T = dict(proposalScore=.25, acceptScore=.25, dotConfidence=.7, nmsIou=.25,
    minCellWidth=.008, minCellHeight=.06, edgeMargin=.008, maxRowDeviation=.55, maxGapRatio=1.2, maxOverlapRatio=.5)


def output(dot=.99, scale=1.):
    value = np.zeros((48, 120, 11), np.float32)
    cx, cy = .5+((60.5/120)-.5)*scale, .5+((24.5/48)-.5)*scale
    gx, gy = int(cx*120), int(cy*48)
    value[gy, gx] = [.9, cx*120-gx, cy*48-gy, .05*scale, .2*scale, dot, .01, .01, .01, .01, .01]
    return value


def raw_flip(value, name):
    return flip.invert_outputs(value[None], name)[0]


class ZoomRuntimeTests(unittest.TestCase):
    def test_method_contract_has_six_raw_views_and_immutable_json_order(self):
        spec = zoom.method_specification()
        self.assertEqual(spec['views'], ['identity', 'horizontal', 'vertical', 'half_turn', 'zoom125', 'zoom150'])
        self.assertEqual(spec['runtime_module'], 'word_overall_v5_zoom')
        self.assertEqual(spec, json.loads(json.dumps(spec)))
        self.assertEqual(spec['policies']['cascade_zoom_fallback'], list(flip.POLICIES['cascade'])+['zoom_logit_mean','zoom_probability_mean'])
        spec['scales']['zoom125'] = 10
        self.assertEqual(zoom.method_specification()['scales']['zoom125'], 1.25)

    def test_bilinear_pixel_center_transform_and_rgb_mean_padding(self):
        ramp = np.broadcast_to(np.arange(8, dtype=np.float32)[None,None,:,None], (1,6,8,3)).copy()
        np.testing.assert_array_equal(zoom.center_zoom(ramp, 1.), ramp)
        np.testing.assert_allclose(zoom.transformed_images(ramp, 'zoom125')[0,3,:,0], (np.arange(8)+.5-4)/1.25+4-.5, atol=1e-6)
        np.testing.assert_array_equal(zoom.center_zoom(ramp, .5)[0,0,0], [3.5]*3)

    def test_projected_clipping_and_complete_view_box_guards(self):
        base = np.zeros((48,120,11), np.float32); base[2,3,5:] = .1
        selected = [((2,3), dict(box=[.15,.4,.25,.6], dots=[.1]*6))]
        other = [((4,5), dict(box=[.174,.4,.264,.6], dots=[.99]*6))]
        np.testing.assert_array_equal(zoom.zoom_average(base, selected, {'zoom150':other}, T), base)
        value = output(); value[24,1] = [.8,.1,.5,.1,.2,.99,.01,.01,.01,.01,.01]
        self.assertEqual(len(zoom.inverse_selected(value, 1.5, T, None, decode)), 1)

    def test_old_three_policies_keep_full_semantics_and_skip_unneeded_zoom(self):
        values = dict(identity=output(.6), **{name:raw_flip(output(),name) for name in flip.VIEWS})
        expected = flip.runtime_decisions(values,T)
        actual = zoom.runtime_decisions(values,T,policies=tuple(flip.POLICIES)+('cascade_zoom_fallback',))
        for name in flip.POLICIES:
            self.assertEqual(actual['decisions'][name], expected['decisions'][name])
            self.assertEqual(actual['routes'][name], expected['routes'][name])
        self.assertEqual(actual['decisions']['cascade_zoom_fallback'], expected['decisions']['cascade'])

    def test_zoom_rescues_only_dots_and_keeps_full_original_boxes(self):
        values = dict(identity=output(.6), **{name:raw_flip(output(.6),name) for name in flip.VIEWS},
                      zoom125=output(scale=1.25), zoom150=output(scale=1.5))
        actual = zoom.runtime_decisions(values,T)
        original = decode(values['identity'],T)
        self.assertEqual(actual['decisions']['cascade']['text'], '')
        for name in ('zoom_fallback','cascade_zoom_fallback'):
            decision = actual['decisions'][name]
            self.assertEqual(decision['text'], 'a')
            self.assertEqual(decision['cells'][0]['mask'], 1)
            self.assertEqual(decision['cells'][0]['box'], original['cells'][0]['box'])
            self.assertEqual(decision['cells'][0]['score'], original['cells'][0]['score'])
            self.assertEqual(actual['routes'][name], 'zoom_logit_mean')
        rejected = zoom.runtime_decisions({'identity':output()},T,quality_rejected=True)
        self.assertTrue(all(value['reason']=='no_grid' for value in rejected['decisions'].values()))

    def test_selected_cascade_zoom_predicts_zooms_only_after_flip_rejects(self):
        pixels = np.random.default_rng(8).uniform(.1,.9,(4,192,480,3)).astype(np.float32); pixels[3] = 0
        supplied = [np.stack([output(),output(.6),output(.6),output()])]
        supplied += [np.stack([raw_flip(output(),name),raw_flip(output(.6),name)]) for name in flip.VIEWS]
        supplied += [output(scale=scale)[None] for scale in zoom.SCALES.values()]
        calls = []
        def predict(images):
            calls.append(images.copy()); return supplied[len(calls)-1]
        result = zoom.infer_decisions(predict,pixels,T,policy='cascade_zoom_fallback',batch_size=8)
        self.assertEqual([len(value) for value in calls], [4,2,2,2,1,1])
        self.assertEqual([value['text'] for value in result['decisions']], ['a','a','a',''])
        for index,name in enumerate(zoom.SCALES,4):
            np.testing.assert_array_equal(calls[index],zoom.transformed_images(pixels[2:3],name))
        self.assertEqual(result['latency']['images_per_view']['zoom125'],1)
        self.assertEqual(result['latency']['mean_forward_count'],3.)

    def test_identity_only_policy_never_requests_augmented_views(self):
        pixels = np.random.default_rng(2).uniform(.1,.9,(1,192,480,3)).astype(np.float32)
        calls=[]
        def predict(images):
            calls.append(images); return output(.6)[None]
        result=zoom.infer_decisions(predict,pixels,T,policy='identity')
        self.assertEqual(len(calls),1)
        self.assertEqual(result['decisions'][0]['text'],'')


if __name__ == '__main__':
    unittest.main()
