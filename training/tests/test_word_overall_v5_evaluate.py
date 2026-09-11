"""No real model or TEST pixels: final-method authorization and denominator checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_overall_v5_evaluate import (validate_choices, reserve_stage, semantic_signature, decision_parity,
    report_for, pixels_for, infer_rows, evaluate, sha, validate_policy_results, validate_method_bindings)
from word_overall_v5_tta import method_specification
from word_overall_v5_protocol import GROUPS, metrics


class FinalEvaluationTests(unittest.TestCase):
    def valid_choices(self):
        thresholds = {'dotConfidence': .7, 'proposalScore': .25}
        selection = dict(selection_scope='validation_only', test_opened=False, sha256='checkpoint')
        config = dict(thresholds=thresholds)
        policy = dict(selection_scope='validation_only', test_opened=False, old_or_fresh_test_used=False,
            checkpoint_sha256='checkpoint', validation_sha256='validation', selected_validation_policy='cascade',
            method=method_specification(), thresholds=thresholds)
        exported = dict(status='passed', allclose=True, checkpoint_sha256='checkpoint', tflite_sha256='tflite',
            validation_sha256='validation', scope='development_validation_only', test_opened=False, samples=540,
            pipeline_decision_mismatches=[], views=['identity','horizontal','vertical','half_turn'],
            inference_count=2160, method=method_specification(), thresholds=thresholds)
        return selection, config, policy, exported

    def test_method_requires_validation_selected_weights_policy_and_passed_export(self):
        choices = self.valid_choices()
        validate_choices(*choices, 'checkpoint', 'tflite', 'validation', method_specification())
        for index, key, value in [(0,'test_opened',True), (0,'sha256','different'),
                (2,'validation_sha256','test'), (2,'selected_validation_policy','new_unvalidated'),
                (2,'old_or_fresh_test_used',True), (3,'tflite_sha256','different'),
                (3,'pipeline_decision_mismatches',[3]), (3,'samples',12), (3,'allclose',False),
                (3,'views',['identity']), (3,'inference_count',540)]:
            changed = deepcopy(choices)
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_choices(*changed, 'checkpoint', 'tflite', 'validation', method_specification())
        changed = deepcopy(choices)
        changed[2]['thresholds'] = {'dotConfidence': .6}
        with self.assertRaisesRegex(ValueError, 'thresholds'):
            validate_choices(*changed, 'checkpoint', 'tflite', 'validation', method_specification())

    def test_once_only_reservation_binds_engine_and_confirmation_to_same_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'final-method-selection.json').write_text('{}')
            method = dict(checkpoint_sha256='a', tflite_sha256='b', primary_engine='tflite', selected_policy='identity')
            with self.assertRaisesRegex(ValueError, 'engine'):
                reserve_stage(root, method, 'primary', 'torch')
            reserve_stage(root, method, 'primary', 'tflite')
            with self.assertRaises(FileExistsError):
                reserve_stage(root, method, 'primary', 'tflite')
            with self.assertRaisesRegex(ValueError, 'finish'):
                reserve_stage(root, method, 'confirm', 'torch')
            (root/'final-test-evaluation.json').write_text(json.dumps(dict(method_sha256=sha(root/'final-method-selection.json'))))
            reserve_stage(root, method, 'confirm', 'torch')
            with self.assertRaises(FileExistsError):
                reserve_stage(root, method, 'confirm', 'torch')
            changed = dict(method, selected_policy='cascade')
            with self.assertRaisesRegex(ValueError, 'reservation'):
                reserve_stage(root, changed, 'regression', 'tflite')

    def test_inference_engine_is_not_constructed_if_reservation_fails(self):
        with patch('word_overall_v5_evaluate.verify_method', return_value={'primary_engine':'tflite'}), \
             patch('word_overall_v5_evaluate.reserve_stage', side_effect=ValueError('reserved')), \
             patch('word_overall_v5_evaluate.make_predictor') as factory:
            with self.assertRaisesRegex(ValueError, 'reserved'):
                evaluate(SimpleNamespace(experiment=Path('.'), command='primary', device='cpu'))
            factory.assert_not_called()

    def test_primary_eighty_overall_is_not_conflated_with_precision(self):
        decisions = []
        for group in GROUPS:
            for index in range(100):
                expected = 'a' if group != 'punctuation' and index >= 95 else 'ab'
                decisions.append(dict(source=f'{group}/{index}', group=group, expected=expected,
                    predicted=expected if index < 80 else 'wrong', reason='', route='identity'))
        decisions += [dict(source=f'invalid/{index}', group='rejection', expected='', predicted='', reason='invalid_sequence', route='identity') for index in range(40)]
        report = report_for(decisions, primary=True)
        self.assertTrue(report['target80_met'])
        self.assertEqual(report['overall_exact_accuracy'], .8)
        self.assertFalse(report['precision95_met'])
        self.assertEqual(report['positive_total'], 500)
        native = report_for(decisions[:10], primary=False)
        self.assertNotIn('target80_met', native)

    def test_probability_decision_parity_includes_ordered_masks_and_roles(self):
        decision = dict(text='Ab', reason='', cells=[dict(mask=np.int64(32), role='capital', output=''), dict(mask=1, role='letter', output='A')])
        signature = semantic_signature(decision)
        json.dumps(signature)
        left = [dict(source='a', group='capital_single', expected='Ab', predicted='Ab', reason='', expected_cells=3,
            detected_cells=3, semantic_signature=signature, image_sha256='same', route='identity')]
        right = deepcopy(left)
        right[0]['semantic_signature'][2][0][0] = 60
        self.assertEqual(decision_parity(left,right)['status'], 'failed')
        right = deepcopy(left)
        right[0]['route'] = 'matched_logit_four'
        report = decision_parity(left,right)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(report['routing_differences']), 1)
        a = semantic_signature(dict(text='', reason='low_confidence', cells=[dict(dots=[.6,.4,.4,.4,.4,.4])]))
        b = semantic_signature(dict(text='', reason='low_confidence', cells=[dict(dots=[.4,.6,.4,.4,.4,.4])]))
        self.assertNotEqual(a,b)  # Retain physical bits even before parser visits masks.

    def test_image_hash_and_runtime_dimensions_fail_before_predictor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'fixture.png'
            Image.new('RGB', (40,20), 'white').save(path)
            rows = [dict(source='a', image=str(path), expected='a')]
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                pixels_for(rows, {str(path): sha(path)})
            self.assertEqual(pixels_for(rows, {str(path):sha(path)}, allow_resize=True).shape, (1,192,480,3))
            with self.assertRaisesRegex(ValueError, 'changed'):
                pixels_for(rows, {str(path): 'incorrect'})

    def test_historical_crops_may_share_original_source_but_not_same_fixture(self):
        rows = [dict(source='same_page', image='one.png', expected='ab'), dict(source='same_page', image='two.png', expected='cd')]
        result = dict(decisions=[dict(text=row['expected'], reason='', cells=[]) for row in rows],
                      routes=['identity']*2, latency={})
        method = dict(image_sha256={'one.png':'one','two.png':'two'}, thresholds={}, selected_policy='identity')
        with patch('word_overall_v5_evaluate.pixels_for', return_value=np.zeros((2,192,480,3),np.float32)), \
             patch('word_overall_v5_tta.infer_decisions', return_value=result):
            values, _ = infer_rows(rows, lambda images: None, method)
        self.assertEqual(len(values), 2)

    def test_recorded_tta_selection_must_match_all_frozen_validation_truth(self):
        decisions = []
        for group in GROUPS:
            for index in range(100):
                text = 'a' if group != 'punctuation' and index >= 95 else 'ab'
                decisions.append(dict(source=f'{group}/{index}', group=group, expected=text, predicted=text))
        decisions += [dict(source=f'invalid/{index}',group='rejection',expected='',predicted='') for index in range(40)]
        rows = [dict(source=row['source'],symbol_group=row['group'],expected=row['expected']) for row in decisions]
        policy = dict(method=method_specification(),selected_validation_policy='identity',variants={name:
            dict(decisions=deepcopy(decisions),metrics=metrics(decisions,require_primary=True)) for name in method_specification()['policies']})
        validate_policy_results(policy,rows)
        wrong = deepcopy(policy)
        wrong['selected_validation_policy'] = 'cascade'
        with self.assertRaisesRegex(ValueError,'rank'): validate_policy_results(wrong,rows)
        wrong = deepcopy(policy)
        wrong['variants']['identity']['decisions'][0]['expected'] = 'changed'
        with self.assertRaisesRegex(ValueError,'truth'): validate_policy_results(wrong,rows)
        wrong = deepcopy(policy)
        wrong['variants']['identity']['metrics']['overall_exact_accuracy'] = .1
        with self.assertRaisesRegex(ValueError,'metric'): validate_policy_results(wrong,rows)

    def test_declared_artifacts_and_pixels_cannot_escape_the_bound_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            policy_path = root/'policy.json'
            policy_path.write_text(json.dumps(dict(checkpoint_sha256='weights',thresholds={'dotConfidence':.7},selected_validation_policy='identity')))
            for name in ('test-image-hashes.json','native-context-image-hashes.json'):
                (root/name).write_text(json.dumps({'image.png':'original'}))
            method = dict(checkpoint=str(root/'model.keras'),checkpoint_sha256='weights',
                tflite=str(root/'model.tflite'),tflite_sha256='tflite',
                primary_manifest=str(root/'test-symbols.json'),native_manifest=str(root/'native-context.json'),
                tta_validation_report=str(policy_path),thresholds={'dotConfidence':.7},selected_policy='identity',
                input_sha256={str(root/name):digest for name,digest in [('model.keras','weights'),('model.tflite','tflite'),
                    ('test-symbols.json','x'),('native-context.json','x'),('policy.json','x'),
                    ('test-image-hashes.json','x'),('native-context-image-hashes.json','x')]},image_sha256={'image.png':'original'})
            validate_method_bindings(method,root)
            for field,value in [('checkpoint',str(root/'unbound.keras')),('primary_manifest',str(root/'easier.json')),
                    ('selected_policy','cascade'),('image_sha256',{'image.png':'changed'})]:
                changed = deepcopy(method)
                changed[field] = value
                with self.subTest(field=field),self.assertRaises(ValueError): validate_method_bindings(changed,root)


if __name__ == '__main__': unittest.main()
