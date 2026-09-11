"""Synthetic CPU contracts for four/six-view dispatch; no model or TEST input."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import word_overall_v5_runtime as dispatch
import word_overall_v5_export as export
import word_overall_v5_evaluate as evaluation

ZOOM = 'word_overall_v5_zoom'
FOUR = ('identity', 'horizontal', 'vertical', 'half_turn')
SIX = (*FOUR, 'zoom125', 'zoom150')
THRESHOLDS = {'dotConfidence': .7, 'proposalScore': .25}


def specification(name):
    return dict(version='synthetic-dispatch-contract', views=list(dispatch.get_views(name)),
        policies={'identity': ['identity'], 'cascade': ['identity', 'matched_logit_four']})


def valid_choices(name):
    selection = dict(selection_scope='validation_only', test_opened=False, sha256='checkpoint')
    config = dict(thresholds=THRESHOLDS)
    policy = dict(selection_scope='validation_only', test_opened=False, old_or_fresh_test_used=False,
        checkpoint_sha256='checkpoint', validation_sha256='validation', selected_validation_policy='cascade',
        method=specification(name), thresholds=THRESHOLDS)
    parity = dict(status='passed', allclose=True, checkpoint_sha256='checkpoint', tflite_sha256='tflite',
        validation_sha256='validation', scope='development_validation_only', test_opened=False, samples=540,
        pipeline_decision_mismatches=[], views=list(dispatch.get_views(name)),
        inference_count=540*len(dispatch.get_views(name)), method=specification(name), thresholds=THRESHOLDS)
    # Missing runtime_module remains compatible only with the original helper.
    if name != dispatch.DEFAULT_RUNTIME:
        policy['runtime_module'] = parity['runtime_module'] = name
    return selection, config, policy, parity


class RuntimeDispatchTests(unittest.TestCase):
    def test_whitelist_and_view_order_reject_arbitrary_imports(self):
        self.assertEqual(dispatch.get_views(), FOUR)
        self.assertEqual(dispatch.get_views(ZOOM), SIX)
        module = object()
        with patch.object(dispatch, 'import_module', return_value=module) as importer:
            self.assertIs(dispatch.get_runtime(), module)
            importer.assert_called_once_with(dispatch.DEFAULT_RUNTIME)
            importer.reset_mock()
            self.assertIs(dispatch.get_runtime(ZOOM), module)
            importer.assert_called_once_with(ZOOM)
            importer.reset_mock()
            for name in ('os', '../word_overall_v5_zoom', 'word_overall_v5_zoom.py', ''):
                with self.subTest(name=name), self.assertRaises(ValueError): dispatch.get_runtime(name)
                with self.assertRaises(ValueError): dispatch.get_views(name)
            importer.assert_not_called()

    def test_source_bindings_include_dispatch_and_chosen_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            scripts = Path(temporary)
            for name in (*export.SOURCE_NAMES, ZOOM+'.py'):
                (scripts/name).write_text('# synthetic source '+name)
            bound = export.bindings(scripts, ZOOM)
            for name in ('word_overall_v5_runtime.py', 'word_overall_v5_tta.py', ZOOM+'.py'):
                self.assertEqual(bound[str(scripts/name)], export.sha(scripts/name))
            export.verify_hashes(bound)
            (scripts/(ZOOM+'.py')).write_text('# changed zoom implementation')
            with self.assertRaises(ValueError): export.verify_hashes(bound)
            with self.assertRaises(ValueError): export.bindings(scripts, 'unapproved_module')

    def test_six_raw_views_require_exact_keys_shape_count_and_float32(self):
        good = {name: np.zeros((2,48,120,11), np.float32) for name in SIX}
        export.validate_view_bundle(good, 2, SIX)
        for name in SIX:
            changed = dict(good); del changed[name]
            with self.subTest(missing=name), self.assertRaises(ValueError): export.validate_view_bundle(changed, 2, SIX)
        for changed in (dict(good, extra=good['identity']),
                dict(good, zoom125=good['zoom125'][:1]),
                dict(good, zoom150=good['zoom150'].astype(np.float64)),
                dict(good, zoom150=np.full((2,48,120,11), np.nan, np.float32))):
            with self.assertRaises(ValueError): export.validate_view_bundle(changed, 2, SIX)
        # The original optional-argument contract still requires exactly four.
        export.validate_view_bundle({name: good[name] for name in FOUR}, 2)
        with self.assertRaises(ValueError): export.validate_view_bundle(good, 2)

    def test_final_choice_requires_3240_six_view_parity_and_matching_runtime(self):
        for name in (dispatch.DEFAULT_RUNTIME, ZOOM):
            choices = valid_choices(name)
            evaluation.validate_choices(*choices, 'checkpoint', 'tflite', 'validation', specification(name))
        good = valid_choices(ZOOM)
        for index, key, value in ((3,'inference_count',2160), (3,'views',list(FOUR)),
                (3,'samples',360), (3,'runtime_module',dispatch.DEFAULT_RUNTIME),
                (3,'method',specification(dispatch.DEFAULT_RUNTIME)), (2,'runtime_module',dispatch.DEFAULT_RUNTIME)):
            changed = deepcopy(good); changed[index][key] = value
            with self.subTest(key=key,index=index), self.assertRaises(ValueError):
                evaluation.validate_choices(*changed, 'checkpoint', 'tflite', 'validation', specification(ZOOM))
        changed = deepcopy(good); del changed[3]['runtime_module']
        with self.assertRaises(ValueError):
            evaluation.validate_choices(*changed, 'checkpoint', 'tflite', 'validation', specification(ZOOM))

    def test_oracle_binds_six_view_module_and_exact_method_before_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(checkpoint=root/'model.keras', validation=root/'validation.json',
                thresholds=root/'thresholds.json', runtime_module=ZOOM)
            args.checkpoint.write_bytes(b'synthetic metadata only')
            args.validation.write_text('{}')
            args.thresholds.write_text(json.dumps(dict(thresholds=THRESHOLDS)))
            frozen = {str(path): export.sha(path) for path in (args.checkpoint,args.validation,args.thresholds)}
            meta = dict(input_sha256=frozen, checkpoint_sha256=frozen[str(args.checkpoint)],
                validation_sha256=frozen[str(args.validation)], samples=540, views=list(SIX),
                orientation='raw_model_output_for_each_transformed_image', method=specification(ZOOM),
                thresholds=THRESHOLDS, scope='development_validation_only', test_opened=False,
                physical_camera_validated=False, oracle_sha256='synthetic archive', runtime_module=ZOOM)
            runtime = SimpleNamespace(method_specification=lambda: specification(ZOOM))
            with patch.object(export, 'get_runtime', return_value=runtime) as importer:
                export.validate_oracle_metadata(meta,args,frozen,540,'synthetic archive')
                importer.assert_called_with(ZOOM)
                for key,value in (('views',list(FOUR)),('runtime_module',dispatch.DEFAULT_RUNTIME),
                        ('method',specification(dispatch.DEFAULT_RUNTIME))):
                    changed = deepcopy(meta); changed[key] = value
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        export.validate_oracle_metadata(changed,args,frozen,540,'synthetic archive')
                del meta['runtime_module']
                with self.assertRaises(ValueError): export.validate_oracle_metadata(meta,args,frozen,540,'synthetic archive')

    def test_pipeline_signatures_call_chosen_runtime_and_keep_route_and_physical_bits(self):
        cell = dict(dots=[.99,.01,.01,.01,.01,.01])
        decision = dict(text='', reason='low_confidence', cells=[cell])
        invoke = Mock(return_value=dict(decisions={'zoom_fallback':decision}, routes={'zoom_fallback':'zoom125'}))
        arrays = {name: np.zeros((48,120,11),np.float32) for name in SIX}
        selection = {'x':.2,'y':.1,'width':.5,'height':.7}
        with patch.object(export, 'runtime_decisions', side_effect=AssertionError('Wrong default runtime')):
            signatures = export.pipeline_signatures(arrays,THRESHOLDS,selection,False,SimpleNamespace(runtime_decisions=invoke))
        invoke.assert_called_once_with(arrays,THRESHOLDS,selection,quality_rejected=False)
        self.assertEqual(signatures['zoom_fallback'][1:], ((1,), 'zoom125'))

    def test_bound_policy_runtime_cannot_differ_from_final_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            policy = dict(checkpoint_sha256='weights', thresholds=THRESHOLDS, selected_validation_policy='cascade',runtime_module=ZOOM)
            (root/'policy.json').write_text(json.dumps(policy))
            for name in ('test-image-hashes.json','native-context-image-hashes.json'):
                (root/name).write_text('{}')
            method = dict(checkpoint=str(root/'model.keras'),checkpoint_sha256='weights',
                tflite=str(root/'model.tflite'),tflite_sha256='tflite', runtime_module=ZOOM,
                primary_manifest=str(root/'test-symbols.json'),native_manifest=str(root/'native-context.json'),
                tta_validation_report=str(root/'policy.json'),thresholds=THRESHOLDS,selected_policy='cascade',image_sha256={},
                input_sha256={str(root/name):digest for name,digest in [('model.keras','weights'),('model.tflite','tflite'),
                    ('test-symbols.json','x'),('native-context.json','x'),('policy.json','x'),
                    ('test-image-hashes.json','x'),('native-context-image-hashes.json','x')]})
            evaluation.validate_method_bindings(method,root)
            for altered in (dict(method,runtime_module=dispatch.DEFAULT_RUNTIME),
                    {key:value for key,value in method.items() if key != 'runtime_module'}):
                with self.assertRaisesRegex(ValueError,'bound validation policy'):
                    evaluation.validate_method_bindings(altered,root)

    def test_primary_confirmation_and_regression_dispatch_same_frozen_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            row = dict(source='synthetic/one',image='synthetic.png',expected='a',symbol_group='lowercase',masks=[1])
            for name in ('primary.json','native.json'):
                (root/name).write_text(json.dumps({'records':[row]}))
            method = dict(primary_engine='tflite',runtime_module=ZOOM,selected_policy='cascade',
                checkpoint_sha256='weights',tflite_sha256='tflite',thresholds=THRESHOLDS,
                primary_manifest=str(root/'primary.json'),native_manifest=str(root/'native.json'),
                image_sha256={'synthetic.png':'synthetic hash'},regression_reference='synthetic baseline')
            (root/'final-method-selection.json').write_text(json.dumps(method))
            selected_engines, routed_engines, allow_resize = [], [], []
            def factory(engine, frozen, device):
                self.assertIs(frozen,method); selected_engines.append(engine)
                return engine, dict(engine=engine,device='synthetic CPU stub')
            def pixels(rows, hashes, **kwargs):
                allow_resize.append(kwargs['allow_resize'])
                return np.zeros((len(rows),192,480,3),np.float32)
            def infer(predictor, images, thresholds, selections, **kwargs):
                routed_engines.append(predictor)
                self.assertEqual(kwargs['policy'],'cascade')
                self.assertEqual(thresholds,THRESHOLDS)
                self.assertEqual(selections,[None])  # No expected text or annotations passed.
                return dict(decisions=[dict(text='a',reason='',cells=[dict(mask=1,role='letter',output='a',dots=[.99,.01,.01,.01,.01,.01])])],
                    routes=['zoom125'],latency={'synthetic':True})
            runtime = SimpleNamespace(infer_decisions=infer)
            cached = {'synthetic_regression':[dict(row, baseline={'text':'a'})]}
            summary = lambda decisions,primary: dict(target80_met=False,**evaluation.summarize_decisions(decisions))
            with patch.object(evaluation,'verify_method',return_value=method), \
                    patch.object(evaluation,'make_predictor',side_effect=factory), \
                    patch.object(evaluation,'get_runtime',return_value=runtime) as importer, \
                    patch.object(evaluation,'pixels_for',side_effect=pixels), \
                    patch.object(evaluation,'report_for',side_effect=summary), \
                    patch.object(evaluation,'load_cached_regression',return_value=cached), redirect_stdout(io.StringIO()):
                evaluation.evaluate(SimpleNamespace(experiment=root,command='primary',device='cpu'))
                evaluation.evaluate(SimpleNamespace(experiment=root,command='confirm',device='cpu'))
                evaluation.regression(SimpleNamespace(experiment=root,device='cpu'))
            self.assertEqual(selected_engines,['tflite','torch','tflite'])
            self.assertEqual(routed_engines,['tflite','tflite','torch','torch','tflite'])
            self.assertEqual(allow_resize,[False,False,False,False,True])
            self.assertEqual([call.args for call in importer.call_args_list],[(ZOOM,)]*5)
            confirmation = json.loads((root/'final-test-torch-confirmation.json').read_text())
            self.assertEqual(confirmation['status'],'passed')
            self.assertEqual(confirmation['results']['primary']['decisions'][0]['route'],'zoom125')


if __name__ == '__main__': unittest.main()
