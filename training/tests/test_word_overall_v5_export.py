"""CPU export contracts with synthetic arrays; no framework inference or TEST."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import word_overall_v5_export as export
from word_overall_v5_tta import invert_outputs, method_specification

THRESHOLDS=dict(proposalScore=.25,acceptScore=.25,dotConfidence=.7,nmsIou=.25,
    minCellWidth=.008,minCellHeight=.06,edgeMargin=.008,maxRowDeviation=.55,
    maxGapRatio=1.2,maxOverlapRatio=.5)


def output(first=.99):
    value=np.zeros((48,120,11),np.float32)
    value[24,40]=[.9,.5,.5,.025,.2,first,.01,.01,.01,.01,.01]
    return value


def views(base,others=None):
    others=base if others is None else others
    return dict(identity=base,**{name:invert_outputs(others[None],name)[0]
        for name in ('horizontal','vertical','half_turn')})


class ExportTests(unittest.TestCase):
    def test_import_does_not_load_torch_tensorflow_or_keras(self):
        script="""
import importlib.abc,sys
class DenyFrameworks(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','keras'}:raise AssertionError('Eager framework import: '+fullname)
sys.meta_path.insert(0,DenyFrameworks())
sys.path.insert(0,sys.argv[1])
import word_overall_v5_export
print('FRAMEWORK_FREE_IMPORT_PASS')
"""
        result=subprocess.run([sys.executable,'-c',script,str(SCRIPTS)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('FRAMEWORK_FREE_IMPORT_PASS',result.stdout)

    def test_method_is_json_canonical_and_does_not_expose_mutable_globals(self):
        specification=method_specification()
        self.assertEqual(specification,json.loads(json.dumps(specification)))
        specification['policies']['cascade'].append('ground_truth_oracle')
        specification['transforms']['horizontal'][2][0]=99
        self.assertNotEqual(specification,method_specification())

    def test_view_bundle_requires_all_four_float32_finite_raw_tensors(self):
        good={name:output()[None] for name in export.VIEWS}
        export.validate_view_bundle(good,1)
        for name in export.VIEWS:
            bad=dict(good);del bad[name]
            with self.assertRaises(ValueError):export.validate_view_bundle(bad,1)
        for value in (output()[None].astype(np.float64),np.full((1,48,120,11),np.nan,np.float32),np.zeros((1,48,120,6),np.float32)):
            bad=dict(good,identity=value)
            with self.assertRaises(ValueError):export.validate_view_bundle(bad,1)

    def test_raw_view_orientation_is_inverted_exactly_once_for_fallback(self):
        base=output(.6);raw=views(base,output())
        before={name:value.copy() for name,value in raw.items()}
        result=export.pipeline_signatures(raw,THRESHOLDS,None,False)
        self.assertEqual(result['cascade'][0][0],'a')
        self.assertEqual(result['cascade'][2],'matched_logit_four')
        for name in raw:np.testing.assert_array_equal(raw[name],before[name])

    def test_same_accepted_text_with_different_routes_is_not_parity(self):
        identity=export.pipeline_signatures(views(output()),THRESHOLDS,None,False)['cascade']
        fallback=export.pipeline_signatures(views(output(.6),output()),THRESHOLDS,None,False)['cascade']
        self.assertEqual(identity[:2],fallback[:2])
        self.assertNotEqual(identity,fallback)

    def test_rejected_text_does_not_hide_later_physical_mask_change(self):
        base=output(.6)
        base[24,45]=[.88,.5,.5,.025,.2,.99,.4999999,.01,.01,.01,.01]
        changed=base.copy();changed[24,45,6]=.5000001
        self.assertTrue(np.allclose(base,changed,atol=1e-4,rtol=1e-4))
        a=export.pipeline_signatures(views(base,np.zeros_like(base)),THRESHOLDS,None,False)
        b=export.pipeline_signatures(views(changed,np.zeros_like(base)),THRESHOLDS,None,False)
        self.assertEqual(a['cascade'][0],b['cascade'][0])
        self.assertNotEqual(a['cascade'][1],b['cascade'][1])
        self.assertNotEqual(a,b)

    def test_quality_rejection_survives_all_augmented_views(self):
        result=export.pipeline_signatures(views(output()),THRESHOLDS,None,True)
        for decision,masks,route in result.values():
            self.assertEqual(decision[:2],('','no_grid'));self.assertEqual(masks,());self.assertEqual(route,'identity')

    def make_oracle(self,path):
        args=SimpleNamespace(checkpoint=path/'best-word.keras',validation=path/'validation.json',thresholds=path/'config.json')
        args.checkpoint.write_bytes(b'fake checkpoint metadata only');args.validation.write_text('{}')
        args.thresholds.write_text(json.dumps(dict(thresholds=THRESHOLDS)))
        frozen={str(p):export.sha(p) for p in (args.checkpoint,args.validation,args.thresholds)}
        meta=dict(input_sha256=frozen,checkpoint_sha256=frozen[str(args.checkpoint)],validation_sha256=frozen[str(args.validation)],samples=540,
            views=list(export.VIEWS),orientation='raw_model_output_for_each_transformed_image',method=method_specification(),thresholds=THRESHOLDS,
            scope='development_validation_only',test_opened=False,physical_camera_validated=False,oracle_sha256='frozen archive hash')
        return args,frozen,meta

    def test_oracle_binds_orientation_thresholds_scope_checkpoint_and_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            args,frozen,meta=self.make_oracle(Path(temporary))
            export.validate_oracle_metadata(json.loads(json.dumps(meta)),args,frozen,540,'frozen archive hash')
            changes=dict(orientation='already_inverted',views=list(reversed(export.VIEWS)),samples=500,
                checkpoint_sha256='other',validation_sha256='other',oracle_sha256='other',scope='test',test_opened=True,
                method=dict(version='different'),thresholds=dict(THRESHOLDS,dotConfidence=.6))
            for key,value in changes.items():
                altered=copy.deepcopy(meta);altered[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):export.validate_oracle_metadata(altered,args,frozen,540,'frozen archive hash')
            args.validation.write_text('{"mutated":true}')
            with self.assertRaises(ValueError):export.validate_oracle_metadata(meta,args,frozen,540,'frozen archive hash')

    def test_selected_run_binds_checkpoint_config_and_both_source_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment=Path(temporary);run=experiment/'run';run.mkdir();snapshot=run/'source';snapshot.mkdir();scripts=experiment/'scripts';scripts.mkdir()
            names=('word_torch_port.py','word_decode_id_v4.py','word_symbols_id_v4.py','word_symbols.py','word_data.py','word_anchor_zoom.py','benchmark_angelina.py')
            for name in names:
                (scripts/name).write_text('# test source '+name);(snapshot/name).write_bytes((scripts/name).read_bytes())
            checkpoint=run/'best-word.keras';checkpoint.write_bytes(b'fake checkpoint')
            config=run/'training-config.json';config.write_text(json.dumps(dict(source_sha256={name:export.sha(scripts/name) for name in names},thresholds=THRESHOLDS)))
            selection=dict(selection_scope='validation_only',test_opened=False,sha256=export.sha(checkpoint),training_config_sha256=export.sha(config))
            (run/'selection.json').write_text(json.dumps(selection))
            args=SimpleNamespace(checkpoint=checkpoint,thresholds=config,experiment=experiment)
            bound=export.selected_run_bindings(args,scripts)
            self.assertIn(str(checkpoint),bound);self.assertIn(str(run/'selection.json'),bound)
            (scripts/'word_data.py').write_text('# silently changed decoder dependency')
            with self.assertRaises(ValueError):export.selected_run_bindings(args,scripts)
            (scripts/'word_data.py').write_bytes((snapshot/'word_data.py').read_bytes())
            selection['selection_scope']='test';(run/'selection.json').write_text(json.dumps(selection))
            with self.assertRaises(ValueError):export.selected_run_bindings(args,scripts)

    def test_cli_rejects_other_manifest_before_any_engine_or_pixel_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment=Path(temporary)
            (experiment/'protocol.json').write_text(json.dumps(dict(development_files={'validation-symbols.json':str(experiment/'frozen-validation.json')})))
            argv=['export','--reference','--checkpoint',str(experiment/'run/best-word.keras'),'--validation',str(experiment/'test.json'),
                '--thresholds',str(experiment/'run/training-config.json'),'--experiment',str(experiment),'--output',str(experiment/'export')]
            with patch.object(sys,'argv',argv),patch.object(export,'reference',side_effect=AssertionError('Must not infer')):
                with self.assertRaisesRegex(ValueError,'frozen development validation'):export.main()


if __name__=='__main__':unittest.main()
