"""Temp-filesystem checks; no source verification/build/model tools execute."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage_source_test',HERE/'stage_source.py')
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
# Import only pure APK helper functions/constants, never its CLI or external tools.
tree=ast.parse((HERE/'verify_apk.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_fixture_pixels' or
       isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REQUIRED_NATIVE_LIBRARIES' for t in n.targets)]
apk={'hashlib':hashlib,'io':io,'Image':Image}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'verify_apk_pure_helpers','exec'),apk)


class SourceIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='release-helper-integrity-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'source';self.root.mkdir()
        self.target=Path(self.temp.name)/'stage';self.target.mkdir()
        (self.root/'app.ts').write_text('original')
        self.paths=['app.ts']
        self.git=patch.object(stage.subprocess,'run',side_effect=lambda *a,**k:types.SimpleNamespace(stdout=('\0'.join(self.paths)+'\0').encode()))
        self.git.start();self.addCleanup(self.git.stop)
        self.evidence=self.root/'ignored-output';self.evidence.mkdir()

    def setup_proof(self):
        before=stage.snapshot(self.root,'run-id')
        bp=self.evidence/'source-snapshot-before.json'
        bp.write_text(json.dumps(before))
        rows=[{'check':name,'exitCode':0,'verificationId':'run-id','sourceSnapshotSha256':stage.sha(bp)} for name in stage.CHECKS]
        cp=self.evidence/'source-checks.json';cp.write_text(json.dumps(rows))
        after=stage.snapshot(self.root,'run-id');after['sourceChecksSha256']=stage.sha(cp)
        (self.evidence/'source-snapshot-after.json').write_text(json.dumps(after))
        return before,after

    def test_unchanged_snapshot_checks_and_exact_copy_pass(self):
        self.setup_proof()
        checked,_=stage.verified_source(self.root,self.evidence)
        stage.copy_verified_source(self.root,self.target,checked)
        self.assertEqual((self.target/'app.ts').read_bytes(),(self.root/'app.ts').read_bytes())

    def test_modified_source_after_successful_checks_rejected(self):
        self.setup_proof();(self.root/'app.ts').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source changed'):
            stage.verified_source(self.root,self.evidence)

    def test_added_or_deleted_source_changes_snapshot(self):
        before=stage.snapshot(self.root,'id')
        (self.root/'new.ts').write_text('new');self.paths.append('new.ts')
        with self.assertRaises(ValueError):stage.require_same_source(before,stage.snapshot(self.root,'id'))
        self.paths=['app.ts'];(self.root/'app.ts').unlink()
        with self.assertRaises(ValueError):stage.require_same_source(before,stage.snapshot(self.root,'id'))

    def test_run_identity_and_result_file_drift_rejected(self):
        self.setup_proof()
        p=self.evidence/'source-checks.json';rows=json.loads(p.read_text());rows[0]['verificationId']='old-run';p.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'not bound'):stage.verified_source(self.root,self.evidence)
        self.setup_proof();p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'results changed'):stage.verified_source(self.root,self.evidence)

    def test_old_unbound_result_format_rejected(self):
        self.setup_proof();p=self.evidence/'source-checks.json'
        p.write_text(json.dumps([{'check':name,'exitCode':0} for name in stage.CHECKS]))
        with self.assertRaisesRegex(ValueError,'not bound'):stage.verified_source(self.root,self.evidence)

    def test_drift_during_copy_detected(self):
        checked=stage.snapshot(self.root,'id');real_copy=stage.shutil.copyfile
        def mutate(source,target):
            real_copy(source,target);source.write_text('changed during copy')
        with patch.object(stage.shutil,'copyfile',side_effect=mutate),self.assertRaisesRegex(ValueError,'while copying'):
            stage.copy_verified_source(self.root,self.target,checked)

    def test_new_file_during_copy_detected_by_final_inventory(self):
        checked=stage.snapshot(self.root,'id');real_copy=stage.shutil.copyfile
        def mutate(source,target):
            real_copy(source,target);(self.root/'added.ts').write_text('added');self.paths.append('added.ts')
        with patch.object(stage.shutil,'copyfile',side_effect=mutate),self.assertRaisesRegex(ValueError,'source changed'):
            stage.copy_verified_source(self.root,self.target,checked)

    def test_stale_previously_staged_file_rejected(self):
        checked=stage.snapshot(self.root,'id')
        (self.target/'evidence').mkdir();(self.target/'old.ts').write_text('old')
        (self.target/'evidence/source-copy.json').write_text(json.dumps({'files':[{'file':'old.ts'}]}))
        with self.assertRaisesRegex(ValueError,'Stale prior'):stage.copy_verified_source(self.root,self.target,checked)

    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid source path'):stage.safe_path(self.target,'../outside')


class ApkIntegrity(unittest.TestCase):
    def test_fixture_hash_precedes_rgba_pixel_comparison(self):
        original=io.BytesIO();Image.new('RGB',(2,3),(19,30,41)).save(original,format='PNG')
        encoded=original.getvalue();path=types.SimpleNamespace(read_bytes=lambda:encoded)
        size,pixels=apk['load_fixture_pixels'](path,hashlib.sha256(encoded).hexdigest())
        self.assertEqual(size,(2,3));self.assertEqual(len(pixels),24)
        with self.assertRaisesRegex(ValueError,'differs from the model manifest'):
            apk['load_fixture_pixels'](path,'0'*64)

    def test_required_libraries_include_direct_expo_loads(self):
        self.assertEqual(len(apk['REQUIRED_NATIVE_LIBRARIES']),14)
        for name in ['expo-modules-core','expo-sqlite','BrailleWordGeometry','hermesvm']:
            self.assertIn(name,apk['REQUIRED_NATIVE_LIBRARIES'])


if __name__=='__main__':
    unittest.main(verbosity=2)
