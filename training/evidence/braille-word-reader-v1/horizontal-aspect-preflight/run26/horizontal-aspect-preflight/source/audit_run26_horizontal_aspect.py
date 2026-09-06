"""Frozen run26 integration/CLI review; only AST + NumPy/Pillow, no ML."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;ROOT=W.parents[1];RUN=W/'run26';SRC=RUN/'source'
REPORT=RUN/'horizontal-aspect-integration-audit.json'
assert not REPORT.exists(), 'Published audit must not be overwritten'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
plan=read(RUN/'run26-plan.json');assert len(plan['source_sha256'])==22
assert all(sha(SRC/n)==h==sha(ROOT/'training/scripts'/n) for n,h in plan['source_sha256'].items())
def nodes(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if getattr(n,'name',None)==name]
    assert len(selected)==1
    return ast.Module(body=selected,type_ignores=[])
private=W/'horizontal-aspect-prototype-v1/source-private'
assert ast.dump(nodes(SRC/'word_train.py','Patches'))==ast.dump(nodes(private/'word_train.py','Patches'))
assert ast.dump(nodes(SRC/'word_horizontal_aspect.py','horizontal_aspect_factors'))==ast.dump(nodes(private/'word_horizontal_aspect.py','horizontal_aspect_factors'))
unchanged=['word_model.py','word_decode.py','word_sampling.py','word_sources.py','word_point_geometry.py','word_point_metadata.py']
assert all(sha(SRC/n)==sha(W/'run25/source'/n) for n in unchanged)
current=ast.parse((SRC/'word_data.py').read_text());prototype=ast.parse((private/'word_data.py').read_text())
aug=next(n for n in current.body if getattr(n,'name',None)=='augment_patch')
guards=[n for n in aug.body if isinstance(n,ast.If) and any(isinstance(v,ast.Constant) and v.value=='Horizontal aspect is a separate illumination-only geometry ablation' for v in ast.walk(n))]
assert len(guards)==1;aug.body.remove(guards[0])
assert ast.dump(current)==ast.dump(prototype), 'Only API profile guard may differ from audited private raster implementation'
spec=importlib.util.spec_from_file_location('actual_aspect_test',W/'horizontal-aspect-prototype-v1/test_private_aspect.py')
tests=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tests;spec.loader.exec_module(tests)
tests.horizontal=tests.load_module('actual_horizontal_stream',SRC/'word_horizontal_aspect.py')
tests.NEW=tests.load_data(SRC);tests.NewPatches,tests.NEW_PATCH=tests.load_patches(SRC,tests.NEW)
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(tests.AspectTests))
assert result.wasSuccessful(),log.getvalue()
ns={'argparse':argparse,'Path':Path,'__doc__':'bounded actual CLI audit'}
exec(compile(nodes(SRC/'word_train.py','parse_args'),str(SRC/'word_train.py'),'exec'),ns)
base=['--data','unused','--output','unused','--resume',plan['parent_checkpoint'],'--geometry-loss','relative_size_huber']
valid=base+['--illumination-augmentation','--horizontal-aspect-augmentation']
assert ns['parse_args'](valid).horizontal_aspect_augmentation
assert not ns['parse_args'](base).horizontal_aspect_augmentation
rejected=[]
def reject(name,call):
    try:call()
    except (ValueError,SystemExit) as e:rejected.append({'case':name,'error':str(e)})
    else:raise AssertionError('Unexpectedly accepted '+name)
with contextlib.redirect_stderr(io.StringIO()):
    reject('aspect without illumination',lambda:ns['parse_args'](base+['--horizontal-aspect-augmentation']))
    for extra in (['--camera-augmentation'],['--polarity-augmentation'],['--half-turn-augmentation'],['--horizontal-mirror-augmentation'],['--vertical-stretch-augmentation'],['--image-normalization','channel_gray_std05'],['--geometry-head-initialization','unused'],['--bn-affine-initialization','unused','--bn-affine-conversion','unused'],['--fine-branch-initialization','unused','--fine-branch-conversion','unused']):
        reject('CLI '+extra[0],lambda extra=extra:ns['parse_args'](valid+extra))
for options in ({},{'illumination_augmentation':True,'camera_augmentation':True},{'illumination_augmentation':True,'polarity_augmentation':True},{'illumination_augmentation':True,'vertical_stretch':1.1}):
    reject('direct API '+str(options),lambda options=options:tests.NEW['augment_patch'](tests.source_image(),tests.np.array([[215.,55.,265.,135.]]),tests.SpyRNG(1),horizontal_aspect=.55,**options))
summary=read(RUN/'preflight-tests.json');assert summary['status']=='passed' and summary['test_sources_unchanged_before_after']
assert sha(RUN/'preflight-tests.log')==summary['log_sha256']
assert 'Ran 141 tests' in (RUN/'preflight-tests.log').read_text()
assert all(sha(RUN/'test-source'/n)==h==sha(ROOT/'training/tests'/n) for n,h in summary['tests_sha256'].items())
assert sha(plan['parent_checkpoint'])==plan['parent_checkpoint_sha256']
assert plan['horizontal_aspect']['range']==[.55,1.25] and plan['horizontal_aspect']['stream_id']==0x48415350
assert plan['horizontal_aspect']['seed_sequence']==['seed','dataset_epoch','expanded_record_index',0x48415350]
assert plan['illumination_augmentation'] and plan['horizontal_aspect_augmentation'] and plan['dsbi_visible_partial_supervision']
for key in ('camera_augmentation','polarity_augmentation','half_turn_augmentation','horizontal_mirror_augmentation','vertical_stretch_augmentation','partial_geometry_supervision','geometry_conditioned_residual','batchnorm_affine','fine_detail_branch'):assert plan[key] is False
assert all(sha(SRC/n)==h==sha(ROOT/'training/scripts'/n) for n,h in plan['source_sha256'].items())
assert not any(n in sys.modules for n in ('torch','tensorflow','keras'))
out=RUN/'horizontal-aspect-audit-source';out.mkdir(exist_ok=True)
script=out/Path(__file__).name;shutil.copyfile(__file__,script)
testlog=out/'tests.log';testlog.write_text(log.getvalue(),encoding='utf-8')
report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Prelaunch final integration; actual copied Patches and parse_args via AST, NumPy/Pillow only; no model or corpus inference',
 'source_sha256':plan['source_sha256'],'script_sha256':sha(script),'script_path':str(script),'tests_log_sha256':sha(testlog),
 'plan_sha256':sha(RUN/'run26-plan.json'),'parent_checkpoint_sha256':plan['parent_checkpoint_sha256'],
 'prototype_report_sha256':sha(W/'horizontal-aspect-prototype-v1/proof-report.json'),
 'independent_prototype_review_sha256':sha(W/'horizontal-aspect-independent-review-v1/report.json'),
 'actual_cpu_tests':result.testsRun,'cli_and_direct_api_rejections':rejected,
 'patches_private_ast_exact':True,'rng_private_ast_exact':True,'raster_private_ast_exact_except_profile_guard':True,
 'all63_masks_actual_patches_off_equal':True,'actual_single_affine_and_same_legacy_rng':True,
 'known_points_all_stage_survival_and_partial_unknown':True,'unknown_not_positive':True,
 'full_word_suite':{'tests':141,'summary_sha256':sha(RUN/'preflight-tests.json'),'log_sha256':summary['log_sha256'],'test_sources_verified':len(summary['tests_sha256'])},
 'active_sources_unchanged':True,'backend_modules_imported':[],
 'limitations':['No model accuracy, optimizer or physical visibility claim.','Config/protocol do not exist yet; this report binds immutable draft plan/source and later release verification must bind actual config/protocol.',
                'Existing zoom makes effective x scale 0.44..2.125; only relative width factor is historical0.55..1.25.','Patches class is executed with a PyDataset stub; CLI/direct augmentation guards are tested, not a native Keras worker.']}
REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(REPORT),'sha256':sha(REPORT),'cpu_tests':result.testsRun,'rejects':len(rejected)}))
