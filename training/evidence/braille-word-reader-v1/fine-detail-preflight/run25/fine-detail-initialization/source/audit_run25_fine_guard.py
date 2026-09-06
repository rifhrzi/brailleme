"""Independent CPU-only source/actual-proof guard audit; no ML backend."""
import argparse
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

W=Path(__file__).resolve().parent; ROOT=W.parents[1]; RUN=W/'run25'; SOURCE=RUN/'source'
assert not (RUN/'fine-branch-guard-audit.json').exists(), 'Published audit is immutable'
OUT=RUN/'fine-guard-audit-work'; OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
plan=read(RUN/'run25-plan.json')
assert len(plan['source_sha256'])==21
assert all(sha(SOURCE/n)==h==sha(ROOT/'training/scripts'/n) for n,h in plan['source_sha256'].items())
def extract(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    assert len(nodes)==1
    return ast.Module(body=nodes,type_ignores=[])
assert ast.dump(extract(SOURCE/'word_train.py','Patches'))==ast.dump(extract(W/'run24/source/word_train.py','Patches'))
unchanged=['word_data.py','word_model.py','word_decode.py','word_sampling.py','word_sources.py','word_point_geometry.py','word_point_metadata.py','word_vertical_stretch.py']
assert all(sha(SOURCE/n)==sha(W/'run24/source'/n) for n in unchanged)
sys.path.insert(0,str(ROOT/'training/scripts'))
from word_fine_validation import require_fine_reports, validated_fine_branch
initpath=Path(plan['fine_initialization_report']); proofpath=Path(plan['fine_conversion_report'])
assert sha(initpath)==plan['fine_initialization_report_sha256'] and sha(proofpath)==plan['fine_conversion_report_sha256']
initial,proof=require_fine_reports(Path(plan['parent_checkpoint']),initpath,proofpath)
assert initial['parent_checkpoint_sha256']==plan['base_parent_checkpoint_sha256']
assert initial['nonzero_diagnostic_sha256']==plan['nonzero_diagnostic_sha256']
rejects=[]
def reject(label,call):
    try:call()
    except (ValueError,AssertionError,SystemExit) as e:rejects.append({'case':label,'reason':str(e)})
    else:raise AssertionError('Unexpected acceptance: '+label)
reject('diagnostic cannot resume',lambda:require_fine_reports(Path(initial['nonzero_diagnostic_checkpoint']),initpath,proofpath))
reject('unextended base cannot resume',lambda:require_fine_reports(Path(initial['parent_checkpoint']),initpath,proofpath))
class Plain: layers=[]
class Detail: layers=[type('L',(),{'name':'fpn_2_zero'})()]
assert validated_fine_branch(Plain(),None,None,None)=={'enabled':False}
reject('detail without explicit proof',lambda:validated_fine_branch(Detail(),None,None,None))
for name in ('initial-conversion-probe.tflite','nonzero-conversion-probe.tflite'):
    shutil.copyfile(proofpath.parent/name,OUT/name)
changes=[('wrong report pair',lambda p:p.update(initialization_report_sha256='0'*64)),
 ('zero diagnostic effect',lambda p:p.update(nonzero_tflite_output_delta=0)),
 ('nonfinite effect',lambda p:p.update(nonzero_tf_output_delta=float('nan'))),
 ('inconsistent numeric pass',lambda p:p['probes']['nonzero'].update(torch_vs_tflite_max_abs=.01)),
 ('custom operator',lambda p:p['probes']['initial'].update(operators=['CUSTOM'])),
 ('missing executed dependency',lambda p:p['source_sha256'].pop('word_bn_affine_parity.py')),
 ('altered output contract',lambda p:p['probes']['nonzero']['output'].update(shape=[1,48,120,10])),
 ('changed saved decisions',lambda p:p['probes']['nonzero']['decisions'][0]['raw'].update(reason='tampered')),
 ('missing full405 obligation',lambda p:p.update(trained_candidate_requires_full405_parity=False))]
for label,change in changes:
    p=copy.deepcopy(proof);change(p); path=OUT/'proof.json';path.write_text(json.dumps(p))
    reject(label,lambda:require_fine_reports(Path(plan['parent_checkpoint']),initpath,path))
ns={'argparse':argparse,'Path':Path,'__doc__':'independent CLI audit'}
exec(compile(extract(SOURCE/'word_train.py','parse_args'),str(SOURCE/'word_train.py'),'exec'),ns)
cli=['--data','unused','--output','unused','--geometry-loss','relative_size_huber','--resume',str(plan['parent_checkpoint']),'--fine-branch-initialization',str(initpath),'--fine-branch-conversion',str(proofpath)]
assert ns['parse_args'](cli).fine_branch_initialization==initpath
with contextlib.redirect_stderr(io.StringIO()):
    for extra in [['--bn-affine-initialization','i','--bn-affine-conversion','p'],['--geometry-head-initialization','i'],['--image-normalization','channel_gray_std05']]:
        reject('CLI incompatible '+extra[0],lambda extra=extra:ns['parse_args'](cli+extra))
    reject('CLI unpaired proof',lambda:ns['parse_args'](cli[:-2]))
actualpath=W/'fine-branch-parent-v2/trainer-validation-preflight-v2.json'; actual=read(actualpath)
assert actual['status']=='passed' and actual['fine_detail_branch']['initial_checkpoint_sha256']==plan['parent_checkpoint_sha256']
assert actual['fine_detail_branch']['spec']==initial['fine_branch']
assert not any(name in sys.modules for name in ('keras','torch','tensorflow'))
script=OUT/Path(__file__).name;shutil.copyfile(__file__,script)
report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Executed actual source-only report validator, actual six saved outputs/decoder decisions, negative proof/CLI cases. No ML/backend/model inference or optimizer.',
 'plan_sha256':sha(RUN/'run25-plan.json'),'source_sha256':plan['source_sha256'],
 'script_sha256':sha(script),'script':str(script.relative_to(ROOT)),
 'initialization_report_sha256':sha(initpath),'conversion_report_sha256':sha(proofpath),
 'base_parent_checkpoint_sha256':plan['base_parent_checkpoint_sha256'],'initial_checkpoint_sha256':plan['parent_checkpoint_sha256'],
 'forbidden_diagnostic_sha256':plan['nonzero_diagnostic_sha256'],
 'actual_report_guard_passed':True,'default_off_returns_disabled':True,'default_Patches_AST_equal_run24':True,
 'unchanged_data_sampling_sources':unchanged,'negative_cases':rejects,
 'actual_model_guard_evidence':{'file':str(actualpath.relative_to(ROOT)),'sha256':sha(actualpath)},
 'actual_model_guard_scope':'Trainer previously loaded exact initial checkpoint and checked actual 52 ordinary frozen BN, normalization, all three new weight shapes/classes/topology. This reviewer checked that source and evidence without loading Keras again.',
 'model_guard_source_review':{'exact_52_plain_frozen_BN':True,'initial_zero_lateral_nonzero_depthwise':True,'linear_depthwise_lateral_Add_before_original_activation':True,'exclusive_other_initializations':True,'guard_before_compile_fit':True},
 'defects':[], 'limits':['Static coordinate status and HDF5 graph/arrays independently checked by preceding preflight; report guard binds that immutable static hash.','Frozen ordinary BN uses Keras trainable=False; unlike affine branch no per-epoch extra moving-stat callback is needed.','Training config/protocol do not yet exist and are bound by archival verifier after launch.']}
path=RUN/'fine-branch-guard-audit.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'rejected_cases':len(rejects),'report_sha256':sha(path)}))
