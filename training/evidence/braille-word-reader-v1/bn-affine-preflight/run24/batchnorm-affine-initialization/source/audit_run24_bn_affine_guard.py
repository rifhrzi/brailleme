"""Read-only actual proof/metadata guard audit, no backend/model execution."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from zipfile import ZipFile
import time
import sys
import numpy as np
import h5py

W=Path('.braille-training-work/word-scan-v1-20260905').resolve()
RUN=W/'run24';SOURCE=RUN/'source'
OUT=RUN/'guard-audit-work';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((RUN/'run24-plan.json').read_text())
assert all(sha(SOURCE/name)==h==sha(Path('training/scripts')/name) for name,h in plan['source_sha256'].items())
def extract(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(selected)==len(names)
    return ast.Module(body=selected,type_ignores=[])
assert ast.dump(extract(SOURCE/'word_train.py',['Patches']))==ast.dump(extract(W/'run23/source/word_train.py',['Patches']))
for name in ['word_data.py','word_model.py','word_decode.py','word_sampling.py','word_sources.py','word_point_geometry.py','word_point_metadata.py','word_vertical_stretch.py']:
    assert sha(SOURCE/name)==sha(W/'run23/source'/name)
init_path=Path(plan['bn_affine_initialization_report'])
proof_path=Path(plan['bn_affine_conversion_report'])
initial=json.loads(init_path.read_text());proof=json.loads(proof_path.read_text())
assert sha(init_path)==plan['bn_affine_initialization_report_sha256']
assert sha(proof_path)==plan['bn_affine_conversion_report_sha256']

# Lightweight adapters exercise exact validation/Callback AST against actual
# saved52BN metadata. Real subclass call/grad/update behavior is independently
# proved in the preceding tiny-model audit and is not reimplemented here.
class Parameter:
    def __init__(self,array,trainable):self.array=np.array(array,copy=True);self.shape=self.array.shape;self.trainable=trainable
    def numpy(self):return self.array
class BN:pass
class FixedBN(BN):pass
class Callback:pass
class Model:
    def __init__(self,config,weights):
        self.config=config;self.trainable=True;self.saved=[]
        graph=next(l for l in config['layers'] if l['config']['name']=='mobilenetv2_half')
        self.backbone=SimpleNamespace(name='mobilenetv2_half',trainable=True,layers=[])
        for entry in graph['config']['layers']:
            if entry['class_name']!='FrozenMomentBatchNormalization':continue
            assert entry['registered_name']=='BrailleMe>FrozenMomentBatchNormalization'
            c=entry['config'];layer=FixedBN();layer.name=c['name'];layer.trainable=c['trainable'];layer.center=c['center'];layer.scale=c['scale']
            for i,name in enumerate(['gamma','beta','moving_mean','moving_variance']):setattr(layer,name,Parameter(weights[layer.name][i],i<2))
            self.backbone.layers.append(layer)
        self.layers=[self.backbone]
    def get_layer(self,name):assert name=='mobilenetv2_half';return self.backbone
    def get_config(self):return self.config
    def save(self,path):self.saved.append(str(path));Path(path).write_bytes(b'not-a-model: callback ordering fixture')
with ZipFile(initial['initial_checkpoint']) as z:
    config=json.loads(z.read('config.json'))['config'];weights={}
    with h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h5:
        def collect(name,node):
            if isinstance(node,h5py.Group) and node.attrs.get('name') in initial['batchnorm_affine']['layer_names']:
                weights[node.attrs['name']]=[node[str(i)][:] for i in range(4)]
        h5.visititems(collect)
model=Model(config,weights);assert len(model.backbone.layers)==52
ns=dict(np=np,json=json,hashlib=hashlib,Path=Path,time=time,keras=SimpleNamespace(layers=SimpleNamespace(BatchNormalization=BN),callbacks=SimpleNamespace(Callback=Callback)),
        FrozenMomentBatchNormalization=FixedBN,__file__=str(Path('training/scripts/word_train.py').resolve()),HEIGHT=192,WIDTH=480)
exec(compile(extract(SOURCE/'word_bn_affine.py',['affine_backbone_spec']),str(SOURCE/'word_bn_affine.py'),'exec'),ns)
exec(compile(extract(SOURCE/'word_train.py',['moving_moments_sha256','validated_bn_affine','WordCheckpoint']),str(SOURCE/'word_train.py'),'exec'),ns)
validate=ns['validated_bn_affine'];moments=ns['moving_moments_sha256']
good=validate(model,Path(initial['initial_checkpoint']),init_path,proof_path)
assert good['moving_moments_sha256']==plan['moving_moments_sha256']
assert good['spec']==initial['batchnorm_affine']
assert good['initial_checkpoint_sha256']==plan['parent_checkpoint_sha256']
assert good['base_parent_checkpoint_sha256']==plan['base_parent_checkpoint_sha256']

rejections=[]
def reject(label,call):
    try:call()
    except (ValueError,RuntimeError) as error:rejections.append({'case':label,'reason':str(error)})
    else:raise AssertionError('Accepted '+label)
reject('diagnostic resume',lambda:validate(model,Path(initial['nonzero_diagnostic_checkpoint']),init_path,proof_path))
reject('base parent resume instead of initialized parent',lambda:validate(model,Path(initial['parent_checkpoint']),init_path,proof_path))
reject('missing explicit proof',lambda:validate(model,Path(initial['initial_checkpoint']),None,None))

fixture=OUT/'mutated-proof';fixture.mkdir()
for name in ['bn-affine-six-oracle.npz','initial-conversion-probe.tflite','nonzero-conversion-probe.tflite']:
    shutil.copyfile(proof_path.parent/name,fixture/name)
oracle_original=json.loads((proof_path.parent/'bn-affine-six-oracle.json').read_text())
def mutated(label,change):
    p=copy.deepcopy(proof);o=copy.deepcopy(oracle_original)
    change(p,o)
    oracle=fixture/'bn-affine-six-oracle.json';oracle.write_text(json.dumps(o))
    p['oracle_report_sha256']=sha(oracle)
    conversion=fixture/'conversion.json';conversion.write_text(json.dumps(p))
    reject(label,lambda:validate(model,Path(initial['initial_checkpoint']),init_path,conversion))
mutated('failed conversion',lambda p,o:p.update(status='failed'))
mutated('wrong proof pair',lambda p,o:p.update(initialization_report_sha256='0'*64))
mutated('non-finite numeric summary',lambda p,o:p['probes']['initial'].update(torch_vs_tf_max_abs=float('nan')))
mutated('out-of-tolerance numeric summary',lambda p,o:p['probes']['nonzero'].update(tf_vs_tflite_max_abs=.5))
mutated('zero diagnostic effect',lambda p,o:o.update(nonzero_max_output_delta=0))
mutated('custom operator',lambda p,o:p['probes']['nonzero']['operators'].append('CUSTOM'))
mutated('mismatched raw/effective decisions',lambda p,o:p['probes']['nonzero']['decisions'][0]['raw'].update(text='A'))
mutated('missing full405 trained parity requirement',lambda p,o:p.update(trained_candidate_requires_full405_parity=False))
mutated('wrong output shape',lambda p,o:p['probes']['initial']['output'].update(shape=[1,48,120,10]))
mutated('source snapshot mismatch',lambda p,o:p.update(source_sha256='0'*64))

# Actual helper guard rejects a partially plain/trainable BN population even
# if the graph metadata still carries training=False.
mixed=Model(copy.deepcopy(config),weights)
old=mixed.backbone.layers[0];plain=BN();plain.__dict__.update(old.__dict__);mixed.backbone.layers[0]=plain
reject('plain trainable BN mixed with fixed-moment class',lambda:validate(mixed,Path(initial['initial_checkpoint']),init_path,proof_path))
frozen=Model(copy.deepcopy(config),weights)
for layer in frozen.backbone.layers:
    layer.__class__=BN;layer.trainable=False
assert validate(frozen,None,None,None)=={'enabled':False}

# Execute real callback code with metadata adapter save spy. A corrupted
# moving array aborts before even last.keras; no TensorFlow/Torch invocation.
def callback(output,expected):
    output.mkdir()
    c=ns['WordCheckpoint'].__new__(ns['WordCheckpoint'])
    c.model=model;c.output=output;c.batch_size=24;c.retain_epochs=True;c.images=None;c.expected_bn_moments=expected
    return c
before=moments(model)
passed=callback(OUT/'clean-checkpoint',before)
passed.on_epoch_end(0,{})
assert len(model.saved)==1 and (passed.output/'retained-epochs/epoch-01.keras').exists()
original=model.backbone.layers[0].moving_mean.array.copy()
model.backbone.layers[0].moving_mean.array[0]+=np.float32(.001)
assert moments(model)!=before
bad=callback(OUT/'corrupt-checkpoint',before);saved_count=len(model.saved)
reject('changed moving moments before first checkpoint write',lambda:bad.on_epoch_end(0,{}))
assert len(model.saved)==saved_count and not list(bad.output.iterdir())
model.backbone.layers[0].moving_mean.array[:]=original
assert moments(model)==before
# Gamma/beta are deliberately outside the moving-statistics digest.
model.backbone.layers[0].gamma.array[0]+=np.float32(.001)
assert moments(model)==before
assert all(sha(SOURCE/name)==h==sha(Path('training/scripts')/name) for name,h in plan['source_sha256'].items())
assert all(module not in sys.modules for module in ['torch','tensorflow','keras'])
helper_review=W/'bn-affine-independent-preflight/report.json'
report={'status':'passed','scope':'Independent actual-report and serialized52BN metadata guard proof with exact source AST; no model/backend/inference. Save spy verifies callback ordering, not a real checkpoint serialization.',
 'script_sha256':sha(__file__),'reviewed_plan_sha256':sha(RUN/'run24-plan.json'),'source_sha256':plan['source_sha256'],
 'initialization_report_sha256':sha(init_path),'conversion_report_sha256':sha(proof_path),
 'independent_helper_runtime_audit_sha256':sha(helper_review),'base_parent_checkpoint_sha256':good['base_parent_checkpoint_sha256'],
 'initial_checkpoint_sha256':good['initial_checkpoint_sha256'],'forbidden_diagnostic_sha256':good['nonzero_diagnostic_sha256'],
 'actual52BN_checkpoint_metadata_and_spec_passed':True,'moving_moments_sha256':before,
 'exact_actual_proof_pair_and_initial_resume_passed':True,'default_disabled_behavior_unchanged':True,
 'patches_ast_identical_run23':True,'data_targets_sampler_point_sources_identical_run23':True,
 'plain_trainable_BN_rejected_despite_saved_false':True,'moving_moment_corruption_aborts_before_any_checkpoint_write':True,
 'unchanged_moments_allow_last_and_retained_save':True,'gamma_beta_changes_do_not_change_moment_digest':True,
 'negative_cases':rejections,'defects':[],
 'limits':['Class call/autograd/save-load mechanics are separately proven by the hash-bound real tiny-model audit; this guard proof uses lightweight type/variable adapters initialized from actual saved52BN HDF5 values.','The final protocol will bind this prelaunch audit, so this report binds the reviewed plan rather than creating a circular protocol hash. Actual config/protocol linkage is verified separately after they exist.','Six-input conversion preflight contains no accepted words; trained candidate still requires full405 parity and unchanged acceptance gates.']}
path=RUN/'bn-affine-guard-audit.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(path),'sha256':sha(path),'negative_cases':len(rejections),'moving_moments_sha256':before}))
