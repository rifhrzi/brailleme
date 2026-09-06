"""Independent serialized graph/weights/oracle audit, no ML/backend execution."""
from pathlib import Path
from zipfile import ZipFile
import ast
import copy
import hashlib
import io
import json
import shutil
import sys
import types
import h5py
import numpy as np
from PIL import Image

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
RUN=WORK/'fine-branch-parent-v2'
OUT=WORK/'fine-branch-independent-preflight'
assert not OUT.exists(), 'Never overwrite published evidence'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def link(path):
    return {'file':str(Path(path).relative_to(ROOT)), 'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_bytes())
init=read(RUN/'fine-initialization.json'); proof=read(RUN/'fine-conversion-parity.json')
static_path=ROOT/init['static_audit']; static=read(static_path)
assert init['status']==proof['status']=='passed'
assert proof['initialization_report_sha256']==sha(RUN/'fine-initialization.json')
assert init['static_audit_sha256']==sha(static_path)
assert static['status']=='static_config_and_arithmetic_passed'
for name,digest in init['source_sha256'].items():
    assert sha(RUN/'source'/name)==sha(ROOT/'training/scripts'/name)==digest
for name,digest in proof['source_sha256'].items():
    assert sha(RUN/'conversion-source'/name)==sha(ROOT/'training/scripts'/name)==digest

def load_serial(path):
    with ZipFile(path) as z:
        cfg=json.loads(z.read('config.json'))['config']
        arrays={}
        with h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h5:
            def collect(path,node):
                if path.startswith('layers/') and isinstance(node,h5py.Group) and node.attrs.get('name') and len(node) and all(isinstance(item,h5py.Dataset) for item in node.values()):
                    name=node.attrs['name'];assert name not in arrays
                    arrays[name]=[node[str(i)][()] for i in range(len(node))]
            h5.visititems(collect)
    return cfg,arrays

models={}
for label,key,hkey in [('parent','parent_checkpoint','parent_checkpoint_sha256'),('initial','initial_checkpoint','initial_checkpoint_sha256'),('nonzero','nonzero_diagnostic_checkpoint','nonzero_diagnostic_sha256')]:
    path=ROOT/init[key];assert sha(path)==init[hkey]
    models[label]=load_serial(path)
parent,initial,nonzero=(models[k] for k in ('parent','initial','nonzero'))
pc,pw=parent;ic,iw=initial;dc,dw=nonzero
assert sum(map(len,pw.values()))==280 and sum(map(len,iw.values()))==283
assert set(iw)-set(pw)=={'fpn_2_zero','fpn_2_depthwise'}
assert set(dw)==set(iw)
for name,arrays in pw.items():
    for before,after,diagnostic in zip(arrays,iw[name],dw[name],strict=True):
        assert before.dtype==after.dtype==diagnostic.dtype
        assert before.shape==after.shape==diagnostic.shape
        assert before.tobytes()==after.tobytes()==diagnostic.tobytes(),name
assert all(not np.any(a) for a in iw['fpn_2_zero'])
assert [a.shape for a in iw['fpn_2_zero']]==[(1,1,16,64),(64,)]
assert iw['fpn_2_depthwise'][0].shape==(3,3,16,1) and np.any(iw['fpn_2_depthwise'][0])
assert iw['fpn_2_depthwise'][0].tobytes()==dw['fpn_2_depthwise'][0].tobytes()
assert all(np.isfinite(a).all() and np.any(a) and abs(float(np.max(abs(a)))-.01)<1e-8 for a in dw['fpn_2_zero'])
assert any(np.any(a) for a in pw['fpn_32_zero'])

def strip_ids(value):
    if isinstance(value,dict):return {k:strip_ids(v) for k,v in value.items() if k!='shared_object_id'}
    if isinstance(value,list):return [strip_ids(v) for v in value]
    return value
def layer_map(c):return {l['name']:l for l in c['layers']}
p_layers,i_layers,d_layers=(layer_map(c) for c in (pc,ic,dc))
assert strip_ids(ic)==strip_ids(dc)
assert set(i_layers)-set(p_layers)=={'fpn_2_depthwise','fpn_2_zero','fpn_4_with_2'}
for name,layer in p_layers.items():
    if name=='mobilenetv2_half':
        p_back,i_back=layer['config'],i_layers[name]['config']
        assert strip_ids(p_back['layers'])==strip_ids(i_back['layers'])
        assert p_back['trainable']==i_back['trainable']
        assert i_back['output_layers']==p_back['output_layers']+[['expanded_conv_depthwise_relu',0,0]]
        assert i_layers[name]['inbound_nodes'][0]['kwargs']['training'] is False
        continue
    assert layer['class_name']==i_layers[name]['class_name']
    assert strip_ids(layer['config'])==strip_ids(i_layers[name]['config']),name
    if name!='activation':assert strip_ids(layer.get('inbound_nodes'))==strip_ids(i_layers[name].get('inbound_nodes')),name
bn=[l for l in p_back['layers'] if l['class_name']=='BatchNormalization']
assert len(bn)==52 and all(l['config']['trainable'] is False for l in bn)
def inputs(layer):
    found=[]
    def visit(v):
        if isinstance(v,dict):
            if v.get('class_name')=='__keras_tensor__':found.append(v['config']['keras_history'])
            else:
                for x in v.values():visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    visit(layer['inbound_nodes']);return found
assert inputs(i_layers['fpn_2_depthwise'])==[['mobilenetv2_half',0,4]]
assert inputs(i_layers['fpn_2_zero'])==[['fpn_2_depthwise',0,0]]
assert inputs(i_layers['fpn_4_with_2'])==[['add_1',0,0],['fpn_2_zero',0,0]]
assert inputs(i_layers['activation'])==[['fpn_4_with_2',0,0]]
for name in ('fpn_2_depthwise','fpn_2_zero'):assert i_layers[name]['config']['activation']=='linear'
assert i_layers['fpn_2_depthwise']['config']['strides']==[2,2] and i_layers['fpn_2_depthwise']['config']['padding']=='same'
assert i_layers['fpn_2_zero']['config']['strides']==[1,1]
assert i_layers['camera_normalization']['config']['target_std']==.05
assert i_layers['mobilenet_normalization']['config']['scale']==2 and i_layers['mobilenet_normalization']['config']['offset']==-1
assert ic['output_layers']==pc['output_layers']==['word_cells',0,0]

# Independent integer SAME recurrence. q=0 is the first pixel-index center.
def same(n,k,s):
    out=(n+s-1)//s;total=max((out-1)*s+k-n,0);return out,total//2,total-total//2
axis=[]
for n in (192,480):
    stage1=same(n,3,2);stage2=same(stage1[0],3,1);stage3=same(stage2[0],3,2)
    origin=0;stride=1;support=1
    for k,s,pad in [(3,2,stage1[1]),(3,1,stage2[1]),(3,2,stage3[1])]:
        origin+=((k-1)/2-pad)*stride;support+=(k-1)*stride;stride*=s
    assert stage1[1:]==(0,1) and stage2[1:]==(1,1) and stage3[1:]==(0,1)
    assert (origin,stride,support)==(3,4,11)
    axis.append({'input':n,'new_depthwise_leading_trailing_padding':list(stage3[1:]),'output':stage3[0],'origin':origin,'stride':stride,'local_support':support})
assert init['fine_branch']['tap_pixel_origin']==1 and init['fine_branch']['output_pixel_origin']==3
assert init['fine_branch']['output_local_support']==11 and init['fine_branch']['added_parameters']==1232
assert sum(a.size for arrays in iw.values() for a in arrays)==830539

assert sha(RUN/'initialization-oracle.npz')==init['oracle_sha256']==proof['oracle_sha256']
with np.load(RUN/'initialization-oracle.npz',allow_pickle=False) as saved:
    images=saved['images'];outputs={k:saved[k] for k in ('parent','initial','nonzero')}
assert images.shape==(6,192,480,3) and images.dtype==np.float32
for values in outputs.values():assert values.shape==(6,48,120,11) and values.dtype==np.float32 and np.isfinite(values).all() and values.min()>=0 and values.max()<=1
assert outputs['parent'].tobytes()==outputs['initial'].tobytes()
delta=float(np.max(abs(outputs['initial']-outputs['nonzero'])))
assert delta==init['nonzero_max_output_delta'] and delta>1e-3
train=read(WORK/'data/train.json');assert sha(WORK/'data/train.json')==init['train_manifest_sha256']
for i,row in enumerate(init['train_inputs']):
    assert train[row['index']]['source']==row['source'] and train[row['index']]['image']==row['image']
    path=WORK/'data'/row['image'];assert sha(path)==row['sha256']
    with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'),np.float32)/255
    assert rgb.tobytes()==images[i].tobytes()
assert np.all(images[3]==0) and np.all(images[4]==1)
assert np.array_equal(images[5],np.random.default_rng(20260905).random(images[5].shape,dtype=np.float32))
assert init['zero_branch_gradients']['lateral_l1']>0 and init['zero_branch_gradients']['depthwise_l1']==0
assert init['after_isolated_lateral_step_gradients']['depthwise_l1']>0

# Execute archived decoder and quality functions only, with literal constants.
def constants(path):
    result={}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign):
            try:v=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):result[node.targets[0].id]=v
            elif len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple):result.update(zip((v.id for v in node.targets[0].elts),v))
    return result
data=constants(RUN/'source/word_data.py');data['GRID_W']=120;data['GRID_H']=48
for name,values in [('word_data',data),('benchmark_angelina',constants(RUN/'conversion-source/benchmark_angelina.py'))]:
    m=types.ModuleType(name);m.__dict__.update(values);sys.modules[name]=m
decoder=types.ModuleType('word_decode');exec(compile((RUN/'conversion-source/word_decode.py').read_text(),'archived_word_decode','exec'),decoder.__dict__);sys.modules['word_decode']=decoder
quality_ns={'np':np,'Image':Image}
node=next(n for n in ast.parse((RUN/'conversion-source/word_benchmark.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quality')
exec(compile(ast.Module(body=[node],type_ignores=[]),'archived_quality','exec'),quality_ns)
decisions={}
for name,values in outputs.items():
    rows=[]
    for value,image in zip(values,images):
        raw=decoder.decode(value,proof['thresholds']);quality=quality_ns['quality'](image)
        raw={k:raw[k] for k in ('text','reason')}|{'cell_count':len(raw['cells'])}
        rows.append({'raw':raw,'quality_reason':quality,'effective':raw if not quality else {'text':'','reason':quality,'cell_count':0}})
    decisions[name]=rows
assert decisions['parent']==proof['torch_parent_decisions']
for name in ('initial','nonzero'):
    probe=proof['probes'][name]
    assert decisions[name]==probe['decisions']
    assert probe['builtins_only'] and not any('CUSTOM' in op or 'Flex' in op for op in probe['operators'])
    assert sha(RUN/f'{name}-conversion-probe.tflite')==probe['artifact_sha256']
    for key in ('torch_vs_tf_max_abs','tf_vs_tflite_max_abs','torch_vs_tflite_max_abs','compiled_vs_eager_max_abs'):assert np.isfinite(probe[key]) and 0<=probe[key]<=2e-4
assert proof['nonzero_tf_output_delta']>1e-3 and proof['nonzero_tflite_output_delta']>1e-3

OUT.mkdir();(OUT/'source').mkdir()
sources=[Path(__file__),*sorted((RUN/'source').glob('*.py')),*sorted((RUN/'conversion-source').glob('*.py'))]
retained=[];seen=set()
for path in sources:
    if path.name in seen:continue
    seen.add(path.name);target=OUT/'source'/path.name;shutil.copyfile(path,target);assert sha(target)==sha(path);retained.append(link(target))
report={'status':'passed','scope':'Independent frozen-source, serialized graph/HDF5 arrays and saved six-input oracle/decoder/quality audit. No graph/backend/model inference or gradients rerun.',
        'inputs':[link(RUN/'fine-initialization.json'),link(RUN/'fine-conversion-parity.json'),link(static_path)],'script_sha256':sha(__file__),'sources':retained,
        'base_parent_checkpoint_sha256':init['parent_checkpoint_sha256'],'initial_checkpoint_sha256':init['initial_checkpoint_sha256'],'forbidden_diagnostic_sha256':init['nonzero_diagnostic_sha256'],
        'initialization_report_sha256':sha(RUN/'fine-initialization.json'),'conversion_report_sha256':sha(RUN/'fine-conversion-parity.json'),'oracle_sha256':init['oracle_sha256'],
        'inherited_arrays_byte_identical':280,'total_arrays':283,'changed_diagnostic_arrays':['fpn_2_zero/kernel','fpn_2_zero/bias'],
        'zero_lateral_all_zero':True,'random_depthwise_nonzero_and_diagnostic_unchanged':True,'inherited_fulltail_lateral_nonzero_preserved':True,
        'ordinary_frozen_BN_layers':52,'backbone_and_layer_trainability_unchanged':True,'inherited_normalization_unchanged':True,
        'serialized_branch_has_no_relu_between_zero_and_add':True,'same_padding_axes':axis,'initial_outputs_byte_identical_parent':6,
        'nonzero_max_output_delta':delta,'exact_saved_input_provenance_passed':True,'raw_and_quality_decisions_recomputed_equal':True,
        'gradient_source_review':'Actual source computes real TRAIN loss backward at zero, requires nonzero new-lateral gradient and zero depthwise gradient; changes only new lateral, then requires positive depthwise gradient. Serialized diagnostic kernel and bias both change by nonzero max.01 while every other array remains identical. Backend gradient computation itself was not repeated by this reviewer.',
        'zero_gradient_report':init['zero_branch_gradients'],'after_lateral_step_gradient_report':init['after_isolated_lateral_step_gradients'],
        'defects':[],'limits':['Convolution origin/support describes normalized-tensor coordinates; global CameraNormalization prevents a claim of an11pixel raw-RGB receptive field.','All six preflight inputs reject; numeric conversion is not recognition accuracy. Full405 trained parity and unchanged gates remain required.','Training guards and final protocol are audited separately after trainer integration.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':str(OUT/'report.json'),'sha256':sha(OUT/'report.json'),'status':'passed'}))
