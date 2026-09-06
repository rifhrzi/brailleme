"""Read-only geometry-head source/ZIP/HDF5/NumPy preflight, no ML imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import h5py
import numpy as np

WORK=Path(__file__).resolve().parent
REPO=WORK.parent.parent
DIR=WORK/'geometry-head-parent-v1'
HEAD=REPO/'training/scripts/word_geometry_head.py'
TEST=REPO/'training/tests/test_word_geometry_head.py'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
assert sha(HEAD)=='f46c8f90e7f4fc9404eed2dc27fbde6e5c823d25d4f88e4127c7e56ed6e8764e'
assert sha(TEST)=='ecf8a9818959d37cd1ddcd4f4de08a034cb87d930d4132d4919e6b747011d454'
init=read(DIR/'geometry-initialization.json')
for name,digest in init['source_sha256'].items():assert sha(DIR/'source'/name)==digest
parent_path=REPO/init['parent_checkpoint'];zero_path=REPO/init['zero_checkpoint']
assert sha(parent_path)==init['parent_checkpoint_sha256']
assert sha(zero_path)==init['zero_checkpoint_sha256']

def saved(path):
    with zipfile.ZipFile(path) as archive:
        configuration=json.loads(archive.read('config.json'));weights={}
        with h5py.File(io.BytesIO(archive.read('model.weights.h5'))) as h5:
            def visit(name,node):
                if name.startswith('layers/') and name.rsplit('/',1)[-1]=='vars' and isinstance(node,h5py.Group) and len(node):
                    label=node.attrs['name'];assert label not in weights
                    weights[label]=[node[key][()] for key in sorted(node,key=int)]
            h5.visititems(visit)
    return configuration,weights

before,old_weights=saved(parent_path);after,new_weights=saved(zero_path)
old_layers={layer['name']:layer for layer in before['config']['layers']}
new_layers={layer['name']:layer for layer in after['config']['layers']}
assert len(new_layers)==len(after['config']['layers'])
for name,values in old_weights.items():
    assert len(values)==len(new_weights[name])
    for old,new in zip(values,new_weights[name]):np.testing.assert_array_equal(old,new)
added=set(new_weights)-set(old_weights)
assert added=={'geometry_feature_projection','geometry_feature_mlp','geometry_residual_zero'}
assert sum(v.size for name in added for v in new_weights[name])==2451
assert all(np.count_nonzero(v)==0 for v in new_weights['geometry_residual_zero'])
for name in ('cell_probability','cell_geometry','dot_probabilities'):
    prior=copy.deepcopy(old_layers[name]['config']);current=new_layers[name]['config']
    assert prior['activation']=='sigmoid' and current['activation']=='linear'
    prior['activation']='linear';assert prior==current
    assert len(new_layers[name]['inbound_nodes'])==1
    history=new_layers[name]['inbound_nodes'][0]['args'][0]['config']['keras_history']
    assert history[0]=='reader_features_1'
for name in old_layers:
    if name in ('cell_probability','cell_geometry','dot_probabilities','word_cells'):continue
    assert old_layers[name]==new_layers[name],name
assert new_layers['word_cells']['class_name']=='Activation' and new_layers['word_cells']['config']['activation']=='sigmoid'
assert after['config']['input_layers']==before['config']['input_layers']==['rgb_0_to_1',0,0]
assert after['config']['output_layers']==['word_cells',0,0]
assert new_layers['rgb_0_to_1']['config']['batch_shape']==[None,192,480,3]
word_input=new_layers['word_cells']['inbound_nodes'][0]['args'][0]['config']
assert word_input['shape']==[None,48,120,11]
bn=[layer for layer in new_layers['mobilenetv2_half']['config']['layers'] if layer['class_name']=='BatchNormalization']
assert bn and all(not layer['config']['trainable'] for layer in bn)
assert new_layers['mobilenetv2_half']['inbound_nodes'][0]['kwargs']['training'] is False
assert new_layers['camera_normalization']==old_layers['camera_normalization']
assert new_layers['mobilenet_normalization']==old_layers['mobilenet_normalization']

# Exact call arithmetic, substituting NumPy operations only. This checks shape,
# slot order and batching separately from Keras/TFLite execution handled by trainer.
tree=ast.parse(HEAD.read_text())
bilinear=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='bilinear_feature_samples')
klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GeometrySlotSampler')
call=copy.deepcopy(next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='call'));call.name='sample_slots'
ops=SimpleNamespace(shape=lambda x:x.shape,clip=np.clip,floor=np.floor,minimum=np.minimum,expand_dims=np.expand_dims,
                    reshape=np.reshape,cast=lambda x,dtype:x.astype(dtype),take_along_axis=np.take_along_axis,
                    convert_to_tensor=lambda x,dtype=None:np.asarray(x,dtype=dtype),arange=np.arange)
module=ast.fix_missing_locations(ast.Module(body=[bilinear,call],type_ignores=[]));namespace={'ops':ops}
exec(compile(module,str(HEAD)+'::numpy-arithmetic','exec'),namespace)
priors=np.asarray(init['sampling']['slot_fractions'],np.float32)
sampler=SimpleNamespace(slot_fractions=priors.tolist(),feature_origin=3.,stride=4)
rng=np.random.default_rng(37)
B,H,W,C=3,5,7,3
features=rng.random((B,H,W,C),dtype=np.float32)+np.arange(B,dtype=np.float32)[:,None,None,None]*10
probabilities=rng.random((B,H,W,11),dtype=np.float32)
actual=namespace['sample_slots'](sampler,[features,probabilities]).reshape(B,H,W,6,C)
expected=np.empty_like(actual);outside=0
for b in range(B):
    for gy in range(H):
        for gx in range(W):
            p=probabilities[b,gy,gx].astype(float)
            for slot,(fx,fy) in enumerate(priors.astype(float)):
                # Independent corner equation: q=box_corner+fraction*size-.5.
                width=p[3]*W*4;height=p[4]*H*4
                bx=(gx+p[1])*4-width/2;by=(gy+p[2])*4-height/2
                qx=bx+fx*width-.5;qy=by+fy*height-.5
                ix=(qx-3)/4;iy=(qy-3)/4
                outside+=ix<0 or ix>W-1 or iy<0 or iy>H-1
                x=float(np.clip(ix,0,W-1));y=float(np.clip(iy,0,H-1))
                x0=int(np.floor(x));y0=int(np.floor(y));x1=min(x0+1,W-1);y1=min(y0+1,H-1)
                expected[b,gy,gx,slot]=((1-(x-x0))*(1-(y-y0))*features[b,y0,x0]+(x-x0)*(1-(y-y0))*features[b,y0,x1]
                                             +(1-(x-x0))*(y-y0)*features[b,y1,x0]+(x-x0)*(y-y0)*features[b,y1,x1])
error=float(np.max(np.abs(actual-expected)))
np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=0)
assert outside>0
for b in range(B):
    single=namespace['sample_slots'](sampler,[features[b:b+1],probabilities[b:b+1]])[0].reshape(H,W,6,C)
    np.testing.assert_array_equal(single,actual[b])

oracle=DIR/'initialization-oracle.npz'
assert sha(oracle)==init['oracle_sha256']
with np.load(oracle) as arrays:
    parent_outputs=arrays['parent_outputs'];zero_outputs=arrays['zero_outputs'];nonzero_outputs=arrays['nonzero_outputs']
    assert parent_outputs.shape==zero_outputs.shape==nonzero_outputs.shape==(6,48,120,11)
    np.testing.assert_array_equal(parent_outputs,zero_outputs)
    nonzero_difference=float(np.max(np.abs(nonzero_outputs-parent_outputs)))
    assert nonzero_difference>0
report={'status':'passed','scope':'Independent source and ZIP/HDF5 checks plus NumPy execution of sampling arithmetic against a scalar oracle and stored six-input output equality. No model/backend execution.',
 'source_sha256':sha(HEAD),'test_source_sha256':sha(TEST),'initialization_report_sha256':sha(DIR/'geometry-initialization.json'),
 'parent_checkpoint_sha256':sha(parent_path),'zero_checkpoint_sha256':sha(zero_path),'oracle_sha256':sha(oracle),'script_sha256':sha(__file__),
 'all_inherited_weight_layers_equal':len(old_weights),'all_inherited_arrays_equal':sum(len(values) for values in old_weights.values()),
 'added_weight_layers':sorted(added),'added_parameters':2451,'final_projection_all_zero':True,
 'inherited_head_filters':[new_layers[name]['config']['filters'] for name in ('cell_probability','cell_geometry','dot_probabilities')],
 'head_kernel_duplication':False,'inverse_sigmoid_reconstruction':False,'frozen_batchnorm_layers':len(bn),'backbone_called_training_false':True,'unchanged_old_normalization_and_feature_graph':True,
 'input_shape':[None,192,480,3],'output_shape':[None,48,120,11],
 'multi_batch_scalar_oracle':{'batch':B,'grid':[H,W],'channels':C,'slots':6,'comparisons':int(actual.size),'max_absolute_error':error,'out_of_range_slots_clamped':int(outside),'batched_equals_separate_calls':True},
 'stored_zero_oracle_equal_inputs':6,'stored_nonzero_max_delta':nonzero_difference,
 'defects':[],
 'limits':['NumPy arithmetic proof does not execute Keras/Torch/TF; trainer owns actual backend gradients and nonzero TFLite conversion proof.',
           'Border semantics replicate the nearest feature sample by clamping coordinates, rather than zero padding or excluding a slot. Out-of-range coordinate gradients are zero.',
           'Origin3 is nominal fine-grid indexing, not an exact receptive-field center for fused coarse features.',
           'The sampler uses approximate DSBI TRAIN-derived priors; no exact Angelina keypoint or accuracy claim.',
           'Only the zero checkpoint is eligible as an initialization parent; the separately updated nonzero probe must not enter training ancestry.'],
 'ml_modules_imported':[name for name in ('keras','torch','tensorflow') if name in sys.modules]}
path=WORK/'independent-geometry-head-audit.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2));print('report_sha256',sha(path))
