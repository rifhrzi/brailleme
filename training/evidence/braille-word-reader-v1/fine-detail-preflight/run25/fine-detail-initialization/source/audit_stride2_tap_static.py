"""Static JSON/AST audit of a proposed stride2 skip; no ML/backend imports."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone

ROOT=Path.cwd().resolve()
WORK=ROOT/'.braille-training-work/word-scan-v1-20260905'
PARENT=WORK/'selected-v15/best-word.keras'
OUT=WORK/'stride2-tap-independent-static-audit'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(data):return hashlib.sha256(data).hexdigest()
assert not OUT.exists()
parent_hash=sha(PARENT)
assert parent_hash=='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
with zipfile.ZipFile(PARENT) as archive:
    config_bytes=archive.read('config.json')
    config=json.loads(config_bytes)
top={layer['config']['name']:layer for layer in config['config']['layers']}
backbone=top['mobilenetv2_half']
layers={layer['config']['name']:layer for layer in backbone['config']['layers']}
keras=Path('C:/bm-envs/word-scan-torch-only/Lib/site-packages/keras/src')
padding_source=keras/'backend/torch/nn.py'
padding_code=padding_source.read_text(encoding='utf-8')
node=next(node for node in ast.parse(padding_code).body if isinstance(node,ast.FunctionDef) and node.name=='_compute_padding_length')
padding_ast=ast.Module(body=[node],type_ignores=[])
namespace={}
exec(compile(padding_ast,str(padding_source),'exec'),namespace)
same=namespace['_compute_padding_length']
assert same(96,3,2)==same(240,3,2)==(0,1)

def inputs(layer):
    result=[]
    def visit(item):
        if isinstance(item,dict):
            if item.get('class_name')=='__keras_tensor__':
                result.append(item['config'])
            else:
                for value in item.values():visit(value)
        elif isinstance(item,list):
            for value in item:visit(value)
    for node in layer.get('inbound_nodes',[]):visit(node['args'])
    return result

states={}
for name,layer in layers.items():
    c=layer['config']; kind=layer['class_name']; ins=inputs(layer)
    if kind=='InputLayer':
        states[name]=dict(shape=c['batch_shape'][1:],jump=1,origin=0,rf=1)
        continue
    prior=[states[item['keras_history'][0]] for item in ins]
    assert prior
    for item,state in zip(ins,prior):assert item['shape'][1:]==state['shape'],(name,item,state)
    state=dict(prior[0]); h,w,channels=state['shape']
    if kind in ('Conv2D','DepthwiseConv2D'):
        ky,kx=c['kernel_size']; sy,sx=c['strides']; dy,dx=c['dilation_rate']
        assert ky==kx and sy==sx and dy==dx
        ph,pw=same(h,ky,sy,dy) if c['padding']=='same' else (0,0),same(w,kx,sx,dx) if c['padding']=='same' else (0,0)
        assert ph==pw
        kernel=(ky-1)*dy+1
        state['shape']=[(h+sum(ph)-kernel)//sy+1,(w+sum(pw)-kernel)//sx+1,c['filters'] if kind=='Conv2D' else channels*c['depth_multiplier']]
        state['origin']+=(.5*(kernel-1)-ph[0])*state['jump']
        state['rf']+=(kernel-1)*state['jump']
        state['jump']*=sy
    elif kind=='ZeroPadding2D':
        py,px=c['padding']; assert py==px
        state['shape']=[h+sum(py),w+sum(px),channels]
        state['origin']-=py[0]*state['jump']
    elif kind=='Add':
        assert all(p['shape']==state['shape'] and p['jump']==state['jump'] and p['origin']==state['origin'] for p in prior)
        state['rf']=max(p['rf'] for p in prior)
    else:
        assert kind in ('BatchNormalization','ReLU'),kind
    states[name]=state
    if name=='block_3_expand_relu':break
tap=states['expanded_conv_depthwise_relu']
low=states['block_3_expand_relu']
assert tap==dict(shape=[96,240,16],jump=2,origin=1.,rf=7)
assert low==dict(shape=[48,120,96],jump=4,origin=3.,rf=19)
assert all(layer['class_name']!='FixedMomentsBatchNormalization' for layer in layers.values())
bn=[layer for layer in layers.values() if layer['class_name']=='BatchNormalization']
assert len(bn)==52 and all(layer['config']['trainable'] is False for layer in bn)
new=dict(shape=[48,120,16],jump=4,origin=tap['origin']+(1-same(96,3,2)[0])*tap['jump'],rf=tap['rf']+2*tap['jump'])
assert new==dict(shape=[48,120,16],jump=4,origin=3.,rf=11)
assert top['fpn_4']['build_config']['input_shape']==[None,48,120,96]
assert [i['keras_history'][0] for i in inputs(top['add_1'])]==['fpn_4','up_sampling2d_1']
assert [i['keras_history'][0] for i in inputs(top['activation'])]==['add_1']
assert top['activation']['config']['activation']=='relu'
for name in ('reader_features_0','reader_features_1'):
    c=top[name]['config']
    assert c['kernel_size']==[3,3] and c['strides']==[1,1] and c['padding']=='same'
assert config['config']['output_layers']==['word_cells',0,0]
assert top['word_cells']['build_config']['input_shape']==[[None,48,120,1],[None,48,120,4],[None,48,120,6]]
pixels=48*120
parameters=dict(depthwise_kernel=3*3*16,depthwise_bias=0,pointwise_kernel=16*64,pointwise_bias=64)
macs=dict(depthwise=pixels*3*3*16,pointwise=pixels*16*64)
assert sum(parameters.values())==1232 and sum(macs.values())==6727680
assert not any(name in sys.modules for name in ('numpy','h5py','keras','torch','tensorflow'))
assert sha(PARENT)==parent_hash
report=dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),status='static_config_and_arithmetic_passed',
    scope='Read-only serialized checkpoint JSON and installed Keras pure padding-function AST. No graph creation, backend import, tensors, inference, training or accuracy claim.',
    parent=dict(path=str(PARENT),sha256=parent_hash,config_sha256=digest(config_bytes),name=config['config']['name'],plain_batchnorm_layers=52,all_batchnorm_frozen=True),
    coordinate_convention='Input pixel-index centers begin q=0. Origins are nominal convolution support centers; edge padding truncates real-pixel support. With pixel-edge centers q=0.5, add0.5 to every reported origin.',
    receptive_field_recurrence='jump_out=jump_in*stride; RF_out=RF_in+(effective_kernel-1)*jump_in; origin_out=origin_in+((effective_kernel-1)/2-pad_before)*jump_in.',
    actual_parent_states=states,proposed_depthwise=new,
    proposed_projection=dict(shape=[48,120,64],jump=4,origin=3,rf=11,kernel_initializer='zeros',bias_initializer='zeros'),
    same_padding=dict(input_shape=[96,240],kernel=[3,3],strides=[2,2],top=0,bottom=1,left=0,right=1,verified_installed_keras_pure_function=True),
    insertion=dict(existing_merge='add_1([fpn_4,up_sampling2d_1])',existing_activation='activation(relu)',
        proposed='Add projected stride2 residual to the unchanged result of add_1, immediately before activation. Keep original add_1 computation order unchanged.',
        old_fpn4_shape=[None,48,120,64],new_residual_shape=[None,48,120,64],output_shape=[None,48,120,11]),
    parameter_count=parameters,total_added_parameters=sum(parameters.values()),float32_parameter_bytes=4*sum(parameters.values()),
    mac_count=macs,total_added_macs=sum(macs.values()),mac_convention='One scalar multiply-accumulate counts as one MAC; count dense scheduled kernel positions including boundary padding. Runtime fusion/sparsity can differ.',
    non_mac_additions=dict(pointwise_bias_adds=pixels*64,residual_adds=pixels*64),
    activation_storage=dict(tap_float32_bytes=96*240*16*4,depthwise_output_float32_bytes=pixels*16*4,projected_residual_float32_bytes=pixels*64*4,
        limit='Individual tensor sizes only, not additive measured peak RAM. Shared tap liveness, backend workspaces, activation gradients and optimizer state depend on execution.'),
    local_path_after_two_reader_convolutions=dict(new_branch_nominal_rf=27,old_low_branch_nominal_rf=35),
    limitations=[
        'CameraNormalization uses global spatial statistics; RF7/11/19/27/35 describe convolution support relative to its normalized output, not a strict local receptive field in raw RGB.',
        'Existing FPN16/8/32 routes remain and contribute broader context; this branch does not shrink the complete model receptive field.',
        'Tap already follows Conv1 stride2 and ReLU6; it cannot restore raw details lost there, and branch still downsamples to stride4 before prediction.',
        'The tap precedes the expanded_conv_project 16-to8 bottleneck and block1 downsample, so it offers a shorter local feature path; detail preservation or Braille accuracy is unproven.',
        'No high-resolution stride2 output head, dot coordinates or label mapping are introduced; six output probabilities and decoder grid remain unchanged.',
        'Both 1x1 kernel and bias must initialize exactly zero. The new depthwise kernel must not also start zero, or the two-layer branch can have dead kernel gradients.',
        'At zero pointwise weights the new depthwise kernel initially receives zero branch gradient; a nonzero pointwise update can unlock it. Actual finite-gradient and save/load/output equality checks remain required.',
        'Fresh learned depthwise downsample is not automatically an antialias filter. Alignment and MAC arithmetic do not prove Keras/TFLite numerical parity or Android runtime performance.'
    ],
    source_sha256={str(path):sha(path) for path in [Path(__file__),padding_source,keras/'backend/tensorflow/nn.py',keras/'applications/mobilenet_v2.py',WORK/'selected-v15/source/word_model.py']})
OUT.mkdir()
(OUT/'parent-config.json').write_bytes(config_bytes)
(OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(OUT/'audit_stride2_tap_static.py').write_bytes(Path(__file__).read_bytes())
(OUT/'keras-padding-function.py.txt').write_text(ast.get_source_segment(padding_code,node)+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'audit.json'),sha256=sha(OUT/'audit.json'),tap=tap,existing_low=low,proposed=new,parameters=sum(parameters.values()),macs=sum(macs.values()))))
