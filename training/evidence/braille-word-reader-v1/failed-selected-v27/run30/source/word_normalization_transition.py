"""Hash-bound std.15 initialization and truthful training normalization metadata."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_digest(weights):
    result=hashlib.sha256()
    for value in weights:
        result.update(str(value.dtype).encode());result.update(str(value.shape).encode())
        result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def training_normalization_metadata(spec):
    if spec['kind']=='none':
        return 'raw','RGB float32 /255; internal Rescaling(2,-1)'
    std=float(spec['targetStd']);floor=float(spec['stdFloor'])
    if not np.isfinite([std,floor]).all() or min(std,floor)<=0:
        raise ValueError('Normalization metadata must describe finite positive scales')
    short=lambda value:format(value,'.10g').removeprefix('0') if 0<value<1 else format(value,'.10g')
    identity='channel_gray_std'+format(std*100,'.10g').zfill(2)
    text=('RGB float32 /255; internal per-image per-channel mean and population std (ddof=0, std floor '+short(floor)+'), '
          'mean standardized channels, .5+'+short(std)+'*gray clipped [0,1], repeat RGB, then Rescaling(2,-1)')
    return identity,text


def validated_normalization_transition(model,resume,initialization_path,parity_path,spec):
    if bool(initialization_path) != bool(parity_path):
        raise ValueError('Initialization and runtime proof must be supplied together')
    if initialization_path is None:
        if spec.get('targetStd',.05)!=.05:
            raise ValueError('A changed embedded targetStd requires its explicit initialization and runtime proof')
        return {'enabled':False}
    initialization_path=Path(initialization_path);parity_path=Path(parity_path)
    initial=json.loads(initialization_path.read_text());parity=json.loads(parity_path.read_text())
    runtime=parity_path.parent;torch_path=runtime/'torch-report.json';torch=json.loads(torch_path.read_text())
    def require(condition,message):
        if not condition:
            raise ValueError(message)
    require(initial['status']=='serialized_parent_and_hdf5_identity_passed_runtime_load_pending','Wrong initializer evidence')
    require(parity['status']==torch['status']=='passed' and parity['phase']=='tensorflow' and torch['phase']=='torch','Incomplete runtime proof')
    require(digest(initialization_path)==parity['initialization_report_sha256']==torch['initialization_report_sha256'],'Initializer report changed')
    require(digest(resume)==initial['initialized_checkpoint_sha256']==parity['initializer_sha256']==torch['initializer_sha256'],'Resume only the actual std.15 initializer')
    require(initial['only_model_configuration_change']=={'path':'config.layers[1].config.target_std','from':.05,'to':.15},'Normalization transition changed')
    require(initial['parent_checkpoint_sha256']==parity['base_checkpoint_sha256']==torch['base_checkpoint_sha256'],'Base checkpoint differs')
    require(digest(initial['parent_checkpoint'])==initial['parent_checkpoint_sha256'],'Base checkpoint bytes changed')
    require(digest(torch_path)==parity['torch_report_sha256'] and digest(runtime/'torch-oracle.npz')==parity['torch_oracle_sha256']==torch['torch_oracle_sha256'],'Torch oracle changed')
    require(digest(runtime/'probe.py')==parity['script_sha256']==torch['script_sha256'],'Runtime probe source changed')
    require(parity['source_sha256']==torch['source_sha256'],'Runtime source sets differ')
    for name,expected in parity['source_sha256'].items():
        require(digest(runtime/'source'/name)==expected and digest(Path(__file__).parent/name)==expected,'Runtime dependency changed: '+name)
    require(digest(runtime/'std15-probe.tflite')==parity['artifact_sha256'] and (runtime/'std15-probe.tflite').stat().st_size==parity['artifact_bytes'],'Converted initializer changed')
    require(parity['builtins_only'] and 'CUSTOM' not in parity['operators'] and not any(x.startswith('Flex') for x in parity['operators']),'Require built-in operators')
    require(parity['tensor_tolerance']=={'atol':1e-4,'rtol':1e-4},'Tolerance changed')
    for key in ('torch_tf_max_abs','tf_tflite_max_abs','torch_tflite_max_abs','compiled_eager_max_abs','normalization_formula_max_abs'):
        value=parity[key];require(isinstance(value,(int,float)) and np.isfinite(value) and 0<=value<=2e-4,'Invalid numeric parity: '+key)
    require(parity['input']=={'count':1,'shape':[1,192,480,3],'dtype':'float32'} and parity['output']=={'count':1,'shape':[1,48,120,11],'dtype':'float32'},'Tensor interface changed')
    require(parity['raw_and_effective_decoder_parity'] and parity['decisions']==torch['decisions'] and len(parity['decisions'])==6,'Decoder parity differs')
    require(parity['thresholds']==torch['thresholds'] and parity['selected_validation_sha256']==torch['selected_validation_sha256'],'Decision gates changed')
    selected=Path(initial['parent_checkpoint']).parent/'selected-validation.json'
    require(digest(selected)==parity['selected_validation_sha256'] and json.loads(selected.read_text())['thresholds']==parity['thresholds'],'Frozen parent decision metadata changed')
    for key in ('compiled_eager_max_abs','normalization_formula_max_abs'):
        value=torch[key];require(isinstance(value,(int,float)) and np.isfinite(value) and 0<=value<=2e-4,'Invalid Torch numeric proof: '+key)
    for key in ('all_weights_and_moments_unchanged','eager_train_eval_exact','all_outputs_finite_and_bounded'):
        require(parity[key] is True and torch[key] is True,'Runtime invariant failed: '+key)
    for key in ('runtime_graph_change_only_target_std','all_280_parent_arrays_exact','save_reload_output_and_weights_exact','constant_black_white_parent_output_exact'):
        require(torch[key] is True,'Initializer invariant failed: '+key)
    deltas=np.asarray(torch['parent_output_max_abs_delta_by_input'],float)
    require(deltas.shape==(6,) and np.isfinite(deltas).all() and np.all((deltas>=0)&(deltas<=1)) and np.all(deltas[[0,1,2,5]]>0) and np.all(deltas[3:5]==0),'The intended normalization effect was not established')
    require(spec==parity['imageNormalization']==torch['imageNormalization'] and spec['targetStd']==.15 and spec['stdFloor']==.001,'Actual normalization differs from runtime proof')
    require(model.count_params()==829307 and len(model.get_weights())==280 and weight_digest(model.get_weights())==parity['weights_sha256']==torch['weights_sha256'],'Loaded initializer weights differ')
    backbone=model.get_layer('mobilenetv2_half');bn=[x for x in backbone.layers if x.__class__.__name__=='BatchNormalization']
    require(len(bn)==52 and not any(x.trainable for x in bn),'Retain ordinary frozen BatchNorm')
    require(sorted(192//int(x.shape[1]) for x in backbone.outputs)==[4,8,16,32],'Full feature pyramid changed')
    return {'enabled':True,'initialization_report':str(initialization_path),'initialization_report_sha256':digest(initialization_path),
            'runtime_proof':str(parity_path),'runtime_proof_sha256':digest(parity_path),
            'base_parent_checkpoint_sha256':initial['parent_checkpoint_sha256'],'initial_checkpoint_sha256':digest(resume),
            'target_std_before':.05,'target_std_after':.15,'imageNormalization':spec,'initial_weights_sha256':parity['weights_sha256'],
            'scope':'Single embedded normalization parameter change, full parent weights/topology retained, fresh optimizer. Exact initializer only; any future trained-checkpoint continuation needs explicit lineage.'}
