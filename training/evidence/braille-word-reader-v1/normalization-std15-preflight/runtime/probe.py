"""Actual std.15 initializer: six fixed TRAIN/analytic inputs, CPU Torch then TF/Lite."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent;WORK=OUT.parent;REPO=OUT.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES=('word_model.py','word_data.py','word_anchor_zoom.py','word_decode.py','word_benchmark.py','benchmark_angelina.py','word_bn_affine_parity.py')
def save(name,data):
    p=OUT/name;assert not p.exists(),p;p.write_text(json.dumps(data,indent=2)+'\n')
def weight_digest(weights):
    h=hashlib.sha256()
    for value in weights:
        h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['torch','tensorflow']);args=parser.parse_args()
    os.environ.update(KERAS_BACKEND=args.phase,KERAS_TORCH_DEVICE='cpu',CUDA_VISIBLE_DEVICES='-1')
    source=OUT/'source'
    if args.phase=='torch':
        assert not (OUT/'torch-report.json').exists() and not (OUT/'torch-oracle.npz').exists()
        source.mkdir()
        for name in SOURCES:(source/name).write_bytes((REPO/'training/scripts'/name).read_bytes())
    else:
        assert not (OUT/'parity-report.json').exists() and not (OUT/'std15-probe.tflite').exists()
    hashes={name:sha(source/name) for name in SOURCES}
    sys.path.insert(0,str(source))
    if args.phase=='torch':
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    else:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        from tensorflow.lite.python import schema_py_generated as schema
    import keras
    from keras import ops
    from word_model import image_normalization_spec
    from word_bn_affine_parity import check_outputs,decision_rows,require_same_decisions
    initialized=WORK/'normalization-std15-parent-v1/std15-parent.keras';base=WORK/'selected-v15/best-word.keras'
    initial_report=WORK/'normalization-std15-parent-v1/initialization-report.json'
    assert sha(initialized)=='918d148a43399642f44c5d18693eb2733eecca338f0332fed8782cbe79beea00'
    assert sha(base)=='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    assert sha(initial_report)=='bb1b0dc9068e4757a3e680c8d31ce7f40fd582ccba3d6c41db8fa94f4f92224b'
    oracle_path=WORK/'geometry-head-parent-v1/initialization-oracle.npz'
    assert sha(oracle_path)=='9330946dafe06c7705b3f2c85cce26f8bbfeca11763ba7465d62e809180d5669'
    images=np.load(oracle_path)['images'];assert images.shape==(6,192,480,3) and images.dtype==np.float32
    assert np.isfinite(images).all() and images.min()>=0 and images.max()<=1
    input_report=WORK/'geometry-head-parent-v1/geometry-initialization.json'
    identities=json.loads(input_report.read_text())['train_inputs']
    for index,row in enumerate(identities):
        p=WORK/'data'/row['image'];assert sha(p)==row['sha256']
        np.testing.assert_array_equal(images[index],np.asarray(Image.open(p).convert('RGB'),np.float32)/255.)
    np.testing.assert_array_equal(images[3],0.);np.testing.assert_array_equal(images[4],1.)
    np.testing.assert_array_equal(images[5],np.random.default_rng(20260905).random((192,480,3),dtype=np.float32))
    selected_path=WORK/'selected-v15/selected-validation.json'
    thresholds=json.loads(selected_path.read_text())['thresholds']
    model=keras.models.load_model(initialized,compile=False)
    spec=image_normalization_spec(model);assert spec['targetStd']==.15 and spec['stdFloor']==.001
    assert model.count_params()==829307 and len(model.get_weights())==280
    assert list(model.input_shape)==[None,192,480,3] and list(model.output_shape)==[None,48,120,11]
    backbone=model.get_layer('mobilenetv2_half');bn=[x for x in backbone.layers if isinstance(x,keras.layers.BatchNormalization)]
    assert len(bn)==52 and not any(x.trainable for x in bn)
    assert sorted(192//int(x.shape[1]) for x in backbone.outputs)==[4,8,16,32]
    assert model.get_layer('fpn_32_zero') is not None
    weights=model.get_weights();before=weight_digest(weights)
    actual=model.predict(images,batch_size=1,verbose=0);check_outputs(actual)
    eager=np.stack([ops.convert_to_numpy(model(x[None],training=False))[0] for x in images])
    training=np.stack([ops.convert_to_numpy(model(x[None],training=True))[0] for x in images])
    np.testing.assert_array_equal(eager,training);np.testing.assert_allclose(actual,eager,atol=1e-4,rtol=1e-4)
    decisions=decision_rows(actual,images,thresholds);require_same_decisions(decisions,decision_rows(eager,images,thresholds))
    normalized=ops.convert_to_numpy(model.get_layer('camera_normalization')(images))
    shifted=images-images[:,:1,:1,:];centered=shifted-shifted.mean(axis=2,keepdims=True).mean(axis=1,keepdims=True)
    variance=np.square(centered).mean(axis=2,keepdims=True).mean(axis=1,keepdims=True)
    expected=np.repeat(np.clip(.5+.15*(centered/np.sqrt(np.maximum(variance,.001**2))).mean(axis=-1,keepdims=True),0,1),3,axis=-1)
    np.testing.assert_allclose(normalized,expected,atol=1e-4,rtol=1e-4)
    np.testing.assert_array_equal(normalized[3:5],.5)
    report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'phase':args.phase,'script_sha256':sha(__file__),
      'initializer':str(initialized),'initializer_sha256':sha(initialized),'initialization_report_sha256':sha(initial_report),'base_checkpoint_sha256':sha(base),
      'input_oracle_sha256':sha(oracle_path),'input_identity_report_sha256':sha(input_report),'train_inputs':identities,
      'selected_validation_sha256':sha(selected_path),'thresholds':thresholds,'source_sha256':hashes,'imageNormalization':spec,
      'model_arrays':280,'parameters':829307,'frozen_batchnorm_layers':52,'feature_pyramid_strides':[32,16,8,4],
      'weights_sha256':before,'eager_train_eval_exact':True,'compiled_eager_max_abs':float(np.max(np.abs(actual-eager))),
      'normalization_formula_max_abs':float(np.max(np.abs(normalized-expected))),'all_outputs_finite_and_bounded':True,'decisions':decisions,
      'scope':'Actual one-field std.15 initializer, not a second perturbed diagnostic model. Six fixed TRAIN/analytic probes only; no checkpoint selection or accuracy claim. Any trained candidate still needs full405 parity and original607/44/66/12/14 gates.'}
    if args.phase=='torch':
        parent=keras.models.load_model(base,compile=False)
        for a,b in zip(parent.get_weights(),weights,strict=True):np.testing.assert_array_equal(a,b)
        old_graph=parent.get_config();new_graph=model.get_config()
        layer=next(x for x in old_graph['layers'] if x['name']=='camera_normalization');assert layer['config']['target_std']==.05
        layer['config']['target_std']=.15;assert old_graph==new_graph
        previous=parent.predict(images,batch_size=1,verbose=0);check_outputs(previous)
        np.testing.assert_array_equal(previous[3:5],actual[3:5])
        deltas=np.max(np.abs(previous-actual),axis=(1,2,3));assert np.all(deltas[[0,1,2,5]]>0)
        model.save(OUT/'reload-check.keras');restored=keras.models.load_model(OUT/'reload-check.keras',compile=False)
        assert restored.get_config()==new_graph and weight_digest(restored.get_weights())==before
        np.testing.assert_array_equal(actual,restored.predict(images,batch_size=1,verbose=0))
        report.update(runtime_graph_change_only_target_std=True,all_280_parent_arrays_exact=True,save_reload_output_and_weights_exact=True,
            constant_black_white_parent_output_exact=True,parent_output_max_abs_delta_by_input=deltas.tolist(),roundtrip_checkpoint_sha256=sha(OUT/'reload-check.keras'))
        np.savez_compressed(OUT/'torch-oracle.npz',images=images,outputs=actual,parent_outputs=previous)
        report['torch_oracle_sha256']=sha(OUT/'torch-oracle.npz')
        assert weight_digest(model.get_weights())==before
        report['all_weights_and_moments_unchanged']=True
        save('torch-report.json',report)
        print(json.dumps(report));return
    oracle_report=json.loads((OUT/'torch-report.json').read_text())
    for k in ('initializer_sha256','initialization_report_sha256','input_oracle_sha256','selected_validation_sha256','thresholds','source_sha256','imageNormalization','weights_sha256'):
        assert report[k]==oracle_report[k],k
    assert sha(OUT/'torch-oracle.npz')==oracle_report['torch_oracle_sha256']
    oracle=np.load(OUT/'torch-oracle.npz');np.testing.assert_array_equal(images,oracle['images'])
    np.testing.assert_allclose(actual,oracle['outputs'],atol=1e-4,rtol=1e-4);require_same_decisions(oracle_report['decisions'],decisions)
    serving=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec([1,192,480,3],tf.float32)])
    frozen=convert_variables_to_constants_v2(serving.get_concrete_function())
    converter=tf.lite.TFLiteConverter.from_concrete_functions([frozen]);converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    artifact=converter.convert();flat=schema.Model.GetRootAsModel(artifact,0)
    names={v:k for k,v in vars(schema.BuiltinOperator).items() if k.isupper() and isinstance(v,int)}
    operators=sorted({names[flat.OperatorCodes(i).BuiltinCode()] for i in range(flat.OperatorCodesLength())});assert 'CUSTOM' not in operators
    interpreter=tf.lite.Interpreter(model_content=artifact,num_threads=4);interpreter.allocate_tensors()
    ins,outs=interpreter.get_input_details(),interpreter.get_output_details();assert len(ins)==len(outs)==1
    inp,out=ins[0],outs[0];assert inp['shape'].tolist()==[1,192,480,3] and out['shape'].tolist()==[1,48,120,11]
    assert inp['dtype']==out['dtype']==np.float32
    lite=[]
    for x in images:
        interpreter.set_tensor(inp['index'],x[None]);interpreter.invoke();lite.append(interpreter.get_tensor(out['index'])[0].copy())
    lite=np.asarray(lite);check_outputs(lite)
    np.testing.assert_allclose(lite,actual,atol=1e-4,rtol=1e-4);np.testing.assert_allclose(lite,oracle['outputs'],atol=1e-4,rtol=1e-4)
    require_same_decisions(decisions,decision_rows(lite,images,thresholds))
    assert weight_digest(model.get_weights())==before
    (OUT/'std15-probe.tflite').write_bytes(artifact);np.save(OUT/'tflite-six-outputs.npy',lite)
    report.update(torch_report_sha256=sha(OUT/'torch-report.json'),torch_oracle_sha256=sha(OUT/'torch-oracle.npz'),
        torch_tf_max_abs=float(np.max(np.abs(actual-oracle['outputs']))),tf_tflite_max_abs=float(np.max(np.abs(actual-lite))),torch_tflite_max_abs=float(np.max(np.abs(lite-oracle['outputs']))),
        tensor_tolerance={'atol':1e-4,'rtol':1e-4},raw_and_effective_decoder_parity=True,builtins_only=True,operators=operators,
        input={'count':1,'shape':inp['shape'].tolist(),'dtype':'float32'},output={'count':1,'shape':out['shape'].tolist(),'dtype':'float32'},
        artifact_sha256=sha(OUT/'std15-probe.tflite'),artifact_bytes=len(artifact),all_weights_and_moments_unchanged=True)
    for name,digest in hashes.items():assert sha(source/name)==digest
    save('parity-report.json',report);print(json.dumps(report))

if __name__=='__main__':main()
