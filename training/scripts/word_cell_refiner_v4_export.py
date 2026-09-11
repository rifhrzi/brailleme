"""Builtin-only export of the optional fixed-architecture six-dot refiner.

Uses an explicit predicted-box VALIDATION crop fixture. This verifies the tiny
CNN only; detector, crop sampling, blending, and word decisions require separate
pipeline confirmation. Never reads held-out TEST or alters the detector model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHAPES = {
    'features.0.weight': (16,1,3,3), 'features.0.bias': (16,),
    'features.3.weight': (32,16,3,3), 'features.3.bias': (32,),
    'features.6.weight': (48,32,3,3), 'features.6.bias': (48,),
    'classifier.0.weight': (64,1920), 'classifier.0.bias': (64,),
    'classifier.2.weight': (6,64), 'classifier.2.bias': (6,),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_weights(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(SHAPES):
            raise ValueError('Refiner checkpoint layer keys differ from the fixed architecture')
        weights = {name:archive[name].copy() for name in SHAPES}
    for name, expected in SHAPES.items():
        value = weights[name]
        if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Refiner weight shape/dtype/finiteness mismatch: '+name)
    return weights


def read_crops(path, validation, detector, source):
    content = json.loads(Path(validation).read_text(encoding='utf-8'))
    if content.get('split') != 'validation':
        raise ValueError('Refiner export requires VALIDATION, never TRAIN or TEST')
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in [('scope','validation_predicted_boxes'), ('validation_sha256',sha(validation)),
                              ('detector_sha256',sha(detector)), ('crop_code_sha256',sha(source))]:
            if key not in archive or str(archive[key]) != expected:
                raise ValueError('Refiner crop fixture binding differs: '+key)
        crops = archive['inputs'].copy()
    if (crops.ndim != 4 or crops.shape[1:] != (64,40,1) or not len(crops)
            or crops.dtype != np.float32 or not np.isfinite(crops).all() or crops.min()<0 or crops.max()>1):
        raise ValueError('Refiner crops must be finite float32[N,64,40,1] in [0,1]')
    return crops


def validate_probabilities(values, count):
    if (values.shape != (count,6) or not np.isfinite(values).all() or values.min()<0 or values.max()>1):
        raise ValueError('Refiner probabilities must be finite[N,6] in [0,1]')


def tensorflow_function(weights):
    import tensorflow as tf
    constants = {name:tf.constant(value) for name,value in weights.items()}
    @tf.function(input_signature=[tf.TensorSpec([1,64,40,1],tf.float32,name='cell_luminance_0_to_1')])
    def infer(inputs):
        centered = inputs-tf.reduce_mean(inputs,axis=[1,2,3],keepdims=True)
        x = centered/tf.sqrt(tf.reduce_mean(centered*centered,axis=[1,2,3],keepdims=True)+.01**2)
        for index in (0,3,6):
            # Torch Conv2d padding=1 is explicit symmetric zero padding.
            x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]])
            kernel = tf.transpose(constants[f'features.{index}.weight'],[2,3,1,0])
            x = tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding='VALID')
            x = tf.nn.relu(tf.nn.bias_add(x,constants[f'features.{index}.bias']))
            x = tf.nn.max_pool2d(x,ksize=2,strides=2,padding='VALID')
        # Preserve PyTorch Linear's channel-major flatten ordering exactly.
        x = tf.reshape(tf.transpose(x,[0,3,1,2]),[1,1920])
        x = tf.nn.relu(tf.matmul(x,constants['classifier.0.weight'],transpose_b=True)+constants['classifier.0.bias'])
        logits = tf.matmul(x,constants['classifier.2.weight'],transpose_b=True)+constants['classifier.2.bias']
        return tf.sigmoid(logits)
    return infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference',action='store_true')
    mode.add_argument('--convert',action='store_true')
    for name in ('checkpoint','crops','validation','detector','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name('word_cell_refiner_v4.py')
    crops = read_crops(args.crops,args.validation,args.detector,source)
    weights = load_weights(args.checkpoint)
    frozen = {str(path.resolve()):sha(path) for path in
              (args.checkpoint,args.crops,args.validation,args.detector,source,Path(__file__))}
    args.output.mkdir(parents=True,exist_ok=True)
    oracle_path, binding_path = args.output/'refiner-torch-oracle.npz', args.output/'refiner-torch-oracle.json'
    if args.reference:
        if oracle_path.exists() or binding_path.exists():
            raise ValueError('Refiner oracle already exists; preserve old evidence')
        import torch
        from word_cell_refiner_v4 import load_refiner
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_refiner(args.checkpoint).cuda().eval()
        outputs = []
        with torch.no_grad():
            for start in range(0,len(crops),64):
                outputs.append(torch.sigmoid(model(torch.from_numpy(crops[start:start+64]).cuda())).cpu().numpy())
        probabilities = np.concatenate(outputs)
        validate_probabilities(probabilities,len(crops))
        for path,digest in frozen.items():
            if sha(path)!=digest:raise ValueError('Input changed during refiner reference: '+path)
        np.savez_compressed(oracle_path,probabilities=probabilities)
        write_new(binding_path,dict(input_sha256=frozen,oracle_sha256=sha(oracle_path),samples=len(crops),
                                   torch=torch.__version__,scope='validation_predicted_boxes',test_opened=False))
        return
    binding = json.loads(binding_path.read_text())
    if binding['input_sha256']!=frozen or binding['oracle_sha256']!=sha(oracle_path) or binding['samples']!=len(crops):
        raise ValueError('Refiner oracle binding changed')
    import tensorflow as tf
    function = tensorflow_function(weights)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([function.get_concrete_function()])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    target, report_path = args.output/'cell-refiner.tflite', args.output/'refiner-export-parity.json'
    if target.exists() or report_path.exists():
        raise ValueError('Refiner conversion evidence already exists')
    target.write_bytes(converter.convert())
    engine = tf.lite.Interpreter(model_path=str(target),num_threads=2)
    engine.allocate_tensors()
    inputs, outputs = engine.get_input_details(),engine.get_output_details()
    if (len(inputs)!=1 or len(outputs)!=1 or list(inputs[0]['shape'])!=[1,64,40,1]
            or list(outputs[0]['shape'])!=[1,6] or inputs[0]['dtype']!=np.float32 or outputs[0]['dtype']!=np.float32):
        raise ValueError('Refiner TFLite interface differs from the fixed protocol')
    with np.load(oracle_path,allow_pickle=False) as archive:
        expected = archive['probabilities'].copy()
    validate_probabilities(expected,len(crops))
    maximum_tf,maximum_lite,numerical,mask_equal = 0.,0.,True,True
    for index,crop in enumerate(crops):
        actual_tf = function(crop[None]).numpy()
        engine.set_tensor(inputs[0]['index'],crop[None]);engine.invoke()
        actual_lite = engine.get_tensor(outputs[0]['index'])
        for actual in (actual_tf,actual_lite):
            validate_probabilities(actual,1)
            numerical &= bool(np.allclose(expected[index],actual[0],atol=1e-4,rtol=1e-4))
            mask_equal &= bool(np.array_equal(expected[index]>=.5,actual[0]>=.5))
        maximum_tf = max(maximum_tf,float(np.max(np.abs(expected[index]-actual_tf[0]))))
        maximum_lite = max(maximum_lite,float(np.max(np.abs(expected[index]-actual_lite[0]))))
    operators = sorted({op['op_name'] for op in engine._get_ops_details()})
    passed = numerical and mask_equal and not any('CUSTOM' in op or 'Flex' in op for op in operators)
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Input changed during refiner conversion: '+path)
    write_new(report_path,dict(status='passed' if passed else 'failed',samples=len(crops),input_sha256=frozen,
        refiner_sha256=sha(args.checkpoint),tflite_sha256=sha(target),bytes=target.stat().st_size,
        torch_tf_max_abs=maximum_tf,torch_tflite_max_abs=maximum_lite,allclose=numerical,bit_decisions_equal=mask_equal,
        atol=1e-4,rtol=1e-4,operators=operators,tensorflow=tf.__version__,scope='validation_predicted_boxes',
        pipeline_word_parity_verified=False,physical_camera_validated=False,test_opened=False))
    print(json.dumps(dict(passed=passed,samples=len(crops),torch_tf_max_abs=maximum_tf,torch_tflite_max_abs=maximum_lite)))
    if not passed:raise ValueError('Refiner conversion parity failed')


if __name__ == '__main__':
    main()
