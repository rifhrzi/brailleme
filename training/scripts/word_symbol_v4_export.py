"""Export the trained checkpoint through TensorFlow primitives, without Keras.

First write a Torch validation oracle with --reference in the CUDA environment;
then --convert in the existing TensorFlow environment. No TEST data is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def runtime_quality_rejected(pixels):
    """Same whole-image precheck used by training evaluation and the runtime."""
    lum = pixels @ np.asarray([.2126, .7152, .0722], np.float32)
    return bool(lum.mean() < .06 or lum.mean() > .985 or lum.std() < .01)


def decision_signature(decision):
    # Confidence/geometry are compared numerically; discrete recognition must
    # preserve the entire ordered semantic cell sequence, including indicators.
    return (decision['text'], decision['reason'],
            [(cell.get('mask'), cell.get('role'), cell.get('output')) for cell in decision['cells']])


def validate_tensor_interface(inputs, outputs):
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('TFLite requires exactly one input and one output')
    for detail, shape in ((inputs[0], [1,192,480,3]), (outputs[0], [1,48,120,11])):
        if list(detail['shape']) != shape or detail['dtype'] != np.float32:
            raise ValueError('TFLite tensor shape/dtype disagrees with runtime')
    return inputs[0], outputs[0]


def validate_outputs(outputs, count):
    if outputs.shape != (count,48,120,11) or not np.isfinite(outputs).all():
        raise ValueError('Output shape or finiteness disagrees with oracle contract')
    if outputs.min() < 0 or outputs.max() > 1:
        raise ValueError('Output probabilities outside [0,1]')


def source_hashes():
    directory = Path(__file__).resolve().parent
    names = ('word_symbol_v4_export.py', 'word_torch_port.py', 'word_decode_id_v4.py',
             'word_symbols_id_v4.py', 'word_symbols.py', 'word_data.py')
    return {name: sha(directory/name) for name in names}


def pixels_for(row):
    with Image.open(row['image']) as image:
        if image.size != (480,192):
            raise ValueError('Validation image must be 480x192')
        return np.asarray(image.convert('RGB'), np.float32)/255.


def load_arrays(checkpoint):
    weights = {}
    with zipfile.ZipFile(checkpoint) as archive:
        config = json.loads(archive.read('config.json'))['config']
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as handle:
            def visit(name, obj):
                if isinstance(obj, h5py.Group) and name.endswith('/vars') and 'optimizer' not in name:
                    layer = obj.attrs.get('name')
                    if isinstance(layer, bytes):
                        layer = layer.decode()
                    values = [np.asarray(obj[key]) for key in sorted(obj, key=int)]
                    if values:
                        if layer in weights:
                            raise ValueError('Duplicate weighted layer name: '+str(layer))
                        weights[layer] = values
            handle.visititems(visit)
    return config, weights


def tensorflow_function(checkpoint):
    import tensorflow as tf
    config, arrays = load_arrays(checkpoint)
    weights = {name: [tf.constant(value) for value in values] for name, values in arrays.items()}
    def activation(x, name):
        if name == 'linear':
            return x
        if name == 'relu':
            return tf.nn.relu(x)
        if name == 'sigmoid':
            return tf.math.sigmoid(x)
        raise ValueError('Unsupported activation '+name)
    def graph(cfg, initial):
        outputs = {}
        def reference(value):
            if isinstance(value, dict) and value.get('class_name') == '__keras_tensor__':
                name, node, index = value['config']['keras_history']
                if node != 0:
                    raise ValueError('Only the frozen single-call graph is supported')
                result = outputs[name]
                return result[index] if isinstance(result, list) else result
            if isinstance(value, list):
                return [reference(item) for item in value]
            return value
        for layer in cfg['layers']:
            cls, c = layer['class_name'], layer['config']
            name = c['name']
            if c.get('data_format', 'channels_last') != 'channels_last':
                raise ValueError('Only channels-last checkpoint configuration is supported')
            if cls == 'InputLayer':
                outputs[name] = initial
                continue
            if len(layer['inbound_nodes']) != 1:
                raise ValueError('Unsupported shared graph node')
            args = reference(layer['inbound_nodes'][0]['args'])
            x = args[0]
            w = weights.get(name, [])
            if cls == 'Functional':
                y = graph(c, x)
            elif cls == 'CameraNormalization':
                shifted = x-x[:, :1, :1, :]
                mean = tf.reduce_mean(tf.reduce_mean(shifted, axis=2, keepdims=True), axis=1, keepdims=True)
                centered = shifted-mean
                variance = tf.reduce_mean(tf.reduce_mean(tf.square(centered), axis=2, keepdims=True), axis=1, keepdims=True)
                std = tf.sqrt(tf.maximum(variance, c['std_floor']**2))
                gray = tf.reduce_mean(centered/std, axis=-1, keepdims=True)
                y = tf.repeat(tf.clip_by_value(.5+c['target_std']*gray, 0., 1.), 3, axis=-1)
            elif cls == 'Rescaling':
                y = x*c['scale']+c['offset']
            elif cls in ('Conv2D', 'DepthwiseConv2D', 'SeparableConv2D'):
                strides = [1, *c['strides'], 1]
                dilation = c.get('dilation_rate', [1, 1])
                if cls == 'Conv2D':
                    if c.get('groups', 1) != 1:
                        raise ValueError('Unsupported grouped convolution')
                    y = tf.nn.conv2d(x, w[0], strides=strides, padding=c['padding'].upper(),
                                     dilations=[1, *dilation, 1])
                    bias = 1
                else:
                    y = tf.nn.depthwise_conv2d(x, w[0], strides=strides, padding=c['padding'].upper(), dilations=dilation)
                    bias = 1
                    if cls == 'SeparableConv2D':
                        y = tf.nn.conv2d(y, w[1], strides=[1,1,1,1], padding='VALID')
                        bias = 2
                if c['use_bias']:
                    y = tf.nn.bias_add(y, w[bias])
                y = activation(y, c['activation'])
            elif cls == 'BatchNormalization':
                if not c.get('scale', True) or not c.get('center', True):
                    raise ValueError('Unexpected BatchNormalization configuration')
                if c.get('axis', -1) not in (-1, 3):
                    raise ValueError('Only final-channel BatchNormalization is supported')
                gamma, beta, mean, variance = w
                y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, c['epsilon'])
            elif cls == 'ReLU':
                if c.get('negative_slope', 0) != 0 or c.get('threshold', 0) != 0:
                    raise ValueError('Unexpected ReLU configuration')
                y = tf.nn.relu(x)
                if c['max_value'] is not None:
                    y = tf.minimum(y, c['max_value'])
            elif cls == 'ZeroPadding2D':
                y = tf.pad(x, [[0,0], *c['padding'], [0,0]])
            elif cls == 'Add':
                y = x[0]
                for part in x[1:]:
                    y = y+part
            elif cls == 'UpSampling2D':
                if c['interpolation'] != 'nearest':
                    raise ValueError('Only exact nearest-neighbor upsampling is supported')
                y = tf.repeat(tf.repeat(x, c['size'][0], axis=1), c['size'][1], axis=2)
            elif cls == 'Activation':
                y = activation(x, c['activation'])
            elif cls == 'Concatenate':
                y = tf.concat(x, axis=c['axis'])
            else:
                raise ValueError('Unsupported layer '+cls)
            outputs[name] = y
        refs = cfg['output_layers']
        def get(ref):
            result = outputs[ref[0]]
            return result[ref[2]] if isinstance(result, list) else result
        return [get(ref) for ref in refs] if isinstance(refs[0], list) else get(refs)
    @tf.function(input_signature=[tf.TensorSpec([1,192,480,3], tf.float32, name='rgb_0_to_1')])
    def infer(x):
        return graph(config, x)
    return infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference', action='store_true')
    mode.add_argument('--convert', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path,
                        help='JSON containing thresholds; defaults to the checkpoint run config, then deployed manifest')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    content = json.loads(args.validation.read_text())
    if content.get('split') != 'validation':
        raise ValueError('Export validation must use the validation split, never held-out TEST')
    rows = content['records']
    if not rows or len({row['source'] for row in rows}) != len(rows):
        raise ValueError('Validation rows must be nonempty with unique source identities')
    if args.reference:
        if (args.output/'torch-oracle.npz').exists() or (args.output/'torch-oracle.json').exists():
            raise ValueError('Oracle already exists; use a new export evidence directory')
        thresholds_path = args.thresholds or args.checkpoint.parent/'training-config.json'
        if not thresholds_path.is_file():
            if args.thresholds:
                raise ValueError('Explicit thresholds file does not exist')
            thresholds_path = Path('assets/models/braille-word-reader-v1/manifest.json')
        thresholds = json.loads(thresholds_path.read_text())['thresholds']
        import torch
        from word_torch_port import load_keras_checkpoint
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_keras_checkpoint(args.checkpoint).cuda().eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), 8):
                pixels = np.stack([pixels_for(r) for r in rows[start:start+8]])
                outputs.append(model(torch.from_numpy(pixels).cuda()).cpu().numpy())
        outputs = np.concatenate(outputs)
        validate_outputs(outputs, len(rows))
        np.savez_compressed(args.output/'torch-oracle.npz', outputs=outputs)
        save(args.output/'torch-oracle.json', dict(checkpoint_sha256=sha(args.checkpoint),
            validation_sha256=sha(args.validation), oracle_sha256=sha(args.output/'torch-oracle.npz'),
            images_sha256={r['image']: sha(r['image']) for r in rows}, samples=len(rows),
            thresholds=thresholds, thresholds_path=str(thresholds_path.resolve()),
            thresholds_sha256=sha(thresholds_path), source_sha256=source_hashes(),
            torch=torch.__version__, scope='validation_only', test_opened=False))
        return
    import tensorflow as tf
    from word_decode_id_v4 import decode
    oracle = json.loads((args.output/'torch-oracle.json').read_text())
    if (oracle['checkpoint_sha256'] != sha(args.checkpoint) or oracle['validation_sha256'] != sha(args.validation)
            or oracle['oracle_sha256'] != sha(args.output/'torch-oracle.npz')
            or oracle['thresholds_sha256'] != sha(oracle['thresholds_path'])
            or oracle['source_sha256'] != source_hashes() or oracle['samples'] != len(rows)
            or set(oracle['images_sha256']) != {row['image'] for row in rows}):
        raise ValueError('Export oracle input binding changed')
    if (args.output/'word-reader.tflite').exists() or (args.output/'export-parity.json').exists():
        raise ValueError('Conversion artifact already exists; use a new export evidence directory')
    function = tensorflow_function(args.checkpoint)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([function.get_concrete_function()])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    target = args.output/'word-reader.tflite'
    target.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = validate_tensor_interface(interpreter.get_input_details(), interpreter.get_output_details())
    with np.load(args.output/'torch-oracle.npz') as archive:
        expected = archive['outputs'].copy()
    validate_outputs(expected, len(rows))
    thresholds = oracle['thresholds']
    max_tf, max_lite, mismatches, quality_mismatches = 0., 0., [], []
    numerical = True
    for i, row in enumerate(rows):
        if sha(row['image']) != oracle['images_sha256'][row['image']]:
            raise ValueError('Validation image changed')
        pixels = pixels_for(row)[None]
        tf_out = function(pixels).numpy()[0]
        interpreter.set_tensor(inp['index'], pixels)
        interpreter.invoke()
        lite = interpreter.get_tensor(out['index'])[0]
        validate_outputs(tf_out[None], 1)
        validate_outputs(lite[None], 1)
        max_tf = max(max_tf, float(np.max(np.abs(expected[i]-tf_out))))
        max_lite = max(max_lite, float(np.max(np.abs(expected[i]-lite))))
        numerical &= bool(np.allclose(expected[i], tf_out, atol=1e-4, rtol=1e-4) and np.allclose(expected[i], lite, atol=1e-4, rtol=1e-4))
        decisions = [decode(v, thresholds, row.get('selection')) for v in (expected[i], tf_out, lite)]
        keys = [decision_signature(d) for d in decisions]
        if keys[0] != keys[1] or keys[0] != keys[2]:
            mismatches.append(dict(index=i, source=row['source'], decisions=keys))
        quality_rejected = runtime_quality_rejected(pixels)
        effective = [np.zeros_like(v) if quality_rejected else v for v in (expected[i], tf_out, lite)]
        quality_keys = [decision_signature(decode(v, thresholds, row.get('selection'))) for v in effective]
        if quality_keys[0] != quality_keys[1] or quality_keys[0] != quality_keys[2]:
            quality_mismatches.append(dict(index=i, source=row['source'], decisions=quality_keys))
        if (i+1) % 100 == 0:
            print('PARITY', i+1, len(rows), flush=True)
    operators = sorted({r['op_name'] for r in interpreter._get_ops_details()})
    passed = numerical and not mismatches and not quality_mismatches and not any('Flex' in op or 'CUSTOM' in op for op in operators)
    save(args.output/'export-parity.json', dict(status='passed' if passed else 'failed', samples=len(rows),
         checkpoint_sha256=sha(args.checkpoint), tflite_sha256=sha(target), bytes=target.stat().st_size,
         torch_tf_max_abs=max_tf, torch_tflite_max_abs=max_lite, allclose=numerical,
         atol=1e-4, rtol=1e-4, decision_mismatches=mismatches,
         quality_effective_decision_mismatches=quality_mismatches, operators=operators,
         input_shape=inp['shape'].tolist(), output_shape=out['shape'].tolist(), dtype='float32', all_finite=True,
         thresholds=thresholds, oracle_sha256=sha(args.output/'torch-oracle.npz'), source_sha256=source_hashes(),
         validation_sha256=sha(args.validation), tensorflow=tf.__version__, physical_camera_validated=False,
         scope='validation_only', test_opened=False, native_runtime_validated=False))
    if not passed:
        raise ValueError('Conversion parity failed; artifact must not be deployed')


if __name__ == '__main__':
    main()
