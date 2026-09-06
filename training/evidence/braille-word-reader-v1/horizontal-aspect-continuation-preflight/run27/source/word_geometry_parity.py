"""Prove the actual nonzero geometry sampler survives builtins-only conversion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import word_geometry_head  # register the serialized layer
import word_model  # register the inherited normalization


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    init_path = args.run/'geometry-initialization.json'
    init = json.loads(init_path.read_text())
    for name, expected in init['source_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError(f'Initialization source changed before parity: {name}')
    archive = np.load(args.run/'initialization-oracle.npz')
    if digest(args.run/'initialization-oracle.npz') != init['oracle_sha256']:
        raise ValueError('Initialization oracle changed')
    images = archive['images']
    parent = keras.models.load_model(init['parent_checkpoint'], compile=False)
    parent_tf = parent.predict(images, batch_size=2, verbose=0)
    np.testing.assert_allclose(parent_tf, archive['parent_outputs'], atol=1e-4, rtol=1e-4)
    operator_names = {value: key for key, value in vars(schema.BuiltinOperator).items()
                      if key.isupper() and isinstance(value, int)}
    reports = {}
    for label, path_key, hash_key, oracle_key in (
            ('zero', 'zero_checkpoint', 'zero_checkpoint_sha256', 'zero_outputs'),
            ('nonzero', 'nonzero_diagnostic_checkpoint', 'nonzero_diagnostic_sha256', 'nonzero_outputs')):
        path = Path(init[path_key])
        if digest(path) != init[hash_key]:
            raise ValueError('Initialized or diagnostic checkpoint changed')
        model = keras.models.load_model(path, compile=False)
        expected = archive[oracle_key]
        actual_tf = model.predict(images, batch_size=2, verbose=0)
        np.testing.assert_allclose(actual_tf, expected, atol=1e-4, rtol=1e-4)
        if label == 'zero':
            np.testing.assert_allclose(actual_tf, parent_tf, atol=1e-6, rtol=1e-6)
        serving = tf.function(lambda image: model(image, training=False),
                              input_signature=[tf.TensorSpec([1, 192, 480, 3], tf.float32)])
        frozen = convert_variables_to_constants_v2(serving.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        artifact = converter.convert()
        output = args.run/f'{label}-conversion-probe.tflite'
        output.write_bytes(artifact)
        flatbuffer = schema.Model.GetRootAsModel(artifact, 0)
        operators = sorted({operator_names[flatbuffer.OperatorCodes(i).BuiltinCode()]
                            for i in range(flatbuffer.OperatorCodesLength())})
        if 'CUSTOM' in operators:
            raise ValueError('Custom/Flex operators are forbidden for this experiment')
        if label == 'nonzero' and not any(name in operators for name in ('GATHER', 'GATHER_ND')):
            raise ValueError('Nonzero sampler branch was unexpectedly removed')
        interpreter = tf.lite.Interpreter(model_content=artifact, num_threads=4)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        assert inp['shape'].tolist() == [1, 192, 480, 3]
        assert out['shape'].tolist() == [1, 48, 120, 11]
        lite = []
        for image in images:
            interpreter.set_tensor(inp['index'], image[None])
            interpreter.invoke()
            lite.append(interpreter.get_tensor(out['index'])[0].copy())
        lite = np.asarray(lite)
        assert np.isfinite(lite).all()
        np.testing.assert_allclose(lite, actual_tf, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(lite, expected, atol=1e-4, rtol=1e-4)
        reports[label] = {'checkpoint_sha256': init[hash_key], 'artifact_sha256': digest(output),
                           'artifact_bytes': len(artifact), 'operators': operators,
                           'torch_vs_tf_max_abs': float(np.max(np.abs(expected-actual_tf))),
                           'tf_vs_tflite_max_abs': float(np.max(np.abs(actual_tf-lite))),
                           'torch_vs_tflite_max_abs': float(np.max(np.abs(expected-lite))),
                           'all_finite': True, 'builtins_only': True, 'samples': len(images)}
    source = Path(__file__)
    (args.run/'source'/source.name).write_bytes(source.read_bytes())
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Six TRAIN/analytic initialization inputs, zero and actual nonzero diagnostic copies; no accuracy or candidate promotion.',
              'initialization_report_sha256': digest(init_path), 'source_sha256': digest(source),
              'oracle_sha256': init['oracle_sha256'], 'absolute_tolerance': 1e-4,
              'relative_tolerance': 1e-4, 'conversion_probes_must_not_be_activated_or_resumed': True,
              'probes': reports, 'trained_candidate_requires_full405_parity': True}
    (args.run/'geometry-conversion-parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
