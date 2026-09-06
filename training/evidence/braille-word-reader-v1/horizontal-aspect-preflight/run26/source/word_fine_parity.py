"""Actual initial and nonzero stride2 branch Torch/TF/TFLite preflight."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np

from word_bn_affine_parity import check_outputs, decision_rows, require_same_decisions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    destination = args.run/'fine-conversion-parity.json'
    if destination.exists() or any(args.run.glob('*-conversion-probe.tflite')):
        raise SystemExit('Existing conversion proofs must not be overwritten')
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import keras
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from word_fine_branch import fine_branch_spec
    import word_model  # register inherited normalization before loading
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    init_path = args.run/'fine-initialization.json'
    init = json.loads(init_path.read_text())
    assert init['status'] == 'passed' and init['nonzero_diagnostic_must_never_be_resumed'] is True
    for name, expected in init['source_sha256'].items():
        assert digest(Path(__file__).parent/name) == expected, name
    oracle_path = args.run/'initialization-oracle.npz'
    assert digest(oracle_path) == init['oracle_sha256']
    oracle = np.load(oracle_path)
    images = oracle['images']
    assert images.shape == (6, 192, 480, 3) and images.dtype == np.float32
    assert np.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    for label in ('parent', 'initial', 'nonzero'):
        check_outputs(oracle[label])
    np.testing.assert_array_equal(oracle['parent'], oracle['initial'])
    assert float(np.max(np.abs(oracle['initial']-oracle['nonzero']))) > 1e-3
    assert digest(init['parent_checkpoint']) == init['parent_checkpoint_sha256']
    selected_path = Path(init['parent_checkpoint']).parent/'selected-validation.json'
    selected = json.loads(selected_path.read_text())
    assert selected['checkpoint_sha256'] == init['parent_checkpoint_sha256']
    assert digest(Path(__file__).parent/'word_decode.py') == digest(selected_path.parent/'source/word_decode.py')
    thresholds = selected['thresholds']
    sources = ('word_fine_parity.py', 'word_bn_affine_parity.py', 'word_decode.py', 'word_benchmark.py', 'benchmark_angelina.py', 'word_data.py')
    source_hashes = {name: digest(Path(__file__).parent/name) for name in sources}
    decisions = {label: decision_rows(oracle[label], images, thresholds) for label in ('parent', 'initial', 'nonzero')}
    require_same_decisions(decisions['parent'], decisions['initial'])
    parent = keras.models.load_model(init['parent_checkpoint'], compile=False)
    parent_tf = parent.predict(images, batch_size=1, verbose=0)
    check_outputs(parent_tf)
    np.testing.assert_allclose(parent_tf, oracle['parent'], atol=1e-4, rtol=1e-4)
    require_same_decisions(decisions['parent'], decision_rows(parent_tf, images, thresholds))
    del parent
    names = {value: key for key, value in vars(schema.BuiltinOperator).items() if key.isupper() and isinstance(value, int)}
    reports = {}
    tf_values, lite_values = {}, {}
    for label, key, hash_key in [('initial', 'initial_checkpoint', 'initial_checkpoint_sha256'),
                                  ('nonzero', 'nonzero_diagnostic_checkpoint', 'nonzero_diagnostic_sha256')]:
        assert digest(init[key]) == init[hash_key]
        model = keras.models.load_model(init[key], compile=False)
        assert fine_branch_spec(model) == init['fine_branch']
        assert model.count_params() == init['parameters'] == 830539
        before = model.get_weights()
        lateral = model.get_layer('fpn_2_zero').get_weights()
        if label == 'initial':
            assert all(not np.any(value) for value in lateral)
        else:
            assert all(np.any(value) and np.isfinite(value).all() for value in lateral)
        assert np.any(model.get_layer('fpn_2_depthwise').get_weights()[0])
        actual = model.predict(images, batch_size=1, verbose=0)
        check_outputs(actual)
        eager = np.stack([model(image[None], training=False).numpy()[0] for image in images])
        train = np.stack([model(image[None], training=True).numpy()[0] for image in images])
        np.testing.assert_array_equal(eager, train)
        np.testing.assert_allclose(actual, eager, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual, oracle[label], atol=1e-4, rtol=1e-4)
        for values in (actual, eager):
            require_same_decisions(decisions[label], decision_rows(values, images, thresholds))
        if label == 'initial':
            np.testing.assert_allclose(actual, parent_tf, atol=1e-4, rtol=1e-4)
        serving = tf.function(lambda pixels: model(pixels, training=False), input_signature=[tf.TensorSpec([1, 192, 480, 3], tf.float32)])
        frozen = convert_variables_to_constants_v2(serving.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        artifact = converter.convert()
        buffer = schema.Model.GetRootAsModel(artifact, 0)
        operators = sorted({names[buffer.OperatorCodes(i).BuiltinCode()] for i in range(buffer.OperatorCodesLength())})
        assert 'CUSTOM' not in operators
        interpreter = tf.lite.Interpreter(model_content=artifact, num_threads=4)
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        assert len(inputs) == len(outputs) == 1
        inp, out = inputs[0], outputs[0]
        assert inp['dtype'] == out['dtype'] == np.float32
        assert inp['shape'].tolist() == [1, 192, 480, 3] and out['shape'].tolist() == [1, 48, 120, 11]
        lite = []
        for image in images:
            interpreter.set_tensor(inp['index'], image[None]); interpreter.invoke()
            lite.append(interpreter.get_tensor(out['index'])[0].copy())
        lite = np.asarray(lite)
        tf_values[label], lite_values[label] = actual, lite
        check_outputs(lite)
        np.testing.assert_allclose(lite, actual, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(lite, oracle[label], atol=1e-4, rtol=1e-4)
        require_same_decisions(decisions[label], decision_rows(lite, images, thresholds))
        for old, new in zip(before, model.get_weights(), strict=True):
            np.testing.assert_array_equal(old, new)
        path = args.run/f'{label}-conversion-probe.tflite'; path.write_bytes(artifact)
        reports[label] = {'checkpoint_sha256': init[hash_key], 'artifact_sha256': digest(path), 'artifact_bytes': len(artifact),
                          'operators': operators, 'builtins_only': True, 'samples': 6, 'all_finite': True,
                          'torch_vs_tf_max_abs': float(np.max(np.abs(oracle[label]-actual))),
                          'tf_vs_tflite_max_abs': float(np.max(np.abs(actual-lite))),
                          'torch_vs_tflite_max_abs': float(np.max(np.abs(oracle[label]-lite))),
                          'compiled_vs_eager_max_abs': float(np.max(np.abs(actual-eager))),
                          'eager_train_eval_exact': True, 'all_moments_and_weights_unchanged': True,
                          'raw_and_quality_effective_decisions_equal': True, 'decisions': decisions[label],
                          'input': {'count': 1, 'shape': inp['shape'].tolist(), 'dtype': 'float32'},
                          'output': {'count': 1, 'shape': out['shape'].tolist(), 'dtype': 'float32'}}
        del interpreter, model
    tf_delta = float(np.max(np.abs(tf_values['initial']-tf_values['nonzero'])))
    lite_delta = float(np.max(np.abs(lite_values['initial']-lite_values['nonzero'])))
    assert tf_delta > 1e-3 and lite_delta > 1e-3
    source = args.run/'conversion-source'; source.mkdir()
    for name in sources:
        assert digest(Path(__file__).parent/name) == source_hashes[name]
        (source/name).write_bytes((Path(__file__).parent/name).read_bytes())
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Six fixed TRAIN/analytic inputs, actual initial and nonzero stride2 copies; no accuracy or activation.',
              'initialization_report': str(init_path), 'initialization_report_sha256': digest(init_path),
              'oracle_sha256': digest(oracle_path), 'source_sha256': source_hashes,
              'selected_validation_report': str(selected_path), 'selected_validation_sha256': digest(selected_path),
              'thresholds': thresholds, 'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-4,
              'probes': reports, 'torch_parent_decisions': decisions['parent'],
              'nonzero_tf_output_delta': tf_delta, 'nonzero_tflite_output_delta': lite_delta,
              'conversion_probes_must_not_be_activated_or_resumed': True, 'trained_candidate_requires_full405_parity': True}
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
