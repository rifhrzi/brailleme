"""Separate CPU phases for six-input initial/nonzero fixed-moment BN parity."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def check_outputs(values):
    if (values.shape != (6, 48, 120, 11) or values.dtype != np.float32
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
        raise ValueError('Require six finite float32 probability outputs [6,48,120,11] in [0,1]')


def decision_rows(values, images, thresholds):
    from word_decode import decode
    from word_benchmark import quality
    rows = []
    for output, image in zip(values, images, strict=True):
        raw = decode(output, thresholds)
        reason = quality(image)
        decision = {'text': raw['text'], 'reason': raw['reason'], 'cell_count': len(raw['cells'])}
        rows.append({'raw': decision, 'quality_reason': reason,
                     'effective': {'text': '', 'reason': reason, 'cell_count': 0} if reason else decision})
    return rows


def require_same_decisions(expected, actual):
    if actual != expected:
        raise ValueError('Raw/effective decoder decisions differ despite numerical tensor tolerance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('torch', 'tensorflow'))
    parser.add_argument('--reference-oracle', type=Path)
    parser.add_argument('--reference-report', type=Path)
    args = parser.parse_args()
    os.environ['KERAS_BACKEND'] = args.phase
    os.environ['KERAS_TORCH_DEVICE'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import keras
    import word_bn_affine  # register fixed-moment BN
    import word_model  # register the inherited image normalization

    init_path = args.run/'bn-affine-initialization.json'
    init = json.loads(init_path.read_text())
    assert init['status'] == 'passed'
    for name, expected in init['source_sha256'].items():
        assert digest(Path(__file__).parent/name) == expected, name
    variants = [('initial', 'initial_checkpoint', 'initial_checkpoint_sha256'),
                ('nonzero', 'nonzero_diagnostic_checkpoint', 'nonzero_diagnostic_sha256')]
    for _, path_key, hash_key in variants:
        assert digest(init[path_key]) == init[hash_key]
    assert digest(init['parent_checkpoint']) == init['parent_checkpoint_sha256']
    selected_path = Path(init['parent_checkpoint']).parent/'selected-validation.json'
    selected = json.loads(selected_path.read_text())
    assert selected['checkpoint_sha256'] == init['parent_checkpoint_sha256']
    decoder_path = Path(__file__).parent/'word_decode.py'
    assert digest(decoder_path) == digest(selected_path.parent/'source/word_decode.py')
    thresholds = selected['thresholds']
    decision_sources = {name: digest(Path(__file__).parent/name)
                        for name in ('word_decode.py', 'word_benchmark.py', 'benchmark_angelina.py', 'word_data.py')}
    oracle_path = args.run/'bn-affine-six-oracle.npz'
    oracle_report_path = args.run/'bn-affine-six-oracle.json'
    if args.phase == 'torch':
        if not args.reference_oracle or not args.reference_report or oracle_path.exists() or oracle_report_path.exists():
            raise ValueError('Require original six-input reference and a new oracle output')
        import torch
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = json.loads(args.reference_report.read_text())
        assert digest(args.reference_oracle) == reference['oracle_sha256']
        assert reference['status'] == 'passed'
        assert len(reference['train_inputs']) == 3 and len(reference['analytic_inputs']) == 3
        images = np.load(args.reference_oracle)['images']
        assert images.shape == (6, 192, 480, 3) and images.dtype == np.float32 and np.isfinite(images).all()
        assert images.min() >= 0 and images.max() <= 1
        one_path = args.run/'initialization-one-train-oracle.npz'
        assert digest(one_path) == init['oracle_sha256']
        np.testing.assert_array_equal(images[:1], np.load(one_path)['images'])
        outputs = {}
        for label, path in [('parent', init['parent_checkpoint'])]+[(label, init[path_key]) for label, path_key, _ in variants]:
            model = keras.models.load_model(path, compile=False)
            before = model.get_weights()
            with torch.no_grad():
                values = np.stack([model(image[None], training=False).numpy()[0] for image in images])
                if label != 'parent':
                    train = np.stack([model(image[None], training=True).numpy()[0] for image in images])
                    np.testing.assert_array_equal(values, train)
            check_outputs(values)
            outputs[label] = values
            for old, new in zip(before, model.get_weights(), strict=True):
                np.testing.assert_array_equal(old, new)
            del model
        np.testing.assert_array_equal(outputs['parent'], outputs['initial'])
        np.testing.assert_array_equal(outputs['initial'][:1], np.load(one_path)['initial'])
        np.testing.assert_array_equal(outputs['nonzero'][:1], np.load(one_path)['nonzero'])
        assert np.max(np.abs(outputs['initial']-outputs['nonzero'])) > 1e-6
        np.savez_compressed(oracle_path, images=images, **outputs)
        report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'initialization_report_sha256': digest(init_path), 'source_sha256': digest(__file__),
                  'reference_oracle': str(args.reference_oracle), 'reference_oracle_sha256': digest(args.reference_oracle),
                  'reference_report': str(args.reference_report), 'reference_report_sha256': digest(args.reference_report),
                  'reference_scope': 'Reuse only the six frozen TRAIN/analytic input images; every parent/initial/nonzero output is recomputed from this initialization, never taken from the older reference checkpoint.',
                  'oracle_sha256': digest(oracle_path), 'samples': 6, 'device': 'cpu', 'threads': 2,
                  'initial_and_parent_exact': True, 'train_and_eval_exact': True, 'all_weights_unchanged': True,
                  'selected_validation_report': str(selected_path), 'selected_validation_sha256': digest(selected_path),
                  'thresholds': thresholds, 'decision_sources_sha256': decision_sources,
                  'decisions': {label: decision_rows(values, images, thresholds) for label, values in outputs.items()},
                  'nonzero_max_output_delta': float(np.max(np.abs(outputs['initial']-outputs['nonzero']))),
                  'train_inputs': reference['train_inputs'], 'analytic_inputs': reference['analytic_inputs']}
        save(oracle_report_path, report)
        print(json.dumps(report, indent=2))
        return
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    destination = args.run/'bn-affine-conversion-parity.json'
    if destination.exists() or any((args.run/f'{label}-conversion-probe.tflite').exists() for label, _, _ in variants):
        raise ValueError('Use a fresh conversion output; previous proof artifacts are immutable')
    oracle_report = json.loads(oracle_report_path.read_text())
    assert oracle_report['status'] == 'passed' and oracle_report['samples'] == 6
    assert digest(init_path) == oracle_report['initialization_report_sha256']
    assert digest(__file__) == oracle_report['source_sha256']
    assert digest(oracle_path) == oracle_report['oracle_sha256']
    assert digest(selected_path) == oracle_report['selected_validation_sha256']
    assert thresholds == oracle_report['thresholds'] and decision_sources == oracle_report['decision_sources_sha256']
    oracle = np.load(oracle_path)
    images = oracle['images']
    parent = keras.models.load_model(init['parent_checkpoint'], compile=False)
    parent_tf = parent.predict(images, batch_size=1, verbose=0)
    check_outputs(parent_tf)
    np.testing.assert_allclose(parent_tf, oracle['parent'], atol=1e-4, rtol=1e-4)
    require_same_decisions(oracle_report['decisions']['parent'], decision_rows(parent_tf, images, thresholds))
    del parent
    operator_names = {value: key for key, value in vars(schema.BuiltinOperator).items() if key.isupper() and isinstance(value, int)}
    reports = {}
    for label, path_key, hash_key in variants:
        model = keras.models.load_model(init[path_key], compile=False)
        assert word_bn_affine.affine_backbone_spec(model) == init['batchnorm_affine']
        before = model.get_weights()
        actual = model.predict(images, batch_size=1, verbose=0)
        check_outputs(actual)
        training = np.stack([model(image[None], training=True).numpy()[0] for image in images])
        np.testing.assert_allclose(actual, training, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual, oracle[label], atol=1e-4, rtol=1e-4)
        expected_decisions = oracle_report['decisions'][label]
        require_same_decisions(expected_decisions, decision_rows(actual, images, thresholds))
        if label == 'initial':
            np.testing.assert_allclose(actual, parent_tf, atol=1e-6, rtol=1e-6)
        serving = tf.function(lambda image: model(image, training=False), input_signature=[tf.TensorSpec([1, 192, 480, 3], tf.float32)])
        frozen = convert_variables_to_constants_v2(serving.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        artifact = converter.convert()
        flatbuffer = schema.Model.GetRootAsModel(artifact, 0)
        operators = sorted({operator_names[flatbuffer.OperatorCodes(i).BuiltinCode()] for i in range(flatbuffer.OperatorCodesLength())})
        assert 'CUSTOM' not in operators
        interpreter = tf.lite.Interpreter(model_content=artifact, num_threads=4)
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        assert len(inputs) == len(outputs) == 1
        inp, out = inputs[0], outputs[0]
        assert inp['shape'].tolist() == [1, 192, 480, 3] and out['shape'].tolist() == [1, 48, 120, 11]
        assert inp['dtype'] == out['dtype'] == np.float32
        lite = []
        for image in images:
            interpreter.set_tensor(inp['index'], image[None]); interpreter.invoke()
            lite.append(interpreter.get_tensor(out['index'])[0].copy())
        lite = np.asarray(lite)
        check_outputs(lite)
        np.testing.assert_allclose(lite, actual, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(lite, oracle[label], atol=1e-4, rtol=1e-4)
        require_same_decisions(expected_decisions, decision_rows(lite, images, thresholds))
        for old, new in zip(before, model.get_weights(), strict=True):
            np.testing.assert_array_equal(old, new)
        path = args.run/f'{label}-conversion-probe.tflite'; path.write_bytes(artifact)
        reports[label] = {'checkpoint_sha256': init[hash_key], 'artifact_sha256': digest(path), 'artifact_bytes': len(artifact),
                          'operators': operators, 'builtins_only': True, 'samples': 6, 'all_finite': True,
                          'torch_vs_tf_max_abs': float(np.max(np.abs(oracle[label]-actual))),
                          'tf_vs_tflite_max_abs': float(np.max(np.abs(actual-lite))),
                          'torch_vs_tflite_max_abs': float(np.max(np.abs(oracle[label]-lite))),
                          'train_eval_absolute_tolerance': 1e-6, 'all_moments_and_weights_unchanged': True,
                          'input': {'count': 1, 'shape': inp['shape'].tolist(), 'dtype': 'float32'},
                          'output': {'count': 1, 'shape': out['shape'].tolist(), 'dtype': 'float32'},
                          'raw_and_quality_effective_decisions_equal': True, 'decisions': expected_decisions}
        del interpreter, model
    assert reports['initial']['operators'] == reports['nonzero']['operators']
    source = Path(__file__); (args.run/'source'/source.name).write_bytes(source.read_bytes())
    for name in decision_sources:
        (args.run/'source'/name).write_bytes((source.parent/name).read_bytes())
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Six fixed TRAIN/analytic inputs, initial and actual nonzero affine copies; no accuracy or candidate activation.',
              'initialization_report_sha256': digest(init_path), 'source_sha256': digest(__file__),
              'oracle_report_sha256': digest(oracle_report_path), 'oracle_sha256': digest(oracle_path),
              'selected_validation_report': str(selected_path), 'selected_validation_sha256': digest(selected_path),
              'thresholds': thresholds, 'decision_sources_sha256': decision_sources,
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-4, 'probes': reports,
              'conversion_probes_must_not_be_activated_or_resumed': True, 'trained_candidate_requires_full405_parity': True}
    save(destination, report); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
