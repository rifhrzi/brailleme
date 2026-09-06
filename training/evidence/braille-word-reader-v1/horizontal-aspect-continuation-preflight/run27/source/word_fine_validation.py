"""Fail-closed initialization lineage for the opt-in stride2 detail ablation."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_fine_reports(resume, initialization_path, conversion_path):
    """Check immutable evidence without importing a model backend."""
    from word_bn_affine_parity import check_outputs, decision_rows
    initial = json.loads(initialization_path.read_text())
    proof = json.loads(conversion_path.read_text())
    if (initial['status'] != 'passed' or proof['status'] != 'passed'
            or digest(resume) != initial['initial_checkpoint_sha256']
            or digest(resume) == initial['nonzero_diagnostic_sha256']
            or digest(initialization_path) != proof['initialization_report_sha256']
            or initial['oracle_sha256'] != proof['oracle_sha256']):
        raise ValueError('Resume only the exact initial stride2 parent after its actual nonzero conversion proof')
    for key in ('parent_checkpoint', 'initial_checkpoint', 'nonzero_diagnostic_checkpoint'):
        hash_key = 'nonzero_diagnostic_sha256' if key == 'nonzero_diagnostic_checkpoint' else key+'_sha256'
        if digest(initial[key]) != initial[hash_key]:
            raise ValueError('Stride2 initialization checkpoint changed')
    for report in (initial, proof):
        required = ({'word_fine_branch.py', 'word_fine_init.py', 'word_model.py', 'word_data.py'} if report is initial else
                    {'word_fine_parity.py', 'word_bn_affine_parity.py', 'word_decode.py', 'word_benchmark.py', 'benchmark_angelina.py', 'word_data.py'})
        if set(report['source_sha256']) != required:
            raise ValueError('Stride2 proof must bind every executed source dependency')
        for name, expected in report['source_sha256'].items():
            if digest(Path(__file__).parent/name) != expected:
                raise ValueError(f'Stride2 proof source changed: {name}')
    if digest(initial['static_audit']) != initial['static_audit_sha256']:
        raise ValueError('Stride2 static coordinate audit changed')
    static = json.loads(Path(initial['static_audit']).read_text())
    if static['parent']['sha256'] != initial['parent_checkpoint_sha256']:
        raise ValueError('Stride2 static coordinate audit must bind the base parent')
    selected_path = Path(proof['selected_validation_report'])
    selected = json.loads(selected_path.read_text())
    if (digest(selected_path) != proof['selected_validation_sha256']
            or selected['checkpoint_sha256'] != initial['parent_checkpoint_sha256']
            or selected['thresholds'] != proof['thresholds']):
        raise ValueError('Stride2 proof must preserve the actual base checkpoint thresholds')
    for key in ('nonzero_diagnostic_must_never_be_resumed', 'all_inherited_arrays_and_trainability_equal',
                'initial_train_eval_parent_outputs_equal', 'initial_save_reload_values_and_outputs_equal',
                'diagnostic_save_reload_outputs_equal', 'diagnostic_only_new_lateral_changed',
                'saved_parent_and_initial_unchanged'):
        if initial[key] is not True:
            raise ValueError('Stride2 initialization proof is incomplete')
    if (initial['ordinary_frozen_batchnorm_layers'] != 52 or initial['inherited_array_count'] != 280
            or initial['array_count'] != 283 or initial['parameters'] != 830539
            or proof['absolute_tolerance'] != 1e-4 or proof['relative_tolerance'] != 1e-4
            or proof['conversion_probes_must_not_be_activated_or_resumed'] is not True
            or proof['trained_candidate_requires_full405_parity'] is not True):
        raise ValueError('Stride2 proof contract changed')
    for value in (initial['nonzero_max_output_delta'], proof['nonzero_tf_output_delta'], proof['nonzero_tflite_output_delta']):
        if not np.isfinite(value) or not 1e-3 < value <= 1.:
            raise ValueError('Stride2 diagnostic must produce finite outputs beyond conversion tolerance')
    first, second = initial['zero_branch_gradients'], initial['after_isolated_lateral_step_gradients']
    if (not np.isfinite(list(first.values())+list(second.values())).all()
            or first['depthwise_l1'] != 0 or first['lateral_l1'] <= 0
            or second['depthwise_l1'] <= 0 or second['lateral_l1'] <= 0):
        raise ValueError('Stride2 gradient reachability proof is incomplete')
    oracle_path = initialization_path.parent/'initialization-oracle.npz'
    if digest(oracle_path) != initial['oracle_sha256']:
        raise ValueError('Stride2 six-input oracle changed')
    with np.load(oracle_path) as oracle:
        images = oracle['images']
        if (images.shape != (6, 192, 480, 3) or images.dtype != np.float32
                or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1):
            raise ValueError('Stride2 oracle must contain six finite RGB float32 inputs')
        for label in ('parent', 'initial', 'nonzero'):
            check_outputs(oracle[label])
        np.testing.assert_array_equal(oracle['parent'], oracle['initial'])
        if float(np.max(np.abs(oracle['nonzero']-oracle['initial']))) != initial['nonzero_max_output_delta']:
            raise ValueError('Stride2 diagnostic delta must match actual saved outputs')
        expected = {label: decision_rows(oracle[label], images, proof['thresholds'])
                    for label in ('parent', 'initial', 'nonzero')}
    if proof['torch_parent_decisions'] != expected['parent']:
        raise ValueError('Stride2 parent decisions do not match the saved oracle')
    for label, checkpoint_hash in [('initial', initial['initial_checkpoint_sha256']), ('nonzero', initial['nonzero_diagnostic_sha256'])]:
        probe = proof['probes'][label]
        if (probe['checkpoint_sha256'] != checkpoint_hash or probe['samples'] != 6
                or not all(probe[key] is True for key in ('builtins_only', 'all_finite', 'eager_train_eval_exact',
                           'all_moments_and_weights_unchanged', 'raw_and_quality_effective_decisions_equal'))
                or probe['input'] != {'count': 1, 'shape': [1, 192, 480, 3], 'dtype': 'float32'}
                or probe['output'] != {'count': 1, 'shape': [1, 48, 120, 11], 'dtype': 'float32'}
                or probe['decisions'] != expected[label]
                or not probe['operators'] or any('CUSTOM' in name or 'Flex' in name for name in probe['operators'])
                or digest(conversion_path.parent/f'{label}-conversion-probe.tflite') != probe['artifact_sha256']):
            raise ValueError('Stride2 initial/nonzero conversion probe is incomplete or changed')
        for key in ('torch_vs_tf_max_abs', 'tf_vs_tflite_max_abs', 'torch_vs_tflite_max_abs', 'compiled_vs_eager_max_abs'):
            if not np.isfinite(probe[key]) or not 0 <= probe[key] <= 2e-4:
                raise ValueError('Stride2 numeric summary contradicts the probability parity tolerance')
    return initial, proof


def validated_fine_branch(model, resume, initialization_path, conversion_path):
    enabled = any(layer.name == 'fpn_2_zero' for layer in model.layers)
    if enabled != bool(initialization_path) or bool(initialization_path) != bool(conversion_path):
        raise ValueError('Stride2 training requires both explicit initialization and conversion reports')
    if not enabled:
        return {'enabled': False}
    initial, _ = require_fine_reports(resume, initialization_path, conversion_path)
    from word_fine_branch import fine_branch_spec
    from word_model import image_normalization_spec
    import keras
    spec = fine_branch_spec(model)
    batchnorm = [layer for layer in model.get_layer('mobilenetv2_half').layers if isinstance(layer, keras.layers.BatchNormalization)]
    if (spec != initial['fine_branch'] or model.count_params() != 830539
            or image_normalization_spec(model) != initial['imageNormalization']
            or len(batchnorm) != 52 or any(type(layer) is not keras.layers.BatchNormalization or layer.trainable for layer in batchnorm)
            or any(np.any(value) for value in model.get_layer('fpn_2_zero').get_weights())
            or not np.any(model.get_layer('fpn_2_depthwise').get_weights()[0])):
        raise ValueError('Stride2 model must preserve plain frozen BN and its exact initial branch')
    depthwise, lateral, addition = [model.get_layer(name) for name in ('fpn_2_depthwise', 'fpn_2_zero', 'fpn_4_with_2')]
    if (type(depthwise) is not keras.layers.DepthwiseConv2D or depthwise.depth_multiplier != 1
            or depthwise.dilation_rate != (1, 1) or depthwise.get_config()['activation'] != 'linear'
            or type(lateral) is not keras.layers.Conv2D or lateral.strides != (1, 1)
            or lateral.get_config()['activation'] != 'linear' or type(addition) is not keras.layers.Add
            or [tuple(value.shape) for value in depthwise.get_weights()] != [(3, 3, 16, 1)]
            or [tuple(value.shape) for value in lateral.get_weights()] != [(1, 1, 16, 64), (64,)]
            or any(not np.isfinite(value).all() for value in depthwise.get_weights())
            or [value._keras_history.operation.name for value in addition.input] != ['add_1', 'fpn_2_zero']
            or model.get_layer('activation').input._keras_history.operation.name != 'fpn_4_with_2'):
        raise ValueError('Stride2 branch must retain its exact linear depthwise/lateral/add topology')
    return {'enabled': True, 'initialization_report': str(initialization_path),
            'initialization_report_sha256': digest(initialization_path),
            'conversion_report': str(conversion_path), 'conversion_report_sha256': digest(conversion_path),
            'base_parent_checkpoint_sha256': initial['parent_checkpoint_sha256'],
            'initial_checkpoint_sha256': initial['initial_checkpoint_sha256'],
            'nonzero_diagnostic_sha256': initial['nonzero_diagnostic_sha256'],
            'nonzero_diagnostic_forbidden': True, 'spec': spec}
