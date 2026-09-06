"""Prepared only: one scoped CPU Torch run after root schedules it."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit('Prepared script only; use --execute solely when root schedules the backend preflight')
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    protocol_path = ROOT / 'fixture-protocol.json'
    assert sha(protocol_path) == '11846237a01fc1fd996cbabfe8a873ef760864f3d10fb57adaa54e5e229eba62'
    p = json.loads(protocol_path.read_text())
    prototype = WORK / 'dot-auxiliary-prototype-v1'
    assert sha(prototype/'index.json') == p['prototype_index_sha256']
    for name, expected in p['prototype_candidate_sha256'].items():
        assert sha(prototype/'candidate'/name) == expected
    for name, expected in p['source_sha256'].items():
        assert sha(ROOT/'source'/name) == expected
    data = WORK / 'data'
    for name, expected in p['data_sha256'].items():
        assert sha(data/name) == expected
    assert sha(p['parent_checkpoint']) == p['parent_checkpoint_sha256']
    assert sha(WORK/'v15-train-batchnorm-activations/protocol.json') == p['cohort_sha256']
    assert sha(WORK/'geometry-head-parent-v1/geometry-initialization.json') == p['negative_cohort_sha256']
    records = json.loads((WORK/'combined-training-v5/combined-training-manifest.json').read_text())
    selected = []
    split = json.loads((data/'source-split.json').read_text())['images']
    for sample in p['samples']:
        row = records[sample['manifest_index']]
        assert hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == sample['record_json_sha256']
        assert row['image'] == sample['image'] and row['source'] == sample['source']
        assert row['source'] not in split['validation'] and row['source'] not in split['external']
        assert sha(data/row['image']) == sample['sha256']
        selected.append(row)

    os.environ.update(KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cpu', CUDA_VISIBLE_DEVICES='-1')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import keras
    from keras import ops
    keras.utils.set_random_seed(20260905)
    sys.path[:0] = [str(prototype/'candidate'), str(ROOT/'source')]
    from word_point_metadata import load_point_states
    from word_model import word_relative_loss, image_normalization_spec
    from word_dot_aux import wrap_dot_training, known_dot_bce
    from word_dot_aux_targets import complete_dot_queries, pad_dot_queries
    baseline = load_module('dot_preflight_baseline_train', ROOT/'source/word_train.py')
    candidate = load_module('dot_preflight_candidate_train', prototype/'candidate/word_train.py')
    states, point_audit = load_point_states('../dsbi-point-metadata-v1/points.json', data,
                                           ['../dsbi-training-contexts-v2/source-index.json'], records)
    del records

    def array_hash(values):
        h = hashlib.sha256()
        for value in values:
            value = np.asarray(value)
            h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()

    def arrays_equal(first, second):
        assert len(first) == len(second)
        for before, after in zip(first, second, strict=True): np.testing.assert_array_equal(before, after)

    def make_batch(module, auxiliary, validation=False):
        chosen = [selected[0], selected[-1]] if validation else selected
        options = dict(illumination_augmentation=not validation, balance_source_families=not validation,
                       point_states=None if validation else states)
        if module is candidate: options['auxiliary_dot_supervision'] = auxiliary
        dataset = module.Patches(data, chosen, len(chosen), not validation, 20260905, **options)
        if not validation: dataset.on_epoch_end()
        traces, actual_states = [], []
        augment = module.augment_patch

        def traced(*args, **kwargs):
            rng = args[2]
            before = deepcopy(rng.bit_generator.state)
            result = augment(*args, **kwargs)
            traces.append({'before': before, 'after': deepcopy(rng.bit_generator.state)})
            actual_states.append({'state': None if kwargs.get('point_state') is None else result[3],
                                  'complete': result[2].copy()})
            return result

        module.augment_patch = traced
        try: batch = dataset[0]
        finally: module.augment_patch = augment
        return dataset, batch, traces, actual_states

    old_dataset, (images, truth), old_rng, _ = make_batch(baseline, False)
    off_dataset, (off_images, off_truth), off_rng, _ = make_batch(candidate, False)
    on_dataset, ((on_images, queries), (on_truth, occupancy)), on_rng, transformed_states = make_batch(candidate, True)
    arrays_equal([images, truth, old_dataset.order], [off_images, off_truth, off_dataset.order])
    arrays_equal([images, truth, old_dataset.order], [on_images, on_truth, on_dataset.order])
    assert old_rng == off_rng == on_rng
    assert old_dataset.epoch == on_dataset.epoch == 2 and np.any(occupancy >= 0)
    assert np.isfinite(queries).all() and set(np.unique(occupancy)).issubset({-1., 0., 1.})
    for batch_index, record_index in enumerate(on_dataset.order):
        row = selected[int(record_index)]
        if not row['source'].startswith('dsbi/'):
            assert np.all(occupancy[batch_index] == -1)
            continue
        # Independent array oracle using the actual augmented source state.
        actual = transformed_states[batch_index]
        state = actual['state']
        assert state is not None
        labels = ((np.asarray(row['masks'])[:, None] >> np.arange(6)) & 1).astype(np.float32)
        positions = (state.points-3.)/4.
        eligible = actual['complete'][:len(labels)]
        eligible = eligible & ((positions >= [0, 0]) & (positions <= [119, 47])).all(axis=(1, 2))
        eligible &= ((labels == 0) | state.survivors).all(axis=1)
        expected_queries = positions[eligible].reshape(-1, 2).astype(np.float32)
        expected_labels = labels[eligible].reshape(-1, 1)
        count = len(expected_labels)
        np.testing.assert_array_equal(queries[batch_index, :count], expected_queries)
        np.testing.assert_array_equal(occupancy[batch_index, :count], expected_labels)
        assert np.all(occupancy[batch_index, count:] == -1)
    _, (val_images, val_truth), _, _ = make_batch(baseline, False, validation=True)
    _, ((val_on_images, val_queries), (val_on_truth, val_occupancy)), _, _ = make_batch(candidate, True, validation=True)
    arrays_equal([val_images, val_truth], [val_on_images, val_on_truth])
    assert np.all(val_occupancy == -1)
    np.savez_compressed(out/'actual-patches.npz', images=images, truth=truth, queries=queries,
                        occupancy=occupancy, val_images=val_images, val_truth=val_truth)
    (out/'rng-traces.json').write_text(json.dumps(old_rng, indent=2)+'\n')
    print('PATCHES equality and explicit TRAIN queries passed', flush=True)

    reader = keras.models.load_model(p['parent_checkpoint'], compile=False)
    assert reader.count_params() == 829307 and len(reader.get_weights()) == 280
    assert image_normalization_spec(reader)['targetStd'] == .05 and image_normalization_spec(reader)['stdFloor'] == .001
    backbone = reader.get_layer('mobilenetv2_half')
    bn = [layer for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)]
    assert len(bn) == 52 and not any(layer.trainable for layer in bn)
    assert sorted(192//int(t.shape[1]) for t in backbone.outputs) == [4, 8, 16, 32]
    graph = deepcopy(reader.get_config())
    before_weights = reader.get_weights()
    bn_before = [value.copy() for layer in bn for value in layer.get_weights()]
    with torch.no_grad(): initial = ops.convert_to_numpy(reader(images, training=False))
    training = wrap_dot_training(reader, word_relative_loss, p['diagnostic_step_learning_rate'])
    assert reader.get_config() == graph
    arrays_equal(before_weights, reader.get_weights())
    assert {id(w) for w in reader.weights} <= {id(w) for w in training.weights}
    assert len(training.weights) == 282
    assert training.count_params() == 829307 + int(reader.get_layer('fpn_4').input.shape[-1]) + 1
    with torch.no_grad(): wrapped_initial = ops.convert_to_numpy(training((images, queries), training=False)[0])
    np.testing.assert_array_equal(initial, wrapped_initial)
    assert float(training.optimizer.iterations.numpy()) == 0
    print('Reader graph, weights and initial main outputs unchanged', flush=True)

    # This probe exposes the exact shared fine tensor, not a copied feature model.
    fine_tensor = reader.get_layer('fpn_4').input
    auxiliary_probe = keras.Model(training.inputs, (fine_tensor, training.outputs[1]))
    training.zero_grad()
    fine_values, logits = auxiliary_probe((images, queries), training=True)
    fine_values.retain_grad()
    aux_loss = known_dot_bce(torch.from_numpy(occupancy), logits)
    aux_loss.backward()
    fine_gradient = float(fine_values.grad.abs().sum())
    head_gradient = float(training.get_layer('auxiliary_dot_logits').kernel.value.grad.abs().sum())
    conv_gradient = float(backbone.get_layer('block_3_expand').kernel.value.grad.abs().sum())
    assert all(np.isfinite(value) and value > 0 for value in (fine_gradient, head_gradient, conv_gradient))
    arrays_equal(before_weights, reader.get_weights())
    del fine_values, logits, aux_loss

    training.zero_grad()
    _, ignored_logits = auxiliary_probe((val_images, val_queries), training=True)
    ignored_loss = known_dot_bce(torch.from_numpy(val_occupancy), ignored_logits)
    assert torch.isfinite(ignored_logits).all() and float(ignored_loss.detach()) == 0.
    ignored_loss.backward()
    for variable in training.trainable_weights:
        gradient = variable.value.grad
        assert gradient is None or (torch.isfinite(gradient).all() and not torch.count_nonzero(gradient))
    arrays_equal(before_weights, reader.get_weights())
    del ignored_logits, ignored_loss

    with torch.no_grad():
        outputs = training((images, queries), training=False)
        main_value = word_relative_loss(torch.from_numpy(truth), outputs[0])
        auxiliary_value = known_dot_bce(torch.from_numpy(occupancy), outputs[1])
        expected_total = main_value + .25*auxiliary_value
        actual_total = training.compute_loss(x=(images, queries), y=(truth, occupancy), y_pred=outputs, training=False)
        torch.testing.assert_close(actual_total, expected_total, rtol=0., atol=1e-6)
        validation_outputs = training((val_images, val_queries), training=False)
        validation_main = word_relative_loss(torch.from_numpy(val_truth), validation_outputs[0])
        validation_aux = known_dot_bce(torch.from_numpy(val_occupancy), validation_outputs[1])
        assert float(validation_aux) == 0.
        validation_total = training.compute_loss(x=(val_images, val_queries), y=(val_truth, val_occupancy), y_pred=validation_outputs, training=False)
        torch.testing.assert_close(validation_total, validation_main, rtol=0., atol=1e-6)
        loss_values = {'main': float(main_value), 'auxiliary': float(auxiliary_value), 'total': float(actual_total),
                       'validation_main': float(validation_main), 'validation_auxiliary': float(validation_aux)}
    training.reset_metrics()
    validation_metrics = training.test_on_batch((val_images, val_queries), (val_truth, val_occupancy), return_dict=True)
    np.testing.assert_allclose(validation_metrics['loss'], loss_values['validation_main'], rtol=0., atol=1e-6)
    arrays_equal(before_weights, reader.get_weights())
    print('Aux-only gradients and main/aux validation loss accounting passed', flush=True)

    training.reset_metrics()
    step_metrics = training.train_on_batch((images, queries), (truth, occupancy), return_dict=True)
    assert all(np.isfinite(value).all() for value in step_metrics.values())
    assert int(training.optimizer.iterations.numpy()) == 1
    after_weights = reader.get_weights()
    assert array_hash(after_weights) != array_hash(before_weights)
    arrays_equal(bn_before, [value for layer in bn for value in layer.get_weights()])
    assert reader.get_config() == graph
    with torch.no_grad():
        after_main = ops.convert_to_numpy(reader(images, training=False))
        after_wrapper = [ops.convert_to_numpy(t) for t in training((images, queries), training=False)]
    np.testing.assert_array_equal(after_main, after_wrapper[0])
    assert np.isfinite(after_main).all() and np.all((after_main >= 0) & (after_main <= 1))
    reader_path = out/'reader-after-one-step.keras'
    wrapper_path = out/'training-after-one-step.keras'
    reader.save(reader_path)
    training.save(wrapper_path)
    with zipfile.ZipFile(reader_path) as archive: saved_config = json.loads(archive.read('config.json'))
    assert 'annotated_dot' not in json.dumps(saved_config) and 'auxiliary_dot' not in json.dumps(saved_config)
    registry = keras.saving.get_custom_objects()
    removed = {name: registry.pop(name) for name in ('BrailleMe>AnnotatedDotSampler', 'BrailleMe>known_dot_bce') if name in registry}
    try: restored_reader = keras.models.load_model(reader_path, compile=False)
    finally: registry.update(removed)
    assert restored_reader.count_params() == 829307
    assert tuple(restored_reader.input_shape) == (None, 192, 480, 3)
    assert tuple(restored_reader.output_shape) == (None, 48, 120, 11)
    assert restored_reader.get_config() == graph
    arrays_equal(after_weights, restored_reader.get_weights())
    with torch.no_grad(): np.testing.assert_array_equal(after_main, ops.convert_to_numpy(restored_reader(images, training=False)))
    restored_wrapper = keras.models.load_model(wrapper_path, compile=True)
    arrays_equal(training.get_weights(), restored_wrapper.get_weights())
    assert int(restored_wrapper.optimizer.iterations.numpy()) == 1
    arrays_equal([v.numpy() for v in training.optimizer.variables], [v.numpy() for v in restored_wrapper.optimizer.variables])
    with torch.no_grad():
        for expected, actual in zip(after_wrapper, restored_wrapper((images, queries), training=False), strict=True):
            np.testing.assert_array_equal(expected, ops.convert_to_numpy(actual))
    assert sha(p['parent_checkpoint']) == p['parent_checkpoint_sha256']
    report = {'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'protocol_sha256': sha(protocol_path),
              'parent_checkpoint_sha256': p['parent_checkpoint_sha256'], 'torch_version': torch.__version__, 'keras_version': keras.__version__,
              'device': 'cpu', 'samples': p['samples'], 'micro_batch_order': on_dataset.order.tolist(),
              'point_metadata_audit': point_audit, 'actual_patches_sha256': sha(out/'actual-patches.npz'),
              'rng_trace_sha256': sha(out/'rng-traces.json'), 'main_arrays_and_rng_exact_off_and_on': True,
              'known_query_points': int(np.sum(occupancy >= 0)), 'dynamic_query_shape': list(queries.shape),
              'training_parameters': training.count_params(), 'reader_parameters': reader.count_params(),
              'initial_graph_weights_outputs_equal': True, 'main_loss_formula_unchanged': True,
              'auxiliary_gradient_l1': {'fine_features': fine_gradient, 'head_kernel': head_gradient, 'shared_conv': conv_gradient},
              'ignored_loss_and_gradients_exact_zero': True, 'loss_values': loss_values,
              'validation_metrics': {key: float(value) for key, value in validation_metrics.items()},
              'one_step_metrics': {key: float(value) for key, value in step_metrics.items()}, 'optimizer_steps': 1,
              'frozen_52_bn_all_208_arrays_exact': True, 'reader_roundtrip_exact_no_auxiliary_classes': True,
              'wrapper_roundtrip_exact_optimizer_state_readable': True,
              'reader_checkpoint_sha256': sha(reader_path), 'wrapper_checkpoint_sha256': sha(wrapper_path),
              'reader_weights_before_sha256': array_hash(before_weights), 'reader_weights_after_sha256': array_hash(after_weights),
              'scope': 'One diagnostic optimizer step on fixed TRAIN inputs only. Reader checkpoints are forbidden for training initialization/activation; no accuracy, held-selection, TF/Lite, full CLI resume or interruption-exact continuation claim.'}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'report_sha256': sha(out/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
