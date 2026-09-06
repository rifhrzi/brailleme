"""CPU-only one-TRAIN-example initialization proof; never train or select here."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KERAS_TORCH_DEVICE'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
from PIL import Image
import torch

from word_bn_affine import FrozenMomentBatchNormalization, affine_backbone_spec, build_bn_affine_parent
from word_data import make_target
from word_model import word_relative_loss, image_normalization_spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_digest(values):
    result = hashlib.sha256()
    for value in values:
        result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--activation-report', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new initialization directory; existing proofs must not be overwritten')
    activation = json.loads(args.activation_report.read_text())
    parent_hash = digest(args.parent)
    assert activation['checkpoint_sha256'] == parent_hash
    assert activation['all_model_values_unchanged'] is True
    records = json.loads((args.data/'train.json').read_text())
    record = records[0]
    assert record['boxes'] and not record['source'].startswith('dsbi/')
    # Exactly one original cached TRAIN example, not a held or external fixture.
    pixels = np.asarray(Image.open(args.data/record['image']).convert('RGB'), np.float32)[None]/255.
    boxes = np.asarray(record['boxes'], np.float32)
    complete = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= 480) & (boxes[:, 3] <= 192)
    if 'originally_complete' in record:
        complete &= np.asarray(record['originally_complete'], bool)
    truth = make_target(boxes, np.asarray(record['masks'], int), complete)[None]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = keras.models.load_model(args.parent, compile=False)
    original = parent.get_weights()
    initial, spec = build_bn_affine_parent(parent)
    assert spec['batchnorm_layers'] == spec['layer_forced_frozen_moments'] == 52
    with torch.no_grad():
        expected = parent(pixels, training=False).numpy()
        eval_output = initial(pixels, training=False).numpy()
        train_output = initial(pixels, training=True).numpy()
    np.testing.assert_array_equal(expected, eval_output)
    np.testing.assert_array_equal(expected, train_output)
    for before, after in zip(original, initial.get_weights(), strict=True):
        np.testing.assert_array_equal(before, after)
    args.output.mkdir(parents=True)
    source_names = ('word_bn_affine.py', 'word_bn_affine_init.py', 'word_model.py', 'word_data.py')
    (args.output/'source').mkdir()
    for name in source_names:
        (args.output/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    initial_path = args.output/'bn-affine-parent.keras'
    initial.save(initial_path)
    reloaded = keras.models.load_model(initial_path, compile=False)
    assert affine_backbone_spec(reloaded) == spec
    with torch.no_grad():
        np.testing.assert_array_equal(reloaded(pixels, training=True).numpy(), expected)
    for before, after in zip(original, reloaded.get_weights(), strict=True):
        np.testing.assert_array_equal(before, after)
    # Real unchanged loss and TRAIN labels establish gradient reachability;
    # no optimizer or weight update is performed by this backward pass.
    prediction = reloaded(pixels, training=True)
    loss = word_relative_loss(torch.from_numpy(truth), prediction)
    loss.backward()
    gradient_rows = []
    for layer in reloaded.get_layer('mobilenetv2_half').layers:
        if not isinstance(layer, FrozenMomentBatchNormalization):
            continue
        row = {'layer': layer.name}
        for name in ('gamma', 'beta'):
            variable = getattr(layer, name)
            gradient = variable.value.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            row[name+'_gradient_l1'] = float(gradient.abs().sum())
        for name in ('moving_mean', 'moving_variance'):
            variable = getattr(layer, name)
            assert not variable.trainable and not variable.value.requires_grad and variable.value.grad is None
        gradient_rows.append(row)
    assert all(row['gamma_gradient_l1'] > 0 and row['beta_gradient_l1'] > 0 for row in gradient_rows)
    for before, after in zip(original, reloaded.get_weights(), strict=True):
        np.testing.assert_array_equal(before, after)
    # A fixed, bounded, analytic perturbation makes a separate nonzero export
    # probe. It is not a learned checkpoint and must never be resumed/selected.
    diagnostic = keras.models.load_model(initial_path, compile=False)
    changed_affine_tensors = 0
    for layer in diagnostic.get_layer('mobilenetv2_half').layers:
        if not isinstance(layer, FrozenMomentBatchNormalization):
            continue
        for offset, name in enumerate(('gamma', 'beta')):
            variable = getattr(layer, name)
            value = variable.numpy()
            delta = .001*np.where((np.arange(value.size)+offset) % 2 == 0, 1., -1.).reshape(value.shape)
            replacement = (value+delta).astype(value.dtype)
            assert np.any(replacement != value)
            variable.assign(replacement)
            changed_affine_tensors += 1
    diagnostic_path = args.output/'nonzero-diagnostic-only.keras'
    diagnostic.save(diagnostic_path)
    with torch.no_grad():
        nonzero = diagnostic(pixels, training=True).numpy()
    assert np.isfinite(nonzero).all() and np.max(np.abs(nonzero-expected)) > 1e-6
    parent_backbone = parent.get_layer('mobilenetv2_half')
    for layer in diagnostic.get_layer('mobilenetv2_half').layers:
        source = parent_backbone.get_layer(layer.name)
        if isinstance(layer, FrozenMomentBatchNormalization):
            np.testing.assert_array_equal(layer.moving_mean.numpy(), source.moving_mean.numpy())
            np.testing.assert_array_equal(layer.moving_variance.numpy(), source.moving_variance.numpy())
        else:
            for before, after in zip(source.get_weights(), layer.get_weights(), strict=True):
                np.testing.assert_array_equal(before, after)
    assert digest(args.parent) == parent_hash and arrays_digest(parent.get_weights()) == arrays_digest(original)
    oracle_path = args.output/'initialization-one-train-oracle.npz'
    np.savez_compressed(oracle_path, images=pixels, truth=truth, parent=expected, initial=eval_output, nonzero=nonzero)
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'One cached TRAIN input on CPU; initial equality/reload and real-loss backward without optimizer. Separate fixed .001 alternating affine perturbation is only a forbidden export diagnostic.',
              'parent_checkpoint': str(args.parent), 'parent_checkpoint_sha256': parent_hash,
              'initial_checkpoint': str(initial_path), 'initial_checkpoint_sha256': digest(initial_path),
              'nonzero_diagnostic_checkpoint': str(diagnostic_path), 'nonzero_diagnostic_sha256': digest(diagnostic_path),
              'nonzero_diagnostic_must_never_be_resumed': True,
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in source_names},
              'activation_report': str(args.activation_report), 'activation_report_sha256': digest(args.activation_report),
              'input': {'image': record['image'], 'source': record['source'], 'image_sha256': digest(args.data/record['image']),
                        'train_manifest_sha256': digest(args.data/'train.json'), 'manifest_index': 0},
              'oracle_sha256': digest(oracle_path), 'backend': 'torch', 'torch_version': torch.__version__,
              'keras_version': keras.__version__, 'device': 'cpu', 'threads': 2,
              'parameters': initial.count_params(), 'batchnorm_affine': spec,
              'imageNormalization': image_normalization_spec(initial),
              'all_initial_arrays_equal_parent': True, 'initial_train_eval_parent_outputs_equal': True,
              'save_reload_values_and_outputs_equal': True, 'all_arrays_unchanged_after_backward': True,
              'parent_unchanged': True, 'array_count': len(original), 'initial_arrays_sha256': arrays_digest(original),
              'gradient_loss': float(loss.detach()), 'gradient_rows': gradient_rows,
              'diagnostic_changed_affine_tensors': changed_affine_tensors,
              'diagnostic_other_weights_and_moments_unchanged': True,
              'nonzero_max_output_delta': float(np.max(np.abs(nonzero-expected))),
              'conversion_required_before_training': 'Six fixed TRAIN/analytic inputs; initial and nonzero Torch/TF/TFLite builtins-only parity. This one-input CPU proof alone never permits a training launch.'}
    (args.output/'bn-affine-initialization.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('gradient_rows', 'source_sha256', 'input')}, indent=2))


if __name__ == '__main__':
    main()
