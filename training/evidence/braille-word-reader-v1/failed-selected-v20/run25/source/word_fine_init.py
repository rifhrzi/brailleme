"""CPU initialization proof for a separate stride2 detail branch experiment."""
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

from word_fine_branch import extend_fine_branch, fine_branch_spec, inherited_weight_pairs
from word_data import make_target
from word_model import image_normalization_spec, word_relative_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--static-audit', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new initialization directory; do not overwrite proofs')
    static = json.loads(args.static_audit.read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    keras.utils.set_random_seed(20260905)
    parent_hash = digest(args.parent)
    assert static['status'] == 'static_config_and_arithmetic_passed'
    assert static['parent']['sha256'] == parent_hash
    assert static['parent']['all_batchnorm_frozen'] is True
    parent = keras.models.load_model(args.parent, compile=False)
    original = parent.get_weights()
    model = extend_fine_branch(parent)
    records = json.loads((args.data/'train.json').read_text())
    indices = [0, len(records)//2, len(records)-1]
    selected = [records[index] for index in indices]
    train = np.stack([np.asarray(Image.open(args.data/row['image']).convert('RGB'), np.float32)/255. for row in selected])
    images = np.concatenate((train, np.stack([np.zeros_like(train[0]), np.ones_like(train[0]),
                              np.random.default_rng(20260905).random(train[0].shape, dtype=np.float32)])))
    def infer(current, training=False):
        with torch.no_grad():
            return np.stack([current(image[None], training=training).numpy()[0] for image in images])
    expected, actual = infer(parent), infer(model)
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(actual, infer(model, True))
    args.output.mkdir(parents=True)
    initial_path = args.output/'fine-parent.keras'
    model.save(initial_path)
    initial_hash = digest(initial_path)
    reloaded = keras.models.load_model(initial_path, compile=False)
    np.testing.assert_array_equal(actual, infer(reloaded))
    for before, after in zip(model.get_weights(), reloaded.get_weights(), strict=True):
        np.testing.assert_array_equal(before, after)
    label = selected[0]
    truth = make_target(np.asarray(label['boxes'], np.float32), np.asarray(label['masks']), label.get('originally_complete'))[None]
    lateral = reloaded.get_layer('fpn_2_zero').trainable_weights
    depthwise = reloaded.get_layer('fpn_2_depthwise').trainable_weights
    def gradients():
        reloaded.zero_grad(set_to_none=True)
        loss = word_relative_loss(torch.from_numpy(truth), reloaded(train[:1], training=True))
        loss.backward()
        def norm(variables):
            assert all(var.value.grad is not None and torch.isfinite(var.value.grad).all() for var in variables)
            return sum(float(var.value.grad.abs().sum().detach()) for var in variables)
        return {'loss': float(loss.detach()), 'lateral_l1': norm(lateral), 'depthwise_l1': norm(depthwise)}
    first = gradients()
    assert first['lateral_l1'] > 0 and first['depthwise_l1'] == 0
    # Only this separate reload receives one bounded new-lateral gradient step.
    # The saved initial parent and all inherited arrays remain untouched.
    with torch.no_grad():
        for variable in lateral:
            gradient = variable.value.grad
            variable.value.add_(gradient, alpha=-1e-2/max(float(gradient.abs().max()), 1e-12))
    second = gradients()
    assert second['depthwise_l1'] > 0 and np.isfinite(list(first.values())+list(second.values())).all()
    for old, copied in inherited_weight_pairs(parent, reloaded):
        for before, after in zip(old.get_weights(), copied.get_weights(), strict=True):
            np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(model.get_layer('fpn_2_depthwise').get_weights()[0],
                                  reloaded.get_layer('fpn_2_depthwise').get_weights()[0])
    diagnostic_path = args.output/'nonzero-diagnostic-only.keras'
    reloaded.save(diagnostic_path)
    nonzero = infer(reloaded)
    np.testing.assert_array_equal(nonzero, infer(reloaded, True))
    diagnostic_reload = keras.models.load_model(diagnostic_path, compile=False)
    np.testing.assert_array_equal(nonzero, infer(diagnostic_reload))
    assert np.isfinite(nonzero).all() and float(np.max(np.abs(nonzero-actual))) > 1e-3
    assert digest(initial_path) == initial_hash and digest(args.parent) == parent_hash
    for before, after in zip(original, parent.get_weights(), strict=True):
        np.testing.assert_array_equal(before, after)
    oracle_path = args.output/'initialization-oracle.npz'
    np.savez_compressed(oracle_path, images=images, parent=expected, initial=actual, nonzero=nonzero)
    names = ('word_fine_branch.py', 'word_fine_init.py', 'word_model.py', 'word_data.py')
    source = args.output/'source'; source.mkdir()
    for name in names:
        (source/name).write_bytes((Path(__file__).parent/name).read_bytes())
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Six TRAIN/analytic initialization inputs, one TRAIN gradient probe; no selection or accuracy claim.',
              'parent_checkpoint': str(args.parent), 'parent_checkpoint_sha256': parent_hash,
              'initial_checkpoint': str(initial_path), 'initial_checkpoint_sha256': initial_hash,
              'nonzero_diagnostic_checkpoint': str(diagnostic_path), 'nonzero_diagnostic_sha256': digest(diagnostic_path),
              'nonzero_diagnostic_must_never_be_resumed': True,
              'source_sha256': {name: digest(source/name) for name in names},
              'static_audit': str(args.static_audit), 'static_audit_sha256': digest(args.static_audit),
              'static_audit_loaded': isinstance(static, dict), 'oracle_sha256': digest(oracle_path),
              'train_manifest_sha256': digest(args.data/'train.json'),
              'train_inputs': [{'index': index, 'image': row['image'], 'source': row['source'],
                                'sha256': digest(args.data/row['image'])} for index, row in zip(indices, selected, strict=True)],
              'analytic_inputs': ['black', 'white', 'seed20260905 uniform random'],
              'fine_branch': fine_branch_spec(model), 'parameters': model.count_params(),
              'inherited_array_count': len(original), 'array_count': len(model.get_weights()),
              'imageNormalization': image_normalization_spec(model), 'ordinary_frozen_batchnorm_layers': 52,
              'all_inherited_arrays_and_trainability_equal': True,
              'initial_train_eval_parent_outputs_equal': True, 'initial_save_reload_values_and_outputs_equal': True,
              'diagnostic_save_reload_outputs_equal': True, 'diagnostic_only_new_lateral_changed': True,
              'saved_parent_and_initial_unchanged': True, 'zero_branch_gradients': first,
              'after_isolated_lateral_step_gradients': second,
              'diagnostic_step': 'New lateral only; per-array gradient normalized to maximum absolute update .01, ensuring the nonzero output exceeds conversion tolerance.',
              'nonzero_max_output_delta': float(np.max(np.abs(nonzero-actual))),
              'backend': 'torch', 'torch_version': torch.__version__, 'keras_version': keras.__version__,
              'device': 'cpu', 'threads': 2, 'seed': 20260905,
              'conversion_required_before_training': 'Initial and actual nonzero Torch/TF/TFLite six-input numerical and raw/effective decision parity; all builtins. Trained winner still requires full405 parity.'}
    (args.output/'fine-initialization.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
