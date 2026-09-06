"""Initialize and audit a zero geometry residual; save a separate nonzero probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import keras
from keras import ops
import numpy as np
from PIL import Image

from word_data import make_target
from word_geometry_head import extend_geometry_head
from word_model import word_relative_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--slot-priors', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if keras.backend.backend() != 'torch' or os.environ.get('KERAS_TORCH_DEVICE') != 'cpu':
        raise SystemExit('Run this initialization and gradient proof with CPU Torch')
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    keras.utils.set_random_seed(20260905)
    zero_file = args.output/'geometry-parent.keras'
    if zero_file.exists():
        raise SystemExit('Use a new initialization folder; never overwrite a frozen parent')
    priors = json.loads(args.slot_priors.read_text())
    if (priors['status'] != 'passed' or priors['slot_order'] !=
            ['left_top', 'left_middle', 'left_bottom', 'right_top', 'right_middle', 'right_bottom']
            or priors['axis_order'] != ['x', 'y']):
        raise ValueError('Use the audited TRAIN-only point-corner slot priors')
    parent_hash = digest(args.parent)
    parent = keras.models.load_model(args.parent, compile=False)
    model = extend_geometry_head(parent, priors['median_corner_fractions'])
    records = json.loads((args.data/'train.json').read_text())
    selected = [records[index] for index in (0, len(records)//2, len(records)-1)]
    images = np.stack([np.asarray(Image.open(args.data/row['image']).convert('RGB'), np.float32)/255. for row in selected])
    expected = ops.convert_to_numpy(parent(images, training=False))
    actual = ops.convert_to_numpy(model(images, training=False))
    np.testing.assert_array_equal(expected, actual)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(zero_file)
    restored = keras.models.load_model(zero_file, compile=False)
    np.testing.assert_array_equal(expected, ops.convert_to_numpy(restored(images, training=False)))
    for before, after in zip(model.get_weights(), restored.get_weights(), strict=True):
        np.testing.assert_array_equal(before, after)
    analytic = np.stack([np.zeros_like(images[0]), np.ones_like(images[0]),
                         np.random.default_rng(20260905).random(images[0].shape, dtype=np.float32)])
    np.testing.assert_array_equal(ops.convert_to_numpy(parent(analytic, training=False)),
                                  ops.convert_to_numpy(restored(analytic, training=False)))
    labels = np.stack([make_target(np.asarray(row['boxes'], np.float32).reshape(-1, 4),
                                   np.asarray(row['masks']), row.get('originally_complete')) for row in selected])
    target = torch.as_tensor(labels, device=restored.weights[0].value.device)
    final = restored.get_layer('geometry_residual_zero').trainable_weights
    projection = restored.get_layer('geometry_feature_projection').trainable_weights
    hidden = restored.get_layer('geometry_feature_mlp').trainable_weights

    def gradients():
        restored.zero_grad(set_to_none=True)
        loss = word_relative_loss(target, restored(images, training=True))
        loss.backward()
        norm = lambda variables: sum(float(variable.value.grad.abs().sum().detach().cpu())
                                     for variable in variables if variable.value.grad is not None)
        return {'loss': float(loss.detach().cpu()), 'final_l1': norm(final),
                'projection_l1': norm(projection), 'hidden_l1': norm(hidden)}

    first = gradients()
    final_channel_l1 = (final[0].value.grad.abs().sum(dim=0)+final[1].value.grad.abs()).detach().cpu().numpy()
    assert np.all(final_channel_l1 > 0)
    assert first['final_l1'] > 0 and first['projection_l1'] == 0 and first['hidden_l1'] == 0
    # Diagnostic copy only: one bounded final-layer step reveals previously
    # zeroed interpolation/features to conversion and gradient tests.
    with torch.no_grad():
        for variable in final:
            gradient = variable.value.grad
            scale = max(float(gradient.abs().max()), 1.)
            variable.value.add_(gradient, alpha=-1e-3/scale)
    second = gradients()
    assert second['projection_l1'] > 0 and second['hidden_l1'] > 0
    assert np.isfinite(list(first.values())+list(second.values())).all()
    nonzero_file = args.output/'nonzero-diagnostic-only.keras'
    restored.save(nonzero_file)
    nonzero = ops.convert_to_numpy(restored(images, training=False))
    assert np.isfinite(nonzero).all() and np.max(np.abs(nonzero-expected)) > 0
    assert digest(args.parent) == parent_hash
    oracle_images = np.concatenate((images, analytic))
    np.savez_compressed(args.output/'initialization-oracle.npz', images=oracle_images,
                        parent_outputs=ops.convert_to_numpy(parent(oracle_images, training=False)),
                        zero_outputs=ops.convert_to_numpy(model(oracle_images, training=False)),
                        nonzero_outputs=ops.convert_to_numpy(restored(oracle_images, training=False)))
    source_dir = args.output/'source'
    source_dir.mkdir(exist_ok=True)
    for name in ('word_geometry_head.py', 'word_geometry_init.py', 'word_model.py', 'word_data.py'):
        (source_dir/name).write_bytes((Path(__file__).parent/name).read_bytes())
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Initialization equality and separate nonzero diagnostic; no training, checkpoint selection or accuracy claim.',
        'parent_checkpoint': str(args.parent), 'parent_checkpoint_sha256': parent_hash,
        'zero_checkpoint': str(zero_file), 'zero_checkpoint_sha256': digest(zero_file),
        'nonzero_diagnostic_checkpoint': str(nonzero_file), 'nonzero_diagnostic_sha256': digest(nonzero_file),
        'nonzero_diagnostic_must_never_be_resumed': True,
        'slot_prior_audit': str(args.slot_priors), 'slot_prior_audit_sha256': digest(args.slot_priors),
        'source_sha256': {p.name: digest(p) for p in source_dir.glob('*.py')},
        'oracle_sha256': digest(args.output/'initialization-oracle.npz'),
        'train_inputs': [{'image': row['image'], 'source': row['source'], 'sha256': digest(args.data/row['image'])} for row in selected],
        'analytic_inputs': ['black', 'white', 'seed20260905 uniform random'],
        'zero_outputs_equal_parent': True, 'zero_save_reload_all_weights_equal': True,
        'zero_branch_gradients': first, 'after_isolated_final_step_gradients': second,
        'initial_final_residual_gradient_l1_by_channel': final_channel_l1.tolist(),
        'nonzero_max_output_delta': float(np.max(np.abs(nonzero-expected))),
        'parameters': model.count_params(), 'added_parameters': model.count_params()-parent.count_params(),
        'added_conv_dense_macs_per_image': 48*120*(64*8+48*32+32*11),
        'sampling': model.get_layer('geometry_slot_samples').get_config(),
        'conversion_required_before_training': 'Actual nonzero diagnostic copy, builtins-only TFLite, all six oracle inputs; zero branch pruning is insufficient.'}
    (args.output/'geometry-initialization.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
