"""Create an auditable, initially identical trainable-BatchNorm branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
import keras
import numpy as np
from PIL import Image

from word_data import save_json
from word_model import image_normalization_spec


def enable_training_batchnorm(model, momentum=.99):
    """Keep learned values while letting the backbone inherit training mode."""
    if not 0 <= momentum < 1:
        raise ValueError('BatchNorm momentum must be in [0,1)')

    def clone_layer(layer):
        config = layer.get_config()
        if isinstance(layer, keras.layers.BatchNormalization):
            config.update(trainable=True, momentum=float(momentum))
        return layer.__class__.from_config(config)

    def call_layer(layer, *args, **kwargs):
        if layer.name == 'mobilenetv2_half':
            kwargs.pop('training', None)
        return layer(*args, **kwargs)

    branch = keras.models.clone_model(model, clone_function=clone_layer,
                                      call_function=call_layer, recursive=True)
    branch.set_weights(model.get_weights())
    if len(branch.weights) != len(model.weights) or any(
            not np.array_equal(a, b) for a, b in zip(model.get_weights(), branch.get_weights())):
        raise ValueError('BatchNorm branch changed initial learned values or weight ordering')
    return branch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--train-manifest', default='train-coverage-v2.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--momentum', type=float, default=.99)
    args = parser.parse_args()
    artifact = args.output/'batchnorm-parent.keras'
    if artifact.exists():
        raise SystemExit('Use a new output directory; an existing branch is immutable')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = keras.models.load_model(args.checkpoint, compile=False)
    branch = enable_training_batchnorm(source, args.momentum)
    manifest_path = args.data/args.train_manifest
    records = json.loads(manifest_path.read_text())
    positives = [row for row in records if row['boxes']]
    sample_indices = np.linspace(0, len(positives)-1, 16, dtype=int)
    sample = [positives[index] for index in sample_indices]
    images = np.stack([np.asarray(Image.open(args.data/row['image']).convert('RGB'), np.float32)/255 for row in sample])
    # This diagnostic uses TRAIN only. Its moment update is reset before saving;
    # no validation/external image contributes to the initial branch values.
    before = branch.get_weights()
    backbone = branch.get_layer('mobilenetv2_half')
    bn_layers = [layer for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)]
    initial_moments = {layer.name: (layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy()) for layer in bn_layers}
    with torch.no_grad():
        original_output = keras.ops.convert_to_numpy(source(images, training=False))
        cloned_output = keras.ops.convert_to_numpy(branch(images, training=False))
        training_output = keras.ops.convert_to_numpy(branch(images, training=True))
    maximum_error = float(np.max(np.abs(original_output-cloned_output)))
    if maximum_error != 0 or not np.isfinite(training_output).all():
        raise ValueError('BN graph branch failed initial inference equality or finite training output')
    changed_moments = [layer.name for layer in bn_layers if any(
        not np.array_equal(old, new) for old, new in zip(initial_moments[layer.name],
            (layer.moving_mean.numpy(), layer.moving_variance.numpy())))]
    if len(changed_moments) != len(bn_layers):
        raise ValueError('Some BN layers still ignore inherited training mode')
    branch.set_weights(before)
    branch.save(artifact, include_optimizer=False)
    restored = keras.models.load_model(artifact, compile=False)
    with torch.no_grad():
        restored_output = keras.ops.convert_to_numpy(restored(images, training=False))
    roundtrip_error = float(np.max(np.abs(original_output-restored_output)))
    if roundtrip_error != 0:
        raise ValueError('Saved BN branch no longer matches initial parent inference')
    source_folder = args.output/'source'
    source_folder.mkdir(exist_ok=True)
    hashes = {}
    for filename in ('word_batchnorm.py', 'word_model.py', 'word_data.py'):
        content = (Path(__file__).parent/filename).read_bytes()
        (source_folder/filename).write_bytes(content)
        hashes[filename] = hashlib.sha256(content).hexdigest()
    report = {
        'scope': 'Initially weight-identical BN-training branch; no optimizer updates or evaluation-set inputs.',
        'parent_checkpoint': str(args.checkpoint),
        'parent_checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'train_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_sha256': hashes, 'imageNormalization': image_normalization_spec(restored),
        'batchnorm': {'trainable': True, 'momentum': args.momentum,
                      'training_call': 'Backbone inherits outer model training mode; inference uses stored moving moments.',
                      'layers': len(bn_layers), 'diagnostic_layers_with_updated_moments': changed_moments},
        'diagnostic_train_images': [{'image': row['image'], 'source': row['source']} for row in sample],
        'initial_inference_max_abs': maximum_error, 'save_reload_inference_max_abs': roundtrip_error,
        'diagnostic_moment_updates_reset_before_save': True,
        'all_initial_weight_arrays_exact_parent': all(np.array_equal(a, b) for a, b in zip(source.get_weights(), restored.get_weights())),
        'physical_camera_validated': False,
    }
    save_json(args.output/'batchnorm-branch.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
