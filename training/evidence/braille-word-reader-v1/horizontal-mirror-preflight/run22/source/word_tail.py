"""Extend a frozen three-scale reader with the ImageNet tail and a zero residual."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import keras
from keras import layers, ops
import numpy as np
from PIL import Image

from word_model import image_normalization_spec, word_relative_loss
from word_data import make_target


def extend_tail(parent, imagenet_weights):
    if any(layer.name == 'fpn_32_zero' for layer in parent.layers):
        raise ValueError('The parent already includes the full tail')
    assert tuple(parent.input_shape) == (None, 192, 480, 3)
    old_backbone = parent.get_layer('mobilenetv2_half')
    backbone = keras.applications.MobileNetV2(input_shape=(192, 480, 3), alpha=.5,
                                             include_top=False, weights=str(imagenet_weights))
    inherited = set()
    for old in old_backbone.layers:
        if isinstance(old, layers.InputLayer):
            continue
        new = backbone.get_layer(old.name)
        new.set_weights(old.get_weights())
        new.trainable = old.trainable
        inherited.add(old.name)
    for layer in backbone.layers:
        if layer.name not in inherited and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    taps = ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu']
    features = keras.Model(backbone.input, [backbone.get_layer(name).output for name in taps],
                           name='mobilenetv2_half')
    inputs = layers.Input((192, 480, 3), name='rgb_0_to_1')
    clones = {}

    def copy_layer(name, value):
        old = parent.get_layer(name)
        new = old.__class__.from_config(old.get_config())
        result = new(value)
        new.set_weights(old.get_weights())
        clones[name] = new
        return result

    pixels = copy_layer('camera_normalization', inputs) if any(layer.name == 'camera_normalization' for layer in parent.layers) else inputs
    low, middle, high, tail = features(copy_layer('mobilenet_normalization', pixels), training=False)
    high = copy_layer('fpn_16', high)
    tail = layers.Conv2D(64, 1, padding='same', kernel_initializer='zeros', bias_initializer='zeros', name='fpn_32_zero')(tail)
    tail = layers.UpSampling2D(2, interpolation='nearest', name='fpn_32_upsample')(tail)
    high = layers.Add(name='fpn_16_with_32')([high, tail])
    middle = copy_layer('add', [copy_layer('fpn_8', middle), copy_layer('up_sampling2d', high)])
    low = copy_layer('add_1', [copy_layer('fpn_4', low), copy_layer('up_sampling2d_1', middle)])
    output = copy_layer('activation', low)
    for index in range(2):
        output = copy_layer(f'reader_features_{index}', output)
    output = copy_layer('word_cells', [copy_layer(name, output) for name in ('cell_probability', 'cell_geometry', 'dot_probabilities')])
    result = keras.Model(inputs, output, name='braille_word_reader_full_tail')
    for old in parent.layers:
        if isinstance(old, layers.InputLayer):
            continue
        pairs = [(child, backbone.get_layer(child.name)) for child in old.layers if child.weights] if old is old_backbone else [(old, clones[old.name])]
        for original, copied in pairs:
            for a, b in zip(original.get_weights(), copied.get_weights(), strict=True):
                np.testing.assert_array_equal(a, b)
    assert tuple(result.output_shape) == (None, 48, 120, 11)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--imagenet-weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact = args.output/'full-tail-parent.keras'
    if artifact.exists():
        raise SystemExit('Use a new output directory; do not overwrite an initialized parent')
    args.output.mkdir(parents=True, exist_ok=True)
    parent = keras.models.load_model(args.parent, compile=False)
    model = extend_tail(parent, args.imagenet_weights)
    records = json.loads((args.data/'train.json').read_text())
    # Initialization equality uses only existing TRAIN pixels, no selection metric.
    selected = [records[index] for index in (0, len(records)//2, len(records)-1)]
    images = np.stack([np.asarray(Image.open(args.data/row['image']).convert('RGB'), np.float32)/255. for row in selected])
    before = ops.convert_to_numpy(parent(images, training=False))
    after = ops.convert_to_numpy(model(images, training=False))
    np.testing.assert_array_equal(before, after)
    model.save(artifact)
    restored = keras.models.load_model(artifact, compile=False)
    np.testing.assert_array_equal(before, ops.convert_to_numpy(restored(images, training=False)))
    if keras.backend.backend() != 'torch':
        raise SystemExit('The initialization audit requires the Torch backend')
    import torch
    labels = np.stack([make_target(np.asarray(row['boxes'], np.float32).reshape(-1, 4),
                                   np.asarray(row['masks']),
                                   np.asarray(row['originally_complete'], bool) if 'originally_complete' in row else None)
                       for row in selected])
    target = torch.as_tensor(labels, device=restored.weights[0].value.device)
    tail_variables = restored.get_layer('mobilenetv2_half').get_layer('block_13_depthwise').trainable_weights
    lateral_variables = restored.get_layer('fpn_32_zero').trainable_weights
    def gradients():
        restored.zero_grad(set_to_none=True)
        loss = word_relative_loss(target, restored(images, training=True))
        loss.backward()
        norm = lambda variables: sum(float(variable.value.grad.abs().sum().detach().cpu())
                                     for variable in variables if variable.value.grad is not None)
        return {'loss': float(loss.detach().cpu()), 'lateral_l1': norm(lateral_variables),
                'new_tail_depthwise_l1': norm(tail_variables)}
    first = gradients()
    assert first['lateral_l1'] > 0 and first['new_tail_depthwise_l1'] == 0
    # This isolated reload is discarded. The saved parent stays exactly zero.
    with torch.no_grad():
        for variable in lateral_variables:
            variable.value.add_(variable.value.grad, alpha=-1e-3)
    second = gradients()
    assert second['new_tail_depthwise_l1'] > 0 and np.isfinite(list(second.values())).all()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sources = args.output/'source'
    sources.mkdir(exist_ok=True)
    for name in ('word_tail.py', 'word_model.py', 'word_data.py'):
        (sources/name).write_bytes((Path(__file__).parent/name).read_bytes())
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Initialization and serialization equality only; no training, accuracy or promotion.',
              'parent_checkpoint': str(args.parent), 'parent_checkpoint_sha256': digest(args.parent),
              'artifact_sha256': digest(artifact), 'artifact': str(artifact),
              'imagenet_weights_sha256': digest(args.imagenet_weights),
              'source_sha256': {p.name: digest(p) for p in sources.glob('*.py')},
              'input_shape': list(model.input_shape), 'output_shape': list(model.output_shape),
              'parent_parameters': parent.count_params(), 'parameters': model.count_params(),
              'imageNormalization': image_normalization_spec(model), 'all_inherited_weights_exact': True,
              'initial_output_exact': True, 'save_load_output_exact': True,
              'diagnostic_gradient_probe': {'scope': 'Discarded reload; one SGD step on the new lateral only, using three TRAIN targets; saved artifact remains zero and is not overwritten.',
                                            'before_step': first, 'after_lateral_step': second},
              'equality_train_images': [{'image': row['image'], 'sha256': digest(args.data/row['image'])} for row in selected],
              'new_branch': 'ImageNet block13 through out_relu, all new BN frozen; zero Conv64 stride32 lateral added before existing FPN16.'}
    (args.output/'full-tail-initialization.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
