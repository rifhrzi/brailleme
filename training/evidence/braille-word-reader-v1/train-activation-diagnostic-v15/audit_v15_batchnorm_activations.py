"""Descriptive CPU-only TRAIN activation audit; never recalibrate or train BN."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from datetime import datetime, timezone

os.environ['KERAS_BACKEND'] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
root = Path.cwd()
work = root / '.braille-training-work/word-scan-v1-20260905'
output = work / 'v15-train-batchnorm-activations'
output.mkdir(exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = work / 'combined-training-v5/combined-training-manifest.json'
checkpoint = work / 'selected-v15/best-word.keras'
rows = json.loads(manifest.read_bytes())
assert isinstance(rows, list)
families = {}
for i, row in enumerate(rows):
    if row['boxes']:
        families.setdefault(str(PurePosixPath(row['source'].replace('\\', '/')).parent), []).append(i)
sample = [(family, indices[(len(indices)-1)*quarter//4])
          for family, indices in sorted(families.items()) for quarter in (1, 2, 3)]
assert len(families) == 12 and len(sample) == 36
protocol = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Descriptive TRAIN-only statistics on the unchanged v15 reader. No validation/external inputs, augmentation, gradients, optimizer, BN update, model save, threshold selection, or activation-based sample selection.',
    'sampling': 'Per positive source family in frozen manifest order, indices floor((n-1)*q/4), q=1,2,3. 36 cached RGB inputs, three per 12 families; source cropping/resize already applied by dataset preparation.',
    'checkpoint_sha256': sha(checkpoint), 'manifest_sha256': sha(manifest),
    'script_sha256': sha(Path(__file__)),
    'samples': [{'family': family, 'manifest_index': i, 'source': rows[i]['source'],
                 'image': rows[i]['image'], 'sha256': sha(work / 'data' / rows[i]['image'])} for family, i in sample],
    'interpretation_limits': 'Activation distribution mismatch and ReLU sparsity are descriptive. Neither proves a recognition cause, an accuracy gain from changing BN, nor representative physical-camera behavior. Stored moments originated in earlier model training and are compared to current activations, not asserted to be optimal moments for this data.',
}
(output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
sys.path.insert(0, str(root / 'training/scripts'))
sys.path.insert(0, str(work / 'selected-v15/source'))
import numpy as np
from PIL import Image
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
assert not torch.cuda.is_available()
import keras
import word_model
model = keras.models.load_model(checkpoint, compile=False)
assert all(weight.value.device.type == 'cpu' for weight in model.weights)
weight_hash = lambda: hashlib.sha256(b''.join(keras.ops.convert_to_numpy(weight).tobytes() for weight in model.weights)).hexdigest()
weights_before = weight_hash()
backbone = model.get_layer('mobilenetv2_half')
bn = [layer for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)]
relus = [layer for layer in backbone.layers if isinstance(layer, keras.layers.ReLU)]
assert len(bn) == 52 and all(not layer.trainable for layer in bn)
prefix = keras.Model(model.inputs, model.get_layer('mobilenet_normalization').output)
probe = keras.Model(backbone.inputs, [layer.input for layer in bn] + [layer.output for layer in relus])
moments, activation_counts, image_stats = {}, {}, []
for record_number, (family, index) in enumerate(sample):
    with Image.open(work / 'data' / rows[index]['image']) as source:
        pixels = np.asarray(source.convert('RGB'), np.float32)[None] / 255.
    with torch.inference_mode():
        normalized = prefix(pixels, training=False)
        arrays = probe(normalized, training=False)
    normalized_array = keras.ops.convert_to_numpy(normalized)
    image_stats.append({'family': family, 'manifest_index': index,
                        'raw_mean': float(pixels.mean()), 'raw_std': float(pixels.std()),
                        'backbone_input_mean': float(normalized_array.mean()),
                        'backbone_input_std': float(normalized_array.std())})
    for layer, value in zip(bn, arrays[:len(bn)], strict=True):
        array = keras.ops.convert_to_numpy(value).reshape(-1, value.shape[-1]).astype(np.float64)
        entry = moments.setdefault(layer.name, {'count': 0, 'sum': np.zeros(array.shape[-1]), 'square_sum': np.zeros(array.shape[-1])})
        entry['count'] += len(array)
        entry['sum'] += array.sum(axis=0)
        entry['square_sum'] += np.square(array).sum(axis=0)
    for layer, value in zip(relus, arrays[len(bn):], strict=True):
        array = keras.ops.convert_to_numpy(value)
        entry = activation_counts.setdefault(layer.name, {'values': 0, 'zero': 0, 'at_or_above_six': 0})
        entry['values'] += array.size
        entry['zero'] += int(np.count_nonzero(array == 0))
        entry['at_or_above_six'] += int(np.count_nonzero(array >= 6))
    del arrays
    if (record_number + 1) % 12 == 0:
        print(f'TRAIN activation audit {record_number+1}/36', flush=True)
stats = []
quantiles = lambda values: dict(zip(('min', 'p05', 'median', 'p95', 'max'), map(float, np.quantile(values, [0, .05, .5, .95, 1]))))
for layer in bn:
    entry = moments[layer.name]
    mean = entry['sum'] / entry['count']
    variance = np.maximum(entry['square_sum'] / entry['count'] - mean**2, 0.)
    saved_mean = keras.ops.convert_to_numpy(layer.moving_mean).astype(np.float64)
    saved_variance = keras.ops.convert_to_numpy(layer.moving_variance).astype(np.float64)
    denominator = np.sqrt(saved_variance + layer.epsilon)
    stats.append({'layer': layer.name, 'channels': len(mean), 'spatial_sample_count_per_channel': entry['count'],
                  'absolute_mean_shift_per_saved_std': quantiles(np.abs(mean-saved_mean)/denominator),
                  'observed_std_per_saved_std': quantiles(np.sqrt(variance)/denominator),
                  'channels_abs_shift_over_one_saved_std': int(np.count_nonzero(np.abs(mean-saved_mean) > denominator)),
                  'channels_observed_std_below_quarter_saved': int(np.count_nonzero(np.sqrt(variance) < .25*denominator)),
                  'observed_mean': mean.tolist(), 'observed_variance': variance.tolist(),
                  'saved_mean': saved_mean.tolist(), 'saved_variance': saved_variance.tolist()})
weights_after = weight_hash()
assert weights_after == weights_before and sha(checkpoint) == protocol['checkpoint_sha256']
for entry in activation_counts.values():
    entry['zero_fraction'] = entry['zero'] / entry['values']
    entry['at_or_above_six_fraction'] = entry['at_or_above_six'] / entry['values']
report = {'status': 'completed_descriptive_no_accuracy_claim', 'protocol_sha256': sha(output / 'protocol.json'),
          'script_sha256': protocol['script_sha256'], 'checkpoint_sha256': protocol['checkpoint_sha256'],
          'word_model_source_sha256': sha(Path(word_model.__file__)),
          'backend': keras.backend.backend(), 'torch_version': torch.__version__, 'keras_version': keras.__version__,
          'device': 'cpu', 'threads': 2, 'all_weight_arrays_before_after_sha256': weights_before,
          'all_model_values_unchanged': True, 'batchnorm_layers': len(bn),
          'image_statistics': image_stats, 'batchnorm_statistics': stats, 'relu_counts': activation_counts,
          'limits': protocol['interpretation_limits']}
(output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'report_sha256': sha(output / 'report.json'), 'images': len(sample), 'batchnorm_layers': len(bn),
                  'first_bn': {key: value for key, value in stats[0].items() if not isinstance(value, list)},
                  'relu_zero_range': quantiles([entry['zero_fraction'] for entry in activation_counts.values()])}), flush=True)
