"""Re-evaluate the protected baseline after a failed promotion; never train."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from word_symbol_train import sha, metrics, np, Image, keras, torch
from word_decode import DECODER_VERSION, summarize
from word_model import image_normalization_spec
from word_data import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    experiment = args.experiment
    config = json.loads((experiment/'run-batch24/training-config.json').read_text())
    source_root = Path(__file__).parent
    for name, expected in config['source_sha256'].items():
        if sha(source_root/name) != expected:
            raise ValueError('Training/decoder source changed: '+name)
    checkpoint = experiment/'baseline.keras'
    if sha(checkpoint) != config['baseline_checkpoint_sha256']:
        raise ValueError('Protected baseline checkpoint changed')
    sources = dict(config['source_sha256'])
    sources[Path(__file__).name] = sha(Path(__file__))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    keras.utils.set_random_seed(20260908)
    records = json.loads((experiment/'selection-fixtures.json').read_text())['records']
    legacy_path = experiment/'legacy-selection-fixtures.json'
    legacy = json.loads(legacy_path.read_text())['records']
    pixels = np.stack([np.asarray(Image.open(row['image']).convert('RGB'), np.float32)/255. for row in records])
    luminance = pixels @ np.asarray([.2126, .7152, .0722], np.float32)
    quality = (luminance.mean((1,2))<.06) | (luminance.mean((1,2))>.985) | (luminance.std((1,2))<.01)
    del luminance
    model = keras.models.load_model(checkpoint, compile=False)
    outputs = model.predict(pixels, batch_size=24, verbose=0)
    if not np.isfinite(outputs).all():
        raise ValueError('Baseline produced non-finite outputs')
    scored = outputs.copy()
    scored[quality] = 0
    groups, decisions = metrics(scored, records, config['thresholds'])
    recorded = json.loads((experiment/'run-batch24/baseline-evaluation.json').read_text())
    if groups != recorded['groups']:
        raise ValueError('Baseline metrics no longer match the frozen pre-training evaluation')
    shutil.copy2(checkpoint, args.output/'best-word.keras')
    oracle = dict(backend='torch', tf32_enabled=False, fixture_file=str(legacy_path),
                  fixture_sha256=sha(legacy_path), checkpoint_sha256=sha(checkpoint),
                  source_sha256=sources, decoder_version=DECODER_VERSION)
    selected = dict(source_checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        decoder_version=DECODER_VERSION, thresholds=config['thresholds'], oracle=oracle,
        groups=groups, epoch=0, imageNormalization=image_normalization_spec(model),
        quality_rejected_before_decode=int(quality[:len(legacy)].sum()),
        status='unverified_candidate', physical_camera_validated=False,
        words=summarize(scored[:len(legacy)], legacy, config['thresholds']),
        validation_groups={name:groups['legacy_'+name] for name in ['natural','stress']},
        threshold_origin='Original frozen baseline thresholds; baseline restored after candidate failed external letter regression promotion gate',
        deployment_note='Training selected epoch10; this separate evaluation restores the protected baseline without another training run or alternate epoch choice')
    for name in ['selected-validation.json', 'selected-validation-fullprecision.json']:
        save_json(args.output/name, selected)
    for name, start, end, fixture in [
        ('backend-fullprecision.npz', 0, len(legacy), legacy_path),
        ('symbol-backend-fullprecision.npz', len(legacy), len(records), experiment/'validation-symbols.json')]:
        np.savez_compressed(args.output/name, outputs=outputs[start:end], quality_rejected=quality[start:end],
                            fixture_sha256=sha(fixture), checkpoint_sha256=sha(checkpoint))
    shutil.copy2(experiment/'selection/checkpoint-selection.json', args.output/'checkpoint-selection.json')
    save_json(args.output/'baseline-restoration-evaluation.json', dict(
        checkpoint_sha256=sha(checkpoint), decoder_version=DECODER_VERSION,
        thresholds=config['thresholds'], groups=groups, decisions=decisions,
        previous_baseline_evaluation_sha256=sha(experiment/'run-batch24/baseline-evaluation.json'),
        frozen_baseline_decisions_reproduced=True, training_performed=False,
        physical_camera_validated=False))
    print(json.dumps(dict(checkpoint_sha256=sha(checkpoint), groups=groups, training_performed=False)), flush=True)


if __name__ == '__main__':
    main()
