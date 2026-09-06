"""Cache the already ranked best new held candidate for failure diagnosis only."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

os.environ['KERAS_BACKEND'] = 'torch'
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(row):
    groups = row['validation_groups']
    precision = min(value['accepted_accuracy'] or 0. for value in groups.values())
    mean = float(np.mean([value['exact_word_accuracy'] for value in groups.values()]))
    return precision >= .95, mean if precision >= .95 else precision, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new held diagnostic directory')
    table_path = args.selection/'checkpoint-selection.json'
    table = json.loads(table_path.read_text())
    selection = json.loads((args.selection/'selected-validation-fullprecision.json').read_text())
    assert table['external_outputs_used'] is False and table['threshold_grid'] == 'weak-peaks-v2'
    fixture = args.data/'validation-stress-selection.json'
    fixture_hash = digest(fixture)
    assert fixture_hash == table['fixture_sha256'] == selection['oracle']['fixture_sha256']
    assert table['validation_words'] == 405
    eligible = json.loads((args.training_run/'sampling-lineage-final.json').read_text())
    assert eligible['status'] == 'passed'
    eligible_hashes = {row['checkpoint_sha256'] for row in eligible['actual']['observed_epochs'] if row['eligible_by_sampling_lineage']}
    candidates = [row for row in table['candidates'] if Path(row['source_checkpoint']).resolve().parent.parent == args.training_run.resolve()]
    chosen = max(candidates, key=rank)
    checkpoint = Path(chosen['source_checkpoint'])
    assert chosen['checkpoint_sha256'] == digest(checkpoint) and digest(checkpoint) in eligible_hashes
    source = Path('training/scripts')
    for filename, expected in selection['oracle']['source_sha256'].items():
        assert digest(source/filename) == expected, filename
    records = json.loads(fixture.read_text())['records']
    image_hashes = {row['image']: digest(args.data/row['image']) for row in records}
    source_outputs_reused = chosen['checkpoint_sha256'] == table['selected_checkpoint_sha256']
    if source_outputs_reused:
        with np.load(args.selection/'backend-fullprecision.npz') as cache:
            outputs, quality = cache['outputs'], cache['quality_rejected']
            assert str(cache['checkpoint_sha256']) == chosen['checkpoint_sha256']
    else:
        import torch
        import keras
        sys.path.insert(0, str(source.resolve()))
        import word_model
        import word_geometry_head
        import word_bn_affine
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('highest')
        images = np.stack([np.asarray(Image.open(args.data/row['image']).convert('RGB'), np.float32)/255. for row in records])
        luminance = images@np.asarray([.2126, .7152, .0722], np.float32)
        means, stds = luminance.mean(axis=(1, 2)), luminance.std(axis=(1, 2))
        quality = (means < .06) | (means > .985) | (stds < .01)
        model = keras.models.load_model(checkpoint, compile=False)
        outputs = model.predict(images, batch_size=24, verbose=0)
    sys.path.insert(0, str(source.resolve()))
    from word_decode import summarize
    assert outputs.shape == (405, 48, 120, 11) and np.isfinite(outputs).all()
    scored = outputs.copy()
    scored[quality] = 0
    actual = summarize(scored, records, chosen['thresholds'])
    assert actual == chosen['words'], 'Cached diagnostic inference disagrees with the existing frozen selection metrics'
    assert digest(fixture) == fixture_hash and digest(checkpoint) == chosen['checkpoint_sha256']
    args.output.mkdir(parents=True)
    output_path = args.output/'held-outputs.npz'
    np.savez_compressed(output_path, outputs=outputs, quality_rejected=quality,
                        fixture_sha256=fixture_hash, checkpoint_sha256=chosen['checkpoint_sha256'])
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'scope': 'The best new candidate was already ranked by frozen held405; this cache adds diagnosis only, no threshold/checkpoint choice and no external inference.',
              'source_script_sha256': digest(__file__), 'selection_table_sha256': digest(table_path),
              'sampling_lineage_sha256': digest(args.training_run/'sampling-lineage-final.json'),
              'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'fixture_sha256': fixture_hash, 'image_sha256': image_hashes,
              'source_sha256': selection['oracle']['source_sha256'], 'candidate': chosen,
              'existing_winner_outputs_reused': source_outputs_reused, 'tf32_enabled': False,
              'outputs_sha256': digest(output_path), 'all_finite': True, 'exact_selection_metrics_reproduced': True}
    (args.output/'cache-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('image_sha256', 'candidate', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
