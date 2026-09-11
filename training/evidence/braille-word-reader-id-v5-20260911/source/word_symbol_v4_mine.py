"""Bounded dot-mask difficulty weights from frozen TRAIN pixels only.

No labels, images, losses or thresholds are changed. No validation/TEST pixels
are read. Weights 1..3 adjust draws within the existing semantic groups only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT, STRIDE = 480, 192, 4
GRID_W, GRID_H = WIDTH//STRIDE, HEIGHT//STRIDE


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_hash(path, expected):
    if sha(path) != expected:
        raise ValueError('Hard-example input binding changed: '+str(path))


def train_rows(manifest, image_index):
    manifest, image_index = Path(manifest).resolve(), Path(image_index).resolve()
    if manifest.name != 'train-symbols.json' or image_index != manifest.parent/'train-image-hashes.json':
        raise ValueError('Mining requires the canonical TRAIN manifest and image index')
    content, hashes = read(manifest), read(image_index)
    rows = content.get('records', [])
    if content.get('split') != 'train' or not rows:
        raise ValueError('Hard-example mining is TRAIN-only')
    if len({row['source'] for row in rows}) != len(rows):
        raise ValueError('TRAIN source identities must be unique')
    if len({row['image'] for row in rows}) != len(rows) or set(hashes) != {row['image'] for row in rows}:
        raise ValueError('TRAIN image index must cover exactly the unique manifest images')
    image_root = (manifest.parent/'data/train').resolve()
    for row in rows:
        image = Path(row['image']).resolve()
        if not image.is_relative_to(image_root):
            raise ValueError('Only TRAIN image paths are allowed')
        if len(row['boxes']) != len(row['masks']):
            raise ValueError('TRAIN cell boxes and masks differ in count')
        if 'originally_complete' in row and len(row['originally_complete']) != len(row['boxes']):
            raise ValueError('TRAIN completion flags differ from cell count')
    return rows, hashes


def difficulty(output, row):
    """Probe six dot probabilities at the same grid centers as make_target."""
    if (output.shape != (GRID_H,GRID_W,11) or not np.isfinite(output).all()
            or output.min() < 0 or output.max() > 1):
        raise ValueError('Mining output must be finite runtime-shaped probabilities')
    cells = []
    flags = row.get('originally_complete', [True]*len(row['boxes']))
    if len(row['boxes']) != len(row['masks']) or len(flags) != len(row['boxes']):
        raise ValueError('Cell annotations differ in count')
    for index, (box, mask) in enumerate(zip(row['boxes'], row['masks'])):
        if (len(box) != 4 or not np.isfinite(box).all() or isinstance(mask, bool)
                or not isinstance(mask, (int, np.integer)) or not 1 <= mask <= 63):
            raise ValueError('Invalid TRAIN cell annotation')
        x0,y0,x1,y1 = np.asarray(box, np.float32)
        if x0 >= x1 or y0 >= y1:
            raise ValueError('Cell annotation requires positive size')
        if not flags[index] or x0 < 0 or y0 < 0 or x1 > WIDTH or y1 > HEIGHT:
            continue
        gx,gy = min(GRID_W-1, int((x0+x1)/2/STRIDE)), min(GRID_H-1, int((y0+y1)/2/STRIDE))
        bits = output[gy,gx,5:]
        predicted = sum(int(bit >= .5) << dot for dot,bit in enumerate(bits))
        confidence = float(np.minimum(np.maximum(bits, 1-bits), 1).min())
        cells.append(dict(index=index, expected=int(mask), predicted=predicted, confidence=confidence))
    incorrect = sum(cell['expected'] != cell['predicted'] for cell in cells)
    low_confidence = sum(cell['confidence'] < .7 for cell in cells)
    weight = 3 if incorrect else 2 if low_confidence else 1
    return dict(weight=weight, complete_cells=len(cells), skipped_cells=len(row['boxes'])-len(cells),
                incorrect_masks=incorrect, low_confidence_cells=low_confidence,
                minimum_confidence=min((cell['confidence'] for cell in cells), default=None))


def load_sampling_weights(path, train_manifest, image_index):
    """Validate mining provenance before sampler integration; never reads pixels."""
    report = read(path)
    if report.get('split') != 'train' or report.get('scope') != 'TRAIN_only_dot_mask_sampling':
        raise ValueError('Sampling weights must come only from TRAIN mining')
    for prefix, actual in [('manifest', train_manifest), ('image_index', image_index)]:
        if Path(report[prefix+'_path']).resolve() != Path(actual).resolve():
            raise ValueError('Sampling weights refer to a different '+prefix)
        assert_hash(actual, report[prefix+'_sha256'])
    assert_hash(report['checkpoint_path'], report['checkpoint_sha256'])
    rows, _ = train_rows(train_manifest, image_index)
    weights = report['weights_by_source']
    if set(weights) != {row['source'] for row in rows}:
        raise ValueError('Sampling weights must cover every TRAIN source exactly once')
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3 for value in weights.values()):
        raise ValueError('Sampling weights must be bounded integers 1..3')
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment, checkpoint, output = args.experiment.resolve(), args.checkpoint.resolve(), args.output.resolve()
    if output.exists() or output.suffix != '.json' or not output.is_relative_to(experiment):
        raise ValueError('Write mining evidence to a new JSON path inside this experiment')
    if (experiment/'test-evaluation-reservation.json').exists() or (experiment/'regression-evaluation-reservation.json').exists():
        raise ValueError('Final evaluation already reserved; no further TRAIN mining/tuning')
    manifest, image_index = experiment/'train-symbols.json', experiment/'train-image-hashes.json'
    rows, hashes = train_rows(manifest, image_index)
    frozen = {str(path): sha(path) for path in (manifest, image_index, checkpoint)}
    source = Path(__file__).resolve().parent
    source_hashes = {name: sha(source/name) for name in ('word_symbol_v4_mine.py', 'word_torch_port.py')}
    import torch
    from word_torch_port import load_keras_checkpoint
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_keras_checkpoint(checkpoint).cuda().eval()
    weights, diagnostics = {}, []
    for start in range(0, len(rows), 8):
        batch, pixels = rows[start:start+8], []
        for row in batch:
            assert_hash(row['image'], hashes[row['image']])
            with Image.open(row['image']) as image:
                if image.size != (WIDTH,HEIGHT):
                    raise ValueError('TRAIN image shape differs from frozen input')
                pixels.append(np.asarray(image.convert('RGB'), np.float32)/255.)
        with torch.no_grad():
            outputs = model(torch.from_numpy(np.stack(pixels)).cuda()).cpu().numpy()
        if len(outputs) != len(batch):
            raise ValueError('Mining output count differs from TRAIN batch')
        for value,row in zip(outputs,batch):
            result = difficulty(value, row)
            weights[row['source']] = result['weight']
            diagnostics.append(dict(source=row['source'], group=row['symbol_group'], **result))
        if start % 400 == 0:
            print(f'MINED_TRAIN {start+len(batch)}/{len(rows)}', flush=True)
    for path,digest in frozen.items():
        assert_hash(path,digest)
    for name,digest in source_hashes.items():
        assert_hash(source/name,digest)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), split='train', scope='TRAIN_only_dot_mask_sampling',
        manifest_path=str(manifest), manifest_sha256=sha(manifest),
        image_index_path=str(image_index), image_index_sha256=sha(image_index),
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint), source_sha256=source_hashes,
        weights_by_source=weights, distribution=dict(Counter(weights.values())),
        groups={group:dict(Counter(row['weight'] for row in diagnostics if row['group']==group))
                for group in sorted({row['group'] for row in diagnostics})}, rows=diagnostics,
        policy='1 correct confident; 2 correct mask with any dot confidence below .7; 3 any incorrect complete-cell mask',
        interpretation='Annotated-center dot difficulty, not detection or recognition accuracy; invalid sequences retain their mask supervision',
        images_copied=0, validation_pixels_read=0, test_pixels_read=0, torch=torch.__version__)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('MINING_COMPLETE '+json.dumps(dict(rows=len(rows), distribution=report['distribution'], groups=report['groups'])), flush=True)


if __name__ == '__main__':
    main()
