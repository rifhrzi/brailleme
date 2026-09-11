"""Bounded TRAIN-only cell refiner pilot; validation selects epoch and fixed blend.

The frozen detector and v4 thresholds remain unchanged. This is a separate
research artifact, never an automatic app promotion. This module has no TEST
evaluation path. Prepare runs on CPU; train must run after detector training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from word_decode_id_v4 import decode
from word_symbol_v4_mine import train_rows
from word_symbol_v4_train import NativeAdam, audit_rows, rank, score, sha, verify_hashes, verified_pixels

HEIGHT, WIDTH = 64, 40
CROP_MARGIN = .15
ALPHAS = (0., .5, 1.)
LUMINANCE = np.asarray([.2126, .7152, .0722], np.float32)
VERSION = 'positional-cell-refiner-id-v4-pilot-1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def assert_unopened(experiment):
    for name in ('test-evaluation-reservation.json', 'regression-evaluation-reservation.json'):
        if (Path(experiment)/name).exists():
            raise ValueError('Final evaluation reserved: no additional tuning is allowed')


def crop_cell(image, box):
    """RGB float image, normalized xyxy -> [64,40,1] float luminance.

    Expand each side by 15% of the original width/height. Pixel centers are at
    i+.5; sample the expanded rectangle at output pixel centers, subtract .5,
    and use explicit bilinear interpolation with border replication. This is
    the evaluation/export/runtime crop contract; it does not use GT geometry.
    """
    image = np.asarray(image, np.float32)
    box = np.asarray(box, np.float64)
    if (image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1 or box.shape != (4,)
            or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('Crop requires finite RGB probabilities and positive normalized xyxy')
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box * [w, h, w, h]
    dx, dy = (x1-x0)*CROP_MARGIN, (y1-y0)*CROP_MARGIN
    xs = np.clip(x0-dx+(np.arange(WIDTH)+.5)*(x1-x0+2*dx)/WIDTH-.5, 0, w-1)
    ys = np.clip(y0-dy+(np.arange(HEIGHT)+.5)*(y1-y0+2*dy)/HEIGHT-.5, 0, h-1)
    left, top = np.floor(xs).astype(int), np.floor(ys).astype(int)
    right, bottom = np.minimum(left+1, w-1), np.minimum(top+1, h-1)
    wx, wy = xs-left, ys-top
    # Interpolate RGB before luma to make the operation order explicit.
    upper = image[top[:, None], left]*(1-wx[None, :, None])+image[top[:, None], right]*wx[None, :, None]
    lower = image[bottom[:, None], left]*(1-wx[None, :, None])+image[bottom[:, None], right]*wx[None, :, None]
    rgb = upper*(1-wy[:, None, None])+lower*wy[:, None, None]
    return (rgb @ LUMINANCE).astype(np.float32)[..., None]


class CellRefiner(nn.Module):
    """Under 250k parameters; positional head preserves all six dot locations."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Linear(48*8*5, 64), nn.ReLU(), nn.Linear(64, 6))

    def forward(self, image):
        centered = image-image.mean(dim=(1, 2, 3), keepdim=True)
        normalized = centered / torch.sqrt((centered*centered).mean(dim=(1, 2, 3), keepdim=True)+.01**2)
        features = self.features(normalized.permute(0, 3, 1, 2))
        return self.classifier(features.flatten(1))


def save_refiner(model, path):
    with Path(path).open('xb') as stream:
        np.savez(stream, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})


def load_refiner(path):
    model = CellRefiner()
    with np.load(path, allow_pickle=False) as archive:
        values = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    model.load_state_dict(values, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Non-finite refiner checkpoint')
    return model


def selected_grids(output, thresholds, selection=None):
    """Attach actual NMS cells to their original grids without rounding centers."""
    decision = decode(output, thresholds, selection)
    if not decision['cells']:
        return []
    lookup = {}
    raw = output.astype(np.float64, copy=False)
    for gy, gx in np.argwhere(raw[..., 0] >= thresholds['proposalScore']):
        row = raw[gy, gx]
        cx, cy = (gx+row[1])/120, (gy+row[2])/48
        x, y = cx-row[3]/2, cy-row[4]/2
        box = (x, y, x+row[3], y+row[4])
        lookup.setdefault((float(row[0]), box), (int(gy), int(gx)))
    return [(lookup[(cell['score'], tuple(cell['box']))], cell['box']) for cell in decision['cells']]


def replace_dots(outputs, attachments, probabilities, alpha):
    if alpha not in ALPHAS:
        raise ValueError('Only predefined alpha 0, .5, and 1 are permitted')
    if outputs.ndim != 4 or outputs.shape[1:] != (48, 120, 11):
        raise ValueError('Expected complete detector output batch')
    if (len(set(tuple(item) for item in attachments)) != len(attachments)
            or any(len(item) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in item)
                   or not (0 <= item[0] < len(outputs) and 0 <= item[1] < 48 and 0 <= item[2] < 120) for item in attachments)):
        raise ValueError('Attachments require unique integer in-range detector grids')
    probabilities = np.asarray(probabilities, np.float32)
    if probabilities.shape != (len(attachments), 6) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Every selected grid requires six finite probabilities')
    result = outputs.copy()
    for (index, gy, gx), dots in zip(attachments, probabilities):
        result[index, gy, gx, 5:] = (1-alpha)*outputs[index, gy, gx, 5:]+alpha*dots
    return result


def balanced_indices(masks, count, rng):
    groups = {int(mask): np.flatnonzero(masks == mask) for mask in np.unique(masks)}
    order = np.resize(rng.permutation(sorted(groups)), count)
    rng.shuffle(order)
    return np.asarray([rng.choice(groups[int(mask)]) for mask in order])


def augment_crops(image):
    """70% light geometry jitter; photometry/noise/blur in TRAIN batches only."""
    batch, device = len(image), image.device
    active = (torch.rand(batch, device=device) < .7).float()
    theta = torch.zeros((batch, 2, 3), device=device)
    theta[:, 0, 0] = 1+active*(torch.rand(batch, device=device)*.24-.12)
    theta[:, 1, 1] = 1+active*(torch.rand(batch, device=device)*.24-.12)
    theta[:, :, 2] = active[:, None]*(torch.rand(batch, 2, device=device)*.24-.12)
    nchw = image.permute(0, 3, 1, 2)
    grid = F.affine_grid(theta, nchw.shape, align_corners=False)
    value = F.grid_sample(nchw, grid, mode='bilinear', padding_mode='border', align_corners=False)
    shape = (batch, 1, 1, 1)
    contrast = .65+torch.rand(shape, device=device)*.7
    exposure = (torch.rand(shape, device=device)-.5)*.25
    value = (value-.5)*contrast+.5+exposure
    ramp = torch.linspace(-1, 1, WIDTH, device=device)[None, None, None, :]
    value = value+ramp*(torch.rand(shape, device=device)-.5)*.15
    value = value+torch.randn_like(value)*torch.rand(shape, device=device)*.015
    blur = (torch.rand(shape, device=device) < .25).float()*.5
    value = value*(1-blur)+F.avg_pool2d(F.pad(value, (1, 1, 1, 1), mode='replicate'), 3, stride=1)*blur
    return value.clamp(0, 1).permute(0, 2, 3, 1)


def prepare(experiment, output):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    assert_unopened(experiment)
    if output.exists() or output.suffix != '.npz' or not output.is_relative_to(experiment):
        raise ValueError('Preparation requires a new NPZ path inside the experiment')
    metadata = output.with_suffix('.json')
    if metadata.exists():
        raise ValueError('Preparation metadata already exists')
    manifest, index = experiment/'train-symbols.json', experiment/'train-image-hashes.json'
    rows, hashes = train_rows(manifest, index)
    frozen = {str(path): sha(path) for path in (manifest, index)}
    crops, masks, cells = [], [], []
    for row_index, row in enumerate(rows):
        verify_hashes({row['image']: hashes[row['image']]})
        with Image.open(row['image']) as source:
            if source.size != (480, 192):
                raise ValueError('TRAIN input shape changed')
            pixels = np.asarray(source.convert('RGB'), np.float32)/255.
        for cell_index, (box, mask) in enumerate(zip(row['boxes'], row['masks'])):
            if (not isinstance(mask, int) or isinstance(mask, bool) or not 1 <= mask <= 63
                    or len(box) != 4 or not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError('Invalid known TRAIN cell')
            flags = row.get('originally_complete', [True]*len(row['boxes']))
            if not flags[cell_index] or box[0] < 0 or box[1] < 0 or box[2] > 480 or box[3] > 192:
                continue
            crops.append(np.rint(crop_cell(pixels, np.asarray(box)/[480, 192, 480, 192])*255).astype(np.uint8))
            masks.append(mask)
            cells.append([row_index, cell_index])
        if row_index % 500 == 0:
            print(f'PREPARE_TRAIN {row_index+1}/{len(rows)} cells={len(crops)}', flush=True)
    verify_hashes(frozen)
    with output.open('xb') as stream:
        np.savez(stream, crops=np.stack(crops), masks=np.asarray(masks, np.uint8), cells=np.asarray(cells, np.int32))
    write_new(metadata, dict(version=VERSION, scope='TRAIN_only_complete_composed_cells', split='train',
        source_manifest=str(manifest), image_index=str(index), input_sha256=frozen,
        image_hashes_verified=len(rows), output_sha256=sha(output), cells=len(crops),
        mask_counts=dict(sorted(Counter(masks).items())), missing_masks=sorted(set(range(1, 64))-set(masks)),
        crop_contract=dict(shape=[64, 40, 1], margin_each_side=CROP_MARGIN, interpolation='half_pixel_bilinear_border_replication',
            stored='uint8_luminance_rounded_once', runtime='float32_luminance'),
        source_sha256=sha(__file__), test_pixels_read=0, validation_pixels_read=0))
    print('PREPARATION_COMPLETE '+str(metadata), flush=True)


def load_prepared(path, experiment):
    path, experiment = Path(path).resolve(), Path(experiment).resolve()
    meta = read(path.with_suffix('.json'))
    if (meta.get('version') != VERSION or meta.get('split') != 'train'
            or meta.get('scope') != 'TRAIN_only_complete_composed_cells'
            or Path(meta['source_manifest']).resolve() != experiment/'train-symbols.json'
            or Path(meta['image_index']).resolve() != experiment/'train-image-hashes.json'):
        raise ValueError('Refiner preparation must bind this canonical TRAIN split')
    verify_hashes(meta['input_sha256'])
    verify_hashes({path: meta['output_sha256']})
    verify_hashes({__file__: meta['source_sha256']})
    with np.load(path, allow_pickle=False) as data:
        crops, masks, cells = data['crops'].copy(), data['masks'].copy(), data['cells'].copy()
    if crops.dtype != np.uint8 or crops.shape != (meta['cells'], HEIGHT, WIDTH, 1) or masks.shape != (len(crops),) or cells.shape != (len(crops), 2) or np.any((masks < 1) | (masks > 63)):
        raise ValueError('Invalid prepared TRAIN tensor shapes or masks')
    return crops, masks, meta


def material_gain(baseline, candidate):
    """Predeclared usefulness check; does not replace whole-word safety gates."""
    old, new = rank(baseline), rank(candidate)
    return new > old and new[1] >= old[1]+.05 and new[2] >= old[2]+.03 and new[3] >= old[3]-.01 and new[4] >= old[4]


def train(args):
    from word_torch_port import load_keras_checkpoint
    experiment, checkpoint = args.experiment.resolve(), args.checkpoint.resolve()
    assert_unopened(experiment)
    run = experiment/args.run_name
    if run.exists() or run.parent != experiment or not args.run_name.startswith('run-cell-refiner'):
        raise ValueError('Use a new run-cell-refiner directory directly inside the experiment')
    if not (1 <= args.max_seconds <= 600 and 1 <= args.epochs <= 30 and args.steps > 0 and args.batch_size > 0 and args.patience > 0):
        raise ValueError('Pilot requires positive limits and at most 600 seconds / 30 epochs')
    parent = checkpoint.parent
    selected = read(parent/'selection.json')
    detector_config = read(parent/'training-config.json')
    if selected.get('selection_scope') != 'validation_only' or selected.get('test_opened') or checkpoint.name != 'best-word.keras':
        raise ValueError('Detector must be frozen by validation-only selection')
    verify_hashes({checkpoint: selected['sha256'], parent/'training-config.json': selected['training_config_sha256']})
    crops, masks, preparation = load_prepared(args.prepared, experiment)
    thresholds = detector_config['thresholds']
    if thresholds['dotConfidence'] != .7:
        raise ValueError('Pilot requires the unchanged .7 dot confidence threshold')
    val_manifest, val_index = experiment/'validation-symbols.json', experiment/'validation-image-hashes.json'
    val = read(val_manifest)
    if val.get('split') != 'validation':
        raise ValueError('Evaluation requires canonical validation split')
    rows = val['records']
    audit_rows(rows)
    if any(not Path(row['image']).resolve().is_relative_to(experiment/'data/validation') for row in rows):
        raise ValueError('Only canonical validation images may be evaluated')
    pixels = verified_pixels(rows, val_index)
    inputs = [checkpoint, parent/'selection.json', parent/'training-config.json', args.prepared, args.prepared.with_suffix('.json'),
              experiment/'train-symbols.json', experiment/'train-image-hashes.json', val_manifest, val_index]
    frozen = {str(Path(path).resolve()): sha(path) for path in inputs}
    run.mkdir()
    source_dir = run/'source'
    source_dir.mkdir()
    scripts = Path(__file__).resolve().parent
    source_hashes = {}
    for source in scripts.glob('*.py'):
        shutil.copy2(source, source_dir/source.name)
        source_hashes[source.name] = sha(source)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = CellRefiner().to(device)
    config = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        input_sha256=frozen, source_sha256=source_hashes, detector_sha256=selected['sha256'],
        thresholds=thresholds, alphas=ALPHAS, parameters=sum(p.numel() for p in model.parameters()),
        optimizer='NativeAdam fresh state, gradient clipping 10', loss='six known bits BCEWithLogitsLoss',
        sampling='uniform mask then uniform complete TRAIN cell; no validation cells',
        geometry_jitter='70% independently scale axes .88..1.12 and normalized translation +/-.12; 30% identity',
        selection='existing whole-word validation rank over fixed alpha 0/.5/1 and all epochs',
        material_gain='worst exact +5 points, macro exact +3 points, precision loss <=1 point, no new false accepts',
        preparation=preparation, scope='experimental_two_stage_public_recompositions_not_physical_camera',
        test_opened=False, max_seconds_includes='detector validation inference, crop preparation, training and validation')
    write_new(run/'training-config.json', config)
    started = time.perf_counter()
    detector = load_keras_checkpoint(checkpoint).to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(pixels), 8):
            outputs.append(detector(torch.from_numpy(pixels[start:start+8]).to(device)).cpu().numpy())
    outputs = np.concatenate(outputs)
    del detector
    luminance = pixels @ LUMINANCE
    rejected = (luminance.mean((1, 2)) < .06) | (luminance.mean((1, 2)) > .985) | (luminance.std((1, 2)) < .01)
    outputs[rejected] = 0
    np.savez(run/'frozen-validation-detector.npz', outputs=outputs)
    attachments, validation_crops = [], []
    for index, (output, row) in enumerate(zip(outputs, rows)):
        for (gy, gx), box in selected_grids(output, thresholds, row.get('selection')):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue  # Decoder already rejects these; never repair geometry.
            attachments.append((index, gy, gx))
            validation_crops.append(crop_cell(pixels[index], box))
    validation_crops = np.stack(validation_crops) if validation_crops else np.empty((0, HEIGHT, WIDTH, 1), np.float32)
    np.savez(run/'validation-predicted-crops.npz', inputs=validation_crops, attachments=np.asarray(attachments, np.int32),
             scope=np.asarray('validation_predicted_boxes'), validation_sha256=np.asarray(sha(val_manifest)),
             detector_sha256=np.asarray(selected['sha256']), crop_code_sha256=np.asarray(sha(__file__)))
    del pixels, luminance
    baseline, baseline_decisions = score(outputs, rows, thresholds)
    write_new(run/'validation-baseline.json', dict(alpha=0, groups=baseline, decisions=baseline_decisions, rank=rank(baseline)))
    best_rank, best_groups = rank(baseline), baseline
    best = dict(epoch=0, alpha=0., checkpoint=None, sha256=None)
    optimizer = NativeAdam(model.parameters(), args.learning_rate)
    rng = np.random.default_rng(args.seed)
    history, stale, stop_reason = [], 0, 'maximum_epochs'
    for epoch in range(1, args.epochs+1):
        if time.perf_counter()-started >= args.max_seconds:
            stop_reason = 'wall_clock_budget'
            break
        model.train()
        losses = []
        indices = balanced_indices(masks, args.steps*args.batch_size, rng)
        for step in range(args.steps):
            if time.perf_counter()-started >= args.max_seconds:
                stop_reason = 'wall_clock_budget'
                break
            chosen = indices[step*args.batch_size:(step+1)*args.batch_size]
            image = torch.from_numpy(crops[chosen].astype(np.float32)/255.).to(device)
            truth = torch.from_numpy(((masks[chosen, None] >> np.arange(6)) & 1).astype(np.float32)).to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(augment_crops(image)), truth)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite refiner training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            break
        model.eval()
        with torch.no_grad():
            probabilities = np.concatenate([model(torch.from_numpy(validation_crops[start:start+256]).to(device)).sigmoid().cpu().numpy()
                for start in range(0, len(validation_crops), 256)]) if len(validation_crops) else np.empty((0, 6), np.float32)
        checkpoint_path = run/f'epoch{epoch:02d}.npz'
        save_refiner(model, checkpoint_path)
        report = dict(epoch=epoch, train_loss=float(np.mean(losses)), training_steps=len(losses),
                      elapsed_seconds=time.perf_counter()-started, checkpoint_sha256=sha(checkpoint_path), variants=[])
        improved = False
        for alpha in ALPHAS:
            if alpha == 0:
                groups, decisions = baseline, baseline_decisions
            else:
                groups, decisions = score(replace_dots(outputs, attachments, probabilities, alpha), rows, thresholds)
            current = rank(groups)
            report['variants'].append(dict(alpha=alpha, groups=groups, rank=current, decisions=decisions))
            if current > best_rank:
                best_rank, best_groups, improved = current, groups, True
                best = dict(epoch=epoch, alpha=alpha, checkpoint=str(checkpoint_path), sha256=sha(checkpoint_path))
        write_new(run/f'validation-epoch{epoch:02d}.json', report)
        history.append({key: value for key, value in report.items() if key != 'variants'} | {'variants':
            [dict(alpha=item['alpha'], rank=item['rank']) for item in report['variants']]})
        stale = 0 if improved else stale+1
        print('REFINER_EPOCH '+json.dumps(history[-1]), flush=True)
        if stop_reason == 'wall_clock_budget' or time.perf_counter()-started >= args.max_seconds:
            stop_reason = 'wall_clock_budget'
            break
        if stale >= args.patience:
            stop_reason = 'validation_patience'
            break
        if (run/'stop-after-epoch').exists():
            stop_reason = 'stop_file'
            break
    verify_hashes(frozen)
    verify_hashes({scripts/name: digest for name, digest in source_hashes.items()})
    assert_unopened(experiment)
    write_new(run/'history.json', history)
    write_new(run/'selection.json', dict(**best, version=VERSION, selection_scope='validation_only',
        detector_checkpoint=str(checkpoint), detector_sha256=selected['sha256'], rank=best_rank, groups=best_groups,
        baseline_rank=rank(baseline), material_gain=material_gain(baseline, best_groups),
        target_met_on_validation=bool(best_rank[0] and best_rank[1] >= .8), test_opened=False,
        training_config_sha256=sha(run/'training-config.json'), elapsed_seconds=time.perf_counter()-started,
        termination_reason=stop_reason, epochs_completed=len(history),
        artifact_sha256={str(path): sha(path) for path in (run/'frozen-validation-detector.npz', run/'validation-predicted-crops.npz')}))
    print('REFINER_SELECTED '+json.dumps(read(run/'selection.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--experiment', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--experiment', type=Path, required=True)
    fit.add_argument('--prepared', type=Path, required=True)
    fit.add_argument('--checkpoint', type=Path, required=True)
    fit.add_argument('--run-name', default='run-cell-refiner-pilot')
    fit.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    fit.add_argument('--epochs', type=int, default=18)
    fit.add_argument('--steps', type=int, default=100)
    fit.add_argument('--batch-size', type=int, default=128)
    fit.add_argument('--learning-rate', type=float, default=.001)
    fit.add_argument('--patience', type=int, default=5)
    fit.add_argument('--max-seconds', type=float, default=600)
    fit.add_argument('--seed', type=int, default=20260911)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.experiment, args.output)
    else:
        train(args)


if __name__ == '__main__':
    main()
