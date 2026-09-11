"""Continue the retained detector with complete-cell anchors and gentle images.

Only TRAIN and previously observed validation are read. The fresh v5 final TEST
is owned by a separate protocol and is never read or ranked by this trainer.
No Android staging or old experiment files are created/modified here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageFilter
import torch

from word_data import affine_patch, augment_patch, make_target
from word_symbol_v4_loss import word_relative_loss
from word_symbol_v4_train import (NativeAdam, audit_rows, evaluate_model, read_rows,
                                  sha, verify_hashes, verified_pixels)
from word_torch_port import load_keras_checkpoint, save_keras_checkpoint

WIDTH, HEIGHT = 480, 192
GROUPS = ('lowercase', 'capital_single', 'capital_word', 'numbers', 'punctuation')
FLIPS = {
    'identity': (False, False, (0, 1, 2, 3, 4, 5)),
    'horizontal': (True, False, (3, 4, 5, 0, 1, 2)),
    'vertical': (False, True, (2, 1, 0, 5, 4, 3)),
    'half_turn': (True, True, (5, 4, 3, 2, 1, 0)),
}


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def contained(boxes):
    boxes = np.asarray(boxes).reshape(-1, 4)
    return ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
            & (boxes[:, 2] <= WIDTH) & (boxes[:, 3] <= HEIGHT))


def annotation(row):
    count = len(row['boxes'])
    boxes = np.asarray(row['boxes']+row.get('ignore_boxes', []), np.float64).reshape(-1, 4)
    complete = contained(boxes)
    if 'originally_complete' in row:
        flags = np.asarray(row['originally_complete'], bool)
        if flags.shape != (count,):
            raise ValueError('Original eligibility must cover every positive annotation')
        complete[:count] &= flags
    complete[count:] = False
    return boxes, complete, count


def rotate_boxes(boxes, angle):
    theta = -np.deg2rad(angle)
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]), axis=1)
    moved = (corners-[WIDTH/2, HEIGHT/2]) @ matrix.T+[WIDTH/2, HEIGHT/2]
    return np.concatenate((moved.min(axis=1), moved.max(axis=1)), axis=1)


def gentle_anchor(image, boxes, complete, rng):
    """Half exact identity; otherwise keep every originally eligible cell whole.

    Containment is checked after each raster operation, not just at the end.
    A rejected geometry draw retries from the untouched input; three failed
    draws fall back to unchanged geometry. Incomplete source cells stay ignored.
    """
    if rng.random() < .5:
        return np.asarray(image, np.uint8), boxes.copy(), complete.copy(), 'identity'
    moved_image, moved_boxes = image, boxes.copy()
    geometry = 'identity_geometry'
    for _ in range(3):
        scale = float(rng.uniform(.9, 1.1))
        tx, ty = float(rng.uniform(-.02, .02))*WIDTH, float(rng.uniform(-.02, .02))*HEIGHT
        candidate_image, candidate_boxes = affine_patch(image, boxes, scale, scale, tx, ty)
        if not contained(candidate_boxes)[complete].all():
            continue
        angle = float(rng.uniform(-2, 2))
        rotated = rotate_boxes(candidate_boxes, angle)
        if not contained(rotated)[complete].all():
            continue
        background = tuple(np.asarray(candidate_image).mean((0, 1)).astype(int))
        moved_image = candidate_image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=background)
        moved_boxes, geometry = rotated, 'gentle_geometry'
        break
    if rng.random() < .2:
        moved_image = moved_image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.15, .7))))
    pixels = np.asarray(moved_image, np.float32)
    center = pixels.mean((0, 1), keepdims=True)
    pixels = (pixels-center)*float(rng.uniform(.8, 1.2))+center+float(rng.uniform(-.05, .05))*255
    return np.clip(pixels, 0, 255).astype(np.uint8), moved_boxes, complete.copy(), geometry


def overall(groups):
    positives = [groups[name] for name in GROUPS]
    total = sum(item['positive_total'] for item in positives)
    if total != 500 or any(item['positive_total'] != 100 for item in positives):
        raise ValueError('Primary development evaluation requires all 500 balanced positives')
    correct = sum(item['correct'] for item in positives)
    accepted = sum(item['accepted'] for item in positives)
    return dict(correct=correct, total=total, exact_accuracy=correct/total,
                accepted=accepted, accepted_precision=correct/accepted if accepted else 0.,
                invalid_false_accepts=groups['rejection']['false_accepts'],
                lowest_group_exact=min(item['exact_accuracy'] for item in positives))


def overall_rank(groups):
    value = overall(groups)
    # The user's new objective is overall exact text. Report precision and
    # rejection independently; never replace the denominator by accepted scans.
    return (value['exact_accuracy'], value['accepted_precision'],
            -value['invalid_false_accepts'])


def cosine_rate(epoch, epochs, peak=3e-5, final=5e-6):
    if epochs < 4 or not 1 <= epoch <= epochs or not 0 < final <= peak:
        raise ValueError('Invalid cosine schedule')
    if epoch <= 2:
        return 1e-5+(peak-1e-5)*(epoch-1)/2
    progress = (epoch-3)/(epochs-3)
    return final+(peak-final)*.5*(1+math.cos(math.pi*progress))


def flip_example(pixels, boxes, masks, name):
    """Transform physical dot labels and all boxes before constructing targets.

    Text transcription is never a gradient label. A reflected word changes its
    reading order, while each reflected physical cell has the remapped six dots.
    This exactly covers the four orientations used by inference-time flips.
    """
    pixels, boxes, masks = np.asarray(pixels), np.asarray(boxes), np.asarray(masks)
    if pixels.shape != (HEIGHT, WIDTH, 3) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('Expected full-size pixels and xyxy boxes')
    if masks.ndim != 1 or not np.issubdtype(masks.dtype, np.integer) or ((masks < 0) | (masks > 63)).any():
        raise ValueError('Expected physical six-dot integer masks')
    horizontal, vertical, permutation = FLIPS[name]
    moved = boxes.copy()
    if horizontal:
        moved[:, 0], moved[:, 2] = WIDTH-boxes[:, 2], WIDTH-boxes[:, 0]
    if vertical:
        moved[:, 1], moved[:, 3] = HEIGHT-boxes[:, 3], HEIGHT-boxes[:, 1]
    remapped = np.zeros_like(masks)
    for destination, source in enumerate(permutation):
        remapped |= ((masks >> source) & 1) << destination
    return np.ascontiguousarray(pixels[::-1 if vertical else 1, ::-1 if horizontal else 1]), moved, remapped


class Curriculum:
    """Fixed batch24:12new,6complete replay anchors,5contexts,1negative."""
    def __init__(self, data_root, replay, symbols, steps, seed, flip_augmentation=False):
        self.data_root, self.replay, self.symbols = Path(data_root), replay, symbols
        self.steps, self.seed = steps, seed
        self.flip_augmentation = flip_augmentation
        self.complete, self.context, self.negative = [], [], []
        for index, row in enumerate(replay):
            _, eligible, count = annotation(row)
            if count == 0:
                self.negative.append(index)
            else:
                self.context.append(index)
                if eligible[:count].all():
                    self.complete.append(index)
        self.groups = {name: [i for i, row in enumerate(symbols) if row['symbol_group'] == name]
                       for name in (*GROUPS, 'rejection')}
        if not self.complete or not self.context or not self.negative or any(not v for v in self.groups.values()):
            raise ValueError('Every curriculum pool must be nonempty')

    def batch(self, epoch, step):
        rng = np.random.default_rng(self.seed+epoch*100000+step)
        choices = [(self.symbols[int(rng.choice(self.groups[GROUPS[(step*11+j) % 5]]))], 'new') for j in range(11)]
        choices += [(self.symbols[int(rng.choice(self.groups['rejection']))], 'invalid')]
        choices += [(self.replay[int(rng.choice(self.complete))], 'complete_replay') for _ in range(6)]
        choices += [(self.replay[int(rng.choice(self.context))], 'context') for _ in range(5)]
        choices += [(self.replay[int(rng.choice(self.negative))], 'negative')]
        rng.shuffle(choices)
        images, targets, stats = [], [], dict(draws={}, augmentation={}, eligible_before=0, eligible_after=0)
        for row, kind in choices:
            with Image.open(self.data_root/row['image']) as source:
                image = source.convert('RGB')
            if image.size != (WIDTH, HEIGHT):
                raise ValueError('Unexpected training input dimensions')
            boxes, complete, count = annotation(row)
            stats['eligible_before'] += int(complete[:count].sum())
            if kind == 'context':
                rng.random()
                pixels, boxes, eligible = augment_patch(image, boxes, rng, illumination_augmentation=True,
                                                       originally_complete=complete)[:3]
                mode = 'original_context_augmentation'
            else:
                pixels, boxes, eligible, mode = gentle_anchor(image, boxes, complete, rng)
                if not np.array_equal(complete, eligible) or not contained(boxes)[complete].all():
                    raise ValueError('Anchor augmentation lost an originally complete cell')
            masks = np.asarray(row['masks'], int)
            if self.flip_augmentation:
                name = tuple(FLIPS)[int(rng.integers(4))]
                pixels, boxes, masks = flip_example(pixels, boxes, masks, name)
                mode += '|'+name
                if kind != 'context' and not contained(boxes)[eligible].all():
                    raise ValueError('Full-image flip lost an eligible anchor')
            stats['eligible_after'] += int(eligible[:count].sum())
            stats['draws'][kind] = stats['draws'].get(kind, 0)+1
            stats['augmentation'][mode] = stats['augmentation'].get(mode, 0)+1
            images.append(pixels.astype(np.float32)/255.)
            targets.append(make_target(boxes[:count], masks, eligible[:count], ignore_boxes=boxes[count:]))
        return np.stack(images), np.stack(targets), stats


def assert_test_unopened(experiment):
    for name in ('final-method-selection.json', 'test-evaluation-reservation.json', 'final-test-evaluation.json'):
        if (experiment/name).exists():
            raise ValueError('Final TEST already opened: no further tuning in this experiment')


def bind_development_inputs(experiment, training_path, validation_path):
    """An editable split label alone cannot authorize a new training source."""
    protocol_path = Path(experiment)/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('version') != 'overall-exact-text-v5-20260911':
        raise ValueError('Expected the frozen v5 overall protocol')
    files = protocol['development_files']
    requested = {'train-symbols.json': Path(training_path).resolve(),
                 'validation-symbols.json': Path(validation_path).resolve()}
    bound = {protocol_path: sha(protocol_path)}
    for name, path in requested.items():
        if path != Path(files[name]).resolve():
            raise ValueError('Requested data differs from protocol development input: '+name)
    for name in ('train-symbols.json', 'validation-symbols.json', 'train-image-hashes.json', 'validation-image-hashes.json'):
        path = Path(files[name]).resolve()
        expected = protocol['input_sha256'].get(str(path))
        if expected is None:
            raise ValueError('Protocol development input has no frozen hash: '+name)
        bound[path] = expected
    verify_hashes(bound)
    return {str(path): digest for path,digest in bound.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('training/checkpoints/braille-word-reader-id-v4/best-word.keras'))
    parser.add_argument('--train-manifest', type=Path, default=Path('C:/bm-symbols-20260908/accuracy80-20260910/train-symbols.json'))
    parser.add_argument('--validation', type=Path, default=Path('C:/bm-symbols-20260908/accuracy80-20260910/validation-symbols.json'))
    parser.add_argument('--run-name', default='run-curriculum')
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--peak-learning-rate', type=float, default=3e-5)
    parser.add_argument('--final-learning-rate', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--flip-augmentation', action='store_true',
                        help='Uniform full-image identity/horizontal/vertical/half-turn with physical dot remapping')
    args = parser.parse_args()
    if args.epochs < 4 or args.steps < 1 or args.patience < 1:
        raise ValueError('Need >=4 epochs, positive steps and patience')
    cosine_rate(1, args.epochs, args.peak_learning_rate, args.final_learning_rate)
    repo, experiment, checkpoint = Path.cwd().resolve(), args.experiment.resolve(), args.checkpoint.resolve()
    assert_test_unopened(experiment)
    run = (experiment/args.run_name).resolve()
    if run.parent != experiment or run.exists():
        raise ValueError('Use a new direct training-run child, preserving all prior evidence')
    training_path, validation_path = args.train_manifest.resolve(), args.validation.resolve()
    protocol_bindings = bind_development_inputs(experiment, training_path, validation_path)
    if json.loads(training_path.read_text())['split'] != 'train' or json.loads(validation_path.read_text())['split'] != 'validation':
        raise ValueError('Only explicit TRAIN and development validation are allowed')
    train_index = training_path.with_name('train-image-hashes.json')
    val_index = validation_path.with_name('validation-image-hashes.json')
    replay_path = repo/'.braille-training-work/word-scan-v1-20260905/combined-training-v5/combined-training-manifest.json'
    data_root = replay_path.parent.parent/'data'
    thresholds_path = repo/'assets/models/braille-word-reader-v1/manifest.json'
    thresholds = json.loads(thresholds_path.read_text())['thresholds']
    sources = Path(__file__).resolve().parent
    inputs = {str(p): sha(p) for p in (training_path, validation_path, train_index, val_index, replay_path, checkpoint, thresholds_path)}
    inputs.update(protocol_bindings)
    verify_hashes(json.loads(train_index.read_text()))
    symbols, rows = read_rows(training_path), read_rows(validation_path)
    audit_rows(symbols); audit_rows(rows)
    curriculum = Curriculum(data_root, read_rows(replay_path), symbols, args.steps, args.seed, args.flip_augmentation)
    pixels = verified_pixels(rows, val_index)
    protected = list((repo/'training/checkpoints').glob('retained-*.json'))
    protected += list((repo/'assets/models/braille-word-reader-v1').glob('*'))
    protected += list((repo/'training/checkpoints').glob('*/best-word.keras'))
    protected = {str(p): sha(p) for p in protected if p.is_file()}
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise ValueError('Existing CUDA environment is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    model = load_keras_checkpoint(checkpoint).cuda()
    run.mkdir(parents=True)
    snapshot = run/'source'; snapshot.mkdir()
    for path in sources.glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  input_sha256=inputs, source_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},
                  thresholds=thresholds, target_overall_exact_accuracy=.8, primary_denominator=500,
                  selection_policy='Overall exact text first; accepted precision and invalid rejection reported independently',
                  data_policy='Original TRAIN replay and v4 TRAIN compositions only; old and fresh TEST excluded from gradients and selection',
                  sampling=dict(batch=24,new_positive=11,new_invalid=1,complete_replay=6,context_replay=5,negative=1,
                                complete_pool=len(curriculum.complete),context_pool=len(curriculum.context),negative_pool=len(curriculum.negative)),
                  optimizer='NativeAdam; fresh moments per run; frozen original BatchNorm statistics',
                  full_image_flips=dict(enabled=args.flip_augmentation, choices=FLIPS,
                      policy='Uniform four orientations after geometric augmentation; recompute box centers and six-dot masks before loss targets; word text is not a gradient label'),
                  learning_rate=dict(schedule='two-epoch warmup then cosine',peak=args.peak_learning_rate,final=args.final_learning_rate),
                  loss=dict(dot_scale=3.,minority_weight=1.5), physical_camera_validated=False, build_folders_created=0)
    save_new(run/'training-config.json', config); save_new(run/'protected-before.json', protected)
    groups, decisions = evaluate_model(model, rows, pixels, thresholds)
    save_new(run/'baseline-validation.json', dict(groups=groups,overall=overall(groups),decisions=decisions))
    best_rank, best_epoch, best_path = overall_rank(groups), 0, checkpoint
    optimizer = NativeAdam(model.parameters(), args.peak_learning_rate)
    history, started, reason = [], time.perf_counter(), 'epoch_limit'
    for epoch in range(1, args.epochs+1):
        model.train(); losses = []
        optimizer.learning_rate = cosine_rate(epoch,args.epochs,args.peak_learning_rate,args.final_learning_rate)
        totals = dict(draws={},augmentation={},eligible_before=0,eligible_after=0)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(curriculum.batch,epoch,0)
            for step in range(args.steps):
                images, truth, stat = future.result()
                if step+1 < args.steps:future=executor.submit(curriculum.batch,epoch,step+1)
                for field in ('draws','augmentation'):
                    for key,value in stat[field].items():totals[field][key]=totals[field].get(key,0)+value
                for field in ('eligible_before','eligible_after'):totals[field]+=stat[field]
                optimizer.zero_grad()
                loss = word_relative_loss(torch.from_numpy(truth).cuda(),model(torch.from_numpy(images).cuda()),1.5,3.)
                if not torch.isfinite(loss).item():raise ValueError('Non-finite training loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        groups, decisions = evaluate_model(model,rows,pixels,thresholds)
        path=run/f'epoch{epoch:02d}.keras';save_keras_checkpoint(model,checkpoint,path)
        metric=overall(groups);current=overall_rank(groups)
        item=dict(epoch=epoch,elapsed_seconds=time.perf_counter()-started,loss=float(np.mean(losses)),
                  learning_rate=optimizer.learning_rate,checkpoint_sha256=sha(path),overall=metric,groups=groups,sampling=totals)
        save_new(run/f'epoch{epoch:02d}-validation.json',dict(item,decisions=decisions))
        history.append(item)
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if current > best_rank:best_rank,best_epoch,best_path=current,epoch,path
        print('EPOCH '+json.dumps(dict(epoch=epoch,overall=metric,groups={k:v['correct'] for k,v in groups.items()},loss=item['loss'],seconds=item['elapsed_seconds'])),flush=True)
        if (run/'stop-after-epoch').exists():reason='stop_file_after_saved_epoch';break
        if epoch-best_epoch >= args.patience:reason='validation_patience';break
    verify_hashes(inputs);verify_hashes(protected);assert_test_unopened(experiment)
    shutil.copy2(best_path,run/'best-word.keras')
    selection=dict(epoch=best_epoch,path=str(best_path),sha256=sha(best_path),rank=best_rank,
                   selection_scope='validation_only',test_opened=False,training_config_sha256=sha(run/'training-config.json'),
                   termination_reason=reason,epochs_completed=len(history),elapsed_seconds=time.perf_counter()-started)
    save_new(run/'selection.json',selection)
    print('COMPLETE '+json.dumps(selection),flush=True)


if __name__ == '__main__':
    main()
