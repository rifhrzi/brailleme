"""Efficient, source-separated continuation; test is opened only after selection.

Reuses source images, Python environments and checkpoints. Creates no Android
build directory. Keeps research selection separate from deployment eligibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
from PIL import Image
import torch

from word_data import save_json, augment_patch, make_target, WIDTH, HEIGHT
from word_decode_id_v4 import decode
from word_symbol_v4_loss import MINORITY_MASKS, validate_minority_weight, validate_dot_loss_scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    return value['records'] if isinstance(value, dict) else value


def score(outputs, rows, thresholds):
    if len(outputs) != len(rows) or not rows:
        raise ValueError('Every evaluation row requires exactly one output')
    if not np.isfinite(outputs).all():
        raise ValueError('Non-finite model output')
    decisions = []
    grouped = {}
    for output, row in zip(outputs, rows):
        decision = decode(output, thresholds, row.get('selection'))
        group = row.get('symbol_group', row.get('validation_group', 'other'))
        item = dict(source=row['source'], group=group, expected=row['expected'],
                    predicted=decision['text'], reason=decision['reason'],
                    expected_cells=len(row.get('masks', [])), detected_cells=len(decision['cells']))
        decisions.append(item)
        grouped.setdefault(group, []).append(item)
    def summarize(items):
        positives = [r for r in items if r['expected']]
        negatives = [r for r in items if not r['expected']]
        accepted = sum(bool(r['predicted']) for r in positives)
        correct = sum(r['predicted'] == r['expected'] for r in positives)
        return dict(total=len(items), positive_total=len(positives), correct=correct,
                    accepted=accepted, exact_accuracy=correct/len(positives) if positives else None,
                    accepted_precision=correct/accepted if accepted else 0.,
                    rejection_total=len(negatives), false_accepts=sum(bool(r['predicted']) for r in negatives))
    report = {name: summarize(items) for name, items in sorted(grouped.items())}
    for name, items in grouped.items():
        multi = [r for r in items if len(r['expected']) >= 2]
        if multi:
            report[name]['multi_character'] = summarize(multi)
        if name == 'punctuation':
            report[name]['by_punctuation'] = {mark: summarize([r for r in items if mark in r['expected']])
                                               for mark in '.,?!:;-()'}
    return report, decisions


def rank(groups):
    positive = [v for v in groups.values() if v['positive_total']]
    if not positive:
        raise ValueError('Checkpoint ranking requires positive evaluation rows')
    # A few successful single letters must not hide unsuccessful whole words.
    slices = positive+[v['multi_character'] for v in positive if 'multi_character' in v]
    precision = min(v['accepted_precision'] for v in slices)
    worst = min(v['exact_accuracy'] for v in slices)
    mean = float(np.mean([v['exact_accuracy'] for v in positive]))
    false = sum(v['false_accepts'] for v in groups.values())
    # Preserve 95% accepted precision and mandatory rejection as a separate gate.
    return (precision >= .95 and false == 0, worst, mean, precision, -false)


REQUIRED_GROUPS = {'lowercase', 'capital_single', 'capital_word', 'numbers', 'punctuation', 'rejection'}


def audit_rows(rows):
    groups = {row.get('symbol_group') for row in rows}
    if not REQUIRED_GROUPS <= groups:
        raise ValueError('Evaluation missing semantic groups: '+str(sorted(REQUIRED_GROUPS-groups)))
    if len({row['source'] for row in rows}) != len(rows):
        raise ValueError('Duplicate evaluation source identity')
    for row in rows:
        if not isinstance(row['expected'], str) or len(row['boxes']) != len(row['masks']):
            raise ValueError('Malformed evaluation row')
        if bool(row['expected']) == (row['symbol_group'] == 'rejection'):
            raise ValueError('Expected text disagrees with semantic group')


def verify_hashes(expected):
    for path, digest in expected.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError('Frozen input changed: '+str(path))


def verified_pixels(rows, index_path):
    expected = json.loads(Path(index_path).read_text(encoding='utf-8'))
    if {row['image'] for row in rows} - expected.keys():
        raise ValueError('Evaluation image missing from frozen hash index')
    images = []
    for row in rows:
        verify_hashes({row['image']: expected[row['image']]})
        with Image.open(row['image']) as source:
            if source.size != (480, 192):
                raise ValueError('Evaluation image must match runtime 480x192 input')
            images.append(np.asarray(source.convert('RGB'), np.float32)/255.)
    return np.stack(images)


def evaluate_model(model, rows, array, thresholds):
    previous_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = np.concatenate([model(torch.from_numpy(array[start:start+8]).to(device)).cpu().numpy()
                                  for start in range(0, len(array), 8)])
    model.train(previous_training)
    if len(outputs) != len(rows) or not np.isfinite(outputs).all():
        raise ValueError('Invalid model output count or non-finite output')
    # Preserve the runtime's whole-image quality precheck.
    lum = array @ np.asarray([.2126, .7152, .0722], np.float32)
    quality = (lum.mean((1, 2)) < .06) | (lum.mean((1, 2)) > .985) | (lum.std((1, 2)) < .01)
    outputs[quality] = 0
    return score(outputs, rows, thresholds)


def reserve_test(experiment, run, selection):
    """Reserve TEST once per experiment before loading its labels or pixels.

    An interrupted evaluation remains reserved: silently retrying after partial
    exposure would conceal that the supposedly unseen set has been inspected.
    """
    reservation = dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
                       checkpoint_sha256=selection['sha256'], selection_sha256=sha(run/'selection.json'))
    with (experiment/'test-evaluation-reservation.json').open('x', encoding='utf-8') as stream:
        json.dump(reservation, stream, indent=2)
    return reservation


def evaluate_selected(experiment, run):
    from word_torch_port import load_keras_checkpoint
    run = run.resolve()
    if run.parent != experiment:
        raise ValueError('Selected run must belong directly to this experiment')
    config = json.loads((run/'training-config.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    if selection['selection_scope'] != 'validation_only' or selection['test_opened']:
        raise ValueError('Selection must have been frozen using validation only')
    verify_hashes(config['input_sha256'])
    verify_hashes({run/'best-word.keras': selection['sha256'],
                   run/'training-config.json': selection['training_config_sha256']})
    scripts = Path(__file__).resolve().parent
    verify_hashes({scripts/name: digest for name, digest in config['source_sha256'].items()})
    verify_hashes(json.loads((run/'protected-before.json').read_text()))
    reservation = reserve_test(experiment, run, selection)
    test = read_rows(experiment/'test-symbols.json')
    audit_rows(test)
    pixels = verified_pixels(test, experiment/'test-image-hashes.json')
    model = load_keras_checkpoint(run/'best-word.keras').to('cuda')
    groups, decisions = evaluate_model(model, test, pixels, config['thresholds'])
    result_rank = rank(groups)
    result = dict(checkpoint_sha256=selection['sha256'],
        test_manifest_sha256=sha(experiment/'test-symbols.json'), groups=groups, decisions=decisions,
        target_met=result_rank[0] and result_rank[1] >= .80, thresholds=config['thresholds'],
        scope='source_disjoint_recomposed_public_photographs_not_physical_camera',
        physical_camera_validated=False, reservation=reservation)
    save_json(run/'test-evaluation.json', result)
    save_json(experiment/'final-test-evaluation.json', dict(run=str(run), report_sha256=sha(run/'test-evaluation.json'),
              checkpoint_sha256=selection['sha256'], target_met=result['target_met']))
    print('FINAL_TEST '+json.dumps(groups), flush=True)


class ReplayPatches:
    """Equal new semantic groups, 50% replay; all source rows remain available."""
    def __init__(self, root, replay, symbols, batch, steps, seed, source_weights=None):
        if not replay or not symbols or steps < 1 or batch < 2:
            raise ValueError('Replay, symbols, positive steps and batch size >= 2 required')
        self.root, self.records = Path(root), replay+symbols
        self.batch_size, self.seed, self.epoch = batch, seed, 0
        self.replay_count = len(replay)
        self.steps = steps
        self.group_indices = {}
        for i, row in enumerate(symbols, len(replay)):
            self.group_indices.setdefault(row['symbol_group'], []).append(i)
        self.group_probabilities = None
        if source_weights is not None:
            if (len({row['source'] for row in symbols}) != len(symbols)
                    or set(source_weights) != {row['source'] for row in symbols}
                    or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3
                           for value in source_weights.values())):
                raise ValueError('Sampling weights require exact unique TRAIN sources with integer weights 1..3')
            self.group_probabilities = {}
            for group, indices in self.group_indices.items():
                weights = np.asarray([source_weights[self.records[index]['source']] for index in indices], np.float64)
                self.group_probabilities[group] = weights / weights.sum()
        self.sampling_log = []
        self.on_epoch_end()

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed+self.epoch*100000+index)
        images, labels = [], []
        for record_index in self.order[index*self.batch_size:(index+1)*self.batch_size]:
            row = self.records[record_index]
            with Image.open(self.root/row['image']) as source:
                image = source.convert('RGB')
            positive_count = len(row['boxes'])
            boxes = np.asarray(row['boxes']+row.get('ignore_boxes', []), np.float32).reshape(-1, 4)
            complete = (boxes[:, 0]>=0) & (boxes[:, 1]>=0) & (boxes[:, 2]<=WIDTH) & (boxes[:, 3]<=HEIGHT)
            if 'originally_complete' in row:
                complete[:positive_count] &= np.asarray(row['originally_complete'], bool)
            complete[positive_count:] = False
            rng.random()  # Preserve the existing illumination profile's RNG stream.
            pixels, boxes, eligible = augment_patch(image, boxes, rng, illumination_augmentation=True,
                                                    originally_complete=complete)[:3]
            images.append(pixels.astype(np.float32)/255.)
            labels.append(make_target(boxes[:positive_count], np.asarray(row['masks'], int),
                                      eligible[:positive_count], ignore_boxes=boxes[positive_count:]))
        return np.stack(images), np.stack(labels)

    def on_epoch_end(self):
        self.epoch += 1
        rng = np.random.default_rng(self.seed+self.epoch)
        batches = []
        counts = {g: 0 for g in self.group_indices}
        groups = sorted(self.group_indices)
        for step in range(self.steps):
            replay_n = self.batch_size//2
            replay = rng.choice(self.replay_count, replay_n, replace=replay_n>self.replay_count).tolist()
            new = []
            for j in range(self.batch_size-replay_n):
                group = groups[(step*(self.batch_size-replay_n)+j) % len(groups)]
                if self.group_probabilities is None:
                    new.append(int(rng.choice(self.group_indices[group])))
                else:
                    new.append(int(rng.choice(self.group_indices[group], p=self.group_probabilities[group])))
                counts[group] += 1
            batch = replay+new
            rng.shuffle(batch)
            batches.extend(batch)
        self.order = np.asarray(batches, dtype=np.int64)
        self.sampling_log.append(dict(dataset_epoch=self.epoch, replay_draws=self.steps*(self.batch_size//2),
                                      symbol_draws=counts, order_sha256=hashlib.sha256(self.order.tobytes()).hexdigest()))


class NativeAdam:
    """Standard bias-corrected Adam without optional dynamo/optree imports.

    Uses beta1=.9, beta2=.999, epsilon=1e-7, no weight decay and global gradient
    norm clipping at 10. This is the documented optimizer, not an import bypass.
    """
    def __init__(self, parameters, learning_rate):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.learning_rate = learning_rate
        self.steps = 0
        self.moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        gradients = [p.grad for p in self.parameters if p.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise ValueError('Missing or non-finite training gradient')
        self.steps += 1
        with torch.no_grad():
            norm = torch.sqrt(sum(torch.sum(g*g) for g in gradients))
            scale = torch.clamp(10./(norm+1e-6), max=1.)
            for parameter, (first, second) in zip(self.parameters, self.moments):
                if parameter.grad is None:
                    continue
                gradient = parameter.grad*scale
                first.mul_(.9).add_(gradient, alpha=.1)
                second.mul_(.999).addcmul_(gradient, gradient, value=.001)
                denominator = (second/(1-.999**self.steps)).sqrt().add_(1e-7)
                parameter.addcdiv_(first/(1-.9**self.steps), denominator, value=-self.learning_rate)


class StepTiming:
    """Opt-in synchronized profiling; disabled by default to avoid overhead."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.seconds = {}
        self.counts = {}

    @contextmanager
    def measure(self, name, synchronize=True):
        if not self.enabled:
            yield
            return
        if synchronize:
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if synchronize:
            torch.cuda.synchronize()
        self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - start
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self):
        return {name: dict(total_seconds=total, steps=self.counts[name],
                           mean_milliseconds=1000 * total / self.counts[name])
                for name, total in self.seconds.items()}


def requested_stop(signal_requested, stop_file):
    if signal_requested:
        return 'sigint_after_completed_epoch'
    if Path(stop_file).is_file():
        return 'stop_file_after_completed_epoch'
    return None


def bind_sampling_checkpoint(weights_path, checkpoint):
    """An optional sampler must derive from TRAIN and the exact resumed weights."""
    metadata = json.loads(Path(weights_path).read_text(encoding='utf-8'))
    if metadata.get('split') != 'train' or metadata.get('scope') != 'TRAIN_only_dot_mask_sampling':
        raise ValueError('Hard-example sampling must be explicitly TRAIN-only')
    if metadata.get('checkpoint_sha256') != sha(checkpoint):
        raise ValueError('Mining checkpoint differs from resumed checkpoint')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('training/checkpoints/braille-word-reader-v1/best-word.keras'))
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--minority-weight', type=validate_minority_weight, default=1.,
                        help='Positive loss weight >=1 for fully labeled capital/number/punctuation cells')
    parser.add_argument('--dot-loss-scale', type=validate_dot_loss_scale, default=1.,
                        help='Scale >=1 for all known positive dot BCE, including the minority extra; leaves localization/geometry unchanged')
    parser.add_argument('--hard-example-weights', type=Path,
                        help='Verified TRAIN-only weights 1..3 mined from this exact resumed checkpoint')
    parser.add_argument('--profile-steps', action='store_true',
                        help='Synchronize CUDA per stage and record timings; adds profiling overhead')
    parser.add_argument('--stop-file', type=Path,
                        help='Stop after saving current complete epoch when this file exists; default RUN/stop-after-epoch')
    parser.add_argument('--run-name', default='run-transfer')
    parser.add_argument('--evaluate-selected', type=Path,
                        help='Evaluate this frozen run on TEST once; does not train')
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1 or args.batch_size < 2 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError('Positive epochs/steps/patience/learning rate and batch size >= 2 required')
    started = time.perf_counter()
    repo = Path.cwd().resolve()
    experiment = args.experiment.resolve()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Existing CUDA environment is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    if args.evaluate_selected:
        evaluate_selected(experiment, args.evaluate_selected)
        return
    if (experiment/'test-evaluation-reservation.json').exists():
        raise ValueError('TEST already reserved/opened; no further tuning within this experiment')
    from word_torch_port import load_keras_checkpoint, save_keras_checkpoint
    from word_symbol_v4_loss import word_relative_loss
    run = (experiment/args.run_name).resolve()
    if run.parent != experiment:
        raise ValueError('Run name must designate a direct experiment child directory')
    run.mkdir(exist_ok=False)
    stop_file = args.stop_file.resolve() if args.stop_file else run/'stop-after-epoch'
    data = repo/'.braille-training-work/word-scan-v1-20260905/data'
    train_path, val_path, test_path = [experiment/f'{split}-symbols.json' for split in ('train','validation','test')]
    replay_path = (data/'../combined-training-v5/combined-training-manifest.json').resolve()
    # Hash test bytes now, without inspecting samples or running test predictions.
    input_paths = [train_path, val_path, test_path, args.checkpoint.resolve(), replay_path,
                   experiment/'corpus-audit.json', *[experiment/f'{split}-image-hashes.json'
                   for split in ('train', 'validation', 'test')]]
    source_weights, sampling_metadata = None, None
    if args.hard_example_weights:
        from word_symbol_v4_mine import load_sampling_weights
        sampling_metadata = bind_sampling_checkpoint(args.hard_example_weights, args.checkpoint)
        source_weights = load_sampling_weights(args.hard_example_weights, train_path, experiment/'train-image-hashes.json')
        input_paths += [args.hard_example_weights.resolve(), Path(sampling_metadata['checkpoint_path']),
                        Path(sampling_metadata['manifest_path']), Path(sampling_metadata['image_index_path'])]
    frozen = {str(p): sha(p) for p in input_paths}
    protected = list((repo/'training/checkpoints').glob('retained-*.json'))
    protected += list((repo/'training/checkpoints').glob('*/best-word.keras'))
    protected += list((repo/'assets/models/braille-word-reader-v1').glob('*'))
    protected = {str(p): sha(p) for p in protected if p.is_file()}
    save_json(run/'protected-before.json', protected)
    symbols, validation = read_rows(train_path), read_rows(val_path)
    audit_rows(symbols)
    audit_rows(validation)
    verify_hashes(json.loads((experiment/'train-image-hashes.json').read_text()))
    replay = read_rows(replay_path)
    # Keep meaningful exposure to existing negative backgrounds during replay.
    negatives = [r for r in replay if not r['boxes']]
    replay += negatives*12
    training = ReplayPatches(data, replay, symbols, args.batch_size, args.steps, 20260910,
                             source_weights=source_weights)
    thresholds = json.loads((repo/'assets/models/braille-word-reader-v1/manifest.json').read_text())['thresholds']
    model = load_keras_checkpoint(args.checkpoint).to('cuda')
    if args.freeze_backbone:
        model.freeze_backbone()
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
                  input_sha256=frozen, thresholds=thresholds, target_exact_accuracy=.80,
                  target_accepted_precision=.95, test_policy='Evaluate once after validation-only checkpoint selection',
                  sampling=('Half replay, half equal semantic groups; verified TRAIN weights 1..3 normalized within each group'
                            if source_weights is not None else 'Half replay, half equal semantic groups; seeded sampling; no source image duplication'),
                  hard_example_weights=(dict(path=str(args.hard_example_weights.resolve()),
                      sha256=sha(args.hard_example_weights), split='train',
                      checkpoint_sha256=sampling_metadata['checkpoint_sha256'],
                      counts={str(weight): sum(value == weight for value in source_weights.values()) for weight in (1, 2, 3)})
                      if source_weights is not None else None),
                  training_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  gpu=torch.cuda.get_device_name(0), engine='native_torch_keras_archive_graph', torch=torch.__version__,
                  packages={name: version(name) for name in ('torch', 'numpy', 'pillow', 'h5py')},
                  optimizer=dict(name='Adam', learning_rate=args.learning_rate, beta1=.9, beta2=.999,
                                 epsilon=1e-7, global_clipnorm=10., weight_decay=0, state='fresh_for_this_run'),
                  loss=dict(name='relative_joint_with_opt_in_minority_positive_weight', minority_weight=args.minority_weight,
                            dot_loss_scale=args.dot_loss_scale, dot_scale_scope='Known positive dot BCE including minority extra only',
                            minority_masks=list(MINORITY_MASKS), normalizers='Original unweighted center and known-dot counts',
                            target_selection='Fully known binary ground-truth masks at positive centers only',
                            negative_and_ignored_terms='Unchanged'),
                  graceful_stop=dict(file=str(stop_file), signal='SIGINT', policy='Finish current epoch, save checkpoint and validation-only selection'),
                  physical_camera_validated=False, deployment='Research artifact only until regression/export/runtime qualification')
    config['arguments'] = {k: str(v) if isinstance(v, Path) else v for k,v in config['arguments'].items()}
    source_dir = run/'source'
    source_dir.mkdir()
    # Small reproducibility snapshot; this is training evidence, not build staging.
    for p in (repo/'training/scripts').glob('*.py'):
        shutil.copy2(p, source_dir/p.name)
    config['source_sha256'] = {p.name: sha(p) for p in source_dir.glob('*.py')}
    save_json(run/'training-config.json', config)
    pixels = verified_pixels(validation, experiment/'validation-image-hashes.json')
    baseline, decisions = evaluate_model(model, validation, pixels, thresholds)
    save_json(run/'baseline-validation.json', dict(groups=baseline, decisions=decisions))
    print('BASELINE '+json.dumps(baseline), flush=True)
    best_rank = rank(baseline)
    best_epoch = 0
    best_path = args.checkpoint.resolve()
    history = []
    optimizer = NativeAdam(model.parameters(), args.learning_rate)
    stop_requested = False
    def request_interrupt(_signal_number, _frame):
        nonlocal stop_requested
        stop_requested = True
    previous_sigint = signal.signal(signal.SIGINT, request_interrupt)
    termination_reason = 'epoch_limit'
    for epoch in range(args.epochs):
        model.train()
        losses = []
        timing = StepTiming(args.profile_steps)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(training.__getitem__, 0)
            for step in range(args.steps):
                with timing.measure('data_wait', synchronize=False):
                    images, targets = future.result()
                if step+1 < args.steps:
                    future = executor.submit(training.__getitem__, step+1)
                optimizer.zero_grad()
                with timing.measure('forward_and_loss_including_transfer'):
                    output = model(torch.from_numpy(images).to('cuda'))
                    loss = word_relative_loss(torch.from_numpy(targets).to('cuda'), output,
                                              args.minority_weight, args.dot_loss_scale)
                    if not torch.isfinite(loss).item():
                        raise ValueError('Non-finite training loss')
                with timing.measure('backward'):
                    loss.backward()
                with timing.measure('optimizer'):
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if step % 20 == 0:
                    print(f'BATCH epoch={epoch+1} step={step+1}/{args.steps} loss={losses[-1]:.6f}', flush=True)
        groups, decisions = evaluate_model(model, validation, pixels, thresholds)
        current_rank = rank(groups)
        path = run/f'epoch{epoch+1:02d}.keras'
        save_keras_checkpoint(model, args.checkpoint, path)
        row = dict(epoch=epoch+1, elapsed_seconds=time.perf_counter()-started,
                   checkpoint_sha256=sha(path), groups=groups, rank=current_rank,
                   logs=dict(loss=float(np.mean(losses))))
        if args.profile_steps:
            row['step_timing'] = timing.summary()
        history.append(row)
        save_json(run/f'epoch{epoch+1:02d}-validation.json', dict(row, decisions=decisions))
        save_json(run/'history.json', history)
        save_json(run/'sampling.json', training.sampling_log)
        if current_rank > best_rank:
            best_rank, best_epoch, best_path = current_rank, epoch+1, path
        print('EPOCH '+json.dumps(row), flush=True)
        stop_reason = requested_stop(stop_requested, stop_file)
        if stop_reason:
            termination_reason = stop_reason
            break
        if current_rank[0] and current_rank[1] >= .80:
            termination_reason = 'validation_target_met'
            break
        if epoch+1-best_epoch >= args.patience:
            termination_reason = 'validation_patience'
            break
        training.on_epoch_end()
    signal.signal(signal.SIGINT, previous_sigint)
    selection = dict(epoch=best_epoch, path=str(best_path), sha256=sha(best_path), rank=best_rank,
                     selection_scope='validation_only', test_opened=False,
                     termination_reason=termination_reason,
                     training_config_sha256=sha(run/'training-config.json'))
    save_json(run/'selection.json', selection)
    shutil.copy2(best_path, run/'best-word.keras')
    verify_hashes(frozen)
    changed = [p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
    save_json(run/'completion.json', dict(elapsed_seconds=time.perf_counter()-started,
        epochs_completed=len(history), selected_epoch=best_epoch, validation_rank=best_rank,
        termination_reason=termination_reason,
        validation_target_met=best_rank[0] and best_rank[1]>=.80, test_opened=False, protected_changed=changed,
        build_folders_created=0, deployed=False))
    if changed:
        raise ValueError('Protected inputs changed: '+str(changed))
    print('COMPLETE_VALIDATION_ONLY '+json.dumps(selection), flush=True)


if __name__ == '__main__':
    main()
