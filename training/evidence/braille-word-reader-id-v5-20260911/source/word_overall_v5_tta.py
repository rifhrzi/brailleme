"""Fixed flip TTA with label-free rejection fallback and original geometry.

Runtime functions use NumPy and an injected decoder/predictor, so the same
functions can consume native Torch or TFLite tensors. The optional CLI imports
Torch lazily and evaluates only the protocol-bound development validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

VIEWS = ('horizontal', 'vertical', 'half_turn')
TRANSFORMS = {
    'identity': (False, False, (0, 1, 2, 3, 4, 5)),
    'horizontal': (True, False, (3, 4, 5, 0, 1, 2)),
    'vertical': (False, True, (2, 1, 0, 5, 4, 3)),
    'half_turn': (True, True, (5, 4, 3, 2, 1, 0)),
}
POLICIES = {
    'identity': ('identity',),
    'simple_fallback': ('identity', 'matched_logit_four'),
    'cascade': ('identity', 'matched_logit_four', 'matched_prob_four',
                'matched_prob_vertical', 'matched_prob_half_turn', 'matched_prob_horizontal'),
}


def method_specification():
    return dict(version='overall-v5-fixed-flip-fallback-1',
        policies={name: list(order) for name, order in POLICIES.items()},
        transforms={name: [horizontal, vertical, list(bits)]
                    for name, (horizontal, vertical, bits) in TRANSFORMS.items()}, minimum_match_iou=.5,
        routing='First nonempty text in the fixed order; never inspect labels or reference geometry',
        selection_rank=['overall_exact_accuracy', 'accepted_precision', 'negative_invalid_false_accepts'],
        tie_break='earlier policy in declared order', threshold_changes=False,
        geometry_changes=False, quality_changes=False,
        derivation='Fixed before new checkpoint evaluation from prior development validation ablation')


def transformed_images(images, name):
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError('Expected NHWC image batch')
    horizontal, vertical, _ = TRANSFORMS[name]
    return np.ascontiguousarray(images[:, ::-1 if vertical else 1, ::-1 if horizontal else 1, :])


def invert_outputs(outputs, name):
    """Undo full-image flips in grid coordinates and physical Braille dot slots."""
    outputs = np.asarray(outputs)
    if (outputs.ndim != 4 or outputs.shape[1:] != (48, 120, 11)
            or not np.isfinite(outputs).all() or (outputs < 0).any() or (outputs > 1).any()):
        raise ValueError('Expected finite detector probabilities [N,48,120,11]')
    horizontal, vertical, bits = TRANSFORMS[name]
    result = outputs[:, ::-1 if vertical else 1, ::-1 if horizontal else 1, :].copy()
    if horizontal:
        result[..., 1] = 1-result[..., 1]
    if vertical:
        result[..., 2] = 1-result[..., 2]
    result[..., 5:] = result[..., 5:][..., list(bits)]
    return result


def quality_rejections(images):
    """Exactly the unchanged whole-image quality precheck of v4 evaluation."""
    images = np.asarray(images)
    if (images.ndim != 4 or images.shape[1:] != (192, 480, 3)
            or not np.isfinite(images).all() or (images < 0).any() or (images > 1).any()):
        raise ValueError('Expected normalized RGB [N,192,480,3]')
    luminance = images @ np.asarray([.2126, .7152, .0722], np.float32)
    means = luminance.mean((1, 2))
    return (means < .06) | (means > .985) | (luminance.std((1, 2)) < .01)


def selected_cells(output, thresholds, selection, decoder):
    decision = decoder(output, thresholds, selection)
    if decision['reason'] == 'malformed':
        return []
    lookup = {}
    raw = output.astype(np.float64, copy=False)
    for gy, gx in np.argwhere(raw[..., 0] >= thresholds['proposalScore']):
        value = raw[gy, gx]
        cx, cy = (gx+value[1])/120, (gy+value[2])/48
        x, y = cx-value[3]/2, cy-value[4]/2
        key = (float(value[0]), (x, y, x+value[3], y+value[4]))
        lookup.setdefault(key, (int(gy), int(gx)))
    # Attach by the exact score/box arithmetic used by NMS, not floor(center).
    return [(lookup[(cell['score'], tuple(cell['box']))], cell) for cell in decision['cells']]


def box_iou(a, b):
    width = max(0., min(a[2], b[2])-max(a[0], b[0]))
    height = max(0., min(a[3], b[3])-max(a[1], b[1]))
    overlap = width*height
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return overlap/(area_a+area_b-overlap) if area_a+area_b > overlap else 0.


def match_cells(base, other, minimum_iou=.5):
    if not 0 < minimum_iou <= 1:
        raise ValueError('Match IoU must be positive and at most one')
    pairs = sorted((-box_iou(a[1]['box'], b[1]['box']), i, j)
                   for i, a in enumerate(base) for j, b in enumerate(other))
    matched, used = {}, set()
    for negative_overlap, i, j in pairs:
        if -negative_overlap < minimum_iou:
            break
        if i not in matched and j not in used:
            matched[i] = j
            used.add(j)
    return matched


def matched_average(base_output, views, base_cells, view_cells, use_logits=False):
    """Replace only six dots on original NMS proposals with matched views."""
    result = base_output.copy()
    gathered = [[np.asarray(cell['dots'])] for _, cell in base_cells]
    matched = 0
    for name in views:
        candidates = view_cells[name]
        for i, j in match_cells(base_cells, candidates).items():
            gathered[i].append(np.asarray(candidates[j][1]['dots']))
            matched += 1
    for (grid, _), parts in zip(base_cells, gathered):
        if len(parts) == 1:
            continue
        values = np.stack(parts)
        if use_logits:
            values = np.clip(values, 1e-6, 1-1e-6)
            means = np.log(values/(1-values)).mean(axis=0)
            dots = 1/(1+np.exp(-means))
        else:
            dots = values.mean(axis=0)
        result[grid[0], grid[1], 5:] = dots
    return result, matched


def first_accepted(decisions):
    """Runtime routing deliberately has no expected-label argument."""
    if not decisions:
        raise ValueError('At least the identity decision is required')
    return next((decision for decision in decisions if decision['text']), decisions[0])


def runtime_decisions(outputs, thresholds, selection=None, *, quality_rejected=False,
                      views_are_inverted=False, decoder=None):
    """Return all three fixed policies from tensors, without access to labels.

    Additional views may be omitted when identity is accepted or quality fails.
    By default transformed-view tensors remain in their model-output orientation.
    """
    if decoder is None:
        from word_decode_id_v4 import decode
        decoder = decode
    base = np.asarray(outputs['identity'])
    if quality_rejected:
        base = np.zeros_like(base)
    identity = decoder(base, thresholds, selection)
    variants = {'identity': identity}
    routes = dict.fromkeys(POLICIES, 'identity')
    if identity['text'] or quality_rejected or identity['reason'] == 'malformed':
        return dict(decisions=dict.fromkeys(POLICIES, identity), routes=routes)
    base_cells = selected_cells(base, thresholds, selection, decoder)
    if not base_cells:
        return dict(decisions=dict.fromkeys(POLICIES, identity), routes=routes)
    aligned = {name: np.asarray(outputs[name]) if views_are_inverted else
               invert_outputs(np.asarray(outputs[name])[None], name)[0] for name in VIEWS}
    view_cells = {name: selected_cells(value, thresholds, selection, decoder)
                  for name, value in aligned.items()}
    for name in POLICIES['cascade'][1:]:
        views = VIEWS if name.endswith('_four') else (name[len('matched_prob_'):],)
        combined, _ = matched_average(base, views, base_cells, view_cells,
                                      use_logits=name == 'matched_logit_four')
        if not np.array_equal(combined[..., :5], base[..., :5]):
            raise ValueError('TTA altered original proposal scores or geometry')
        variants[name] = decoder(combined, thresholds, selection)
        if variants[name]['text']:
            break
    decisions = {}
    for policy, order in POLICIES.items():
        available = [(name, variants[name]) for name in order if name in variants]
        decision = first_accepted([value for _, value in available])
        decisions[policy] = decision
        routes[policy] = next(name for name, value in available if value is decision)
    return dict(decisions=decisions, routes=routes)


def infer_policies(predict_batch, pixels, thresholds, selections=None, *,
                   policies=tuple(POLICIES), batch_size=8, decoder=None):
    """Engine-independent inference; no labels or annotation boxes enter here.

    Selections are user crop rectangles in normalized original-image xyxy.
    The return contains decoder decisions/routes per requested fixed policy.
    """
    if not len(pixels) or batch_size < 1 or not policies or any(name not in POLICIES for name in policies):
        raise ValueError('Need nonempty pixels, valid policies and positive batch size')
    selections = [None]*len(pixels) if selections is None else selections
    if len(selections) != len(pixels):
        raise ValueError('Selection count differs from pixel count')
    if decoder is None:
        from word_decode_id_v4 import decode
        decoder = decode
    began = time.perf_counter()
    quality = quality_rejections(pixels)
    elapsed = dict.fromkeys(('identity', *VIEWS), 0.)

    def infer(indices, name):
        batches = []
        for start in range(0, len(indices), batch_size):
            inputs = transformed_images(pixels[indices[start:start+batch_size]], name)
            before = time.perf_counter()
            output = np.asarray(predict_batch(inputs))
            elapsed[name] += time.perf_counter()-before
            if output.shape != (len(inputs), 48, 120, 11):
                raise ValueError('Predictor returned incorrect batch shape')
            batches.append(invert_outputs(output, name))
        return np.concatenate(batches) if batches else np.empty((0, 48, 120, 11), np.float32)

    identity = infer(np.arange(len(pixels)), 'identity')
    original = [decoder(np.zeros_like(value) if quality[i] else value, thresholds, selections[i])
                for i, value in enumerate(identity)]
    use_tta = any(name != 'identity' for name in policies)
    rejected = np.asarray([i for i, value in enumerate(original)
                           if use_tta and not value['text'] and not quality[i] and value['cells']], int)
    # Inputs below are image indices determined only by image quality and model
    # decisions; no expected text, masks or annotation boxes enter this branch.
    augmented = {name: infer(rejected, name) for name in VIEWS}
    positions = {int(index): pos for pos, index in enumerate(rejected)}
    decisions, routes = {name: [] for name in policies}, {name: [] for name in policies}
    for i in range(len(pixels)):
        outputs = {'identity': identity[i]}
        if i in positions:
            outputs.update({name: values[positions[i]] for name, values in augmented.items()})
        result = runtime_decisions(outputs, thresholds, selections[i],
            quality_rejected=bool(quality[i]), views_are_inverted=True, decoder=decoder) if use_tta else dict(
                decisions={'identity': original[i]}, routes={'identity': 'identity'})
        for policy in policies:
            decisions[policy].append(result['decisions'][policy])
            routes[policy].append(result['routes'][policy])
    return dict(decisions=decisions, routes=routes,
        latency=dict(inference_and_routing_seconds=time.perf_counter()-began, batched_prediction_seconds=elapsed,
            identity_images=len(pixels), augmented_images_per_view=len(rejected), batch_size=batch_size,
            mean_forward_count=(len(pixels)+3*len(rejected))/len(pixels),
            limitation='Batched predictor throughput plus CPU routing; not single-image or Android camera latency'))


def infer_decisions(predict_batch, pixels, thresholds, selections=None, *, policy='cascade', batch_size=8, decoder=None):
    """Predict one frozen policy with lazy TTA; suitable for Torch and TFLite."""
    result = infer_policies(predict_batch, pixels, thresholds, selections,
                            policies=(policy,), batch_size=batch_size, decoder=decoder)
    return dict(decisions=result['decisions'][policy], routes=result['routes'][policy], latency=result['latency'])


def evaluate_validation(pixels, rows, thresholds, predict_batch, *, batch_size=8, decoder=None):
    """Score all fixed policies only after label-free inference is complete."""
    from word_overall_v5_protocol import metrics, validation_rank
    if len(rows) != len(pixels):
        raise ValueError('Validation rows/pixels disagree')
    scoring_rows = [dict(source=row['source'], group=row['symbol_group'], expected=row['expected'], predicted='') for row in rows]
    metrics(scoring_rows, require_primary=True)
    result = infer_policies(predict_batch, pixels, thresholds, [row.get('selection') for row in rows],
                            batch_size=batch_size, decoder=decoder)
    result_rows = {policy: [dict(row, predicted=decision['text'], reason=decision['reason'],
                                cells=len(decision['cells']), route=route)
                           for row, decision, route in zip(scoring_rows, result['decisions'][policy], result['routes'][policy])]
                   for policy in POLICIES}
    variants = {name: dict(metrics=metrics(decisions, require_primary=True),
                          routing_counts=dict(Counter(row['route'] for row in decisions)), decisions=decisions)
                for name, decisions in result_rows.items()}
    for name, decisions in result_rows.items():
        variants[name]['groups'] = {}
        for group in sorted({row['group'] for row in decisions}):
            group_rows = [row for row in decisions if row['group'] == group]
            positives = [row for row in group_rows if row['expected']]
            accepted = sum(bool(row['predicted']) for row in positives)
            correct = sum(row['predicted'] == row['expected'] for row in positives)
            variants[name]['groups'][group] = dict(positive_total=len(positives), correct=correct,
                exact_accuracy=correct/len(positives) if positives else None, accepted=accepted,
                accepted_precision=correct/accepted if accepted else None,
                invalid_total=len(group_rows)-len(positives),
                invalid_false_accepts=sum(bool(row['predicted']) for row in group_rows if not row['expected']))
    selected = max(POLICIES, key=lambda name: validation_rank(result_rows[name]))
    return dict(variants=variants, selected_validation_policy=selected, latency=result['latency'])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assert_validation_manifest(document):
    if document.get('split') != 'validation':
        raise ValueError('Only development VALIDATION may be evaluated by this utility')
    return document['records']


def assert_development_open(experiment):
    for name in ('final-method-selection.json', 'test-evaluation-reservation.json', 'final-test-evaluation.json'):
        if (Path(experiment)/name).exists():
            raise ValueError('Final method frozen or TEST opened; further method selection is forbidden')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    experiment, run = args.experiment.resolve(), args.run.resolve()
    output = (args.output or run/'tta-validation.json').resolve()
    if not output.is_relative_to(experiment) or output.exists():
        raise ValueError('Write a new report inside the new experiment only')
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    if run.parent != experiment:
        raise ValueError('Evaluate a selected run directly inside this new experiment')
    assert_development_open(experiment)
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    protocol_path, config_path, selection_path = experiment/'protocol.json', run/'training-config.json', run/'selection.json'
    protocol, config, selection = read(protocol_path), read(config_path), read(selection_path)
    if protocol['version'] != 'overall-exact-text-v5-20260911':
        raise ValueError('Expected the frozen v5 overall-exact protocol')
    manifest = Path(protocol['development_files']['validation-symbols.json']).resolve()
    image_index = Path(protocol['development_files']['validation-image-hashes.json']).resolve()
    checkpoint = run/'best-word.keras'
    if selection.get('selection_scope') != 'validation_only' or selection.get('test_opened') is not False:
        raise ValueError('Checkpoint must have been selected using validation only')
    if sha(checkpoint) != selection['sha256']:
        raise ValueError('Selected checkpoint changed')
    bindings = {str(path): sha(path) for path in (protocol_path, config_path, selection_path, checkpoint, manifest, image_index)}
    for path in (manifest, image_index):
        if bindings[str(path)] != protocol['input_sha256'][str(path)]:
            raise ValueError('Protocol development inputs changed')
    if sha(config_path) != selection['training_config_sha256']:
        raise ValueError('Selected training configuration changed')
    # The complete threshold dictionary is frozen in selected run configuration.
    # The v5 protocol has no second adjustable copy of these thresholds.
    source = run/'source'
    dependencies = ('word_torch_port.py', 'word_data.py', 'word_decode_id_v4.py',
                    'word_symbols_id_v4.py', 'word_overall_v5_protocol.py',
                    'word_symbols.py', 'word_anchor_zoom.py', 'benchmark_angelina.py')
    for name in dependencies:
        if sha(source/name) != config['source_sha256'][name]:
            raise ValueError('Frozen training source changed: '+name)
        bindings[str(source/name)] = sha(source/name)
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    rows = assert_validation_manifest(read(manifest))
    indexed = read(image_index)
    if set(indexed) != {row['image'] for row in rows}:
        raise ValueError('Validation image index differs from manifest')
    plan_path = output.with_suffix('.plan.json')
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='previously_observed_development_validation_only',
        method=method_specification(), thresholds=config['thresholds'], input_sha256=bindings,
        old_or_fresh_test_used=False, app_changed=False, build_folders_created=0)
    with plan_path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    sys.path.insert(0, str(source))
    from PIL import Image
    import torch
    from word_torch_port import load_keras_checkpoint
    from word_decode_id_v4 import decode
    pixels = []
    for row in rows:
        path = Path(row['image'])
        if sha(path) != indexed[row['image']]:
            raise ValueError('Validation image changed: '+str(path))
        with Image.open(path) as image:
            pixels.append(np.asarray(image.convert('RGB'), np.float32)/255.)
    pixels = np.stack(pixels)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_keras_checkpoint(checkpoint).cuda().eval()
    def predict(inputs):
        with torch.no_grad():
            return model(torch.from_numpy(inputs).cuda()).cpu().numpy()
    result = evaluate_validation(pixels, rows, config['thresholds'], predict,
                                 batch_size=args.batch_size, decoder=decode)
    del model
    torch.cuda.empty_cache()
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('Bound evidence changed during evaluation: '+path)
    assert_development_open(experiment)
    result.update(plan_sha256=sha(plan_path), method=method_specification(), thresholds=config['thresholds'],
        checkpoint_sha256=sha(checkpoint), validation_sha256=sha(manifest),
        validation_image_index_sha256=sha(image_index), source_sha256=sha(__file__),
        selection_scope='validation_only', test_opened=False,
        old_or_fresh_test_used=False, scope=plan['scope'], physical_camera_validated=False)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({name: value['metrics'] for name, value in result['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()
