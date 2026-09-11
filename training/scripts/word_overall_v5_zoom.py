"""Fixed flip and center-zoom fallback, independent of the frozen flip module.

All runtime routing uses predictions and image quality only. No answer labels
or annotation boxes enter runtime APIs. The CLI evaluates development validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

import word_overall_v5_tta as flip
from word_overall_v5_protocol import metrics, read, sha, validation_rank, verify_hashes, write_new

RUNTIME_MODULE = 'word_overall_v5_zoom'
SCALES = {'zoom125': 1.25, 'zoom150': 1.5}
VIEWS = ('identity', *flip.VIEWS, *SCALES)
POLICIES = dict(flip.POLICIES,
    zoom_fallback=('identity', 'zoom_logit_mean', 'zoom_probability_mean'),
    cascade_zoom_fallback=(*flip.POLICIES['cascade'], 'zoom_logit_mean', 'zoom_probability_mean'))


def method_specification():
    return dict(version='overall-v5-fixed-flip-zoom-fallback-1', runtime_module=RUNTIME_MODULE,
        views=list(VIEWS), policies={name: list(order) for name, order in POLICIES.items()},
        flip_transforms=flip.method_specification()['transforms'], scales=dict(SCALES),
        pixel_transform='Bilinear inverse sampling at pixel centers; float64 coordinates and float32 interpolation; outside=source RGBmean',
        box_transform='zoom=(original-.5)*scale+.5; inverse=(zoom-.5)/scale+.5',
        matching='Actual NMS proposals only; both zoom proposal and projected original box complete within frozen edgeMargin; one-to-one greedy IoU>=.5',
        dot_combination='Original and matched zoom dots: logit mean then probability mean; unmatched original dots stay unchanged',
        routing='First accepted text in fixed order; original geometry, scores and quality check unchanged; no GT access',
        selection_rank=['overall_exact_accuracy', 'accepted_precision', 'negative_invalid_false_accepts'],
        tie_break='earlier policy in declared order', threshold_changes=False, geometry_changes=False,
        quality_changes=False, actual_selected_zoom_execution='Only if the selected policy still rejects after its identity/flip prefix')


def center_zoom(images, scale):
    """Same pixel-center bilinear transform as the fixed two-scale diagnostic."""
    images = np.asarray(images)
    if (images.ndim != 4 or images.shape[-1] != 3 or not np.isfinite(images).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Expected finite NHWC RGB and a positive scale')
    _, height, width, _ = images.shape
    xs = (np.arange(width, dtype=np.float64)+.5-width/2)/scale+width/2-.5
    ys = (np.arange(height, dtype=np.float64)+.5-height/2)/scale+height/2-.5
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    wx, wy = (xs-x0).astype(np.float32), (ys-y0).astype(np.float32)
    fill = images.mean((1, 2), keepdims=True)
    result = np.zeros_like(images, dtype=np.float32)
    for x_offset, x_weight in ((0, 1-wx), (1, wx)):
        xindex = x0+x_offset
        for y_offset, y_weight in ((0, 1-wy), (1, wy)):
            yindex = y0+y_offset
            valid = ((yindex[:, None] >= 0) & (yindex[:, None] < height)
                     & (xindex[None, :] >= 0) & (xindex[None, :] < width))
            sample = images[:, np.clip(yindex, 0, height-1)[:, None], np.clip(xindex, 0, width-1)[None, :], :]
            result += np.where(valid[None, :, :, None], sample, fill)*(y_weight[:, None]*x_weight[None, :])[None, :, :, None]
    return np.ascontiguousarray(result)


def transformed_images(images, name):
    return center_zoom(images, SCALES[name]) if name in SCALES else flip.transformed_images(images, name)


def inverse_selected(output, scale, thresholds, selection, decoder):
    selection = np.asarray([0., 0., 1., 1.] if selection is None else selection, np.float64)
    view_selection = (selection-.5)*scale+.5
    candidates = flip.selected_cells(output, thresholds, view_selection.tolist(), decoder)
    margin = thresholds['edgeMargin']
    bounds = [max(margin, view_selection[0]), max(margin, view_selection[1]),
              min(1-margin, view_selection[2]), min(1-margin, view_selection[3])]
    result = []
    for grid, cell in candidates:
        x0, y0, x1, y1 = cell['box']
        if x0 < bounds[0] or y0 < bounds[1] or x1 > bounds[2] or y1 > bounds[3]:
            continue
        result.append((grid, dict(cell, box=((np.asarray(cell['box'], np.float64)-.5)/scale+.5).tolist())))
    return result


def zoom_average(base, base_cells, zoom_cells, thresholds, use_logits=False):
    gathered = [[np.asarray(cell['dots'])] for _, cell in base_cells]
    margin = thresholds['edgeMargin']
    for name, candidates in zoom_cells.items():
        indices, eligible = [], []
        for index, item in enumerate(base_cells):
            box = (np.asarray(item[1]['box'])-.5)*SCALES[name]+.5
            if box[0] < margin or box[1] < margin or box[2] > 1-margin or box[3] > 1-margin:
                continue
            indices.append(index); eligible.append(item)
        for i, j in flip.match_cells(eligible, candidates).items():
            gathered[indices[i]].append(np.asarray(candidates[j][1]['dots']))
    result = base.copy()
    for (grid, _), parts in zip(base_cells, gathered):
        if len(parts) == 1:
            continue
        values = np.stack(parts)
        if use_logits:
            values = np.clip(values, 1e-6, 1-1e-6)
            dots = 1/(1+np.exp(-np.log(values/(1-values)).mean(0)))
        else:
            dots = values.mean(0)
        result[grid[0], grid[1], 5:] = dots
    if not np.array_equal(result[..., :5], base[..., :5]):
        raise ValueError('Zoom changed original proposal geometry/objectness')
    return result


def _decoder(decoder):
    if decoder is None:
        from word_decode_id_v4 import decode
        return decode
    return decoder


def _policies(policies):
    if not policies or len(set(policies)) != len(policies) or any(name not in POLICIES for name in policies):
        raise ValueError('Request distinct predefined policies only')


def runtime_decisions(outputs, thresholds, selection=None, *, quality_rejected=False,
                      views_are_inverted=False, decoder=None, policies=tuple(POLICIES)):
    """Consume named RAW tensors; inversion flag affects only the three flips.

    Zoom tensors are always raw zoom-view outputs: their selected boxes are
    inverse-mapped, never densely resampled. Unneeded views may be omitted.
    """
    _policies(policies)
    decoder = _decoder(decoder)
    base = np.asarray(outputs['identity'])
    if quality_rejected:
        base = np.zeros_like(base)
    identity = decoder(base, thresholds, selection)
    if identity['text'] or quality_rejected or not identity['cells'] or identity['reason'] == 'malformed':
        return dict(decisions=dict.fromkeys(policies, identity), routes=dict.fromkeys(policies, 'identity'))
    needs_flip = any(name in ('simple_fallback', 'cascade', 'cascade_zoom_fallback') for name in policies)
    previous = flip.runtime_decisions(outputs, thresholds, selection, quality_rejected=False,
        views_are_inverted=views_are_inverted, decoder=decoder) if needs_flip else dict(
            decisions={'identity': identity}, routes={'identity': 'identity'})
    prefixes = {name: previous['decisions']['cascade' if name == 'cascade_zoom_fallback' else
                    'identity' if name == 'zoom_fallback' else name] for name in policies}
    prefix_routes = {name: previous['routes']['cascade' if name == 'cascade_zoom_fallback' else
                    'identity' if name == 'zoom_fallback' else name] for name in policies}
    needs_zoom = any(name in ('zoom_fallback', 'cascade_zoom_fallback') and not value['text']
                     for name, value in prefixes.items())
    candidates = []
    if needs_zoom:
        base_cells = flip.selected_cells(base, thresholds, selection, decoder)
        zoom_cells = {name: inverse_selected(outputs[name], scale, thresholds, selection, decoder)
                      for name, scale in SCALES.items()}
        for name, logits in (('zoom_logit_mean', True), ('zoom_probability_mean', False)):
            value = decoder(zoom_average(base, base_cells, zoom_cells, thresholds, logits), thresholds, selection)
            candidates.append((name, value))
            if value['text']:
                break
    decisions, routes = {}, {}
    for policy, prefix in prefixes.items():
        decision, route = prefix, prefix_routes[policy]
        if policy in ('zoom_fallback', 'cascade_zoom_fallback') and not prefix['text']:
            for name, candidate in candidates:
                if candidate['text']:
                    decision, route = candidate, name
                    break
        decisions[policy], routes[policy] = decision, route
    return dict(decisions=decisions, routes=routes)


def infer_policies(predict_batch, pixels, thresholds, selections=None, *, policies=tuple(POLICIES), batch_size=8, decoder=None):
    """Label-free staged inference: identity, needed flips, then needed zooms."""
    _policies(policies)
    if not len(pixels) or batch_size < 1:
        raise ValueError('Need nonempty pixels and positive batch size')
    selections = [None]*len(pixels) if selections is None else selections
    if len(selections) != len(pixels):
        raise ValueError('Selection count differs from pixel count')
    decoder, began = _decoder(decoder), time.perf_counter()
    quality = flip.quality_rejections(pixels)
    durations, counts = dict.fromkeys(VIEWS, 0.), dict.fromkeys(VIEWS, 0)
    def infer(indices, name):
        batches = []
        for start in range(0, len(indices), batch_size):
            inputs = transformed_images(pixels[indices[start:start+batch_size]], name)
            before = time.perf_counter()
            outputs = np.asarray(predict_batch(inputs))
            durations[name] += time.perf_counter()-before
            if outputs.shape != (len(inputs), 48, 120, 11):
                raise ValueError('Predictor returned incorrect batch shape')
            # Identity inversion validates raw probabilities without changing orientation.
            batches.append(flip.invert_outputs(outputs, 'identity'))
        counts[name] = len(indices)
        return {int(index): value for index, value in zip(indices, np.concatenate(batches))} if batches else {}
    values = {'identity': infer(np.arange(len(pixels)), 'identity')}
    identity = [decoder(np.zeros_like(values['identity'][i]) if quality[i] else values['identity'][i], thresholds, selections[i])
                for i in range(len(pixels))]
    eligible = [i for i, value in enumerate(identity) if not value['text'] and not quality[i] and value['cells']]
    needs_flip = any(name in ('simple_fallback', 'cascade', 'cascade_zoom_fallback') for name in policies)
    if needs_flip:
        values.update({name: infer(np.asarray(eligible, int), name) for name in flip.VIEWS})
    prefixes = []
    for i in range(len(pixels)):
        if needs_flip:
            outputs = {name: cache[i] for name, cache in values.items() if i in cache}
            prefix = flip.runtime_decisions(outputs, thresholds, selections[i], quality_rejected=bool(quality[i]), decoder=decoder)
        else:
            prefix = dict(decisions={'identity': identity[i]}, routes={'identity': 'identity'})
        prefixes.append(prefix)
    zoom_indices = [i for i in eligible if 'zoom_fallback' in policies or
                    ('cascade_zoom_fallback' in policies and not prefixes[i]['decisions']['cascade']['text'])]
    if any(name in ('zoom_fallback', 'cascade_zoom_fallback') for name in policies):
        values.update({name: infer(np.asarray(zoom_indices, int), name) for name in SCALES})
    decisions, routes = {name: [] for name in policies}, {name: [] for name in policies}
    for i in range(len(pixels)):
        outputs = {name: cache[i] for name, cache in values.items() if i in cache}
        result = runtime_decisions(outputs, thresholds, selections[i], quality_rejected=bool(quality[i]), decoder=decoder, policies=policies)
        for policy in policies:
            decisions[policy].append(result['decisions'][policy]); routes[policy].append(result['routes'][policy])
    return dict(decisions=decisions, routes=routes, latency=dict(inference_and_routing_seconds=time.perf_counter()-began,
        batched_prediction_seconds=durations, images_per_view=counts, batch_size=batch_size,
        mean_forward_count=sum(counts.values())/len(pixels),
        limitation='Batched model and CPU routing measurements; not Android camera latency'))


def infer_decisions(predict_batch, pixels, thresholds, selections=None, *, policy='cascade_zoom_fallback', batch_size=8, decoder=None):
    result = infer_policies(predict_batch, pixels, thresholds, selections, policies=(policy,), batch_size=batch_size, decoder=decoder)
    return dict(decisions=result['decisions'][policy], routes=result['routes'][policy], latency=result['latency'])


def evaluate_validation(pixels, rows, thresholds, predict_batch, *, batch_size=8, decoder=None):
    if len(rows) != len(pixels):
        raise ValueError('Validation rows and pixels differ')
    score_rows = [dict(source=row['source'], group=row['symbol_group'], expected=row['expected'], predicted='') for row in rows]
    metrics(score_rows, require_primary=True)
    result = infer_policies(predict_batch, pixels, thresholds, [row.get('selection') for row in rows], batch_size=batch_size, decoder=decoder)
    variants = {}
    for name in POLICIES:
        decisions = [dict(row, predicted=value['text'], reason=value['reason'], cells=len(value['cells']), route=route)
                     for row, value, route in zip(score_rows, result['decisions'][name], result['routes'][name])]
        variants[name] = dict(metrics=metrics(decisions, require_primary=True),
                             routing_counts=dict(Counter(row['route'] for row in decisions)), decisions=decisions)
    selected = max(POLICIES, key=lambda name: validation_rank(variants[name]['decisions']))
    return dict(variants=variants, selected_validation_policy=selected, latency=result['latency'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    experiment, run = args.experiment.resolve(), args.run.resolve()
    output = (args.output or run/'zoom-validation.json').resolve()
    if run.parent != experiment or not output.is_relative_to(experiment) or output.exists() or args.batch_size < 1:
        raise ValueError('Use a selected direct child, new report in experiment and positive batch size')
    flip.assert_development_open(experiment)
    protocol_path, config_path, selection_path = experiment/'protocol.json', run/'training-config.json', run/'selection.json'
    protocol, config, selection = read(protocol_path), read(config_path), read(selection_path)
    if protocol['version'] != 'overall-exact-text-v5-20260911' or selection.get('selection_scope') != 'validation_only' or selection.get('test_opened') is not False:
        raise ValueError('Expected frozen v5 protocol and validation-selected candidate')
    manifest = Path(protocol['development_files']['validation-symbols.json']).resolve()
    index = Path(protocol['development_files']['validation-image-hashes.json']).resolve()
    checkpoint = run/'best-word.keras'
    verify_hashes({checkpoint: selection['sha256'], config_path: selection['training_config_sha256'],
                   manifest: protocol['input_sha256'][str(manifest)], index: protocol['input_sha256'][str(index)]})
    rows = flip.assert_validation_manifest(read(manifest)); image_hashes = read(index)
    if set(image_hashes) != {row['image'] for row in rows}:
        raise ValueError('Validation image index coverage differs')
    paths = (protocol_path, config_path, selection_path, manifest, index, checkpoint, Path(__file__).resolve(), Path(flip.__file__).resolve())
    bindings = {str(path): sha(path) for path in paths}
    source = run/'source'
    for name in ('word_torch_port.py', 'word_decode_id_v4.py', 'word_symbols_id_v4.py', 'word_symbols.py',
                 'word_data.py', 'word_anchor_zoom.py', 'benchmark_angelina.py', 'word_overall_v5_protocol.py'):
        bindings[str(source/name)] = config['source_sha256'][name]
        bindings[str(Path(__file__).with_name(name).resolve())] = config['source_sha256'][name]
    verify_hashes(bindings)
    plan_path = output.with_suffix('.plan.json')
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), runtime_module=RUNTIME_MODULE,
        scope='previously_observed_development_validation_only', method=method_specification(), thresholds=config['thresholds'],
        input_sha256=bindings, old_or_fresh_test_used=False, app_changed=False, build_folders_created=0)
    write_new(plan_path, plan)
    sys.path.insert(0, str(source))
    from PIL import Image
    import torch
    from word_decode_id_v4 import decode
    from word_torch_port import load_keras_checkpoint
    pixels = []
    for row in rows:
        verify_hashes({row['image']: image_hashes[row['image']]})
        with Image.open(row['image']) as image:
            if image.size != (480, 192):
                raise ValueError('Validation dimensions changed')
            pixels.append(np.asarray(image.convert('RGB'), np.float32)/255.)
    pixels = np.stack(pixels)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = load_keras_checkpoint(checkpoint).cuda().eval()
    def predict(images):
        with torch.no_grad():
            return model(torch.from_numpy(images).cuda()).cpu().numpy()
    result = evaluate_validation(pixels, rows, config['thresholds'], predict, batch_size=args.batch_size, decoder=decode)
    del model; torch.cuda.empty_cache()
    verify_hashes(bindings); flip.assert_development_open(experiment)
    result.update(plan_sha256=sha(plan_path), runtime_module=RUNTIME_MODULE, method=method_specification(), thresholds=config['thresholds'],
        checkpoint_sha256=sha(checkpoint), validation_sha256=sha(manifest), validation_image_index_sha256=sha(index), source_sha256=sha(__file__),
        selection_scope='validation_only', test_opened=False, old_or_fresh_test_used=False, scope=plan['scope'], physical_camera_validated=False)
    write_new(output, result)
    print(json.dumps({name: value['metrics'] for name, value in result['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()
