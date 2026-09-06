"""Read-only cached held405 diagnosis; approximate box matching, not AP."""
from __future__ import annotations
import ast
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path.cwd().resolve()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
BASELINE = SELECTION = CANDIDATE_CACHE = DATA = OUT = None



def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def constants(path):
    result = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                result[node.targets[0].id] = value
            elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                result.update(zip((item.id for item in node.targets[0].elts), value))
    return result


def load_decoder():
    """Execute byte-exact archived decoder with only its literal constants."""
    source = SELECTION / 'source'
    data = constants(source / 'word_data.py')
    assert [data[k] for k in ('WIDTH', 'HEIGHT', 'STRIDE')] == [480, 192, 4]
    data['GRID_W'], data['GRID_H'] = data['WIDTH']//data['STRIDE'], data['HEIGHT']//data['STRIDE']
    letters = constants(ROOT / 'training/scripts/benchmark_angelina.py')['LETTER_BY_MASK']
    for module_name, values in [('word_data', data), ('benchmark_angelina', {'LETTER_BY_MASK': letters})]:
        module = types.ModuleType(module_name)
        module.__dict__.update(values)
        sys.modules[module_name] = module
    module = types.ModuleType('archived_word_decode')
    code = (source / 'word_decode.py').read_text(encoding='utf-8')
    exec(compile(code, str(source / 'word_decode.py'), 'exec'), module.__dict__)
    return module, letters


def box_iou(a, b):
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def match(boxes, cells):
    """Greedy maximum IoU >= .25, one-to-one, mask/confidence blind."""
    edges = sorted((-box_iou(box, cell['box_px']), i, j) for i, box in enumerate(boxes) for j, cell in enumerate(cells))
    gt_used, pred_used, pairs = set(), set(), []
    for neg_iou, gt, pred in edges:
        if -neg_iou < .25:
            break
        if gt in gt_used or pred in pred_used:
            continue
        pairs.append((gt, pred, -neg_iou))
        gt_used.add(gt)
        pred_used.add(pred)
    return sorted(pairs), sorted(set(range(len(boxes)))-gt_used), sorted(set(range(len(cells)))-pred_used)


def stats(values):
    return {'count': len(values), 'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None,
            'p90': float(np.percentile(values, 90)) if values else None}


def word_metrics(rows):
    accepted = sum(bool(row['text']) for row in rows)
    correct = sum(row['correct'] for row in rows)
    return {'total': len(rows), 'accepted': accepted, 'correct': correct,
            'whole_word_accuracy': correct/len(rows) if rows else None,
            'accepted_accuracy': correct/accepted if accepted else None,
            'rejected': len(rows)-accepted,
            'rejection_reasons': dict(Counter(row['reason'] for row in rows if not row['text'])),
            'post_nms_detections': sum(len(row['cells']) for row in rows),
            'unmatched_gt_cells': sum(len(row['unmatched_gt']) for row in rows),
            'unmatched_prediction_cells': sum(len(row['unmatched_predictions']) for row in rows)}


def cell_metrics(cells):
    matched = [cell for cell in cells if cell['matched']]
    return {'gt_cells': len(cells), 'matched': len(matched),
            'unmatched': len(cells)-len(matched),
            'matched_correct_mask': sum(cell['mask_correct'] for cell in matched),
            'matched_wrong_mask': sum(not cell['mask_correct'] for cell in matched),
            'matched_wrong_dot_bits': sum(cell['wrong_dot_bits'] for cell in matched),
            'matched_below_objectness_accept': sum(not cell['objectness_pass'] for cell in matched),
            'matched_below_dot_confidence': sum(not cell['dot_confidence_pass'] for cell in matched),
            **{key: stats([cell[key] for cell in matched]) for key in
               ('iou', 'center_error_px', 'center_dx_px', 'center_dy_px', 'width_ratio', 'height_ratio', 'score', 'dot_confidence')}}


def main():
    global ROOT, WORK, BASELINE, SELECTION, CANDIDATE_CACHE, DATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--baseline',type=Path,required=True,help='Selected baseline folder with backend-fullprecision.npz and its report/source')
    parser.add_argument('--selection',type=Path,required=True,help='Current selected-run folder with candidate table and archived evaluator source')
    parser.add_argument('--candidate-cache',type=Path,required=True,help='Current already-ranked best-new held cache folder')
    parser.add_argument('--data',type=Path,required=True,help='Folder containing frozen validation-stress-selection.json and its405 images')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; existing evidence is never overwritten')
    args=parser.parse_args()
    ROOT=args.repo.resolve()
    WORK=ROOT/'.braille-training-work/word-scan-v1-20260905'
    BASELINE,SELECTION,CANDIDATE_CACHE,DATA,OUT=(getattr(args,key).resolve() for key in ('baseline','selection','candidate_cache','data','output'))
    assert all(path.is_relative_to(ROOT) for path in (BASELINE,SELECTION,CANDIDATE_CACHE,DATA,OUT)), 'Keep evidence within repository'
    assert not OUT.exists(), 'Refuse to overwrite diagnosis'
    decoder, letters = load_decoder()
    fixture = DATA / 'validation-stress-selection.json'
    records = read(fixture)['records']
    assert len(records) == 405
    table_path = SELECTION / 'checkpoint-selection.json'
    table = read(table_path)
    control_report = read(BASELINE / 'selected-validation-fullprecision.json')
    mirror_report = read(CANDIDATE_CACHE / 'cache-report.json')
    assert table['external_outputs_used'] is False and table['validation_words'] == 405
    assert mirror_report['status'] == 'passed' and mirror_report['exact_selection_metrics_reproduced'] is True
    assert sha(table_path) == mirror_report['selection_table_sha256']
    assert sha(fixture) == table['fixture_sha256'] == mirror_report['fixture_sha256'] == control_report['oracle']['fixture_sha256']
    selected_report = read(SELECTION/'selected-validation-fullprecision.json')
    assert control_report['oracle']['source_sha256']['word_decode.py'] == selected_report['oracle']['source_sha256']['word_decode.py'], 'Decoder semantics changed between baseline and current'
    assert sha(BASELINE/'source/word_data.py') == control_report['oracle']['source_sha256']['word_data.py']
    old_dims = constants(BASELINE/'source/word_data.py')
    new_dims = constants(SELECTION/'source/word_data.py')
    assert all(old_dims[key] == new_dims[key] for key in ('WIDTH','HEIGHT','STRIDE'))
    for source, digest in selected_report['oracle']['source_sha256'].items():
        assert sha(SELECTION / 'source' / source) == digest
        assert mirror_report['source_sha256'][source] == digest
    assert len(mirror_report['image_sha256']) == 405
    for record in records:
        assert sha(DATA / record['image']) == mirror_report['image_sha256'][record['image']]
        assert len(record['boxes']) == len(record['masks']) == len(record['expected'])
        assert ''.join(letters[mask].upper() for mask in record['masks']) == record['expected']
    # Matcher guards: no duplicated assignment, label-independent and no overlap fails.
    assert match([[0, 0, 10, 10]], [{'box_px': [0, 0, 10, 10]}, {'box_px': [0, 0, 10, 10]}]) == ([(0, 0, 1.)], [], [1])
    assert match([[0, 0, 10, 10]], [{'box_px': [11, 0, 21, 10]}]) == ([], [0], [0])
    assert match([], []) == ([], [], [])
    paths = {'baseline': BASELINE/'backend-fullprecision.npz', 'candidate': CANDIDATE_CACHE/'held-outputs.npz'}
    candidates = {'baseline': control_report, 'candidate': mirror_report['candidate']}
    assert sha(paths['candidate']) == mirror_report['outputs_sha256']
    rows_by_model, cells_by_model, quality_masks = {}, {}, {}
    for name in ('baseline', 'candidate'):
        candidate = candidates[name]
        assert any(row['checkpoint_sha256'] == candidate['checkpoint_sha256'] and row['thresholds'] == candidate['thresholds'] for row in table['candidates'])
        checkpoint = ROOT / candidate['source_checkpoint']
        assert sha(checkpoint) == candidate['checkpoint_sha256']
        with np.load(paths[name], allow_pickle=False) as cache:
            outputs, quality = cache['outputs'], cache['quality_rejected']
            assert outputs.shape == (405, 48, 120, 11) and outputs.dtype == np.float32
            assert quality.shape == (405,) and quality.dtype == np.bool_
            assert str(cache['fixture_sha256']) == sha(fixture)
            assert str(cache['checkpoint_sha256']) == candidate['checkpoint_sha256']
        quality_masks[name] = quality.copy()
        assert np.isfinite(outputs).all() and outputs.min() >= 0 and outputs.max() <= 1
        outputs[quality] = 0  # Identical quality rule used by archived selector.
        recomputed = decoder.summarize(outputs, records, candidate['thresholds'])
        assert recomputed == candidate['words'], name
        rows, all_cells = [], []
        t = candidate['thresholds']
        for index, (output, record) in enumerate(zip(outputs, records)):
            result = decoder.decode(output, t, record.get('selection'))
            predicted = []
            for cell in result['cells']:
                mask = sum(int(dot >= .5) << bit for bit, dot in enumerate(cell['dots']))
                predicted.append({'box_px': (np.asarray(cell['box'])*[480,192,480,192]).tolist(),
                    'score': cell['score'], 'mask': mask, 'letter': letters.get(mask, '?').upper(),
                    'dots': cell['dots'], 'dot_confidence': min(max(dot, 1-dot) for dot in cell['dots'])})
            pairs, missed, extras = match(record['boxes'], predicted)
            matches = {gt: (pred, iou) for gt, pred, iou in pairs}
            gt_rows = []
            for gi, (box, mask) in enumerate(zip(record['boxes'], record['masks'])):
                gt = {'word_index': index, 'gt_index': gi, 'gt_mask': mask, 'gt_letter': letters[mask].upper(),
                      'length': len(record['expected']), 'group': record['validation_group'],
                      'matched': gi in matches}
                if gi in matches:
                    pi, iou = matches[gi]
                    pred = predicted[pi]
                    p, g = np.asarray(pred['box_px']), np.asarray(box)
                    delta = (p[:2]+p[2:]-g[:2]-g[2:])/2
                    gt.update(prediction_index=pi, iou=iou, predicted_mask=pred['mask'], predicted_letter=pred['letter'],
                        mask_correct=pred['mask']==mask, wrong_dot_bits=(mask^pred['mask']).bit_count(),
                        center_error_px=float(np.linalg.norm(delta)), center_dx_px=float(delta[0]), center_dy_px=float(delta[1]),
                        width_ratio=float((p[2]-p[0])/(g[2]-g[0])), height_ratio=float((p[3]-p[1])/(g[3]-g[1])),
                        score=pred['score'], dot_confidence=pred['dot_confidence'],
                        objectness_pass=pred['score'] >= t['acceptScore'], dot_confidence_pass=pred['dot_confidence'] >= t['dotConfidence'])
                gt_rows.append(gt)
                all_cells.append(gt)
            rows.append({'index': index, 'image': record['image'], 'source': record['source'], 'expected': record['expected'],
                'length': len(record['expected']), 'group': record['validation_group'],
                'variant': record.get('stress_transform', {}).get('name', 'natural'),
                'quality_rejected': bool(quality[index]), 'text': result['text'], 'reason': result['reason'],
                'correct': result['text']==record['expected'], 'cells': predicted, 'gt_cells': gt_rows,
                'unmatched_gt': missed, 'unmatched_predictions': extras})
        del outputs
        rows_by_model[name], cells_by_model[name] = rows, all_cells
    assert np.array_equal(quality_masks['baseline'], quality_masks['candidate'])
    a, b = rows_by_model['baseline'], rows_by_model['candidate']
    lost = [i for i in range(405) if a[i]['correct'] and not b[i]['correct']]
    gained = [i for i in range(405) if b[i]['correct'] and not a[i]['correct']]
    shared = [(x, y) for x, y in zip(cells_by_model['baseline'], cells_by_model['candidate']) if x['matched'] and y['matched']]
    def paired_deltas(pairs):
        return {key: stats([y[key]-x[key] for x, y in pairs]) for key in
                ('iou','center_error_px','center_dx_px','center_dy_px','width_ratio','height_ratio','score','dot_confidence')}
    summary = {}
    for name in ('baseline','candidate'):
        rows, cells = rows_by_model[name], cells_by_model[name]
        summary[name] = {'checkpoint_sha256': candidates[name]['checkpoint_sha256'], 'thresholds': candidates[name]['thresholds'],
            'words': word_metrics(rows), 'cells': cell_metrics(cells),
            'by_length': {str(length): {'words': word_metrics([r for r in rows if r['length']==length]),
                                      'cells': cell_metrics([r for r in cells if r['length']==length])} for length in range(1,13)},
            'by_group': {group: {'words': word_metrics([r for r in rows if r['group']==group]),
                                 'cells': cell_metrics([r for r in cells if r['group']==group])} for group in ('natural','stress')},
            'by_pattern': {str(mask): {'letter': letters[mask].upper(), **cell_metrics([r for r in cells if r['gt_mask']==mask])} for mask in sorted(set(r['gt_mask'] for r in cells))},
            'mask_confusions': [{'truth': gt, 'prediction': pred, 'count': count} for (gt, pred), count in
                Counter((r['gt_letter'], r['predicted_letter']) for r in cells if r['matched'] and not r['mask_correct']).most_common()],
            'bit_errors': [sum(bool((r['gt_mask']^r['predicted_mask']) & (1<<bit)) for r in cells if r['matched']) for bit in range(6)],
            'horizontal_reflection_mask_errors': sum(r['predicted_mask'] == (((r['gt_mask']&7)<<3)|((r['gt_mask']>>3)&7)) for r in cells if r['matched'] and not r['mask_correct'])}
    transition = {'baseline_correct_candidate_wrong': len(lost), 'baseline_wrong_candidate_correct': len(gained),
        'both_correct': sum(x['correct'] and y['correct'] for x,y in zip(a,b)),
        'both_wrong': sum(not x['correct'] and not y['correct'] for x,y in zip(a,b)),
        'lost_indices': lost, 'gained_indices': gained,
        'lost_outcomes': dict(Counter(b[i]['reason'] if not b[i]['text'] else 'accepted_wrong' for i in lost)),
        'lost_words_with_unmatched_gt': sum(bool(b[i]['unmatched_gt']) for i in lost),
        'lost_words_with_extra_predictions': sum(bool(b[i]['unmatched_predictions']) for i in lost),
        'lost_words_with_wrong_matched_mask': sum(any(c['matched'] and not c['mask_correct'] for c in b[i]['gt_cells']) for i in lost),
        'lost_words_with_low_dot_confidence': sum(any(c['matched'] and not c['dot_confidence_pass'] for c in b[i]['gt_cells']) for i in lost),
        'lost_words_with_low_objectness': sum(any(c['matched'] and not c['objectness_pass'] for c in b[i]['gt_cells']) for i in lost),
        'both_model_matched_gt_cells': len(shared), 'paired_geometry_and_confidence_deltas_candidate_minus_baseline': paired_deltas(shared),
        'paired_lost_words_deltas': paired_deltas([(x,y) for x,y in shared if x['word_index'] in lost]),
        'gt_cell_match_transitions': dict(Counter(f'{int(x["matched"])}->{int(y["matched"])}' for x,y in zip(cells_by_model['baseline'], cells_by_model['candidate'])))}
    bindings = {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in [
        Path(__file__), fixture, table_path, CANDIDATE_CACHE/'cache-report.json', BASELINE/'selected-validation-fullprecision.json',
        *paths.values(), BASELINE/'source/word_data.py', ROOT/'training/scripts/benchmark_angelina.py', *sorted((SELECTION/'source').glob('*.py'))]}
    report = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Approximate post-NMS diagnostic on cached held405; not AP, not new inference, not external607, not physical proof.',
        'matching': {'rule': 'Greedy highest pixel-box IoU >=0.25, one-to-one; ties by GT then prediction index; no labels/scores in matching.',
            'post_nms': 'Exact archived decoder at each candidate original selected thresholds, before its rejection gates.',
            'quality': 'Identical cached input-quality rejections zeroed as in original selector, counted as failures and unmatched cells.',
            'geometry': '480x192 pixel coordinates; errors measured against complete expected-word boxes, not context neighbors.',
            'limits': 'IoU cutoff may conflate severe geometry errors with missing detections. Unmatched predictions may be fragments, duplicates, context, or false detections. Groups share 135 source crops with 2 stress variants; 405 rows are not 405 independent sources. Selected thresholds may differ; this does not isolate causal effects of the training intervention.'},
        'model_execution': False, 'external_outputs_used': False, 'source_bindings': bindings,
        'quality_rejected': int(quality_masks['baseline'].sum()), 'summary': summary, 'transitions': transition,
        'rows': rows_by_model, 'lightweight_matcher_checks': 3}
    OUT.mkdir(parents=True)
    report_path = OUT/'comparison.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    text = ['# Current candidate versus baseline: cached held405 diagnosis', '',
        'Approximate box matching; not AP or independent physical validation. No model execution, threshold search, or external607 data.', '',
        '| Model | Correct / 405 | Accepted | Rejected | Accepted precision |', '|---|---:|---:|---:|---:|']
    for name in ('baseline','candidate'):
        m=summary[name]['words']
        text.append(f'| {name} | {m["correct"]} | {m["accepted"]} | {m["rejected"]} | {100*m["accepted_accuracy"]:.2f}% |')
    text += ['', f'Lost correct blocks: {len(lost)}; gained: {len(gained)}. Lost outcomes: `{json.dumps(transition["lost_outcomes"],sort_keys=True)}`.', '',
        'Both models use their existing selected thresholds, recorded in JSON; no thresholds are searched or changed. Matching is greedily one-to-one at IoU >=0.25, without reading labels. Unmatched predictions are not automatically false positives.', '',
        '| Length | Blocks | Baseline correct | Candidate correct | Baseline unmatched GT | Candidate unmatched GT |', '|---|---:|---:|---:|---:|---:|']
    for length in range(1,13):
        x,y=(summary[n]['by_length'][str(length)] for n in ('baseline','candidate'))
        text.append(f'| {length} | {x["words"]["total"]} | {x["words"]["correct"]} | {y["words"]["correct"]} | {x["cells"]["unmatched"]} | {y["cells"]["unmatched"]} |')
    text += ['', '| Pattern | GT cells | Baseline matched / correct mask | Candidate matched / correct mask |', '|---|---:|---:|---:|']
    for mask in sorted(summary['baseline']['by_pattern'],key=lambda k:summary['baseline']['by_pattern'][k]['letter']):
        x,y=(summary[n]['by_pattern'][mask] for n in ('baseline','candidate'))
        text.append(f'| {x["letter"]} ({mask}) | {x["gt_cells"]} | {x["matched"]} / {x["matched_correct_mask"]} | {y["matched"]} / {y["matched_correct_mask"]} |')
    text += ['', f'Full row-level diagnosis: [comparison.json](comparison.json), SHA-256 `{sha(report_path)}`.', '',
        f'Fixture SHA-256 `{sha(fixture)}`. {int(quality_masks['baseline'].sum())} identical quality rejections remain failures. Lengths10–12 have no examples in this held set; natural135 + stress270 share source crops.', '',
        'Source/model/cache hashes, selected thresholds, matching rules, paired geometry statistics, all 405 decisions per model, and per-cell masks/boxes are recorded in JSON. Production/APK/physical gates are unchanged.', '']
    (OUT/'README.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'sha256': sha(report_path), 'words': {n: summary[n]['words'] for n in summary},
                      'cells': {n: summary[n]['cells'] for n in summary}, 'transitions': transition}, indent=2))


if __name__ == '__main__':
    main()
