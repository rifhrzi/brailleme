"""One frozen candidate versus the protected reader; regression can veto only.

Uses already observed public data, never fresh symbol TEST or epoch selection.
No model copies, deployed assets or Android staging directories are written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_angelina import read_split
from word_decode_id_v4 import decode
from word_symbols_id_v4 import DECODER_VERSION


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_hash(path, expected):
    if sha(path) != expected:
        raise ValueError('Frozen binding changed: '+str(path))


def test_exposure_state(experiment):
    """Report previous experiment exposure separately from this regression."""
    experiment = Path(experiment)
    return dict(opened_by_this_evaluation=False,
                reserved_before_this_evaluation=(experiment/'test-evaluation-reservation.json').is_file(),
                completed_before_this_evaluation=(experiment/'final-test-evaluation.json').is_file())


def lowercase_rows(rows, root):
    result = []
    identities = set()
    for row in rows:
        text = row['expected']
        if not text or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for char in text):
            raise ValueError('Only literal plain Latin fixture labels may be migrated to lowercase')
        image = str((root/row['image']).resolve())
        identity = (row['source'], image)
        if identity in identities:
            raise ValueError('Duplicate fixture identity')
        identities.add(identity)
        result.append(dict(row, expected=text.lower(), image=image))
    return result


def load_bound_oracle(path, fixture, checkpoint, count, model=None):
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in [('checkpoint_sha256', sha(checkpoint)), ('fixture_sha256', sha(fixture))]:
            if key not in archive or str(archive[key]) != expected:
                raise ValueError('Cached baseline binding differs: '+key)
        if model is not None:
            if str(archive['oracle_kind']) != 'tflite' or str(archive['model_sha256']) != sha(model):
                raise ValueError('Cached baseline does not bind the actual TFLite model')
        outputs = archive['outputs'].copy()
    if outputs.shape != (count,48,120,11) or not np.isfinite(outputs).all():
        raise ValueError('Cached baseline is incomplete or non-finite')
    if outputs.min() < 0 or outputs.max() > 1:
        raise ValueError('Cached baseline has invalid probabilities')
    return outputs


def quality_reason(pixels):
    luminance = pixels @ np.asarray([.2126,.7152,.0722], np.float32)
    mean, std = float(luminance.mean()), float(luminance.std())
    return 'dark' if mean < .06 else 'glare' if mean > .985 else 'low_confidence' if std < .01 else ''


def compare_decisions(rows, baseline, candidate):
    if not rows or len(rows) != len(baseline) or len(rows) != len(candidate):
        raise ValueError('Every fixture requires both decisions')
    summaries = {}
    positives = [i for i, row in enumerate(rows) if row['expected']]
    negatives = [i for i, row in enumerate(rows) if not row['expected']]
    for name, decisions in [('baseline', baseline), ('candidate', candidate)]:
        correct = sum(decisions[i]['text'] == rows[i]['expected'] for i in positives)
        accepted = sum(bool(decisions[i]['text']) for i in positives)
        summaries[name] = dict(positive_total=len(positives), correct=correct, accepted=accepted,
            exact_accuracy=correct/len(positives) if positives else None,
            accepted_precision=correct/accepted if accepted else None,
            negative_total=len(negatives), false_accepts=sum(bool(decisions[i]['text']) for i in negatives))
    lost = sum(baseline[i]['text'] == rows[i]['expected'] and candidate[i]['text'] != rows[i]['expected'] for i in positives)
    gained = sum(baseline[i]['text'] != rows[i]['expected'] and candidate[i]['text'] == rows[i]['expected'] for i in positives)
    before, after = summaries['baseline'], summaries['candidate']
    reasons = []
    if after['correct'] < before['correct']:
        reasons.append('exact_accuracy_regressed')
    if before['accepted'] and after['accepted'] and after['correct']*before['accepted'] < before['correct']*after['accepted']:
        reasons.append('accepted_precision_regressed')
    if after['false_accepts'] > before['false_accepts']:
        reasons.append('negative_false_accepts_regressed')
    if after['false_accepts']:
        reasons.append('mandatory_negative_rejection_failed')
    return dict(total=len(rows), **summaries, lost=lost, gained=gained,
                regression_veto=bool(reasons), reason_codes=reasons)


def evaluate_rows(model, rows, thresholds, reference=None, baseline_model=None):
    import torch
    if reference is not None and len(reference) != len(rows):
        raise ValueError('Cached reference count differs from fixture')
    baseline, candidate, records = [], [], []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        pixels, hashes = [], []
        for row in batch:
            hashes.append(sha(row['image']))
            with Image.open(row['image']) as image:
                pixels.append(np.asarray(image.convert('RGB').resize((480,192), Image.Resampling.BILINEAR), np.float32)/255.)
        array = np.stack(pixels)
        with torch.no_grad():
            tensor = torch.from_numpy(array).cuda()
            predicted = model(tensor).cpu().numpy()
            previous = reference[start:start+len(batch)] if reference is not None else baseline_model(tensor).cpu().numpy()
        if (predicted.shape != (len(batch),48,120,11) or predicted.shape != previous.shape
                or not np.isfinite(predicted).all() or predicted.min() < 0 or predicted.max() > 1):
            raise ValueError('Candidate output is malformed')
        for offset, row in enumerate(batch):
            reason = quality_reason(array[offset])
            pair = [dict(text='', reason=reason, cells=[]) if reason else decode(output, thresholds, row.get('selection'))
                    for output in (previous[offset], predicted[offset])]
            baseline.append(pair[0])
            candidate.append(pair[1])
            records.append(dict(index=start+offset, source=row['source'], image=row['image'],
                image_sha256=hashes[offset], expected=row['expected'], selection=row.get('selection'),
                baseline=dict(text=pair[0]['text'], reason=pair[0]['reason'], cells=len(pair[0]['cells'])),
                candidate=dict(text=pair[1]['text'], reason=pair[1]['reason'], cells=len(pair[1]['cells']))))
        if start % 104 == 0:
            print(f'REGRESSION {start+len(batch)}/{len(rows)}', flush=True)
    summary = compare_decisions(rows, baseline, candidate)
    groups = {}
    for group in sorted({row.get('validation_group', 'all') for row in rows}):
        chosen = [i for i,row in enumerate(rows) if row.get('validation_group', 'all') == group]
        groups[group] = compare_decisions([rows[i] for i in chosen], [baseline[i] for i in chosen], [candidate[i] for i in chosen])
    summary['groups'] = groups
    summary['regression_veto'] |= any(group['regression_veto'] for group in groups.values())
    return dict(summary=summary, rows=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    repo, run, experiment = Path.cwd().resolve(), args.run.resolve(), args.experiment.resolve()
    if run.parent != experiment:
        raise ValueError('Selected run must belong directly to experiment')
    selection, config = read(run/'selection.json'), read(run/'training-config.json')
    checkpoint = run/'best-word.keras'
    if selection['selection_scope'] != 'validation_only' or selection['test_opened']:
        raise ValueError('Use the immutable validation-selected checkpoint')
    assert_hash(checkpoint, selection['sha256'])
    assert_hash(run/'training-config.json', selection['training_config_sha256'])
    scripts = Path(__file__).resolve().parent
    names = ('word_torch_port.py', 'word_decode_id_v4.py', 'word_symbols_id_v4.py', 'word_symbols.py', 'word_data.py')
    for name in names:
        assert_hash(scripts/name, config['source_sha256'][name])
    baseline_dir = Path('C:/bm-symbols-20260908/deployment-baseline')
    baseline_checkpoint = repo/'training/checkpoints/braille-word-reader-v1/best-word.keras'
    assert_hash(baseline_dir/'best-word.keras', sha(baseline_checkpoint))
    external_fixture = baseline_dir/'external-words.json'
    legacy_fixture = Path('C:/bm-symbols-20260908/legacy-selection-fixtures.json')
    data = repo/'.braille-training-work/word-scan-v1-20260905/data'
    external = lowercase_rows(read(external_fixture)['records'], data)
    legacy = lowercase_rows(read(legacy_fixture)['records'], data)
    if len(external) != 607 or len(legacy) != 405:
        raise ValueError('Regression corpus count changed')
    public_cache, legacy_cache = baseline_dir/'public-word-outputs.npz', baseline_dir/'backend-fullprecision.npz'
    external_outputs = load_bound_oracle(public_cache, external_fixture, baseline_checkpoint, 607, baseline_dir/'word_reader.tflite')
    legacy_outputs = load_bound_oracle(legacy_cache, legacy_fixture, baseline_checkpoint, 405)
    dataset = Path('C:/bm-datasets/AngelinaDataset')
    negative_files = read_split(dataset, 'not_braille/train.txt')
    if len(negative_files) != 44:
        raise ValueError('Public negative corpus count changed')
    negatives = [dict(image=str(path.resolve()), source=path.relative_to(dataset).as_posix(), expected='',
                      selection=[0.,.1875,1.,.8125]) for path in negative_files]
    frozen = {str(path): sha(path) for path in [checkpoint, run/'selection.json', run/'training-config.json',
        baseline_checkpoint, baseline_dir/'best-word.keras', baseline_dir/'word_reader.tflite',
        external_fixture, legacy_fixture, public_cache, legacy_cache, dataset/'not_braille/train.txt']}
    report_path = run/'public-regression-v4.json'
    if report_path.exists():
        raise ValueError('Regression evidence already exists')
    test_reservation = experiment/'test-evaluation-reservation.json'
    if test_reservation.exists() and read(test_reservation)['checkpoint_sha256'] != selection['sha256']:
        raise ValueError('Regression candidate differs from the frozen final TEST candidate')
    exposure = test_exposure_state(experiment)
    with (experiment/'regression-evaluation-reservation.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
            selected_sha256=selection['sha256'], selection_sha256=sha(run/'selection.json'),
            policy='External results veto deployment only; never train, tune thresholds or rerank epochs afterward'), stream, indent=2)
    import torch
    from word_torch_port import load_keras_checkpoint
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    candidate_model = load_keras_checkpoint(checkpoint).cuda().eval()
    results = dict(external607=evaluate_rows(candidate_model, external, config['thresholds'], external_outputs),
                   legacy405=evaluate_rows(candidate_model, legacy, config['thresholds'], legacy_outputs))
    baseline_model = load_keras_checkpoint(baseline_checkpoint).cuda().eval()
    results['negative44'] = evaluate_rows(candidate_model, negatives, config['thresholds'], baseline_model=baseline_model)
    for path, digest in frozen.items():
        assert_hash(path, digest)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate_sha256=selection['sha256'],
        baseline_sha256=sha(baseline_checkpoint), thresholds=config['thresholds'], decoder_version=DECODER_VERSION,
        input_sha256=frozen, source_sha256={name: sha(scripts/name) for name in (*names, Path(__file__).name)},
        regression_veto=any(result['summary']['regression_veto'] for result in results.values()), results=results,
        baseline_engines=dict(external607='cached_tflite', legacy405='cached_torch', negative44='native_torch'),
        candidate_engine='native_torch', torch=torch.__version__,
        policy='Veto only; no alternate checkpoint selected or trained; original selection remains unchanged',
        scope='previously_observed_public_regression_not_independent_holdout', fresh_symbol_test_opened=False,
        fresh_symbol_test=exposure,
        fresh_symbol_test_opened_field_scope='Compatibility field describes this regression only; prior experiment exposure is reported in fresh_symbol_test',
        physical_camera_validated=False, deployment_authorized=False,
        limitations=['Legacy caches bind model and fixture bytes; their historical records do not include per-image hashes.',
                    'Image hashes here record current regression inputs; PIL resize is not Android native resize proof.',
                    'External607 baseline uses cached TFLite while candidate uses Torch; final candidate export parity is separate.'])
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('REGRESSION_COMPLETE '+json.dumps(dict(regression_veto=report['regression_veto'],
          summaries={name: value['summary'] for name,value in results.items()})), flush=True)


if __name__ == '__main__':
    main()
