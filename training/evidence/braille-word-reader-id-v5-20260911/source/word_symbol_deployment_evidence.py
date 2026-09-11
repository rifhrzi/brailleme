"""Record an external promotion veto while preserving the original training selection."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_words(baseline, candidate, expected_count=607):
    for field in ('decoder_version', 'thresholds', 'fixture_manifest_sha256'):
        if baseline[field] != candidate[field]:
            raise ValueError(f'External comparisons differ in {field}')
    if len(baseline['words']) != expected_count or len(candidate['words']) != expected_count:
        raise ValueError('External comparison is incomplete')
    rows, identities = [], set()
    counts = dict(total=expected_count, baseline_correct=0, candidate_correct=0,
                  lost=0, gained=0, both_correct=0, neither_correct=0)
    for index, (before, after) in enumerate(zip(baseline['words'], candidate['words'], strict=True)):
        identity_before = {key: value for key, value in before.items() if key not in ('predicted', 'rejection', 'detected_cells')}
        identity_after = {key: value for key, value in after.items() if key not in ('predicted', 'rejection', 'detected_cells')}
        if identity_before != identity_after or not before['expected']:
            raise ValueError(f'External ground-truth identity changed at row {index}')
        identity = (before['source'], before['image'])
        if identity in identities:
            raise ValueError('Duplicate external row identity')
        identities.add(identity)
        good_before = before['predicted'] == before['expected']
        good_after = after['predicted'] == after['expected']
        transition = 'both_correct' if good_before and good_after else 'lost' if good_before else 'gained' if good_after else 'neither_correct'
        counts['baseline_correct'] += good_before
        counts['candidate_correct'] += good_after
        counts[transition] += 1
        rows.append(dict(index=index, image=before['image'], source=before['source'], expected=before['expected'],
                         selection=before.get('selection'), transition=transition,
                         baseline=dict(text=before['predicted'], reason=before['rejection'], correct=good_before),
                         candidate=dict(text=after['predicted'], reason=after['rejection'], correct=good_after)))
    for report, name in ((baseline, 'baseline'), (candidate, 'candidate')):
        if report['whole_words']['total'] != expected_count or report['whole_words']['correct'] != counts[name + '_correct']:
            raise ValueError('External aggregate differs from independently counted rows')
    if counts['candidate_correct'] >= counts['baseline_correct']:
        raise ValueError('No external exact-accuracy decrease; this deployment veto does not apply')
    return counts, rows


def symbol_results(fixtures, reference):
    if len(fixtures) != 516 or len(reference['rows']) != len(fixtures):
        raise ValueError('Baseline symbol replay is incomplete')
    rows = []
    for index, (fixture, replay) in enumerate(zip(fixtures, reference['rows'], strict=True)):
        if replay['index'] != index or fixture['image'] != replay['image'] or fixture['expected'] != replay['expected']:
            raise ValueError('Baseline symbol replay identity changed')
        rows.append(dict(image=replay['image'], source_kind=fixture['origin'], group=fixture['symbol_group'],
                         expected=replay['expected'], predicted=replay['python_text'], rejection=replay['python_reason']))
    groups = {}
    for group in ('lowercase', 'capital_single', 'capital_word', 'numbers'):
        values = [row for row in rows if row['group'] == group]
        accepted = sum(bool(row['predicted']) for row in values)
        correct = sum(bool(row['expected']) and row['predicted'] == row['expected'] for row in values)
        groups[group] = dict(total=len(values), accepted=accepted, correct=correct,
                             exact_word_accuracy=correct / len(values), accepted_accuracy=correct / accepted if accepted else None)
    mandatory = [row for row in rows if row['group'] == 'rejection']
    false_accepts = sum(bool(row['predicted']) for row in mandatory)
    groups['rejection'] = dict(total=len(mandatory), false_acceptances=false_accepts, all_rejected=false_accepts == 0)
    gates = false_accepts == 0 and all(value['exact_word_accuracy'] >= .75 and (value['accepted_accuracy'] or 0) >= .95
                                      for name, value in groups.items() if name != 'rejection')
    return rows, groups, mandatory, gates


def generate(experiment, run):
    experiment, run = Path(experiment).resolve(), Path(run).resolve()
    candidate_dir = experiment / 'selection'
    destinations = [run / name for name in ('external-baseline-comparison.json', 'deployment-selection.json', 'symbol-readiness.json')]
    if any(path.exists() for path in destinations):
        raise ValueError('Use new deployment evidence paths; existing reports are immutable')
    if run == candidate_dir or not run.is_dir():
        raise ValueError('Use the separate existing baseline export directory')
    baseline_public_path, candidate_public_path = run / 'public-benchmark.json', candidate_dir / 'public-benchmark.json'
    training_selection_path = candidate_dir / 'checkpoint-selection.json'
    original_readiness_path = candidate_dir / 'symbol-readiness.json'
    reference_path = run / 'replay-symbols-516/python-reference.json'
    fixture_path = experiment / 'validation-symbols.json'
    baseline_public, candidate_public = read(baseline_public_path), read(candidate_public_path)
    selection, original, reference = read(training_selection_path), read(original_readiness_path), read(reference_path)
    baseline_checkpoint = {key.replace('\\', '/'): value for key, value in read(experiment / 'baseline-freeze.json')['files'].items()}['training/checkpoints/braille-word-reader-v1/best-word.keras']
    candidate_checkpoint = selection['selected_sha256']
    if digest(run / 'best-word.keras') != baseline_checkpoint or digest(experiment / 'baseline.keras') != baseline_checkpoint:
        raise ValueError('Original baseline checkpoint is not retained')
    if digest(candidate_dir / 'best-word.keras') != candidate_checkpoint or original['training']['selected_checkpoint_sha256'] != candidate_checkpoint:
        raise ValueError('Original training-selected checkpoint changed')
    if selection['baseline_retained'] or original['training']['baseline_retained'] or original['training']['epochs_completed'] != 10:
        raise ValueError('Expected the preserved ten-epoch candidate selection history')
    if original['baseline_groups'] != selection['baseline'] or original['candidate_comparisons'] != selection['candidates']:
        raise ValueError('Original training evidence envelope changed')
    baseline_model, candidate_model = digest(run / 'word_reader.tflite'), digest(candidate_dir / 'word_reader.tflite')
    if baseline_public['model_sha256'] != baseline_model or reference['model_sha256'] != baseline_model:
        raise ValueError('Baseline reports do not bind the actual deployment artifact')
    if candidate_public['model_sha256'] != candidate_model or original['model_sha256'] != candidate_model:
        raise ValueError('Candidate reports do not bind the retained candidate artifact')
    if baseline_public['decoder_version'] != 'six-dot-id-v3' or reference['decoderVersion'] != baseline_public['decoder_version']:
        raise ValueError('Deployment decoder contract differs')
    if reference['thresholds'] != baseline_public['thresholds'] or selection['configuration']['thresholds'] != baseline_public['thresholds']:
        raise ValueError('Deployment thresholds changed after training')
    if reference['fixture_manifest_sha256'] != digest(fixture_path) or original['provenance']['fixture_sha256'] != digest(fixture_path):
        raise ValueError('Symbol fixture labels changed')
    source_root = Path(__file__).resolve().parents[2]
    for name, expected in reference['source_sha256'].items():
        if digest(source_root / name) != expected:
            raise ValueError(f'Stale baseline replay source: {name}')
    if digest(reference_path.parent / 'inputs.rgb') != reference['raw_inputs_sha256'] or digest(reference_path.parent / 'outputs.f32le') != reference['raw_outputs_sha256']:
        raise ValueError('Baseline replay pixel/tensor bytes changed')
    counts, comparison_rows = compare_words(baseline_public, candidate_public)
    note = 'External results veto promotion only; no gradients, additional training, threshold tuning or epoch reranking. Original epoch10 training selection remains unchanged.'
    common = dict(decoder_version=baseline_public['decoder_version'], thresholds=baseline_public['thresholds'],
                  fixture_manifest_sha256=baseline_public['fixture_manifest_sha256'],
                  training_selection_sha256=digest(training_selection_path),
                  baseline=dict(checkpoint_sha256=baseline_checkpoint, model_sha256=baseline_model, benchmark_sha256=digest(baseline_public_path)),
                  candidate=dict(checkpoint_sha256=candidate_checkpoint, model_sha256=candidate_model, benchmark_sha256=digest(candidate_public_path)))
    comparison = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(), **common,
                      counts=counts, rows=comparison_rows, generator_sha256=digest(__file__), note=note,
                      independent_holdout=False, physical_camera_validated=False)
    comparison_bytes = encode(comparison)
    deployment = dict(schema_version=1, **common, selected_checkpoint_sha256=baseline_checkpoint,
                      selected_model_sha256=baseline_model, status='baseline_retained_after_external_regression',
                      reason_codes=['external_exact_accuracy_regressed'], comparison_sha256=hashlib.sha256(comparison_bytes).hexdigest(), note=note)
    deployment_bytes = encode(deployment)
    symbol_rows, groups, mandatory, gates = symbol_results(read(fixture_path)['records'], reference)
    # Initial baseline metrics remain historical; current baseline execution must agree.
    for group, values in groups.items():
        expected = selection['baseline'][group]
        if any((value or 0) != (expected.get(key) or 0) for key, value in values.items()):
            raise ValueError(f'Fresh baseline symbol group differs from original baseline: {group}')
    readiness = copy.deepcopy(original)
    readiness.update(model_sha256=baseline_model, decoder_version=reference['decoderVersion'], symbol_protocol=reference['symbolProtocol'],
                     thresholds=reference['thresholds'], rows=symbol_rows, groups=groups, mandatory_rejections=mandatory,
                     numeric_symbol_gates_pass=gates, deployment_selection_sha256=hashlib.sha256(deployment_bytes).hexdigest())
    readiness['provenance'].update(python_reference_sha256=digest(reference_path),
                                   deployment_selection_sha256=hashlib.sha256(deployment_bytes).hexdigest(),
                                   deployment_evidence_script_sha256=digest(__file__))
    readiness['limitations'].append(note)
    for path, data in zip(destinations, (comparison_bytes, deployment_bytes, encode(readiness)), strict=True):
        with path.open('xb') as stream:
            stream.write(data)
    return {'counts': counts, 'groups': groups, 'numeric_symbol_gates_pass': gates,
            'selected_model_sha256': baseline_model, 'reports': [str(path) for path in destinations]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.experiment, args.run), indent=2), flush=True)


if __name__ == '__main__':
    main()
