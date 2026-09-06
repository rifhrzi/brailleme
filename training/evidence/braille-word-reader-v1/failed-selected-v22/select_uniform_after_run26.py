"""Evaluate one predeclared weight average after run26, before downstream export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


WORK = Path(__file__).resolve().parent
PREPARATION = WORK / 'run20-uniform-average-preparation'
PLAN_SHA = '6520a4e0d41e7bbbb8e5e7ffd9aaed5b6a80d4d54ccb38b96f29341db76c1f98'
CANDIDATE_REPORT_SHA = '0ce6817ef003e8fe86c03e1e183ebf8f5c3c602bb18eeaeb5d825559ed0ced1b'
CANDIDATE_SHA = '7c831094852512ef79d791fa31570e6016d15e513004defdeadb091c401b9cda'
INDEPENDENT_AUDIT_SHA = 'f71e485c232d314df1d363de994ac77768be51d42211f17f8c8872e7f85e8cd3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified(reference):
    path = Path(reference['path'])
    if sha(path) != reference['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    return path


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-audit', required=True, type=Path)
    args = parser.parse_args()
    result_path = WORK / 'run26/training-process-result.json'
    if not result_path.exists():
        raise SystemExit('Run26 is still active; monitor session 98797 without restarting.')
    if json.loads(result_path.read_text())['exit_code'] != 0:
        raise SystemExit('Run26 did not complete successfully.')
    selected_path = WORK / 'selected-v21/selected-validation.json'
    table_path = WORK / 'selected-v21/checkpoint-selection.json'
    if not selected_path.exists() or not table_path.exists():
        raise SystemExit('Finish the frozen run26 selection into selected-v21 first.')
    protocol_path = PREPARATION / 'evaluation-protocol.json'
    output = WORK / 'selected-v22'
    if protocol_path.exists() or output.exists():
        raise SystemExit('Evaluation has already been frozen or started; never overwrite or rerun it.')

    plan_path = PREPARATION / 'plan.json'
    report_path = PREPARATION / 'candidate-report.json'
    if sha(plan_path) != PLAN_SHA or sha(report_path) != CANDIDATE_REPORT_SHA:
        raise ValueError('Frozen uniform preparation changed.')
    plan = json.loads(plan_path.read_text())
    report = json.loads(report_path.read_text())
    read_verified(report['plan'])
    read_verified(report['script'])
    candidate = read_verified(report['candidate'])
    assert sha(candidate) == CANDIDATE_SHA
    assert report['training_resume_allowed'] is False
    assert report['optimizer_datasets_after'] == 0 and report['compile_config_empty'] is True
    assert report['original_checkpoints_unchanged'] is True
    assert plan['input_epoch_order'] == [8, 9, 10]
    assert plan['model_array_count'] == 280 and plan['bn_array_count_preserved'] == 208
    assert plan['arrays_to_average'] == 72
    for item in plan['inputs']:
        read_verified(item)
        assert item['weight'] == {'numerator': 1, 'denominator': 3}
        assert item['sampling']['eligible_by_sampling_lineage'] is True
    for key in ('source_training_config', 'source_sampling_lineage', 'source_sampling_history', 'inventory'):
        read_verified(plan[key])
    lineage = json.loads(Path(plan['source_sampling_lineage']['path']).read_text())
    assert lineage['status'] == 'passed'
    assert lineage['eligible_retained_epochs'] == [8, 9, 10]
    for item in plan['inputs']:
        observed = lineage['actual']['observed_epochs'][item['epoch'] - 1]
        assert observed['checkpoint_sha256'] == item['sha256']

    if sha(args.independent_audit) != INDEPENDENT_AUDIT_SHA:
        raise ValueError('The frozen independent whole-array audit changed.')
    audit = json.loads(args.independent_audit.read_text())
    if audit.get('status') != 'passed':
        raise ValueError('Independent whole-array preparation audit did not pass.')
    assert audit['plan_sha256'] == PLAN_SHA and audit['candidate_report_sha256'] == CANDIDATE_REPORT_SHA
    assert audit['candidate_sha256'] == CANDIDATE_SHA
    assert audit['all_three_parents_sampling_eligible'] and audit['all_original_hashes_unchanged']
    assert audit['graph_equal_to_all_three'] and audit['model_values_checked'] == 829307
    assert audit['optimizer_dataset_count'] == 0 and audit['compile_config_empty'] is True
    for path, expected in audit['source_bindings'].items():
        if sha(path) != expected:
            raise ValueError(f'Independent audit dependency changed: {path}')
    assert sha(WORK / 'audit_uniform_average.py') == audit['script_sha256']
    audit_reference = {'path': str(args.independent_audit.resolve()), 'sha256': sha(args.independent_audit)}
    selection = plan['selection']
    fixture = read_verified(selection['fixture'])
    assert selection['expected_records'] == 405
    assert selection['threshold_grid'] == 'weak-peaks-v2'
    assert selection['external_selection'] is False
    for name, reference in selection['sources'].items():
        read_verified(reference)
        if sha(Path('training/scripts') / name) != reference['sha256']:
            raise ValueError(f'Selection source changed: {name}')
    control = Path(plan['control_sequence']['control_path'])
    assert control == WORK / 'selected-v21/best-word.keras'
    assert Path(plan['control_sequence']['output']) == output
    selected = json.loads(selected_path.read_text())
    table = json.loads(table_path.read_text())
    assert sha(control) == selected['checkpoint_sha256'] == table['selected_checkpoint_sha256']
    assert table['fixture_sha256'] == selection['fixture']['sha256']
    assert table['threshold_grid'] == 'weak-peaks-v2' and table['external_outputs_used'] is False
    assert table['threshold_grid_values'] == selection['threshold_grid_values']
    assert table['validation_words'] == 405
    # Enforce the predeclared sequencing: only the final selected-v22 winner
    # can proceed to fresh export/external tests.
    for stage in ('selected-v21', 'selected-v22'):
        for name in ('word_reader.tflite', 'export-report.json', 'public-benchmark.json'):
            if (WORK / stage / name).exists():
                raise ValueError(f'Downstream artifact exists before this selection: {stage}/{name}')
    command = [sys.executable, '-u', 'training/scripts/word_select.py',
               '--candidates', str(control), str(candidate), '--data', str(fixture.parent),
               '--output', str(output), '--batch-size', '24',
               '--validation-manifest', fixture.name, '--grouped-selection',
               '--threshold-grid', 'weak-peaks-v2']
    protocol = {
        'schema_version': 1, 'status': 'frozen_before_evaluation',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan': {'path': str(plan_path), 'sha256': PLAN_SHA},
        'candidate_report': {'path': str(report_path), 'sha256': CANDIDATE_REPORT_SHA},
        'independent_preparation_audit': audit_reference,
        'control_selection': {'path': str(table_path), 'sha256': sha(table_path)},
        'control_validation': {'path': str(selected_path), 'sha256': sha(selected_path)},
        'ordered_candidates': [{'path': str(control), 'sha256': sha(control)}, report['candidate']],
        'fixture': selection['fixture'], 'threshold_grid': selection['threshold_grid'],
        'threshold_grid_values': selection['threshold_grid_values'],
        'tie_rule': 'Control first; replace only on strictly higher existing grouped rank.',
        'derivation': 'Exactly one uniform average of already eligible run20 epochs 8/9/10; no new training epoch or sampling row.',
        'source_sampling_lineage': plan['source_sampling_lineage'],
        'sources': selection['sources'], 'orchestrator': {'path': str(Path(__file__)), 'sha256': sha(__file__)},
        'external_outputs_used': False, 'command': command,
    }
    save_new(protocol_path, protocol)
    env = dict(os.environ, KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    with (PREPARATION / 'selection.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    save_new(PREPARATION / 'selection-process-result.json', {
        'exit_code': code, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_protocol_sha256': sha(protocol_path),
    })
    raise SystemExit(code)


if __name__ == '__main__':
    main()
