"""Replay completed balanced-sampling histories and verify checkpoint lineage on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(run):
    """Check actual recorded consumption; do not infer coverage from future orders."""
    assert not (run/'ABORTED.json').exists(), 'An aborted run cannot establish eligibility'
    config_path = run/'training-config.json'
    config = json.loads(config_path.read_text())
    for name, expected in config['source_sha256'].items():
        assert sha256(run/'source'/name) == expected, ('archived source changed', name)
    data = run.parent/'data'
    for name, expected in config['data_sha256'].items():
        assert sha256(data/name) == expected, ('training data or metadata changed', name)
    manifest = data/config['train_manifest']
    manifest_hash = sha256(manifest)
    assert manifest_hash == config['data_sha256'][config['train_manifest']]
    assert sha256(data/'source-split.json') == config['data_sha256']['source-split.json']
    split = json.loads((data/'source-split.json').read_text())['images']
    original = json.loads(manifest.read_text())
    supplemental_sources = set()
    if config.get('supplemental_source_indexes'):
        source_spec = importlib.util.spec_from_file_location('audited_sources', run/'source/word_sources.py')
        source_module = importlib.util.module_from_spec(source_spec)
        source_spec.loader.exec_module(source_module)
        supplemental_audit = source_module.audit_supplemental_sources(original, data, config['base_train_manifest'], config['supplemental_source_indexes'])
        assert supplemental_audit == config['supplemental_source_audit']
        for index in config['supplemental_source_indexes']:
            index_rows = json.loads((data/index).read_text())['sources']
            supplemental_sources.update(row['source'] for row in index_rows)
    assert len({row['image'] for row in original}) == len(original)
    for row in original:
        if row['boxes']:
            assert row['source'] in set(split['train']) | supplemental_sources
            assert row['source'] not in split['validation'] and row['source'] not in split['external']
    negative = [row for row in original if not row['boxes']]
    if supplemental_sources:
        records, negative_expansion = source_module.expand_negative_records(original, config['negative_repeat'],
                                                                            supplemental_sources, config['supplemental_negative_repeat'])
        assert negative_expansion == config['negative_expansion']
    else:
        records = original + negative*(config['negative_repeat']-1)
    assert len(records) == config['train_patches']
    sampler_source = run/'source/word_sampling.py'
    spec = importlib.util.spec_from_file_location('audited_sampler', sampler_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.BalancedFamilyOrder(records, config['seed'])
    description = sampler.describe()
    assert config['sampling'] == dict(description, keras_batch_shuffle=False)
    quotas = [row['records_per_epoch'] for row in description['families'].values()]
    assert max(quotas)-min(quotas) <= 1
    positive_indices = {i for i, row in enumerate(records) if row['boxes']}
    negative_indices = set(range(len(records)))-positive_indices
    history_path = run/'sampling-history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    affine = config.get('batchnorm_affine', {})
    validation_path = run/'validation-history.json'
    validation_history = json.loads(validation_path.read_text()) if affine.get('enabled') else []
    if affine.get('enabled'):
        assert len(validation_history) == len(history)
    seen, observed = set(), []
    for number, row in enumerate(history, 1):
        assert row['epoch'] == number and row['dataset_epoch'] == number+1
        expected_batches = math.ceil(len(records)/config['batch_size'])
        assert row['expected_batches'] == expected_batches
        # Eligibility requires a complete epoch. Partial histories cannot be
        # replayed because the callback does not persist arbitrary batch IDs.
        assert row['completed_batches'] == expected_batches and row['complete_epoch'] is True
        if affine.get('enabled'):
            assert validation_history[number-1]['epoch'] == number
            assert validation_history[number-1]['batchnorm_moving_moments_sha256'] == affine['moving_moments_sha256']
        order = sampler.order(row['dataset_epoch']-1)
        counts = Counter(map(int, order))
        assert set(counts) <= set(range(len(records)))
        assert {i: counts[i] for i in negative_indices} == dict.fromkeys(negative_indices, 1)
        actual = sampler.audit(order, seen)
        for key, value in actual.items():
            assert row[key] == value, ('history mismatch', number, key)
        for family, values in actual['families'].items():
            assert values['records_this_epoch'] == description['families'][family]['records_per_epoch']
        checkpoint = run/f'retained-epochs/epoch-{number:02d}.keras'
        assert checkpoint.is_file()
        observed.append({'epoch': number, 'checkpoint_sha256': sha256(checkpoint),
                         'completed_batches': expected_batches,
                         'actual_cumulative_positive_records': len(seen & positive_indices),
                         'all_positive_records_seen_in_this_run': positive_indices <= seen})
    return {'run': str(run), 'config_sha256': sha256(config_path),
            'batchnorm_moment_validation_history_sha256': sha256(validation_path) if affine.get('enabled') else None,
            'manifest_sha256': manifest_hash,
            'sampling_history_sha256': sha256(history_path) if history_path.exists() else None,
            'positive_records': len(positive_indices), 'negative_draws_per_epoch': len(negative_indices),
            'observed_epochs': observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-eligible-epoch', type=int)
    args = parser.parse_args()
    actual = replay(args.run)
    config = json.loads((args.run/'training-config.json').read_text())
    protocol = json.loads(args.protocol.read_text())
    assert datetime.fromisoformat(protocol['created_at_utc']) < datetime.fromisoformat(config['started_at_utc'])
    assert protocol['training_manifest_sha256'] == actual['manifest_sha256']
    assert protocol['source_sha256'] == config['source_sha256']
    for key in ('batch_size', 'learning_rate', 'seed', 'negative_repeat',
                'geometry_loss', 'illumination_augmentation', 'horizontal_mirror_augmentation'):
        assert protocol[key] == config[key], ('protocol/config mismatch', key)
    assert protocol['planned_epochs'] == config['epochs']
    if 'vertical_stretch_augmentation' in config:
        assert protocol.get('vertical_stretch_augmentation', False) == config['vertical_stretch_augmentation']
    if 'horizontal_aspect_augmentation' in config:
        assert protocol.get('horizontal_aspect_augmentation', False) == config['horizontal_aspect_augmentation']
    assert protocol['partial_geometry_supervision'] == config['partial_geometry_supervision']['enabled']
    if config.get('supplemental_source_indexes'):
        assert protocol['supplemental_negative_repeat'] == config['supplemental_negative_repeat']
        assert protocol['supplemental_source_indexes'] == config['supplemental_source_indexes']
    if config.get('dsbi_visible_partial_supervision', {}).get('enabled'):
        assert protocol['dsbi_visible_partial_supervision'] is True
        assert protocol['point_metadata_sha256'] == config['data_sha256'][config['dsbi_point_metadata']]
        assert protocol['point_metadata_sha256'] == config['dsbi_visible_partial_supervision']['metadata']['sha256']
    if config.get('geometry_conditioned_residual', {}).get('enabled'):
        geometry = config['geometry_conditioned_residual']
        assert protocol['geometry_conditioned_residual'] is True
        assert protocol['geometry_initialization_report_sha256'] == geometry['initialization_report_sha256']
        assert protocol['geometry_conversion_report_sha256'] == geometry['conversion_report_sha256']
        assert protocol['parent_checkpoint_sha256'] == geometry['zero_checkpoint_sha256'] == config['resume_sha256']
        assert sha256(Path(geometry['initialization_report'])) == geometry['initialization_report_sha256']
        assert sha256(Path(geometry['conversion_report'])) == geometry['conversion_report_sha256']
    parent = None
    if config.get('fine_detail_branch', {}).get('enabled'):
        fine = config['fine_detail_branch']
        assert protocol['fine_detail_branch'] is True
        assert protocol['fine_initialization_report_sha256'] == fine['initialization_report_sha256']
        assert protocol['fine_conversion_report_sha256'] == fine['conversion_report_sha256']
        assert protocol['parent_checkpoint_sha256'] == fine['initial_checkpoint_sha256'] == config['resume_sha256']
        assert protocol['base_parent_checkpoint_sha256'] == fine['base_parent_checkpoint_sha256']
        assert sha256(Path(fine['initialization_report'])) == fine['initialization_report_sha256']
        assert sha256(Path(fine['conversion_report'])) == fine['conversion_report_sha256']
        assert 'parent_sampling_audit' not in protocol, 'Stride2 ablation eligibility requires its own completed full cycle'
    if config.get('batchnorm_affine', {}).get('enabled'):
        affine = config['batchnorm_affine']
        assert protocol['batchnorm_affine'] is True
        assert protocol['bn_affine_initialization_report_sha256'] == affine['initialization_report_sha256']
        assert protocol['bn_affine_conversion_report_sha256'] == affine['conversion_report_sha256']
        assert protocol['parent_checkpoint_sha256'] == affine['initial_checkpoint_sha256'] == config['resume_sha256']
        assert protocol['base_parent_checkpoint_sha256'] == affine['base_parent_checkpoint_sha256']
        assert sha256(Path(affine['initialization_report'])) == affine['initialization_report_sha256']
        assert sha256(Path(affine['conversion_report'])) == affine['conversion_report_sha256']
        assert 'parent_sampling_audit' not in protocol, 'Affine ablation eligibility requires its own completed full cycle'
    if 'parent_sampling_audit' in protocol:
        audit_path = Path(protocol['parent_sampling_audit'])
        assert sha256(audit_path) == protocol['parent_sampling_audit_sha256']
        inherited_audit = json.loads(audit_path.read_text())
        assert inherited_audit['status'] == 'passed'
        parent = replay(audit_path.parent)
        assert parent['config_sha256'] == inherited_audit['config_sha256']
        assert parent['sampling_history_sha256'] == inherited_audit['history_sha256']
        assert parent['manifest_sha256'] == actual['manifest_sha256']
        expected_hash = protocol['parent_checkpoint_sha256']
        assert sha256(Path(protocol['parent_checkpoint'])) == config['resume_sha256'] == expected_hash
        parent_epoch = next(row for row in parent['observed_epochs'] if row['epoch'] == protocol['parent_sampling_eligible_epoch'])
        assert parent_epoch['checkpoint_sha256'] == expected_hash
        assert parent_epoch['all_positive_records_seen_in_this_run']
        parent['inherited_epoch'] = parent_epoch
    for row in actual['observed_epochs']:
        row['eligible_by_sampling_lineage'] = bool(parent) or row['all_positive_records_seen_in_this_run']
    eligible = [row['epoch'] for row in actual['observed_epochs'] if row['eligible_by_sampling_lineage']]
    passed = args.require_eligible_epoch is None or args.require_eligible_epoch in eligible
    report = {'status': 'passed' if passed else 'required_epoch_not_eligible',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'audit_script_sha256': sha256(__file__), 'protocol_sha256': sha256(args.protocol),
              'actual': actual, 'verified_parent': parent, 'eligible_retained_epochs': eligible,
              'required_epoch': args.require_eligible_epoch,
              'limitations': ['This verifies complete record consumption and source separation, not that every cell survived stochastic augmentation.',
                              'Inherited coverage is tied to a re-audited retained parent checkpoint. New partial-cell supervision exposure is reported separately.',
                              'No images, model outputs or external metrics participate in this audit.']}
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'eligible_epochs': eligible,
                      'observed_epochs': actual['observed_epochs']}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
