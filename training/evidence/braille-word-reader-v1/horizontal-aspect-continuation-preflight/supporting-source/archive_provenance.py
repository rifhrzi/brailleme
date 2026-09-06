"""Archive the actual training/selection sources, never substitute current scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--selected', type=Path, required=True)
parser.add_argument('--training-runs', type=Path, nargs='+', required=True)
parser.add_argument('--wrapper-folders', type=Path, nargs='*', default=[])
parser.add_argument('--graph-branches', type=Path, nargs='*', default=[])
parser.add_argument('--supplemental-preparations', type=Path, nargs='*', default=[])
parser.add_argument('--data-root', type=Path, help='Original data root for hash-bound point metadata referenced by training configs.')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path.cwd()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
selected = json.loads((args.selected / 'selected-validation-fullprecision.json').read_bytes())
assert sha(args.selected / 'best-word.keras') == selected['oracle']['checkpoint_sha256']


def archive(source, relative):
    target = args.output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError(f'Refusing to overwrite retained evidence: {target}')
    shutil.copyfile(source, target)
    return {'file': target.relative_to(root).as_posix(), 'sha256': sha(target)}


def archive_geometry_initialization(config, run_folder):
    """Retain the zero-initialized branch and its original conversion/coordinate proofs."""
    geometry = config.get('geometry_conditioned_residual', {})
    if not geometry.get('enabled'):
        return None
    init_path = Path(geometry['initialization_report'])
    conversion_path = Path(geometry['conversion_report'])
    assert sha(init_path) == geometry['initialization_report_sha256']
    assert sha(conversion_path) == geometry['conversion_report_sha256']
    init = json.loads(init_path.read_bytes())
    conversion = json.loads(conversion_path.read_bytes())
    branch_folder = init_path.parent
    work_folder = branch_folder.parent
    assert config['resume_sha256'] == geometry['zero_checkpoint_sha256'] == init['zero_checkpoint_sha256']
    assert geometry['base_parent_checkpoint_sha256'] == init['parent_checkpoint_sha256']
    assert init['nonzero_diagnostic_must_never_be_resumed'] is True
    assert config['resume_sha256'] != init['nonzero_diagnostic_sha256']
    assert init['status'] == conversion['status'] == 'passed'
    assert conversion['initialization_report_sha256'] == sha(init_path)
    assert conversion['oracle_sha256'] == init['oracle_sha256']
    assert conversion['probes']['nonzero']['checkpoint_sha256'] == init['nonzero_diagnostic_sha256']
    assert conversion['probes']['nonzero']['builtins_only'] is True
    protocol_path = run_folder / 'geometry-residual-protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['zero_parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['initialization_report_sha256'] == sha(init_path)
    assert protocol['conversion_report_sha256'] == sha(conversion_path)
    for audit in protocol['preflight_audits']:
        assert sha(Path(audit['path'])) == audit['sha256']
    slot_path = Path(init['slot_prior_audit'])
    assert sha(slot_path) == geometry['slot_prior_audit_sha256'] == init['slot_prior_audit_sha256']
    report_paths = {
        'initializationReport': init_path,
        'nonzeroConversionProof': conversion_path,
        'slotPriorAudit': slot_path,
        'featureLatticeAudit': work_folder / 'feature-lattice-audit.json',
        'backboneLatticeIdentity': work_folder / 'v15-backbone-lattice-identity.json',
        'independentCoordinateAudit': work_folder / 'independent-coordinate-preflight.json',
        'independentHeadAudit': work_folder / 'independent-geometry-head-audit.json',
        'gpuBatchPreflight': branch_folder / 'gpu-batch-preflight.json',
        'protocol': protocol_path,
    }
    relative = Path(run_folder.name) / 'geometry-initialization'
    record = {key: archive(path, relative / path.name) for key, path in report_paths.items()}
    record.update({
        'baseParentCheckpointSha256': init['parent_checkpoint_sha256'],
        'zeroParentCheckpointSha256': init['zero_checkpoint_sha256'],
        'forbiddenDiagnosticCheckpointSha256': init['nonzero_diagnostic_sha256'],
        'sources': [],
    })
    sources = [(branch_folder / 'source' / name, digest) for name, digest in init['source_sha256'].items()]
    sources.append((branch_folder / 'source' / 'word_geometry_parity.py', conversion['source_sha256']))
    audit_scripts = {
        'slotPriorAudit': 'audit_slot_priors.py',
        'featureLatticeAudit': 'audit_feature_lattice.py',
        'independentCoordinateAudit': 'independent-coordinate-preflight.py',
        'independentHeadAudit': 'independent-geometry-head-audit.py',
        'gpuBatchPreflight': 'probe_geometry_training_batch.py',
    }
    for key, name in audit_scripts.items():
        report = json.loads(report_paths[key].read_bytes())
        sources.append((work_folder / name, report['script_sha256']))
    head_audit = json.loads(report_paths['independentHeadAudit'].read_bytes())
    sources.append((root / 'training/tests/test_word_geometry_head.py', head_audit['test_source_sha256']))
    for source, digest in sources:
        if sha(source) != digest:
            raise ValueError(f'Geometry branch source archive mismatch: {source}')
        record['sources'].append(archive(source, relative / 'source' / source.name))
    return record


def archive_horizontal_mirror(config, run_folder):
    if not config.get('horizontal_mirror_augmentation'):
        return None
    protocol_path = run_folder / 'horizontal-mirror-protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['horizontal_mirror_augmentation'] is True
    assert protocol['parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['source_sha256'] == config['source_sha256']
    assert not config['half_turn_augmentation'] and not config['polarity_augmentation']
    assert protocol['mirror']['probability'] == .5
    assert protocol['mirror']['slot_permutation_zero_based'] == [3, 4, 5, 0, 1, 2]
    relative = Path(run_folder.name) / 'horizontal-mirror-preflight'
    record = {'protocol': archive(protocol_path, relative / protocol_path.name),
              'parentCheckpointSha256': config['resume_sha256'], 'audits': [], 'sources': []}
    for entry in protocol['preflight_audits']:
        source = Path(entry['path'])
        retained = run_folder / 'preflight-source-retention' / source.name
        if retained.is_file():
            source = retained
        assert sha(source) == entry['sha256'], f'Mirror preflight changed: {source}'
        record['audits'].append(archive(source, relative / source.name))
        report = json.loads(source.read_bytes())
        if 'script_sha256' in report:
            script = run_folder / 'preflight-source-retention/audit_run22_mirror.py'
            assert sha(script) == report['script_sha256']
            record['sources'].append(archive(script, relative / 'source' / script.name))
    return record


def archive_vertical_stretch(config, run_folder):
    if not config.get('vertical_stretch_augmentation'):
        return None
    protocol_path = run_folder / 'vertical-stretch-protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['vertical_stretch_augmentation'] is True
    assert protocol['parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['source_sha256'] == config['source_sha256']
    assert config['illumination_augmentation'] is True
    assert not config['horizontal_mirror_augmentation']
    assert not config['half_turn_augmentation'] and not config['polarity_augmentation']
    assert protocol['vertical_stretch']['range'] == [.9, 1.1]
    assert protocol['vertical_stretch']['stream_id'] == 0x56535452
    assert protocol['vertical_stretch']['existing_affine_probability'] == .85
    assert protocol['preflight_audits']
    relative = Path(run_folder.name) / 'vertical-stretch-preflight'
    record = {'protocol': archive(protocol_path, relative / protocol_path.name),
              'parentCheckpointSha256': config['resume_sha256'], 'audits': [], 'sources': []}
    source_candidates = sorted((run_folder / 'preflight-source-retention').glob('*.py'))
    source_candidates += sorted(run_folder.parent.glob('*.py'))
    archived_scripts = set()
    for entry in protocol['preflight_audits']:
        source = Path(entry['path'])
        retained = run_folder / 'preflight-source-retention' / source.name
        if retained.is_file():
            source = retained
        assert sha(source) == entry['sha256'], f'Vertical stretch preflight changed: {source}'
        record['audits'].append(archive(source, relative / source.name))
        report = json.loads(source.read_bytes())
        digest = report.get('script_sha256')
        if digest and digest not in archived_scripts:
            script = next((path for path in source_candidates if sha(path) == digest), None)
            assert script is not None, f'Missing original vertical stretch audit script: {digest}'
            record['sources'].append(archive(script, relative / 'source' / script.name))
            archived_scripts.add(digest)
    return record


def archive_horizontal_aspect(config, run_folder):
    if not config.get('horizontal_aspect_augmentation'):
        return None
    if (run_folder / 'horizontal-aspect-continuation-protocol.json').is_file():
        return None  # The separate continuation proof must retain its original ancestor.
    protocol_path = run_folder / 'horizontal-aspect-protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['horizontal_aspect_augmentation'] is True
    assert protocol['parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['source_sha256'] == config['source_sha256']
    assert config['illumination_augmentation'] is True
    for name in ('camera_augmentation', 'horizontal_mirror_augmentation',
                 'vertical_stretch_augmentation', 'half_turn_augmentation', 'polarity_augmentation'):
        assert not config[name] and not protocol[name]
    aspect = protocol['horizontal_aspect']
    assert aspect['range'] == [.55, 1.25] and aspect['base_zoom'] == [.8, 1.7]
    assert aspect['stream_id'] == 0x48415350 and aspect['existing_affine_probability'] == .85
    assert aspect['seed_sequence'] == ['seed', 'dataset_epoch', 'expanded_record_index', 0x48415350]
    assert protocol['preflight_audits']
    relative = Path(run_folder.name) / 'horizontal-aspect-preflight'
    record = {'protocol': archive(protocol_path, relative / protocol_path.name),
              'parentCheckpointSha256': config['resume_sha256'], 'audits': [], 'sources': []}
    for entry in protocol['preflight_audits']:
        source = Path(entry['path'])
        assert sha(source) == entry['sha256'], f'Horizontal aspect preflight changed: {source}'
        record['audits'].append(archive(source, relative / source.name))
        report = json.loads(source.read_bytes())
        if report.get('script_sha256'):
            script = Path(report['script_path'])
            assert sha(script) == report['script_sha256']
            record['sources'].append(archive(script, relative / 'source' / script.name))
        log_digest = report.get('tests_log_sha256') or report.get('log_sha256')
        if log_digest:
            log = (Path(report['script_path']).parent / 'tests.log'
                   if report.get('script_path') else run_folder / 'preflight-tests.log')
            assert sha(log) == log_digest
            record['sources'].append(archive(log, relative / 'logs' / (source.stem + '.log')))
        for name, digest in report.get('tests_sha256', {}).items():
            test = run_folder / 'test-source' / name
            assert sha(test) == digest
            record['sources'].append(archive(test, relative / 'test-source' / name))
    prototype = Path(protocol['private_prototype']['directory'])
    assert sha(prototype / 'index.json') == protocol['private_prototype']['index_sha256']
    assert sha(prototype / 'proof-report.json') == protocol['private_prototype']['proof_sha256']
    review = run_folder.parent / 'horizontal-aspect-independent-review-v1'
    for source_folder, target_folder in ((prototype, 'private-prototype'), (review, 'independent-prototype-review')):
        for source in sorted(source_folder.rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                record['sources'].append(archive(source, relative / target_folder / source.relative_to(source_folder)))
    independent_script = run_folder.parent / 'audit_horizontal_aspect_independent.py'
    record['sources'].append(archive(independent_script, relative / 'source' / independent_script.name))
    return record


def archive_horizontal_aspect_continuation(config, run_folder):
    """Retain a weight-only continuation without relabeling its ancestor's proof."""
    protocol_path = run_folder / 'horizontal-aspect-continuation-protocol.json'
    if not protocol_path.is_file():
        return None
    assert config['horizontal_aspect_augmentation'] is True
    assert not (run_folder / 'horizontal-aspect-protocol.json').exists()
    protocol = json.loads(protocol_path.read_bytes())
    inherited = protocol['continuation']
    assert inherited['kind'] == 'weight_only_same_sample_realizations'
    assert inherited['parent_retained_epoch'] == 10 and inherited['parent_is_selected_control'] is False
    parent_config_path = Path(inherited['parent_training_config']['path'])
    parent_folder = parent_config_path.parent
    for name in ('parent_training_config', 'parent_sampling_lineage', 'parent_sampling_history',
                 'parent_selection', 'parent_held_diagnosis', 'inherited_horizontal_aspect_protocol',
                 'inherited_transform_audit', 'inherited_source_bound_tests'):
        entry = inherited[name]
        assert sha(Path(entry['path'])) == entry['sha256'], f'Continuation ancestor evidence changed: {name}'
    previous = json.loads(parent_config_path.read_bytes())
    without_resume = lambda value: {key: item for key, item in value.items()
                                   if key not in ('resume', 'resume_sha256', 'started_at_utc')}
    assert without_resume(previous) == without_resume(config)
    assert protocol['source_sha256'] == config['source_sha256'] == previous['source_sha256']
    assert protocol['parent_training_config_sha256'] == sha(parent_config_path)
    assert protocol['parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['parent_checkpoint'] == config['resume']
    assert sha(Path(config['resume'])) == config['resume_sha256']
    assert protocol['control_checkpoint_sha256'] == previous['resume_sha256'] != config['resume_sha256']
    assert inherited['original_initial_checkpoint_sha256'] == previous['resume_sha256']
    original_protocol_path = parent_folder / 'horizontal-aspect-protocol.json'
    assert original_protocol_path.resolve() == Path(inherited['inherited_horizontal_aspect_protocol']['path']).resolve()
    original_protocol = json.loads(original_protocol_path.read_bytes())
    assert original_protocol['parent_checkpoint_sha256'] == previous['resume_sha256']
    assert protocol['horizontal_aspect'] == original_protocol['horizontal_aspect']
    assert original_protocol['source_sha256'] == config['source_sha256']
    lineage_path = parent_folder / 'sampling-lineage-final.json'
    history_path = parent_folder / 'sampling-history.json'
    assert sha(lineage_path) == inherited['parent_sampling_lineage']['sha256'] == protocol['parent_sampling_lineage_sha256']
    assert sha(history_path) == inherited['parent_sampling_history']['sha256']
    lineage = json.loads(lineage_path.read_bytes())
    assert lineage['status'] == 'passed' and lineage['actual']['config_sha256'] == sha(parent_config_path)
    assert lineage['actual']['sampling_history_sha256'] == sha(history_path)
    epoch = next(row for row in lineage['actual']['observed_epochs'] if row['epoch'] == 10)
    assert epoch['checkpoint_sha256'] == config['resume_sha256'] and epoch['eligible_by_sampling_lineage'] is True
    assert epoch['all_positive_records_seen_in_this_run'] is True and epoch['completed_batches'] == 1037
    assert 10 in lineage['eligible_retained_epochs']
    guard_path = run_folder / 'continuation-guard-audit.json'
    guard = json.loads(guard_path.read_bytes())
    audits = {entry['sha256']: Path(entry['path']) for entry in protocol['preflight_audits']}
    assert len(audits) == 2 and sha(guard_path) in audits
    parent_audit_path = next(path for digest, path in audits.items() if digest != sha(guard_path))
    for digest, path in audits.items():
        assert sha(path) == digest
    artifact = json.loads(parent_audit_path.read_bytes())
    for audit in (guard, artifact):
        assert audit['status'] == 'passed' and audit['source_sha256'] == config['source_sha256']
        assert audit['parent_checkpoint_sha256'] == config['resume_sha256']
        assert audit['backend_modules_imported'] == []
    assert artifact['bindings']['config']['sha256'] == sha(parent_config_path)
    assert artifact['bindings']['lineage']['sha256'] == sha(lineage_path)
    assert artifact['bindings']['sampling']['sha256'] == sha(history_path)
    assert artifact['sampling_epoch10'] == epoch
    assert artifact['identical_complete_serialized_model_config'] is True
    assert artifact['model_arrays'] == 280 and artifact['model_parameters'] == 829307
    assert artifact['frozen_batchnorm_arrays_byte_identical'] == 208
    assert guard['plan_sha256'] == protocol['reviewed_plan_sha256'] == sha(run_folder / f'{run_folder.name}-plan.json')
    relative = Path(run_folder.name) / 'horizontal-aspect-continuation-preflight'
    record = {
        'protocol': archive(protocol_path, relative / protocol_path.name),
        'guardAudit': archive(guard_path, relative / guard_path.name),
        'parentArtifactAudit': archive(parent_audit_path, relative / 'parent-artifact-audit/report.json'),
        'parentCheckpointSha256': config['resume_sha256'],
        'inheritedRunConfigurationSha256': sha(parent_config_path),
        'inheritedHorizontalAspectProtocolSha256': sha(original_protocol_path),
        'sources': [],
    }
    source_specs = [
        (run_folder.parent / f'audit_{run_folder.name}_continuation.py', guard['script_sha256']),
        (run_folder.parent / f'launch_{run_folder.name}.py', guard['launcher_sha256']),
        (run_folder.parent / f'prepare_{run_folder.name}.py', guard['preparation_script_sha256']),
        (parent_folder / 'launch-command.json', guard['parent_launch_sha256']),
        (parent_audit_path.parent / 'audit.py', artifact['script_sha256']),
    ]
    for source, digest in source_specs:
        assert sha(source) == digest, f'Continuation source changed: {source}'
        folder = ('parent-artifact-audit' if source.parent == parent_audit_path.parent
                  else 'parent-source' if source.parent == parent_folder else 'source')
        record['sources'].append(archive(source, relative / folder / source.name))
    for source in sorted(parent_audit_path.parent.iterdir()):
        if source.is_file() and source.name not in ('audit.py', 'report.json'):
            record['sources'].append(archive(source, relative / 'parent-artifact-audit' / source.name))
    for name in (f'{run_folder.name}-plan.json', 'launch-command.json'):
        record['sources'].append(archive(run_folder / name, relative / name))
    for name, digest in config['source_sha256'].items():
        source = run_folder / 'source' / name
        assert sha(source) == digest == sha(parent_folder / 'source' / name)
        record['sources'].append(archive(source, relative / 'training-source' / name))
    return record


def archive_batchnorm_affine(config, run_folder):
    branch = config.get('batchnorm_affine', {})
    if not branch.get('enabled'):
        return None
    init_path = Path(branch['initialization_report'])
    conversion_path = Path(branch['conversion_report'])
    assert sha(init_path) == branch['initialization_report_sha256']
    assert sha(conversion_path) == branch['conversion_report_sha256']
    init = json.loads(init_path.read_bytes())
    conversion = json.loads(conversion_path.read_bytes())
    assert init['status'] == conversion['status'] == 'passed'
    assert config['resume_sha256'] == branch['initial_checkpoint_sha256'] == init['initial_checkpoint_sha256']
    assert branch['base_parent_checkpoint_sha256'] == init['parent_checkpoint_sha256']
    assert branch['nonzero_diagnostic_sha256'] == init['nonzero_diagnostic_sha256']
    assert branch['nonzero_diagnostic_forbidden'] is True
    assert init['nonzero_diagnostic_must_never_be_resumed'] is True
    assert config['resume_sha256'] != init['nonzero_diagnostic_sha256']
    assert conversion['initialization_report_sha256'] == sha(init_path)
    assert conversion['probes']['initial']['checkpoint_sha256'] == config['resume_sha256']
    assert conversion['probes']['nonzero']['checkpoint_sha256'] == init['nonzero_diagnostic_sha256']
    assert all(probe['builtins_only'] and probe['all_moments_and_weights_unchanged']
               and probe['eager_train_eval_exact'] for probe in conversion['probes'].values())
    branch_folder, work_folder = init_path.parent, run_folder.parent
    activation_path = Path(init['activation_report'])
    assert sha(activation_path) == init['activation_report_sha256']
    assert json.loads(activation_path.read_bytes())['checkpoint_sha256'] == init['parent_checkpoint_sha256']
    report_paths = {
        'initializationReport': init_path,
        'nonzeroConversionProof': conversion_path,
        'sixInputOracle': conversion_path.parent / 'bn-affine-six-oracle.json',
        'activationReport': activation_path,
        'independentActivationReview': work_folder / 'v15-batchnorm-activation-independent-review.json',
        'independentTrainingAudit': work_folder / 'bn-affine-independent-preflight/report.json',
        'independentGuardAudit': run_folder / 'bn-affine-guard-audit.json',
        'paritySourceReview': work_folder / 'bn-affine-independent-preflight/parity-source-review-v2.json',
        'protocol': run_folder / 'bn-affine-protocol.json',
    }
    protocol = json.loads(report_paths['protocol'].read_bytes())
    assert protocol['batchnorm_affine'] is True
    assert protocol['parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['base_parent_checkpoint_sha256'] == init['parent_checkpoint_sha256']
    assert protocol['bn_affine_initialization_report_sha256'] == sha(init_path)
    assert protocol['bn_affine_conversion_report_sha256'] == sha(conversion_path)
    assert protocol['moving_moments_sha256'] == branch['moving_moments_sha256']
    assert protocol['source_sha256'] == config['source_sha256']
    for audit in protocol['preflight_audits']:
        assert sha(Path(audit['path'])) == audit['sha256']
    guard = json.loads(report_paths['independentGuardAudit'].read_bytes())
    assert guard['status'] == 'passed' and guard['source_sha256'] == config['source_sha256']
    assert sha(report_paths['independentGuardAudit']) in {entry['sha256'] for entry in protocol['preflight_audits']}
    assert sha(report_paths['sixInputOracle']) == conversion['oracle_report_sha256']
    relative = Path(run_folder.name) / 'batchnorm-affine-initialization'
    record = {key: archive(path, relative / (f'{key}.json')) for key, path in report_paths.items()}
    record.update({
        'baseParentCheckpointSha256': init['parent_checkpoint_sha256'],
        'initialParentCheckpointSha256': init['initial_checkpoint_sha256'],
        'forbiddenDiagnosticCheckpointSha256': init['nonzero_diagnostic_sha256'],
        'sources': [],
    })
    source_candidates = sorted((branch_folder / 'source').glob('*.py'))
    source_candidates += sorted((conversion_path.parent / 'source').glob('*.py'))
    source_candidates += sorted((work_folder / 'bn-affine-independent-preflight').glob('*.py'))
    source_candidates += sorted(work_folder.glob('*.py'))
    source_candidates += sorted(run_folder.glob('*.py'))
    source_candidates += sorted((root / 'training/scripts').glob('word_*.py'))
    expected_sources = list(init['source_sha256'].values()) + [conversion['source_sha256']]
    expected_sources += list(conversion['decision_sources_sha256'].values())
    for path in report_paths.values():
        report = json.loads(path.read_bytes())
        if report.get('script_sha256'):
            expected_sources.append(report['script_sha256'])
    for digest in dict.fromkeys(expected_sources):
        source = next((path for path in source_candidates if sha(path) == digest), None)
        assert source is not None, f'Missing original BatchNorm affine proof source: {digest}'
        record['sources'].append(archive(source, relative / 'source' / source.name))
    return record


def archive_fine_detail(config, run_folder):
    """Retain the separate stride2 branch initialization and real nonzero proof."""
    branch = config.get('fine_detail_branch', {})
    if not branch.get('enabled'):
        return None
    init_path = Path(branch['initialization_report'])
    conversion_path = Path(branch['conversion_report'])
    assert sha(init_path) == branch['initialization_report_sha256']
    assert sha(conversion_path) == branch['conversion_report_sha256']
    init = json.loads(init_path.read_bytes())
    conversion = json.loads(conversion_path.read_bytes())
    assert init['status'] == conversion['status'] == 'passed'
    assert config['resume_sha256'] == branch['initial_checkpoint_sha256'] == init['initial_checkpoint_sha256']
    assert branch['base_parent_checkpoint_sha256'] == init['parent_checkpoint_sha256']
    assert branch['nonzero_diagnostic_sha256'] == init['nonzero_diagnostic_sha256']
    assert branch['nonzero_diagnostic_forbidden'] is True
    assert init['nonzero_diagnostic_must_never_be_resumed'] is True
    assert config['resume_sha256'] != init['nonzero_diagnostic_sha256']
    assert conversion['initialization_report_sha256'] == sha(init_path)
    assert conversion['oracle_sha256'] == init['oracle_sha256']
    assert conversion['probes']['initial']['checkpoint_sha256'] == config['resume_sha256']
    assert conversion['probes']['nonzero']['checkpoint_sha256'] == init['nonzero_diagnostic_sha256']
    assert all(probe['builtins_only'] and probe['all_moments_and_weights_unchanged']
               and probe['eager_train_eval_exact'] for probe in conversion['probes'].values())
    assert init['fine_branch'] == branch['spec']
    assert init['fine_branch']['added_parameters'] == 1232 and init['parameters'] == 830539
    assert init['ordinary_frozen_batchnorm_layers'] == 52
    assert min(init['nonzero_max_output_delta'], conversion['nonzero_tf_output_delta'],
               conversion['nonzero_tflite_output_delta']) > 1e-3
    work_folder = run_folder.parent
    static_path = Path(init['static_audit'])
    assert sha(static_path) == init['static_audit_sha256']
    assert json.loads(static_path.read_bytes())['parent']['sha256'] == init['parent_checkpoint_sha256']
    report_paths = {
        'initializationReport': init_path,
        'nonzeroConversionProof': conversion_path,
        'independentStaticAudit': static_path,
        'independentPreflightAudit': work_folder / 'fine-branch-independent-preflight/report.json',
        'independentGuardAudit': run_folder / 'fine-branch-guard-audit.json',
        'protocol': run_folder / 'fine-branch-protocol.json',
    }
    protocol = json.loads(report_paths['protocol'].read_bytes())
    assert protocol['parent_checkpoint_sha256'] == config['resume_sha256']
    assert protocol['base_parent_checkpoint_sha256'] == init['parent_checkpoint_sha256']
    assert protocol['source_sha256'] == config['source_sha256']
    audit_hashes = set()
    for entry in protocol['preflight_audits']:
        assert sha(Path(entry['path'])) == entry['sha256']
        audit_hashes.add(entry['sha256'])
    for key in ('independentPreflightAudit', 'independentGuardAudit'):
        path = report_paths[key]
        assert sha(path) in audit_hashes and json.loads(path.read_bytes())['status'] == 'passed'
    relative = Path(run_folder.name) / 'fine-detail-initialization'
    record = {key: archive(path, relative / f'{key}.json') for key, path in report_paths.items()}
    record.update({
        'baseParentCheckpointSha256': init['parent_checkpoint_sha256'],
        'initialParentCheckpointSha256': init['initial_checkpoint_sha256'],
        'forbiddenDiagnosticCheckpointSha256': init['nonzero_diagnostic_sha256'],
        'sources': [],
    })
    source_candidates = []
    for folder in (init_path.parent / 'source', conversion_path.parent / 'source',
                   static_path.parent, work_folder / 'fine-branch-independent-preflight',
                   work_folder / 'fine-branch-independent-preflight/source',
                   work_folder, run_folder, root / 'training/scripts'):
        source_candidates.extend(sorted(folder.glob('*.py')))
    expected_sources = list(init['source_sha256'].values()) + list(conversion['source_sha256'].values())
    static_sources = json.loads(static_path.read_bytes())['source_sha256']
    for filename, digest in static_sources.items():
        source = Path(filename)
        assert sha(source) == digest, f'Stride2 static proof dependency changed: {source}'
        source_candidates.append(source)
        expected_sources.append(digest)
    for path in report_paths.values():
        report = json.loads(path.read_bytes())
        if report.get('script_sha256'):
            expected_sources.append(report['script_sha256'])
    used_source_names = {}
    for digest in dict.fromkeys(expected_sources):
        source = next((path for path in source_candidates if sha(path) == digest), None)
        assert source is not None, f'Missing original stride2 proof source: {digest}'
        filename = source.name
        if filename in used_source_names and used_source_names[filename] != digest:
            filename = digest[:12] + '-' + filename
        used_source_names[filename] = digest
        record['sources'].append(archive(source, relative / 'source' / filename))
    return record


original_sources = {}
for folder in [*args.training_runs, args.selected]:
    for source in (folder / 'source').glob('*.py'):
        original_sources.setdefault(sha(source), source)
runs = []
for folder in args.training_runs:
    config_path = folder / 'training-config.json'
    config = json.loads(config_path.read_bytes())
    record = {'run': folder.name, 'configuration': archive(config_path, Path(folder.name) / config_path.name), 'sources': [], 'unavailable_original_sources': []}
    for name, digest in config.get('source_sha256', {}).items():
        source = folder / 'source' / name
        if not source.is_file():
            source = original_sources.get(digest)
            if source is None:
                record['unavailable_original_sources'].append({'file': name, 'sha256_recorded_at_training': digest})
                continue
        if sha(source) != digest:
            raise ValueError(f'Training source archive mismatch: {source}')
        record['sources'].append(archive(source, Path(folder.name) / 'source' / name))
    record['source_hashes_recorded_at_training'] = bool(config.get('source_sha256'))
    record['resume_checkpoint_sha256_recorded_at_training'] = config.get('resume_sha256')
    point_metadata = config.get('dsbi_point_metadata')
    if point_metadata:
        if args.data_root is None:
            raise ValueError('--data-root is required to retain exact point metadata')
        source = args.data_root / point_metadata
        if sha(source) != config['data_sha256'][point_metadata]:
            raise ValueError(f'Exact point metadata changed: {source}')
        record['dsbiPointMetadata'] = archive(source, Path(folder.name) / 'dsbi-point-metadata.json')
    geometry_initialization = archive_geometry_initialization(config, folder)
    if geometry_initialization:
        record['geometryInitialization'] = geometry_initialization
    mirror_preflight = archive_horizontal_mirror(config, folder)
    if mirror_preflight:
        record['horizontalMirrorPreflight'] = mirror_preflight
    vertical_preflight = archive_vertical_stretch(config, folder)
    if vertical_preflight:
        record['verticalStretchPreflight'] = vertical_preflight
    aspect_continuation = archive_horizontal_aspect_continuation(config, folder)
    if aspect_continuation:
        assert any(sha(candidate / 'training-config.json') == aspect_continuation['inheritedRunConfigurationSha256']
                   for candidate in args.training_runs if candidate != folder), 'Continuation must retain its original ancestor run'
        record['horizontalAspectContinuation'] = aspect_continuation
    aspect_preflight = archive_horizontal_aspect(config, folder)
    if aspect_preflight:
        record['horizontalAspectPreflight'] = aspect_preflight
    batchnorm_affine = archive_batchnorm_affine(config, folder)
    if batchnorm_affine:
        record['batchnormAffineInitialization'] = batchnorm_affine
    fine_detail = archive_fine_detail(config, folder)
    if fine_detail:
        record['fineDetailInitialization'] = fine_detail
    for name in ('validation-history.json', 'history.json', 'source-balance-protocol.json',
                 'full-tail-protocol.json', 'partial-geometry-protocol.json', 'dsbi-addition-protocol.json',
                 'dsbi-visible-partial-protocol.json', 'geometry-residual-protocol.json',
                 'horizontal-mirror-protocol.json', 'vertical-stretch-protocol.json', 'horizontal-aspect-protocol.json', 'horizontal-aspect-continuation-protocol.json', 'bn-affine-protocol.json', 'fine-branch-protocol.json',
                 'sampling-history.json', 'sampling-lineage-final.json',
                 'actual-weight-update-epoch1.json'):
        source = folder / name
        if source.is_file():
            record[name] = archive(source, Path(folder.name) / name)
    runs.append(record)

wrappers = []
for folder in args.wrapper_folders:
    manifest_path = folder / 'normalization-wrappers.json'
    manifest = json.loads(manifest_path.read_bytes())
    record = {'manifest': archive(manifest_path, Path(folder.name) / manifest_path.name), 'sources': []}
    for name, digest in manifest['source_sha256'].items():
        source = folder / name
        if sha(source) != digest:
            raise ValueError(f'Normalization source archive mismatch: {source}')
        record['sources'].append(archive(source, Path(folder.name) / name))
    wrappers.append(record)

branches = []
for manifest_path in args.graph_branches:
    manifest = json.loads(manifest_path.read_bytes())
    folder = manifest_path.parent
    record = {'manifest': archive(manifest_path, Path(folder.name) / manifest_path.name), 'sources': []}
    for name, digest in manifest['source_sha256'].items():
        source = folder / 'source' / name
        if sha(source) != digest:
            raise ValueError(f'Graph branch source archive mismatch: {source}')
        record['sources'].append(archive(source, Path(folder.name) / 'source' / name))
    for name in ('independent-core-review.json', 'independent_core_review.py',
                 'formal-builder-review.json'):
        source = folder / name
        if source.is_file():
            record[name] = archive(source, Path(folder.name) / name)
    branches.append(record)

supplements = []
for folder in args.supplemental_preparations:
    index_path = folder/'source-index.json'
    index = json.loads(index_path.read_bytes())
    snapshot = json.loads((folder/'source-archive.json').read_bytes())
    assert snapshot['source_index_sha256'] == sha(index_path)
    assert snapshot['manifest_sha256'] == index['generatedManifest']['sha256']
    assert snapshot['preparation_protocol_sha256'] == sha(folder/'preparation-protocol.json')
    record = {'dataset': index['dataset'], 'revision': index['revision'],
              'sourceIndex': archive(index_path, Path(folder.name)/index_path.name),
              'generatedManifestSha256': index['generatedManifest']['sha256'], 'sources': [],
              'sourceAnnotations': []}
    dataset_root = Path(index['root'])
    if not dataset_root.is_absolute():
        dataset_root = index_path.parent / dataset_root
    dataset_root = dataset_root.resolve()
    for ordinal, source_record in enumerate(index['sources']):
        original = (dataset_root / source_record['original_annotation']).resolve()
        if not original.is_relative_to(dataset_root):
            raise ValueError(f'Annotation outside dataset root: {original}')
        if sha(original) != source_record['annotation_sha256']:
            raise ValueError(f'Original DSBI annotation changed: {original}')
        retained = archive(original, Path(folder.name) / 'source-annotations' / f'{ordinal:02d}.txt')
        record['sourceAnnotations'].append({'source': source_record['source'], **retained})
    for source in snapshot['files']:
        original = folder/source['archived']
        assert sha(original) == source['sha256']
        record['sources'].append(archive(original, Path(folder.name)/source['archived']))
    for name in ('source-archive.json', 'preparation-protocol.json', 'independent-context-audit.json',
                 'v1-v2-delta-audit.json'):
        source = folder/name
        if source.is_file():
            record[name] = archive(source, Path(folder.name)/name)
    supplements.append(record)

selection_sources = []
for name, digest in selected['oracle']['source_sha256'].items():
    source = args.selected / 'source' / name
    if sha(source) != digest:
        raise ValueError(f'Selection source archive mismatch: {source}')
    selection_sources.append(archive(source, Path('selection-source') / name))
report = {
    'scope': 'Retained training configurations and original source snapshots. This is provenance, not a claim of bitwise retraining reproducibility. Missing originals or contemporaneous resume hashes are marked explicitly. An absent run snapshot may be recovered only from an archive with exactly the recorded SHA-256.',
    'selectedCheckpointSha256': selected['oracle']['checkpoint_sha256'],
    'selectedSourceCheckpoint': selected['source_checkpoint'],
    'trainingRuns': runs,
    'normalizationWrappers': wrappers,
    'graphBranches': branches,
    'supplementalPreparations': supplements,
    'selectionSources': selection_sources,
    'selectionReport': archive(args.selected / 'checkpoint-selection.json', Path('checkpoint-selection.json')),
}
target = args.output / 'training-provenance.json'
target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'file': str(target), 'sha256': sha(target), 'runs': len(runs)}))
