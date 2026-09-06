"""Independent stdlib-only prelaunch review; never imports a training backend."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
RUN = WORK / 'run30'
SOURCE = Path('training/scripts')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol_path = RUN / 'learning-rate-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    launcher_path = WORK / 'launch_run30.py'
    tree = ast.parse(launcher_path.read_text())
    cutoff = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'env' for target in node.targets))
    # Execute only imports, read-only launch assertions and the concrete command.
    prelaunch = ast.Module(body=tree.body[:cutoff], type_ignores=[])
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ('Popen', 'write_text', 'write_bytes', 'open')
                   for node in ast.walk(prelaunch))
    namespace = {'__file__': str(launcher_path)}
    exec(compile(prelaunch, str(launcher_path), 'exec'), namespace)
    command = namespace['command']

    train_tree = ast.parse((SOURCE / 'word_train.py').read_text())
    parse_node = next(node for node in train_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
    parsed_namespace = {'argparse': argparse, 'Path': Path, '__doc__': 'Isolated source CLI review'}
    exec(compile(ast.Module(body=[parse_node], type_ignores=[]), 'isolated_word_train_parse_args', 'exec'), parsed_namespace)
    args = vars(parsed_namespace['parse_args'](command[3:]))
    expected_enabled = {'illumination_augmentation', 'balance_source_families', 'retain_epochs', 'dsbi_visible_partial_supervision'}
    actual_enabled = {key for key, value in args.items() if value is True}
    assert actual_enabled == expected_enabled, (actual_enabled, expected_enabled)
    assert args['epochs'] == 10 and args['batch_size'] == 24 and args['learning_rate'] == 5e-5 and args['seed'] == 20260905
    assert args['geometry_loss'] == 'relative_size_huber' and args['limit_train'] is None
    assert args['image_normalization'] is None and args['normalization_initialization'] is None and args['normalization_runtime_proof'] is None
    assert args['resume'] == Path(protocol['parent_checkpoint'])
    assert args['data'] == WORK / 'data' and args['output'] == RUN
    assert args['train_manifest'] == '../combined-training-v5/combined-training-manifest.json'
    assert args['base_train_manifest'] == '../complete-source-training-v3/combined-training-manifest.json'
    assert args['supplemental_source_index'] == ['../dsbi-training-contexts-v2/source-index.json']
    assert args['dsbi_point_metadata'] == '../dsbi-point-metadata-v1/points.json'
    assert args['negative_repeat'] == 27 and args['supplemental_negative_repeat'] == 2

    baseline_path = WORK / 'run20/training-config.json'
    baseline = json.loads(baseline_path.read_text())
    assert sha(baseline_path) == protocol['parent_training_config_sha256']
    assert baseline['data_sha256'] == protocol['data_sha256']
    assert baseline['imageNormalization'] == protocol['imageNormalization']
    assert baseline['parameters'] == 829307 and baseline['feature_pyramid_strides'] == [32, 16, 8, 4]
    assert baseline['batch_normalization']['layers'] == 52 and baseline['batch_normalization']['trainable_affine_layers'] == 0
    assert baseline['batch_normalization']['backbone_call_training_keywords'] == [False]
    source_names = next(ast.literal_eval(node.value) for node in train_tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'TRAINING_SOURCES' for target in node.targets))
    assert len(source_names) == 24 and set(source_names) == set(protocol['source_sha256'])
    assert sha(WORK / 'prepare_run30.py') == protocol['preparation_script_sha256']
    assert protocol['imageNormalization']['stdFloor'] == .001 and protocol['imageNormalization']['targetStd'] == .05

    inherited = []
    allowed_deltas = {'created_at_utc', 'source_sha256', 'parent_checkpoint', 'parent_checkpoint_sha256',
                      'planned_epochs', 'planned_eligible_epochs', 'batch_size', 'learning_rate', 'seed',
                      'selection', 'scope', 'inherited_protocol'}
    for name in ('dsbi-addition-protocol.json', 'dsbi-visible-partial-protocol.json'):
        path = RUN / name
        current = json.loads(path.read_text())
        ancestor_path = Path(current['inherited_protocol']['path'])
        ancestor = json.loads(ancestor_path.read_text())
        assert sha(ancestor_path) == current['inherited_protocol']['sha256']
        changed = sorted(key for key in current.keys() | ancestor.keys() if current.get(key) != ancestor.get(key))
        assert set(changed) <= allowed_deltas, changed
        assert current['preflight_audits'] == ancestor['preflight_audits']
        for item in current['preflight_audits']:
            assert sha(item['path']) == item['sha256']
        inherited.append({'path': str(path), 'sha256': sha(path), 'ancestor_sha256': sha(ancestor_path), 'changed_keys': changed})

    train_text = (SOURCE / 'word_train.py').read_text()
    assert 'keras.models.load_model(args.resume, compile=False)' in train_text
    assert 'optimizer=keras.optimizers.Adam(args.learning_rate, clipnorm=10.)' in train_text
    assert 'keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=.5, min_lr=1e-5)' in train_text
    selector_text = (SOURCE / 'word_select.py').read_text()
    assert 'minimum_precision>=.95' in selector_text and 'if best is None or rank>best[0]:' in selector_text
    assert protocol['decoder_version'] == 'six-dot-word-v2' and protocol['threshold_grid'] == 'weak-peaks-v2'
    assert protocol['planned_eligible_epochs'] == [8, 9, 10] and protocol['no_activation_before_original_gates']
    selector_launcher_path = WORK / 'select_run30.py'
    selector_launcher = selector_launcher_path.read_text()
    ast.parse(selector_launcher)
    assert "'parent_sampling_audit'" not in json.dumps(protocol)
    for required in ("audit['eligible_retained_epochs'] == [8, 9, 10]", "'--require-eligible-epoch', '8'",
                     "candidates = [parent]+", "for epoch in (8, 9, 10)", "'--grouped-selection'",
                     "'--threshold-grid', 'weak-peaks-v2'", "config['parameters'] == 829307",
                     "config['batch_normalization']['trainable_affine_layers'] == 0"):
        assert required in selector_launcher, required

    output = WORK / 'run30-independent-review'
    output.mkdir(exist_ok=False)
    report = {
        'status': 'passed_prelaunch_source_config_review',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__),
        'protocol_sha256': sha(protocol_path), 'launcher_sha256': sha(launcher_path),
        'preparer_sha256': sha(WORK / 'prepare_run30.py'),
        'selection_launcher_sha256': sha(selector_launcher_path),
        'parent_checkpoint_sha256': protocol['parent_checkpoint_sha256'],
        'control_checkpoint_sha256': protocol['control_checkpoint_sha256'],
        'baseline_config_sha256': sha(baseline_path),
        'training_sources_verified': len(source_names), 'source_sha256': protocol['source_sha256'],
        'selector_source_sha256': protocol['selector_source_sha256'],
        'data_sha256': protocol['data_sha256'],
        'inherited_protocols': inherited,
        'isolated_cli_boolean_flags': sorted(actual_enabled),
        'concrete_command': command,
        'training_change': 'Fresh Adam initial learning rate 5e-5, previously 2e-5; unchanged scheduler, loss, source sampler, data, augmentation and exact plain v15 initial model bytes.',
        'evaluation': 'Common six-dot-word-v2 decoder for control and candidates. Frozen 405, weak-peaks-v2 grid, per-group accepted precision >=95%, maximize equal-group exact word mean, strict rank comparison preserves first control on ties.',
        'pending': 'Actual training-config.json and launch-command.json are generated during startup. Own completed source cycle and retained epoch 8-10 eligibility require the post-training lineage audit. Selection wrapper source reviewed, not executed before training.',
        'limitations': 'Read-only source/JSON/SHA-256 review and isolated argparse execution only. No model load, HDF5 reread, gradient step, inference, parity run, physical testing or APK activation. Exact known parent byte hash binds existing full-tail/frozen-BN/normalization evidence.',
        'findings': [],
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'), 'sha256': sha(output / 'report.json')}))


if __name__ == '__main__':
    main()
