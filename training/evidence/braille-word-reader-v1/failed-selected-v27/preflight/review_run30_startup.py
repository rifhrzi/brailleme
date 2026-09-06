"""Bind run30's generated startup configuration to the prior read-only review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
RUN = WORK / 'run30'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    preflight_path = WORK / 'run30-independent-review/report.json'
    assert sha(preflight_path) == '2ed436ff0fed1ee7a716c3ad0adc3f94201eef0a6229e32af7682807087397a7'
    preflight = json.loads(preflight_path.read_text())
    protocol_path = RUN / 'learning-rate-protocol.json'
    config_path = RUN / 'training-config.json'
    launch_path = RUN / 'launch-command.json'
    protocol = json.loads(protocol_path.read_text())
    config = json.loads(config_path.read_text())
    launch = json.loads(launch_path.read_text())
    baseline = json.loads((WORK / 'run20/training-config.json').read_text())
    assert sha(config_path) == '28e649bc9cef62afc535446f0675fa5c7fc8a8b98946d1a9507eeb9539a6e3c2'
    assert sha(protocol_path) == preflight['protocol_sha256'] == launch['protocol_sha256']
    assert sha(WORK / 'launch_run30.py') == preflight['launcher_sha256'] == launch['launcher_sha256']
    assert launch['command'] == preflight['concrete_command']
    assert datetime.fromisoformat(protocol['created_at_utc']) < datetime.fromisoformat(preflight['created_at_utc']) < datetime.fromisoformat(launch['started_at_utc']) < datetime.fromisoformat(config['started_at_utc'])
    assert config['source_sha256'] == protocol['source_sha256'] == preflight['source_sha256']
    for name, expected in config['source_sha256'].items():
        assert sha(RUN / 'source' / name) == sha(Path('training/scripts') / name) == expected
    assert config['data_sha256'] == protocol['data_sha256'] == baseline['data_sha256']
    for name, expected in config['data_sha256'].items():
        assert sha(WORK / 'data' / name) == expected
    assert config['resume_sha256'] == sha(config['resume']) == protocol['parent_checkpoint_sha256'] == preflight['parent_checkpoint_sha256']
    assert config['learning_rate'] == protocol['learning_rate'] == 5e-5
    assert config['epochs'] == protocol['planned_epochs'] == 10
    expected_off = {'vertical_stretch_augmentation': False, 'horizontal_aspect_augmentation': False,
                    'anchor_zoom_cap': False, 'geometry_conditioned_residual': {'enabled': False},
                    'batchnorm_affine': {'enabled': False}, 'normalization_transition': {'enabled': False},
                    'fine_detail_branch': {'enabled': False}}
    for key, value in expected_off.items():
        assert config[key] == value, key
    deliberate_changes = {'learning_rate', 'resume', 'resume_sha256', 'started_at_utc', 'source_sha256'}
    for key, value in baseline.items():
        if key not in deliberate_changes:
            assert config[key] == value, ('unexpected baseline difference', key)
    assert set(config) - set(baseline) == set(expected_off)
    assert config['parameters'] == 829307 and config['feature_pyramid_strides'] == [32, 16, 8, 4]
    assert config['batch_normalization']['layers'] == 52 and config['batch_normalization']['trainable_affine_layers'] == 0
    assert config['batch_normalization']['backbone_call_training_keywords'] == [False]
    assert config['imageNormalization']['targetStd'] == .05 and config['imageNormalization']['stdFloor'] == .001
    assert config['sampling']['epoch_records'] == 24882 and config['sampling']['complete_cycle_epochs'] == 8
    assert config['backend'] == 'torch' and config['device'] == 'NVIDIA GeForce RTX 5060 Laptop GPU'
    assert not config['physical_camera_validated']

    # Snapshot the currently completed history bytes once; later epochs may append.
    history_bytes = (RUN / 'sampling-history.json').read_bytes()
    historical_bytes = (WORK / 'run20/sampling-history.json').read_bytes()
    first = json.loads(history_bytes)[0]
    original = json.loads(historical_bytes)[0]
    assert first == original
    assert first['epoch'] == 1 and first['dataset_epoch'] == 2 and first['complete_epoch']
    assert first['records'] == 24882 and first['completed_batches'] == first['expected_batches'] == 1037
    assert first['order_int64_sha256'] == 'eb35dfcfdbb4989223e61d02e307dff472752e5dc1939cc41ad940b72406f8dc'

    output = WORK / 'run30-independent-startup-review'
    output.mkdir(exist_ok=False)
    (output / 'sampling-history-at-review.json').write_bytes(history_bytes)
    report = {
        'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'preflight_report_sha256': sha(preflight_path),
        'training_config_sha256': sha(config_path), 'launch_command_sha256': sha(launch_path),
        'protocol_sha256': sha(protocol_path), 'launcher_sha256': sha(WORK / 'launch_run30.py'),
        'training_started_at_utc': config['started_at_utc'], 'sources_verified': len(config['source_sha256']),
        'data_references_verified': len(config['data_sha256']), 'parent_checkpoint_sha256': config['resume_sha256'],
        'learning_rate': config['learning_rate'], 'parameters': config['parameters'],
        'batch_normalization': config['batch_normalization'], 'imageNormalization': config['imageNormalization'],
        'new_config_keys_all_disabled': expected_off,
        'baseline_config_sha256': sha(WORK / 'run20/training-config.json'),
        'baseline_changed_keys': sorted(deliberate_changes),
        'first_sampling_row_equal_to_run20': True, 'first_sampling_row': first,
        'sampling_history_snapshot_sha256': hashlib.sha256(history_bytes).hexdigest(),
        'historical_run20_sampling_history_sha256': hashlib.sha256(historical_bytes).hexdigest(),
        'findings': [],
        'limitations': 'Actual generated metadata and completed callback row verified using only stdlib and file hashes. No model/backend load, gradient or inference. First epoch is ineligible; final own-cycle audit still required. Equal record order does not prove identical pixel tensors, gradients, GPU arithmetic or accuracy.',
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'report': str(output / 'report.json'), 'sha256': sha(output / 'report.json')}))


if __name__ == '__main__':
    main()
