"""Run the frozen, independently audited weight-only continuation exactly once."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

WORK = Path(__file__).resolve().parent
RUN = WORK / 'run27'
PARENT_LAUNCH_SHA = 'a3d065c1b287dbd34fc374bf1877b3ddcb251098a7033de93862f76e1e5e5c46'
REVIEWED_PLAN_SHA = 'f3b9f59cf8acaf7fa39fe3fe36c75279c0423d3b5bc50fd1c552469bc0030dbb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_ref(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return json.loads(Path(reference['path']).read_text())


def validate(protocol):
    reviewed_path = RUN / 'run27-plan.json'
    assert sha(reviewed_path) == REVIEWED_PLAN_SHA == protocol['reviewed_plan_sha256']
    reviewed = json.loads(reviewed_path.read_text())
    assert set(protocol) == set(reviewed) | {'reviewed_plan_sha256'}
    for key, expected in reviewed.items():
        if key not in ('created_at_utc', 'preflight_audits'):
            assert protocol[key] == expected, ('Reviewed plan drift', key)
    assert sha(protocol['parent_checkpoint']) == protocol['parent_checkpoint_sha256'] == '6113867b21316669169391fa7216f9d1b9bf147a13c153889ff6de1c26e9a4ea'
    for name, expected in protocol['source_sha256'].items():
        assert sha(Path('training/scripts') / name) == sha(RUN / 'source' / name) == expected
    assert len(protocol['source_sha256']) == 22
    continuation = protocol['continuation']
    assert continuation['kind'] == 'weight_only_same_sample_realizations'
    assert continuation['parent_retained_epoch'] == 10 and continuation['parent_is_selected_control'] is False
    parent_config = verify_ref(continuation['parent_training_config'])
    parent_lineage = verify_ref(continuation['parent_sampling_lineage'])
    verify_ref(continuation['parent_sampling_history'])
    parent_selection = verify_ref(continuation['parent_selection'])
    verify_ref(continuation['parent_held_diagnosis'])
    verify_ref(continuation['inherited_horizontal_aspect_protocol'])
    verify_ref(continuation['inherited_transform_audit'])
    tests = verify_ref(continuation['inherited_source_bound_tests'])
    assert tests['status'] == 'passed'
    assert parent_config['source_sha256'] == protocol['source_sha256']
    assert parent_lineage['status'] == 'passed' and parent_lineage['eligible_retained_epochs'] == [8, 9, 10]
    assert parent_lineage['actual']['config_sha256'] == continuation['parent_training_config']['sha256']
    epoch = parent_lineage['actual']['observed_epochs'][9]
    assert epoch['checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
    assert epoch['all_positive_records_seen_in_this_run'] and epoch['completed_batches'] == 1037
    assert any(row['checkpoint_sha256'] == protocol['parent_checkpoint_sha256'] for row in parent_selection['candidates'])
    assert parent_selection['external_outputs_used'] is False
    assert 'parent_sampling_audit' not in protocol
    for key in ('learning_rate', 'batch_size', 'seed', 'negative_repeat', 'supplemental_negative_repeat',
                'geometry_loss', 'illumination_augmentation', 'horizontal_mirror_augmentation',
                'vertical_stretch_augmentation', 'horizontal_aspect_augmentation', 'camera_augmentation',
                'half_turn_augmentation', 'polarity_augmentation'):
        assert protocol[key] == parent_config[key], key
    assert protocol['planned_epochs'] == parent_config['epochs'] == 10
    assert protocol['planned_eligible_epochs'] == [8, 9, 10]
    assert parent_config['parameters'] == 829307
    assert parent_config['batch_normalization']['layers'] == 52
    assert parent_config['batch_normalization']['trainable_affine_layers'] == 0
    for key in ('geometry_conditioned_residual', 'batchnorm_affine', 'fine_detail_branch'):
        assert parent_config[key]['enabled'] is False and protocol[key] is False
    for name, expected in parent_config['data_sha256'].items():
        assert sha(WORK / 'data' / name) == expected
    assert sha(protocol['control_checkpoint']) == protocol['control_checkpoint_sha256'] == parent_selection['selected_checkpoint_sha256']
    assert len(protocol['preflight_audits']) >= 2
    for reference in protocol['preflight_audits']:
        assert verify_ref(reference)['status'] == 'passed'
    assert sha(RUN / 'run27-plan.json') == protocol['reviewed_plan_sha256']


def command_for(protocol):
    parent_launch = WORK / 'run26/launch-command.json'
    assert sha(parent_launch) == PARENT_LAUNCH_SHA
    original = json.loads(parent_launch.read_text())['command']
    command = original.copy()
    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert command[command.index('--resume')+1].endswith('selected-v15\\best-word.keras')
    command[command.index('--output')+1] = str(RUN)
    command[command.index('--resume')+1] = protocol['parent_checkpoint']
    assert len([i for i, (a, b) in enumerate(zip(original, command, strict=True)) if a != b]) == 2
    return command


def main():
    if any((RUN/name).exists() for name in ('training-config.json', 'training.log', 'last.keras')):
        raise SystemExit('Run27 has already started; monitor its existing process.')
    protocol_path = RUN / 'horizontal-aspect-continuation-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    validate(protocol)
    command = command_for(protocol)
    env = dict(os.environ, KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    launch = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
              'environment': {key: env[key] for key in ('KERAS_BACKEND', 'KERAS_TORCH_DEVICE', 'CUDA_VISIBLE_DEVICES')},
              'protocol_sha256': sha(protocol_path), 'launcher_sha256': sha(__file__),
              'parent_launch_sha256': sha(WORK / 'run26/launch-command.json'),
              'changed_command_arguments': ['--output', '--resume'],
              'optimizer_and_augmentation_reset_declared': True}
    with (RUN / 'launch-command.json').open('x', encoding='utf-8') as stream:
        json.dump(launch, stream, indent=2); stream.write('\n')
    with (RUN / 'training.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        code = process.wait()
    with (RUN / 'training-process-result.json').open('x', encoding='utf-8') as stream:
        json.dump({'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': code}, stream, indent=2)
        stream.write('\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
