"""Check run30's completed own cycle, then compare eligible checkpoints with v26."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

work = Path(__file__).resolve().parent
run = work/'run30'
output = work/'selected-v27'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if output.exists():
    raise SystemExit('V27 output already exists; never overwrite or rerun it')
if not (run/'training-process-result.json').exists():
    raise SystemExit('Run30 is still active; monitor its existing session')
assert json.loads((run/'training-process-result.json').read_text())['exit_code'] == 0
history = json.loads((run/'validation-history.json').read_text())
config = json.loads((run/'training-config.json').read_text())
protocol_path = run/'learning-rate-protocol.json'
protocol = json.loads(protocol_path.read_text())
assert len(history) == config['epochs'] == 10
assert config['source_sha256'] == protocol['source_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(run/'source'/name) == sha(Path('training/scripts')/name) == expected
for name, expected in protocol['selector_source_sha256'].items():
    assert sha(Path('training/scripts')/name) == expected
assert sha(protocol['validation_fixture']) == protocol['validation_fixture_sha256']
assert config['learning_rate'] == protocol['learning_rate'] == 5e-5
assert config['resume_sha256'] == protocol['parent_checkpoint_sha256']
assert config['imageNormalization'] == protocol['imageNormalization']
assert not config['normalization_transition']['enabled']
assert config['parameters'] == 829307 and config['feature_pyramid_strides'] == [32, 16, 8, 4]
assert config['batch_normalization']['layers'] == 52 and config['batch_normalization']['trainable_affine_layers'] == 0
audit_path = run/'sampling-lineage-final.json'
if not audit_path.exists():
    subprocess.run([sys.executable, 'training/scripts/word_sampling_audit.py', '--run', str(run),
                    '--protocol', str(protocol_path), '--report', str(audit_path),
                    '--require-eligible-epoch', '8'], check=True)
audit = json.loads(audit_path.read_text())
assert audit['status'] == 'passed' and audit['eligible_retained_epochs'] == [8, 9, 10]
assert audit['actual']['config_sha256'] == sha(run/'training-config.json')
parent = Path(protocol['control_checkpoint'])
assert sha(parent) == protocol['control_checkpoint_sha256']
candidates = [parent]+[run/f'retained-epochs/epoch-{epoch:02d}.keras' for epoch in (8, 9, 10)]
for checkpoint, row in zip(candidates[1:], audit['actual']['observed_epochs'][7:], strict=True):
    assert sha(checkpoint) == row['checkpoint_sha256']
command = [sys.executable, '-u', 'training/scripts/word_select.py', '--candidates', *map(str, candidates),
           '--data', str(work/'data'), '--output', str(output), '--batch-size', '24',
           '--validation-manifest', 'validation-stress-selection.json', '--grouped-selection',
           '--threshold-grid', 'weak-peaks-v2']
(run/'selection-command.json').write_text(json.dumps({'command': command, 'audit_sha256': sha(audit_path),
    'protocol_sha256': sha(protocol_path), 'selector_launcher_sha256': sha(__file__),
    'started_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2)+'\n')
env = dict(os.environ, KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
with (run/'selection.log').open('w', encoding='utf-8') as log:
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = process.wait()
(run/'selection-process-result.json').write_text(json.dumps({'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': code}, indent=2)+'\n')
sys.exit(code)
