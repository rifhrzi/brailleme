"""Launch the frozen plain-v15 higher-rate continuation once."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

work = Path(__file__).resolve().parent
run = work/'run30'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol_path = run/'learning-rate-protocol.json'
assert sha(protocol_path) == 'dacada918c7473c3a2c81bd10c3197447ffddf68c834b32774a1945cc03187d8'
p = json.loads(protocol_path.read_text())
assert not any((run/name).exists() for name in ('training-config.json', 'training.log', 'last.keras')), 'Already started'
assert p['parent_checkpoint_sha256'] == sha(p['parent_checkpoint']) == p['control_checkpoint_sha256'] == sha(p['control_checkpoint']) == 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
assert (p['planned_epochs'], p['planned_eligible_epochs'], p['batch_size'], p['learning_rate'], p['seed']) == (10, [8, 9, 10], 24, 5e-5, 20260905)
assert p['imageNormalization']['targetStd'] == .05 and p['frozen_batchnorm'] and p['illumination_augmentation'] and p['dsbi_visible_partial_supervision']
assert p['decoder_version'] == 'six-dot-word-v2' and p['threshold_grid'] == 'weak-peaks-v2'
assert not any(p[key] for key in ('anchor_zoom_cap', 'horizontal_aspect_augmentation', 'vertical_stretch_augmentation', 'horizontal_mirror_augmentation', 'camera_augmentation', 'half_turn_augmentation', 'polarity_augmentation', 'partial_geometry_supervision', 'geometry_conditioned_residual', 'batchnorm_affine', 'fine_detail_branch', 'normalization_transition'))
for name, digest in p['source_sha256'].items():
    assert sha(Path('training/scripts')/name) == sha(run/'source'/name) == digest
for name, digest in p['selector_source_sha256'].items():
    assert sha(Path('training/scripts')/name) == digest
for name, digest in p['data_sha256'].items():
    assert sha(work/'data'/name) == digest
assert sha(p['validation_fixture']) == p['validation_fixture_sha256']
assert sha(p['control_selection']) == p['control_selection_sha256']
assert sha(p['decoder_integration_audit']) == p['decoder_integration_audit_sha256']
command = [sys.executable, '-u', 'training/scripts/word_train.py', '--data', str(work/'data'), '--output', str(run),
           '--epochs', '10', '--batch-size', '24', '--learning-rate', '5e-5', '--seed', '20260905', '--resume', p['parent_checkpoint'],
           '--geometry-loss', 'relative_size_huber', '--illumination-augmentation', '--balance-source-families', '--retain-epochs',
           '--train-manifest', '../combined-training-v5/combined-training-manifest.json', '--base-train-manifest', '../complete-source-training-v3/combined-training-manifest.json',
           '--supplemental-source-index', '../dsbi-training-contexts-v2/source-index.json', '--negative-repeat', '27', '--supplemental-negative-repeat', '2',
           '--dsbi-visible-partial-supervision', '--dsbi-point-metadata', '../dsbi-point-metadata-v1/points.json']
env = dict(os.environ, KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
(run/'launch-command.json').write_text(json.dumps({'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': command, 'protocol_sha256': sha(protocol_path), 'launcher_sha256': sha(__file__)}, indent=2)+'\n')
with (run/'training.log').open('w', encoding='utf-8') as log:
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = process.wait()
(run/'training-process-result.json').write_text(json.dumps({'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': code}, indent=2)+'\n')
sys.exit(code)
