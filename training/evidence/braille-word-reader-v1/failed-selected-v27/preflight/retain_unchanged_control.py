"""Retain a completed ablation whose held validation kept its exact control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--selected', required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--control', default='selected-v15')
parser.add_argument('--description', required=True)
parser.add_argument('--preflight-reference')
args = parser.parse_args()
root = Path.cwd()
work = root / '.braille-training-work/word-scan-v1-20260905'
for name in (args.selected, args.run, args.control):
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected a direct work folder name')
selected, run = work / args.selected, work / args.run
target = root / 'training/evidence/braille-word-reader-v1' / f'failed-{args.selected}'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
selection = json.loads((selected / 'checkpoint-selection.json').read_bytes())
control = json.loads((work / args.control / 'selected-validation-fullprecision.json').read_bytes())
result = json.loads((selected / 'selected-validation-fullprecision.json').read_bytes())
assert selection['external_outputs_used'] is False
assert selection['selected_checkpoint_sha256'] == control['checkpoint_sha256'] == result['checkpoint_sha256']
assert sha(selected / 'best-word.keras') == control['checkpoint_sha256']
assert result['thresholds'] == control['thresholds']
assert result['words'] == control['words']
lineage = json.loads((run / 'sampling-lineage-final.json').read_bytes())
assert lineage['status'] == 'passed'
entries = []
sources = [*selected.glob('*.json'), *(selected / 'source').glob('*.py'),
           *(path for path in run.glob('*.json') if path.name != 'RUNNING-HANDOFF.json'),
           *(run / 'source').glob('*.py')]
for source in sorted(sources):
    relative = source.relative_to(work)
    dest = target / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_bytes() != source.read_bytes():
        raise ValueError(f'Refusing changed retained evidence: {dest}')
    shutil.copyfile(source, dest)
    entries.append({'file': relative.as_posix(), 'sha256': sha(dest), 'sizeBytes': dest.stat().st_size})
report = {
    'scope': f"{args.description} Frozen validation selected the unchanged {args.control} checkpoint and thresholds, with identical aggregate word metrics. No new export or external inference was performed.",
    'internalAcceptancePassed': False,
    'physicalCameraValidated': False,
    'externalResultReference': f'../failed-{args.control}/public-benchmark.json',
    'selectedCheckpointSha256': control['checkpoint_sha256'],
    'preflightReference': args.preflight_reference,
    'files': entries,
}
archive = target / 'archive.json'
payload = (json.dumps(report, indent=2) + '\n').encode('utf-8')
if archive.exists() and archive.read_bytes() != payload:
    raise ValueError('Refusing to replace a different retained archive index')
archive.write_bytes(payload)
print(json.dumps({'files': len(entries), 'bytes': sum(row['sizeBytes'] for row in entries), 'archive_sha256': sha(archive)}))
