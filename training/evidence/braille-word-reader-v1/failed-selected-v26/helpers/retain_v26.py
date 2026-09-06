"""Keep the failed decoder-v2 evaluation without publishing candidate binaries."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path.cwd()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
RUN = WORK / 'selected-v26'
DEST = ROOT / 'training/evidence/braille-word-reader-v1/failed-selected-v26'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


completed = json.loads((RUN / 'COMPLETED-HANDOFF.json').read_text(encoding='utf-8'))
for name, expected in completed['files_sha256'].items():
    assert sha(RUN / name) == expected, name
assert sha(RUN / 'word_reader.tflite') == completed['model_sha256']
assert sha(RUN / 'best-word.keras') == completed['checkpoint_sha256']

paths = [(p, p.relative_to(RUN)) for p in RUN.rglob('*')
         if p.is_file() and p.suffix in ('.json', '.py', '.log')
         and p.name != 'RUNNING-HANDOFF.json']
paths += [(WORK / name, Path('helpers') / name) for name in (
    'evaluate_position_v26.py', 'evaluate_position_v26_retry.py', 'retain_v26.py')]
paths += [(WORK / 'selected-v15' / 'export-report.json', Path('historical') / 'v15-export-report.json'),
          (WORK / 'selected-v15-independent-diagnosis' / 'protocol.json', Path('historical') / 'v15-raw607-protocol.json')]
files = []
for source, relative in paths:
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == sha(source), f'Refusing changed retained file: {target}'
    else:
        shutil.copyfile(source, target)
    files.append({'file': relative.as_posix(), 'sha256': sha(target), 'sizeBytes': target.stat().st_size})

large_paths = [RUN / name for name in ('best-word.keras', 'word_reader.tflite',
                                     'backend-fullprecision.npz', 'backend-tflite-v2.npz')]
large_paths.append(WORK / 'selected-v15' / 'public-word-outputs.npz')
archive = {
    'scope': 'Completed failed v26 evaluation with six-dot-word-v2. No active model or main APK.',
    'internalAcceptancePassed': False,
    'physicalCameraValidated': False,
    'modelSha256': completed['model_sha256'],
    'checkpointSha256': completed['checkpoint_sha256'],
    'inferenceScope': completed['new_model_invocations'],
    'tensorflowProof': completed['tensorflow_proof'],
    'files': files,
    'largeLocalArtifactsNotCopied': [
        {'file': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'sizeBytes': p.stat().st_size}
        for p in large_paths],
}
content = (json.dumps(archive, indent=2) + '\n').encode('utf-8')
index = DEST / 'archive.json'
assert not index.exists() or index.read_bytes() == content, 'Refusing changed archive index'
index.write_bytes(content)
print(json.dumps({'files': len(files), 'bytes': sum(p['sizeBytes'] for p in files), 'archive_sha256': sha(index)}))
