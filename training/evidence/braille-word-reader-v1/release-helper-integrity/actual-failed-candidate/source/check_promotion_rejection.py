"""Prove a real failed candidate cannot alter the active app through review."""
import hashlib
import json
from pathlib import Path
from review_word_promotion import review_promotion

root = Path.cwd()
work = root / '.braille-training-work/word-scan-v1-20260905'
asset = root / 'assets/models/braille-word-reader-v1'
module = root / 'src/ml/word-model-assets.ts'
sha = lambda value: hashlib.sha256(value).hexdigest()
original = {path: path.read_bytes() for path in [*asset.rglob('*'), module] if path.is_file()}
manifest = json.loads((asset / 'manifest.json').read_bytes())
model = (work / 'selected-v15/word_reader.tflite').read_bytes()
conversion = json.loads((work / 'selected-v15/export-report.json').read_bytes())
manifest['artifact'].update(status='unverified_candidate', sha256=sha(model), sizeBytes=len(model))
manifest['imageNormalization'] = conversion['imageNormalization']
manifest['thresholds'] = conversion['thresholds']
assert manifest['internalAcceptance']['passed'] is False
proposed = {asset / 'word-reader.tflite': model,
            asset / 'manifest.json': (json.dumps(manifest, indent=2)+'\n').encode()}
for key, source, name in [('publicBenchmark', work / 'selected-v15/public-benchmark.json', 'public-benchmark.json'),
                          ('wholeWordFixtures', work / 'data/external-words.json', 'external-words.json')]:
    target = root / 'training/evidence/braille-word-reader-v1' / name
    value = source.read_bytes()
    proposed[target] = value
    manifest.setdefault('evidence', {})[key] = {'file': target.relative_to(root).as_posix(), 'sha256': sha(value)}
proposed[asset / 'manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
try:
    review_promotion(root, proposed)
except ValueError as error:
    stage = Path(str(error).split('Review: ', 1)[1])
else:
    raise AssertionError('Failed candidate unexpectedly passed prospective asset review')
assert all(path.read_bytes() == before for path, before in original.items())
assert not (asset / 'word-reader.tflite').exists()
result = json.loads((stage / 'promotion-review.json').read_bytes())
assert result['exit_code'] != 0 and result['source_changed_during_review'] == []
log = (stage / 'promotion-verifier.log').read_text()
assert 'actual: false' in log and 'expected: true' in log
report = {
    'status': 'passed', 'scope': 'Actual v15 binary with internal acceptance false was rejected by the unchanged complete verifier in a separate prospective tree; active app assets and imports remained byte-identical.',
    'script_sha256': sha(Path(__file__).read_bytes()),
    'review_helper_sha256': sha((work / 'review_word_promotion.py').read_bytes()),
    'promotion_helper_sha256': sha((work / 'promote_candidate.py').read_bytes()),
    'review_directory': str(stage),
    'review_report_sha256': sha((stage / 'promotion-review.json').read_bytes()),
    'review_log_sha256': sha((stage / 'promotion-verifier.log').read_bytes()),
    'verifier_sha256': result['verifier_sha256'],
    'failed_model_sha256': sha(model),
    'active_files_unchanged': {str(path.relative_to(root)): sha(value) for path, value in original.items()},
    'word_binary_remains_absent': True,
}
(work / 'promotion-rejection-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
