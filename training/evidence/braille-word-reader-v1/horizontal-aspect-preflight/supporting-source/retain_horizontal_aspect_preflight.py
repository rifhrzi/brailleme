"""Retain run26 geometry augmentation proofs and the exact prelaunch sources."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

root = Path.cwd()
work = root / '.braille-training-work/word-scan-v1-20260905'
run = work / 'run26'
output = root / 'training/evidence/braille-word-reader-v1/horizontal-aspect-preflight'
helper = work / 'archive_provenance.py'
tree = ast.parse(helper.read_text(encoding='utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
args = SimpleNamespace(output=output)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(helper), 'exec'), globals())
config_path = run / 'training-config.json'
config = json.loads(config_path.read_bytes())
record = archive_horizontal_aspect(config, run)
protocol = json.loads((run / 'horizontal-aspect-protocol.json').read_bytes())
audit_path = run / 'horizontal-aspect-integration-audit.json'
audit = json.loads(audit_path.read_bytes())
assert audit['status'] == 'passed'
assert audit['source_sha256'] == config['source_sha256'] == protocol['source_sha256']
assert audit['parent_checkpoint_sha256'] == config['resume_sha256']
assert sha(run / 'run26-plan.json') == audit['plan_sha256'] == protocol['reviewed_plan_sha256']
report = {
    'scope': 'Run26 horizontal width augmentation implementation and prelaunch checks. No trained-model accuracy, native inference, APK or physical-camera claim.',
    'configuration': archive(config_path, Path('run26/training-config.json')),
    'horizontalAspectPreflight': record,
    'sources': [], 'supportingProofs': [], 'initialMarkerTestFailure': [],
}
for name in ('run26-plan.json', 'dsbi-addition-protocol.json', 'dsbi-visible-partial-protocol.json'):
    report[name] = archive(run / name, Path('run26') / name)
for name, digest in config['source_sha256'].items():
    source = run / 'source' / name
    assert sha(source) == digest
    report['sources'].append(archive(source, Path('run26/source') / name))
tests = json.loads((run / 'preflight-tests.json').read_bytes())
assert tests['status'] == 'passed' and tests['exit_code'] == 0
assert tests['test_sources_unchanged_before_after'] is True
assert sha(run / 'preflight-tests.log') == tests['log_sha256']
report['preflightTestLog'] = archive(run / 'preflight-tests.log', Path('run26/preflight-tests.log'))
assert sha(run / 'horizontal-aspect-audit-source/tests.log') == audit['tests_log_sha256']
report['integrationTestLog'] = archive(run / 'horizontal-aspect-audit-source/tests.log', Path('run26/integration-tests.log'))
prototype = work / 'horizontal-aspect-prototype-v1'
assert sha(prototype / 'proof-report.json') == audit['prototype_report_sha256'] == protocol['private_prototype']['proof_sha256']
assert sha(prototype / 'index.json') == protocol['private_prototype']['index_sha256']
review = work / 'horizontal-aspect-independent-review-v1'
assert sha(review / 'report.json') == audit['independent_prototype_review_sha256']
for folder, target in ((run / 'initial-marker-test-failure', 'initial-marker-test-failure'),):
    for source in sorted(folder.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            entry = archive(source, Path(target) / source.relative_to(folder))
            key = 'initialMarkerTestFailure' if target == 'initial-marker-test-failure' else 'supportingProofs'
            report[key].append(entry)
for source in (Path(protocol['historical_audit']), helper, Path(__file__).resolve()):
    report['supportingProofs'].append(archive(source, Path('supporting-source') / source.name))
assert sha(Path(protocol['historical_audit'])) == protocol['historical_audit_sha256']
report['initialMarkerTestFailureNote'] = 'The first marker measurement region clipped the probe Gaussian near the image border. The probe was moved into the interior; augmentation code was not altered for the failure. Original failing test and log remain separate from the passing 141-test prelaunch suite.'
target = output / 'preflight.json'
payload = (json.dumps(report, indent=2) + '\n').encode('utf-8')
if target.exists() and target.read_bytes() != payload:
    raise ValueError('Refusing to replace a changed preflight report')
target.write_bytes(payload)
files = [{'file': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
         for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'archive-index.json']
index = output / 'archive-index.json'
index.write_text(json.dumps({'scope': report['scope'], 'files': files}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'report_sha256': sha(target), 'archive_sha256': sha(index), 'files': len(files), 'bytes': sum(row['bytes'] for row in files)}))
