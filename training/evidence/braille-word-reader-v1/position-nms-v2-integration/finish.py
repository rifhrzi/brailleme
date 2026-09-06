"""Freeze actual decoder integration checks without executing a model."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WORK = OUT.parent
PRIVATE = WORK / 'position-nms-prototype-v1'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return {'file': str(Path(path).relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(path)}
assert not (OUT / 'report.json').exists()
tests = json.loads((OUT / 'test-execution.json').read_text())
assert tests['source_unchanged_after_tests']
assert all(row['exit_code'] == 0 for row in tests['executions'].values())
for name, digest in tests['source_sha256'].items():
    assert sha(ROOT / name) == digest
assert 'Ran 52 tests' in (OUT / 'python.log').read_text()
assert '57 passed, 57 total' in (OUT / 'jest.log').read_text()
python_path = 'training/scripts/word_decode.py'
typescript_path = 'src/ml/word-decoder.ts'
actual = ast.parse((ROOT / python_path).read_text())
private = ast.parse((PRIVATE / 'source-private' / python_path).read_text())
version = [node for node in actual.body if isinstance(node, ast.Assign) and
           any(isinstance(target, ast.Name) and target.id == 'DECODER_VERSION' for target in node.targets)]
assert len(version) == 1 and ast.literal_eval(version[0].value) == 'six-dot-word-v2'
actual.body.remove(version[0])
assert ast.dump(actual, include_attributes=False) == ast.dump(private, include_attributes=False)
assert (ROOT / typescript_path).read_bytes() == (PRIVATE / 'source-private' / typescript_path).read_bytes()
spec = json.loads((PRIVATE / 'prototype-spec.json').read_text())
for name in ['training/scripts/word_data.py', 'training/scripts/word_model.py',
             'training/scripts/word_anchor_zoom.py', 'training/scripts/benchmark_angelina.py']:
    assert sha(ROOT / name) == spec['source_original_sha256'][name]
fixture = WORK / 'data/validation-stress-selection.json'
assert sha(fixture) == 'e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625'
report = {
    'schema_version': 1, 'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'decoder_version': 'six-dot-word-v2', 'source_sha256': tests['source_sha256'],
    'changes': ['Remove local 3x3 grid suppression before existing positional IoU NMS in both runtimes.',
                'Calibration caches retain every finite raw proposal at or above minimum score; old peak-filtered caches cannot be reused.',
                'Test-only ideal predictions score exact supervised centers; source Gaussian loss targets are preserved.'],
    'checks': {'python_actual_tests': 52, 'jest_actual_tests': 57,
               'python_semantics_exact_private_plus_version_constant': True,
               'typescript_byte_exact_private': True,
               'unchanged_training_data_model_loss_anchor_mapping_sources': True,
               'frozen_405_manifest_unchanged': True,
               'golden_sparse_17_case_fixture_unchanged': True,
               'test_source_unchanged_during_execution': True},
    'proofs': {'actual_test_execution': ref(OUT / 'test-execution.json'),
               'private_specification': ref(PRIVATE / 'prototype-spec.json'),
               'private_python_20_cases_1600_cache_trials': ref(PRIVATE / 'python-test-report.json'),
               'private_typescript_20_case_parity': ref(PRIVATE / 'typescript-test-report.json'),
               'fixed_raw405_comparison': ref(PRIVATE / 'held405-comparison/report.json'),
               'independent_private_source_review': ref(WORK / 'position-nms-independent-source-review-v1/report.json')},
    'limits': ['Correctness fix, not an accuracy claim. Prior fixed raw405 comparison changed 357 correct to 355 and accepted 362 to 359.',
               'No model execution, new calibration, image/label modification, release gate change or checkpoint promotion in this integration.',
               'Existing golden fixture remains historical and was executed unchanged; final model/runtime parity still belongs to subsequent selected artifacts.'],
    'script_sha256': sha(__file__),
}
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
index = {'files': [{'file': str(path.relative_to(OUT)).replace('\\', '/'), 'sha256': sha(path), 'bytes': path.stat().st_size}
                   for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'index.json']}
(OUT / 'index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'report': ref(OUT / 'report.json'), 'index': ref(OUT / 'index.json'),
                  'files': len(index['files']), 'bytes': sum(row['bytes'] for row in index['files'])}, indent=2))
